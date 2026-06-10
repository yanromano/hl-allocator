"""automation.scripts.position_health — standalone READ-ONLY position-health monitor.

A SAFETY/observability tool the operator runs by hand, INDEPENDENT of the trading
daemon.  It gathers an account/venue health snapshot, runs the pure
:func:`automation.monitoring.position_health.assess` checks, prints a clear
per-check report, and routes every finding to the alert sink (so the webhook
fires).  It is **READ-ONLY**: it NEVER signs, places, or closes an order.

    python -m automation.scripts.position_health --config automation.yaml

Design — mirror ``leverage_guard``
----------------------------------
* ``main(argv, *, client=None)`` is the injectable seam: tests pass a fake
  duck-typed client; production builds a real ``HLClient`` via ``_build_client``.
* ``_build_client`` mirrors ``leverage_guard._build_client`` EXACTLY (keystore →
  ``HLClient.for_trading``).  HL has no read-only constructor, and ``for_trading``
  acquires the nonce-file lock; this tool only READS, but ``close()`` in the
  ``finally`` releases that lock.  The agent key never leaves the local scope.
* The alert sink is built from the SAME composition root the daemon uses
  (``run_daemon._build_alert_sink``): ``LogAlertSink`` by default, or
  ``MultiSink(Log + Webhook)`` when ``cfg.alert_webhook_url`` is set.

Safety invariants
-----------------
* SIGNS NOTHING — no ``update_leverage`` / ``market_close`` / ``submit_ioc``.
* ``cfg.health.enabled`` False → prints "health monitor disabled", emits NOTHING,
  exits 0.
* The client is ALWAYS closed in a ``finally`` (releases the nonce-file lock).

NEVER print or log the agent key / passphrase — ``get_logger`` redacts and the
key never leaves ``_build_client``'s local scope.
"""

from __future__ import annotations

import argparse
from typing import Any

from automation.core.config import Config
from automation.core.hl_client import HLClient
from automation.core.redaction import get_logger
from automation.core.secrets import KeystoreKeyProvider
from automation.monitoring.position_health import (
    HealthFinding,
    HealthSnapshot,
    assess,
    gather_health_snapshot,
)
from automation.reporting.alerts import Alert, AlertSink
from automation.scripts.run_daemon import (
    _build_alert_sink,
    _build_passphrase_backend,
)

_logger = get_logger(__name__)


# ===========================================================================
# IO — build the client (mirrors leverage_guard._build_client) + main
# ===========================================================================
def _build_client(cfg: Config) -> HLClient:
    """Construct a real trading ``HLClient`` from *cfg* (master-direct, keystore).

    Mirrors ``leverage_guard._build_client`` EXACTLY.  HL exposes no read-only
    constructor; ``for_trading`` acquires the nonce-file lock.  This tool only
    READS, so the lock is incidental — ``main``'s ``finally`` releases it via
    ``client.close()``.  The agent key lives ONLY in the local ``key`` variable —
    never logged, stored, or returned.  Monkeypatch seam for tests (or inject a
    fake client via ``main(..., client=...)``).
    """
    if not cfg.agent_keystore_path:
        raise ValueError(
            "no client injected and `agent_keystore_path` is unset — cannot load "
            "the agent key; set agent_keystore_path (and the passphrase backend) "
            "in the config, or inject a client (tests)."
        )
    backend = _build_passphrase_backend(cfg)
    provider = KeystoreKeyProvider(cfg.agent_keystore_path, backend, cfg.env)
    key = provider.load_agent_key()
    route_via_vault = cfg.trade_mode == "subaccount"
    client = HLClient.for_trading(
        cfg.env,
        cfg.subaccount_address,
        key,
        cfg.nonce_state_path,
        enable_spot=bool(cfg.spot_routing),
        route_via_vault=route_via_vault,
    )
    del key
    return client


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="position_health",
        description=(
            "Standalone READ-ONLY position-health monitor. Gathers an "
            "account/venue snapshot, runs the health checks (funding drag, "
            "maintenance margin / liquidation proximity, venue degradation, "
            "idle spot collateral), prints a per-check report, and routes every "
            "finding to the alert sink. SIGNS NOTHING."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the YAML config")
    return parser


def _print_report(snap: HealthSnapshot, findings: list[HealthFinding]) -> None:
    """Print a clear per-check report + every finding (no secrets)."""
    print("=== position_health [READ-ONLY — SIGNS NOTHING] ===")
    print(f"  equity         : {snap.equity:.2f}")

    # --- funding ---
    total_funding = sum(p.funding_since_open for p in snap.positions)
    funding_pct = (total_funding / snap.equity * 100) if snap.equity > 0 else 0.0
    print(f"  funding total  : {total_funding:.2f} ({funding_pct:.2f}% of equity)")

    # --- margin ---
    maint_frac = (snap.maint_margin / snap.equity) if snap.equity > 0 else 0.0
    print(
        f"  margin ratio   : maint_margin={snap.maint_margin:.2f} "
        f"frac={maint_frac:.3f}"
    )

    # --- venue-degraded coins ---
    degraded = [
        p.coin
        for p in snap.positions
        if p.coin not in snap.venue_coins
        or snap.venue_coins[p.coin].get("delisted")
    ]
    print(f"  venue-degraded : {degraded if degraded else 'none'}")

    # --- spot collateral ---
    print(f"  spot USDC      : {snap.spot_usdc:.2f}")

    # --- findings ---
    if not findings:
        print("  findings       : none (all checks pass)")
    else:
        print(f"  findings       : {len(findings)}")
        for f in findings:
            print(f"    [{f.severity.value}] {f.alert_type.value}: {f.message}")


def _route_findings(findings: list[HealthFinding], sink: AlertSink) -> None:
    """Emit every finding to the alert sink (so the webhook fires)."""
    for f in findings:
        sink.emit(
            Alert(
                type=f.alert_type,
                severity=f.severity,
                message=f.message,
                context=f.context,
            )
        )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    """CLI entry point.  Returns a process exit code (0 = success).

    ``client`` is an injectable seam for tests (a fake duck-typed client); when
    ``None`` a real ``HLClient`` is built from the keystore via ``_build_client``.

    Flow
    ----
    1. Load config.  If ``cfg.health.enabled`` is False → print "health monitor
       disabled", emit NOTHING, exit 0 (no client built).
    2. Build the client (READ-ONLY use), gather the health snapshot, run the
       pure ``assess`` checks.
    3. Print the per-check report + each finding; route every finding to the
       alert sink (LogAlertSink, or MultiSink(Log+Webhook) when configured).
    4. finally: ``client.close()`` (releases the nonce-file lock).
    """
    args = _build_parser().parse_args(argv)
    cfg = Config.from_yaml(args.config)

    if not cfg.health.enabled:
        print("health monitor disabled (cfg.health.enabled is False) — nothing to do.")
        return 0

    if client is None:
        client = _build_client(cfg)

    try:
        snap = gather_health_snapshot(client)
        findings = assess(snap, cfg)
        _print_report(snap, findings)

        sink = _build_alert_sink(cfg)
        _route_findings(findings, sink)

        print(f"\n=== summary === findings={len(findings)} (routed to alert sink)")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
