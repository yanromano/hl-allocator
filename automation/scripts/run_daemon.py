"""automation.scripts.run_daemon — the E1 COMPOSITION ROOT (deploy entry point).

This is the single deploy blocker resolved: the one runnable command that
assembles every building block into a live daemon.  Every block below is
unit-tested in isolation; this module is the place they are finally WIRED:

    Config.from_yaml
        → KeystoreKeyProvider.load_agent_key   (passphrase backend from config)
        → HLClient.for_trading(enable_spot=...) (perp-only ⇒ enable_spot=False)
        → build_source(cfg)                     (generic, dispatched on signal_source.type)
        → IntentLedger / State / AuditLog / AlertSink
        → Daemon(...)  →  boot()  →  run_once()/run_forever()  →  shutdown()

Run it::

    python -m automation.scripts.run_daemon --config automation.yaml            # forever
    python -m automation.scripts.run_daemon --config automation.yaml --once      # one cycle
    python -m automation.scripts.run_daemon --config automation.yaml --once --dry-run  # SENDS NOTHING

``--dry-run`` runs the full snapshot → validate → size → plan pipeline via
``dry_run_cycle`` and prints the plan; it NEVER submits an order, never boots,
never mutates state.  Use it for config / connectivity sanity before a live run.

The strategy-free invariant (no closed-strategy import)
-------------------------------------------------------
``automation`` (the OPEN, strategy-agnostic executor package) must NEVER import a
closed strategy module — and neither does this deploy script.  ``build_daemon``
builds its allocation source GENERICALLY via
``automation.allocation.source_factory.build_source`` (dispatched on
``cfg.signal_source.type`` in {remote, equal_weight, single_asset}), so this
module names NO closed strategy at all.

The operator's closed-strategy path lives on the CLOSED side: a sibling
composition root builds the closed allocation source and INJECTS it via
``build_daemon(cfg, source=...)``.  That closed root is the one place allowed to
import BOTH the closed strategy AND ``automation.*``.

NEVER print or log secrets — the redacted ``get_logger`` is the backstop and the
key never leaves ``build_daemon``'s local scope.
"""

from __future__ import annotations

import argparse
import datetime
import signal
import time
from collections.abc import Callable
from typing import Any

from automation.core.config import Config
from automation.core.hl_client import HLClient
from automation.core.redaction import get_logger
from automation.core.secrets import (
    EnvBackend,
    KeyProvider,
    KeystoreKeyProvider,
    OsKeyringBackend,
    PassphraseBackend,
    PromptBackend,
)
from automation.reporting import state as state_mod
from automation.reporting.alerts import (
    AlertSink,
    LogAlertSink,
    MultiSink,
    WebhookAlertSink,
)
from automation.reporting.audit import AuditLog
from automation.runtime.daemon import Daemon
from automation.runtime.dry_run import dry_run_cycle
from automation.runtime.health import Heartbeat, start_health_server
from automation.runtime.intent_ledger import Intent, IntentLedger

_logger = get_logger(__name__)


def make_chain_resolver(client: Any) -> Callable[[Intent], bool]:
    """Resolve a PENDING intent against the chain: True iff its cloid shows a fill.

    Queries ``client.query_by_cloid(intent.cloid)`` (which wraps the SDK
    ``info.query_order_by_cloid`` ⇒ the HL ``orderStatus`` info endpoint) and
    interprets the response as filled ONLY when the order reached the terminal
    ``"filled"`` state.

    HL ``orderStatus`` response shape
    ---------------------------------
    * order found:   ``{"status": "order",
                        "order": {"order": {...}, "status": "filled"|"open"|"canceled"|...}}``
                     — the INNER ``order["status"]`` is the order lifecycle state;
                     ``"filled"`` is the terminal fill.
    * order absent:  ``{"status": "unknownOid"}`` — not found ⇒ NOT filled.

    Conservative on uncertainty (Layer-4 hygiene, not safety): on ANY query error
    OR an unrecognised / malformed response, return ``False`` — leave the intent
    PENDING, never falsely mark it DONE.  (Layer-1 live-position reconciliation is
    what prevents double-orders; this resolver only drains orphan PENDINGs whose
    order actually filled.)
    """

    def _resolver(intent: Intent) -> bool:
        try:
            resp = client.query_by_cloid(intent.cloid)
        except Exception:  # noqa: BLE001 — any query failure ⇒ stay PENDING
            return False
        if not isinstance(resp, dict):
            return False
        # Top-level must report a concrete order (not "unknownOid"/etc).
        if resp.get("status") != "order":
            return False
        order = resp.get("order")
        if not isinstance(order, dict):
            return False
        # The order's lifecycle status; "filled" is the terminal fill.
        return order.get("status") == "filled"

    return _resolver


def _build_passphrase_backend(cfg: Config) -> PassphraseBackend:
    """Construct the configured :class:`PassphraseBackend` from *cfg*.

    Raises a clear ``ValueError`` when the chosen backend is missing its
    required companion field (e.g. ``env`` without ``passphrase_env_var``).
    NEVER reads or logs the passphrase itself.
    """
    backend = cfg.passphrase_backend
    if backend == "os_keyring":
        if not cfg.keyring_service or not cfg.keyring_username:
            raise ValueError(
                "passphrase_backend='os_keyring' requires both `keyring_service` and "
                "`keyring_username` to be set in the config."
            )
        return OsKeyringBackend(cfg.keyring_service, cfg.keyring_username)
    if backend == "prompt":
        return PromptBackend()
    if backend == "env":
        if not cfg.passphrase_env_var:
            raise ValueError(
                "passphrase_backend='env' requires `passphrase_env_var` to name the "
                "environment variable holding the keystore passphrase."
            )
        return EnvBackend(cfg.passphrase_env_var)
    # Unreachable: pydantic Literal already constrains the value.
    raise ValueError(f"unknown passphrase_backend: {backend!r}")


def _build_alert_sink(cfg: Config) -> AlertSink:
    """LogAlertSink by default; MultiSink(Log + Webhook) when a webhook is set."""
    if cfg.alert_webhook_url:
        return MultiSink([LogAlertSink(), WebhookAlertSink(cfg.alert_webhook_url)])
    return LogAlertSink()


def build_daemon(
    cfg: Config,
    *,
    source: Any = None,
    now_fn: Any = None,
    client: Any = None,
) -> Daemon:
    """Assemble a wired :class:`Daemon` from *cfg* — the testable factory.

    All network / strategy work is INJECTABLE so tests stay offline:

    * ``client`` — when ``None``, the agent key is loaded from the configured V3
      keystore and a real ``HLClient.for_trading`` is constructed.  Inject a fake
      to skip both the keystore and the network handshake.
    * ``source`` — the :class:`AllocationSource`.  When ``None`` it is built
      GENERICALLY from ``cfg.signal_source.type`` via
      ``automation.allocation.source_factory.build_source`` (NO closed-strategy
      import).  The operator's closed-strategy path injects its allocation source
      here from the sibling closed composition root.
    * ``now_fn`` — clock forwarded to the generic source factory (e.g. the remote
      source's freshness gate).

    Wiring sequence
    ---------------
    1. (client is None) load key from keystore → ``HLClient.for_trading`` with
       ``enable_spot = bool(cfg.spot_routing)`` (perp-only ⇒ False).
    2. ``source or build_source(cfg, now_fn=now_fn)`` (generic, strategy-free).
    3. ``IntentLedger`` / ``State`` / ``AuditLog`` / alert sink at the cfg paths.
    4. ``Daemon(cfg, client, source, ledger, state, state_path, audit, alert_sink,
       strategy_id, kill_flag_path)``.

    The agent key is loaded into a LOCAL variable, passed to ``for_trading``, and
    never logged, stored, or returned.
    """
    # --- Step 1: client (load agent key from keystore unless injected) --------
    if client is None:
        if cfg.key_source.type == "env_raw":
            from automation.core.secrets import RawEnvKeyProvider

            provider: KeyProvider = RawEnvKeyProvider(cfg.key_source.env_var, cfg.env)
        else:
            if not cfg.agent_keystore_path:
                raise ValueError(
                    "no client injected and `agent_keystore_path` is unset — cannot load "
                    "the agent key; set agent_keystore_path (and the passphrase backend) "
                    "in the config, or inject a client (tests / dry-run)."
                )
            backend = _build_passphrase_backend(cfg)
            provider = KeystoreKeyProvider(cfg.agent_keystore_path, backend, cfg.env)
        # `key` is the ONLY place the secret lives — never logged, never returned.
        key = provider.load_agent_key()
        # trade_mode → routing: subaccount ⇒ vault routing; master_direct ⇒
        # account_address (no vault).  Default 'subaccount' keeps legacy wiring.
        route_via_vault = cfg.trade_mode == "subaccount"
        client = HLClient.for_trading(
            cfg.env,
            cfg.subaccount_address,
            key,
            cfg.nonce_state_path,
            enable_spot=bool(cfg.spot_routing),
            route_via_vault=route_via_vault,
        )
        # Drop the local reference promptly (for_trading has handed it to the SDK).
        del key

    # --- Step 2: allocation source (generic, strategy-free; injectable) -------
    # When no source is injected, build it from config by dispatching on
    # ``cfg.signal_source.type`` — NO closed-strategy import.  The operator's
    # closed-strategy path injects its source from the sibling closed root.
    if source is None:
        from automation.allocation.source_factory import build_source  # noqa: PLC0415

        source = build_source(cfg, now_fn=now_fn)

    # --- Step 3: ledger / state / audit / alert sink at the cfg paths ---------
    ledger = IntentLedger(cfg.ledger_path)
    state = state_mod.load(cfg.state_path)
    audit = AuditLog(cfg.audit_path)
    alert_sink = _build_alert_sink(cfg)

    # --- Step 4: Daemon -------------------------------------------------------
    return Daemon(
        cfg,
        client,
        source,
        ledger,
        state,
        cfg.state_path,
        audit,
        alert_sink,
        strategy_id=cfg.strategy_id,
        kill_flag_path=cfg.kill_flag_path,
        # Layer 4: resolve orphan PENDINGs against the chain (boot reconcile +
        # per-cycle).  Without this the resolver defaults to lambda _: False and
        # orphan PENDINGs accumulate (ledger drift).  See make_chain_resolver.
        is_filled=make_chain_resolver(client),
    )


def _print_plan(plan: Any) -> None:
    """Print a compact dry-run plan summary to stdout (NO secrets)."""
    print("=== DRY RUN plan (SENDS NOTHING) ===")
    print(f"  rebalance     : {plan.rebalance}")
    print(f"  reason        : {plan.reason}")
    print(f"  n_orders      : {len(plan.orders)}")
    print(f"  gross_notional: {round(plan.gross_notional, 2)}")
    print(f"  scaled_down   : {plan.scaled_down}")
    print(f"  n_skipped     : {len(plan.skipped)}")
    print(f"  equity        : {round(plan.equity, 2)}")
    for op in plan.orders:
        # SizePlan order ops carry only coin/side/size/notional — no secrets.
        print(f"    order: {op}")


def build_arg_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Shared ``--config / --once / --dry-run`` parser (generic + closed roots)."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single rebalance cycle then exit (boot → run_once → shutdown)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the planning pipeline read-only (SENDS NOTHING) and print the plan",
    )
    return parser


def _serve(daemon: Daemon, cfg: Config, args: argparse.Namespace) -> int:
    """Drive an already-built ``daemon`` per ``args`` (dry-run / once / forever).

    Shared boot/run loop used by BOTH ``run_daemon.main`` (generic) and the
    sibling closed-strategy root's ``main`` so the run-forever body lives in ONE place.

    * ``--dry-run`` → ``dry_run_cycle`` (SENDS NOTHING), print the plan, shutdown.
    * ``--once``    → boot → run_once → shutdown.
    * default       → install SIGTERM/SIGINT handlers, boot, run_forever, shutdown.

    Always ``shutdown`` in a finally block.  Returns a process exit code.
    """
    # --- DRY RUN: plan only, submit nothing, no boot/state mutation -----------
    if args.dry_run:
        plan = dry_run_cycle(cfg, daemon.client, daemon.source, daemon.state)
        _print_plan(plan)
        # Release any held resources (nonce lock) without booting/trading.
        daemon.shutdown()
        return 0

    now = datetime.datetime.now(datetime.UTC)
    now_ts = now.isoformat()

    # --- SINGLE CYCLE ---------------------------------------------------------
    if args.once:
        try:
            daemon.boot(now_ts)
            daemon.run_once(now, now_ts)
        finally:
            daemon.shutdown()
        return 0

    # --- FOREVER --------------------------------------------------------------
    stop_requested = {"flag": False}

    def _request_stop(signum: int, _frame: Any) -> None:
        _logger.info("run_daemon: stop signal received", signum=signum)
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    # --- Liveness: /healthz heartbeat (FOREVER path only; not --once/--dry-run).
    # When position-health monitoring is enabled, attach a Heartbeat the daemon
    # beats every tick and start a tiny HTTP server so a supervisor (Railway)
    # can detect a STUCK daemon (no beats → /healthz 503) — not just a crashed
    # process.  --once / --dry-run return above, so neither starts a server.
    if cfg.health.enabled:
        hb = Heartbeat(cfg.health.heartbeat_max_silence_seconds)
        daemon.heartbeat = hb
        start_health_server(
            hb,
            port=cfg.health.http_port,
            clock=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        )
        _logger.info("run_daemon: /healthz liveness server started", port=cfg.health.http_port)

    try:
        daemon.boot(now_ts)
        daemon.run_forever(
            clock=lambda: datetime.datetime.now(datetime.UTC),
            sleep=time.sleep,
            stop=lambda: stop_requested["flag"],
        )
    finally:
        daemon.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code (0 = success).

    Flags
    -----
    --config PATH   (required) — YAML config (Config.from_yaml).
    --once          run ONE cycle (boot → run_once → shutdown) then exit.
    --dry-run       run ``dry_run_cycle`` (SENDS NOTHING), print the plan, exit.
                    No boot, no submit, no state mutation.  Implies single-shot.

    Default (no --once / --dry-run): boot, install SIGTERM/SIGINT handlers, then
    ``run_forever`` until signalled; always ``shutdown`` in a finally block.
    """
    parser = build_arg_parser(
        "run_daemon", "hl-allocator generic perp-only composition root."
    )
    args = parser.parse_args(argv)

    # --- Load + validate config ----------------------------------------------
    try:
        cfg = Config.from_yaml(args.config)
    except FileNotFoundError as exc:
        _logger.error("run_daemon: config file not found", error=str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 — surface validation errors as exit 2
        _logger.error("run_daemon: config failed to load/validate", error=str(exc))
        return 2

    _logger.info(
        "run_daemon: starting",
        env=cfg.env,
        strategy_id=cfg.strategy_id,
        signal_source_type=cfg.signal_source.type,
        once=args.once,
        dry_run=args.dry_run,
        perp_only=not bool(cfg.spot_routing),
    )

    # --- Build the daemon (loads the key unless dry-run injects nothing) ------
    daemon = build_daemon(cfg)
    return _serve(daemon, cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
