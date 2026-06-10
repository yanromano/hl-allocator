"""automation.scripts.leverage_guard — ON-DEMAND leverage safety cap (standalone).

A SAFETY tool the operator runs by hand, INDEPENDENT of the trading daemon, to
ensure NO open position carries a leverage VALUE above the target.  Capping the
leverage VALUE of an open position WITHIN THE SAME MARGIN MODE is allowed by
Hyperliquid and is NOT a trade — it is a pure margin-setting, reduce-risk action
(no exposure change, no spread, no fees).  This is the only thing the tool does
by default.

Switching the margin TYPE (cross↔isolated) on an open position is REFUSED by HL
(it requires flattening first), so it is OUT OF SCOPE by default: the tool merely
REPORTS type mismatches.  ONLY the explicit ``--enforce-type`` flag will flatten a
mode-wrong position (reduce-only ``market_close``) so the daemon reopens it at the
correct mode on its next cycle.

    python -m automation.scripts.leverage_guard --config automation.yaml            # ENFORCE values
    python -m automation.scripts.leverage_guard --config automation.yaml --check     # READ-ONLY (signs nothing)
    python -m automation.scripts.leverage_guard --config automation.yaml --max-leverage 2
    python -m automation.scripts.leverage_guard --config automation.yaml --enforce-type  # ALSO flatten mode-wrong

Design — the PURE planner is separated from the IO
--------------------------------------------------
``plan_leverage`` is a pure function (no client, no config, no network) that
classifies the held leverage view into ``to_lower`` / ``type_mismatch`` / ``ok``.
It is exhaustively unit-tested.  ``main`` is the thin IO shell: build the client,
read ``positions_leverage()``, call the planner, print the report, and (unless
``--check``) execute the plan.

Safety invariants
-----------------
* ``--check`` signs NOTHING (read-only).
* ``to_lower`` KEEPS each position's CURRENT margin mode (``is_cross`` read from
  ``held``) — it ONLY lowers the VALUE.  It NEVER switches mode (HL would reject
  it anyway; we are explicit).
* Flatten happens ONLY with ``--enforce-type`` and ONLY via the reduce-only
  ``market_close`` primitive — it can never OPEN or flip a position.
* The client is ALWAYS closed in a ``finally`` (releases the nonce-file lock).

NEVER print or log the agent key / passphrase — ``get_logger`` redacts and the
key never leaves ``_build_client``'s local scope.
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass, field
from typing import Any

from automation.core.config import Config
from automation.core.hl_client import HLClient
from automation.core.redaction import get_logger
from automation.core.secrets import KeystoreKeyProvider
from automation.execution.executor import make_cloid
from automation.scripts.run_daemon import _build_passphrase_backend

_logger = get_logger(__name__)


# ===========================================================================
# PURE PLANNER — the testable core (no IO)
# ===========================================================================
@dataclass(frozen=True)
class LeveragePlan:
    """Classification of every OPEN position against the target leverage.

    * ``to_lower``      — coins whose VALUE > target ⇒ cap the VALUE in place
                          (``update_leverage`` keeping the CURRENT mode).
    * ``type_mismatch`` — coins whose margin MODE != target mode (regardless of
                          value).  These need a flatten to converge the mode and
                          are only REPORTED unless ``--enforce-type`` is set.
    * ``ok``            — coins already at/below the target VALUE AND in the
                          correct mode.

    A coin can appear in BOTH ``to_lower`` and ``type_mismatch`` (over-value AND
    mode-wrong): we cap its VALUE in place to remove the immediate over-leverage
    risk, AND list it as a type mismatch so the operator knows the MODE still
    needs the boot flatten.
    """

    to_lower: list[str] = field(default_factory=list)
    type_mismatch: list[str] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)


def plan_leverage(
    held: dict[str, dict[str, Any]],
    *,
    target_leverage: int,
    target_is_cross: bool,
) -> LeveragePlan:
    """Classify each OPEN position in *held* (= ``{coin: {"type","value"}}``).

    Rule (applied per coin, independently):

    * ``value > target_leverage``                 → ``to_lower``  (cap VALUE in place)
    * ``type != target mode`` (regardless of value) → ``type_mismatch``
    * ``value <= target_leverage`` AND ``type == target mode`` → ``ok``

    The first two are NOT mutually exclusive: a coin that is BOTH over-value and
    mode-wrong lands in BOTH ``to_lower`` and ``type_mismatch``.  Such a coin is
    NEVER also in ``ok``.  A coin is in ``ok`` ONLY when it is neither over-value
    nor mode-wrong.

    Pure: no IO, no mutation of *held*.  Order follows ``held`` iteration order.
    """
    target_type = "cross" if target_is_cross else "isolated"
    to_lower: list[str] = []
    type_mismatch: list[str] = []
    ok: list[str] = []
    for coin, lev in held.items():
        over_value = int(lev["value"]) > target_leverage
        mode_wrong = lev["type"] != target_type
        if over_value:
            to_lower.append(coin)
        if mode_wrong:
            type_mismatch.append(coin)
        if not over_value and not mode_wrong:
            ok.append(coin)
    return LeveragePlan(to_lower=to_lower, type_mismatch=type_mismatch, ok=ok)


# ===========================================================================
# IO — build the client (mirrors run_daemon.build_daemon Step 1) + main
# ===========================================================================
def _build_client(cfg: Config) -> HLClient:
    """Construct a real trading ``HLClient`` from *cfg* (master-direct, keystore).

    Mirrors ``run_daemon.build_daemon`` Step 1 EXACTLY: passphrase backend from
    config → ``KeystoreKeyProvider.load_agent_key`` → ``HLClient.for_trading`` with
    perp-only spot routing and vault routing keyed off ``trade_mode``.

    The agent key lives ONLY in the local ``key`` variable — never logged, stored,
    or returned.  This function is the monkeypatch seam tests replace (or tests
    inject a fake client directly via ``main(..., client=...)``).
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
        prog="leverage_guard",
        description=(
            "ON-DEMAND leverage safety cap. Ensures no OPEN position carries a "
            "leverage VALUE above the target by capping it IN PLACE (same margin "
            "mode — NOT a trade). Default ENFORCES values; --check is read-only; "
            "--enforce-type additionally FLATTENS mode-mismatched positions "
            "(reduce-only) so they reopen at the correct mode."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument(
        "--check",
        action="store_true",
        help="READ-ONLY: print the plan, sign NOTHING, exit 0.",
    )
    parser.add_argument(
        "--max-leverage",
        type=int,
        default=None,
        help=(
            "Target max leverage VALUE (int). Defaults to cfg.target_leverage. "
            "Any open position above this is capped IN PLACE."
        ),
    )
    parser.add_argument(
        "--enforce-type",
        action="store_true",
        help=(
            "ALSO flatten margin-TYPE mismatches (cross↔isolated) via reduce-only "
            "market_close so the daemon reopens them at the target mode next cycle. "
            "LOUD. Default OFF (type mismatches are only REPORTED)."
        ),
    )
    return parser


def _print_report(
    held: dict[str, dict[str, Any]],
    plan: LeveragePlan,
    *,
    max_leverage: int,
    target_is_cross: bool,
    check: bool,
    enforce_type: bool,
) -> None:
    """Print a clear per-coin report (no secrets)."""
    target_type = "cross" if target_is_cross else "isolated"
    mode = "CHECK (read-only — SIGNS NOTHING)" if check else "ENFORCE"
    print(f"=== leverage_guard [{mode}] ===")
    print(
        f"  target: max_leverage={max_leverage} mode={target_type} "
        f"enforce_type={enforce_type}"
    )
    to_lower = set(plan.to_lower)
    type_mismatch = set(plan.type_mismatch)
    if not held:
        print("  no open positions — nothing to do.")
    for coin, lev in held.items():
        actions: list[str] = []
        if coin in to_lower:
            actions.append(f"LOWER value -> {max_leverage} (keep {lev['type']})")
        if coin in type_mismatch:
            actions.append(
                "FLATTEN (mode mismatch)"
                if enforce_type
                else "mode mismatch (REPORT ONLY)"
            )
        if not actions:
            actions.append("ok")
        print(
            f"  {coin:<6} current={lev['type']}/{lev['value']}x  ->  "
            f"{'; '.join(actions)}"
        )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    """CLI entry point.  Returns a process exit code (0 = success).

    ``client`` is an injectable seam for tests (a fake duck-typed client); when
    ``None`` a real ``HLClient`` is built from the keystore via ``_build_client``.

    Flags
    -----
    --config PATH    (required) — YAML config (Config.from_yaml).
    --check          READ-ONLY: print the plan, sign NOTHING, exit 0.
    --max-leverage N target max VALUE (default cfg.target_leverage).
    --enforce-type   ALSO flatten mode-mismatched positions (reduce-only).

    Always closes the client in a ``finally`` (releases the nonce-file lock).
    """
    args = _build_parser().parse_args(argv)

    cfg = Config.from_yaml(args.config)
    max_leverage = args.max_leverage if args.max_leverage is not None else cfg.target_leverage
    target_is_cross = cfg.leverage_is_cross

    if client is None:
        client = _build_client(cfg)

    n_lowered = 0
    n_flattened = 0
    try:
        held = client.positions_leverage()  # {coin: {"type","value"}}
        plan = plan_leverage(
            held, target_leverage=max_leverage, target_is_cross=target_is_cross
        )
        _print_report(
            held,
            plan,
            max_leverage=max_leverage,
            target_is_cross=target_is_cross,
            check=args.check,
            enforce_type=args.enforce_type,
        )

        if args.check:
            # READ-ONLY — sign nothing.
            print(
                "\n--check: SIGNED NOTHING. "
                f"(would lower {len(plan.to_lower)}, "
                f"type_mismatch {len(plan.type_mismatch)}, ok {len(plan.ok)})"
            )
            return 0

        now_ts = datetime.datetime.now(datetime.UTC).isoformat()

        # --- 1) Cap over-value positions IN PLACE (keep CURRENT mode) ----------
        for coin in plan.to_lower:
            current_is_cross = held[coin]["type"] == "cross"
            try:
                client.update_leverage(
                    coin, max_leverage, is_cross=current_is_cross
                )
                n_lowered += 1
                _logger.info(
                    "leverage_guard: capped leverage value in place",
                    coin=coin,
                    new_value=max_leverage,
                    mode=held[coin]["type"],
                )
            except Exception as exc:  # noqa: BLE001 — isolate per-coin failures
                _logger.error(
                    "leverage_guard: update_leverage failed (continuing)",
                    coin=coin,
                    error=str(exc),
                )

        # --- 2) Flatten mode mismatches ONLY with --enforce-type ---------------
        if args.enforce_type:
            for coin in plan.type_mismatch:
                try:
                    cloid = make_cloid(cfg.strategy_id, "lev-guard", coin, 0, now_ts)
                    client.market_close(coin, cfg.caps.max_order_slippage, cloid)
                    n_flattened += 1
                    _logger.warning(
                        "leverage_guard: FLATTENED mode-mismatched position "
                        "(reduce-only) — daemon reopens at target mode next cycle",
                        coin=coin,
                        current_mode=held[coin]["type"],
                    )
                except Exception as exc:  # noqa: BLE001 — isolate per-coin failures
                    _logger.error(
                        "leverage_guard: market_close (flatten) failed (continuing)",
                        coin=coin,
                        error=str(exc),
                    )
        elif plan.type_mismatch:
            _logger.warning(
                "leverage_guard: margin-TYPE mismatch(es) REPORTED but NOT enforced "
                "(pass --enforce-type to flatten and reopen at the target mode)",
                coins=plan.type_mismatch,
            )

        print(
            f"\n=== summary === lowered={n_lowered} "
            f"type_mismatch={len(plan.type_mismatch)} "
            f"flattened={n_flattened} ok={len(plan.ok)}"
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
