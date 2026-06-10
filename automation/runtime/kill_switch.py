"""
automation.runtime.kill_switch — 3-tier emergency position-flattening.

Tier 1 — In-process reduce-only flatten via the TRADING agent
--------------------------------------------------------------
``flatten_all`` iterates ``client.positions()`` and calls ``client.market_close``
(reduce-only, routed to the sub-account via ``vault_address``) for each coin
with a non-zero position.  Per-coin errors are isolated so a single-coin failure
never blocks the rest.  Every intent is written to the ledger BEFORE the
market_close call (MF-7 write-before-submit invariant).

``scheduleCancel`` is NOT used here — it only cancels RESTING orders and is
volume-gated; it is a hygiene-only operation, not a position-safety action.

Tier 2 — Independent-host watchdog (stub)
-----------------------------------------
``Watchdog`` holds a SEPARATE HLClient built with its OWN dedicated agent key
and checks heartbeat liveness / equity floor.  When triggered it delegates to
``flatten_all`` with the kill client.

CRITICAL — never two concurrent signers on the same agent key.
Two processes signing with the same agent/nonce set will collide: Hyperliquid
uses the nonce to prevent replay, so whichever process has the higher nonce
will invalidate the other's in-flight orders, and both may end up with
corrupted nonce state on restart.  The Watchdog MUST use a dedicated agent key
registered as a SEPARATE agent on the account (provisioned via task B8).

Tier 3 — Nuclear: master-signed agent revocation (runbook only)
---------------------------------------------------------------
``revoke_runbook`` returns a documented string.  It is NEVER executed by the
daemon; it is operator runbook text for the human emergency response.
"""

from __future__ import annotations

import contextlib
from typing import Any

from automation.core.redaction import get_logger
from automation.execution.executor import make_cloid
from automation.reporting.alerts import Alert, AlertSink, AlertType, Severity
from automation.runtime.intent_ledger import Intent, IntentLedger

_logger = get_logger(__name__)

# Filled sizes below this are treated as effectively zero (matches the
# executor's _FILL_EPS — fill precision can exceed sz_decimals).
_FILL_EPS: float = 1e-8


def _extract_filled_sz(statuses: list[dict[str, Any]]) -> float:
    """Sum ``totalSz`` across all ``filled`` status dicts (mirror of the
    executor's helper).  Returns 0.0 when nothing filled."""
    total = 0.0
    for st in statuses:
        if "filled" in st:
            raw = st["filled"].get("totalSz", "0")
            with contextlib.suppress(ValueError, TypeError):
                total += float(raw)
    return total


# ---------------------------------------------------------------------------
# Tier 1 — In-process reduce-only flatten
# ---------------------------------------------------------------------------


def flatten_all(
    client: Any,
    cfg: Any,
    ledger: IntentLedger,
    *,
    strategy_id: str,
    now_ts: str,
    alert_sink: AlertSink | None = None,
) -> list[dict[str, Any]]:
    """Reduce-only flatten of ALL non-zero positions via the TRADING agent.

    This is the AUTHORITATIVE position-safety action for Tier 1.  It:

    1. Reads ``client.positions()`` for all open positions.
    2. For each coin with ``szi != 0``: writes a PENDING intent to the ledger
       BEFORE the market_close call (MF-7), then calls ``client.market_close``
       (reduce-only — can NEVER flip into a new short), then marks the intent
       DONE.
    3. Per-coin exceptions are caught and recorded; remaining coins are still
       flattened (one failure MUST NOT stop the rest).
    4. CONFIRMS the close (F3): extracts the actual filled size from the
       response.  A ``filled`` status that fully covers the position →
       ``"filled"``; a partial → ``"partial"``; a ``resting`` / empty / zero-fill
       response → ``"resting"`` (NEVER mislabelled ``"filled"``).  After all
       coins, re-queries ``client.positions()`` and surfaces any residual
       non-zero szi as a CRITICAL alert (the kill silently half-worked).
    5. Emits a KILL_SWITCH alert via ``alert_sink`` (if provided).

    ``scheduleCancel`` is deliberately NOT used: it only cancels RESTING orders
    and is volume-gated.  It is hygiene-only and does not close positions.

    Parameters
    ----------
    client:
        Trading HLClient wired to the TRADING agent (vault_address = sub).
    cfg:
        Config; uses ``cfg.caps.max_order_slippage``.
    ledger:
        Intent ledger for write-before-submit (MF-7).
    strategy_id:
        Strategy identifier forwarded to ``make_cloid``.
    now_ts:
        ISO timestamp of this kill-switch invocation.
    alert_sink:
        Optional alert sink; receives a KILL_SWITCH CRITICAL alert.

    Returns
    -------
    list[dict]
        Per-coin outcome records with keys
        ``{coin, cloid, status, filled_size, requested_size, error}`` where
        ``status`` is one of ``"filled" | "partial" | "resting" | "error"``.
    """
    from automation.core.safety import parse_order_response  # noqa: PLC0415

    positions = client.positions()
    results: list[dict[str, Any]] = []

    for coin, szi in positions.items():
        # Ignore coins that are already flat.
        if szi == 0:
            continue

        cloid = make_cloid(strategy_id, f"KILL-{now_ts}", coin, 0, now_ts)

        # MF-7: write PENDING intent BEFORE the market_close call.
        intent = Intent(
            cloid=cloid,
            as_of=f"KILL-{now_ts}",
            coin=coin,
            target_w=0.0,
            attempt_seq=0,
            status="PENDING",
            ts=now_ts,
        )
        requested = abs(float(szi))
        try:
            ledger.record_pending(intent)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "kill_switch.flatten_all: ledger.record_pending failed",
                coin=coin,
                cloid=cloid,
                error=str(exc),
            )
            results.append(
                {
                    "coin": coin,
                    "cloid": cloid,
                    "status": "error",
                    "filled_size": 0.0,
                    "requested_size": requested,
                    "error": str(exc),
                }
            )
            continue

        # Reduce-only close — routed to the sub via vault_address in the client.
        # F3: NEVER assume success.  Determine status from the ACTUAL filled size.
        status = "error"
        filled = 0.0
        error: str | None = None
        try:
            resp = client.market_close(coin, cfg.caps.max_order_slippage, cloid)
            try:
                statuses = parse_order_response(resp)
                filled = _extract_filled_sz(statuses)
                if filled >= requested - _FILL_EPS:
                    status = "filled"
                elif filled > _FILL_EPS:
                    status = "partial"
                else:
                    # resting / empty statuses / zero-fill — the position is
                    # NOT confirmed closed.  Do NOT call this "filled".
                    status = "resting"
            except Exception as parse_exc:  # noqa: BLE001
                status = "error"
                error = str(parse_exc)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "kill_switch.flatten_all: market_close failed",
                coin=coin,
                cloid=cloid,
                error=str(exc),
            )
            status = "error"
            error = str(exc)

        # Always attempt to mark DONE so the ledger doesn't accumulate stale PENDING.
        try:
            ledger.mark_done(cloid)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "kill_switch.flatten_all: mark_done failed",
                coin=coin,
                cloid=cloid,
                error=str(exc),
            )

        results.append(
            {
                "coin": coin,
                "cloid": cloid,
                "status": status,
                "filled_size": filled,
                "requested_size": requested,
                "error": error,
            }
        )
        _logger.info(
            "kill_switch.flatten_all: coin close attempted",
            coin=coin,
            cloid=cloid,
            status=status,
            filled=filled,
        )

    # F3: re-query positions after all closes; surface any residual non-zero szi
    # as a CRITICAL alert — a kill that silently half-worked must be visible.
    residual: dict[str, float] = {}
    try:
        post_positions = client.positions()
        residual = {c: float(s) for c, s in post_positions.items() if float(s) != 0.0}
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "kill_switch.flatten_all: post-close positions re-query failed",
            error=str(exc),
        )

    if residual and alert_sink is not None:
        residual_alert = Alert(
            type=AlertType.KILL_SWITCH,
            severity=Severity.CRITICAL,
            message=f"Kill-switch INCOMPLETE: {len(residual)} positions still OPEN after close",
            context={
                "strategy_id": strategy_id,
                "now_ts": now_ts,
                "residual_positions": residual,
            },
        )
        with contextlib.suppress(Exception):
            alert_sink.emit(residual_alert)

    # Emit KILL_SWITCH alert regardless of per-coin outcomes.
    if alert_sink is not None:
        alert = Alert(
            type=AlertType.KILL_SWITCH,
            severity=Severity.CRITICAL,
            message=f"Kill-switch executed: attempted to flatten {len(results)} positions",
            context={
                "strategy_id": strategy_id,
                "now_ts": now_ts,
                "n_positions": len(results),
                "outcomes": [r["status"] for r in results],
                "n_residual": len(residual),
            },
        )
        try:
            alert_sink.emit(alert)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "kill_switch.flatten_all: alert_sink.emit failed (swallowed)",
                error=str(exc),
            )

    return results


# ---------------------------------------------------------------------------
# Tier 2 — Independent-host watchdog stub
# ---------------------------------------------------------------------------


class Watchdog:
    """Independent-host Tier 2 watchdog.

    This class holds a SEPARATE HLClient built with its OWN dedicated agent key
    (separate nonce set — see module docstring for why this is critical: never
    run two concurrent signers on the same agent/nonce set).

    The watchdog checks two conditions and calls ``flatten_all`` with its
    kill_client when either trips:
    - **Liveness**: ``now_ts_float - last_heartbeat_ts > liveness_deadline_s``
      (the main daemon has gone silent).
    - **Equity floor**: ``current_equity < equity_floor``
      (catastrophic drawdown / account drain).

    In production this runs on a SEPARATE host from the main daemon (hence
    "independent-host") to guard against main-host crashes.  The in-repo logic
    here is the trigger decision + flatten path; the cross-host wiring is
    operator infrastructure (systemd, supervisord, etc.).

    Parameters
    ----------
    kill_client:
        A SEPARATE HLClient built with a dedicated watchdog agent key.
        MUST NOT share an agent key or nonce space with the trading daemon.
    cfg:
        Config; forwarded to ``flatten_all``.
    ledger:
        Shared intent ledger; forwarded to ``flatten_all``.
    equity_floor:
        Trigger flatten when ``current_equity`` falls below this USD value.
        ``None`` disables the equity check.
    liveness_deadline_s:
        Trigger flatten when the daemon heartbeat is this many seconds stale.
        ``None`` disables the liveness check.
    alert_sink:
        Optional alert sink for KILL_SWITCH alerts.
    """

    def __init__(
        self,
        kill_client: Any,
        cfg: Any,
        ledger: IntentLedger,
        *,
        equity_floor: float | None = None,
        liveness_deadline_s: float | None = None,
        alert_sink: AlertSink | None = None,
    ) -> None:
        self._kill_client = kill_client
        self._cfg = cfg
        self._ledger = ledger
        self._equity_floor = equity_floor
        self._liveness_deadline_s = liveness_deadline_s
        self._alert_sink = alert_sink

    def check(
        self,
        *,
        last_heartbeat_ts: float,
        now_ts_float: float,
        current_equity: Any,
    ) -> bool:
        """Check watchdog conditions and flatten all if triggered.

        Parameters
        ----------
        last_heartbeat_ts:
            Unix timestamp (float seconds) of the daemon's last heartbeat.
        now_ts_float:
            Current unix timestamp (float seconds).
        current_equity:
            Current account equity (USD; Decimal or float).

        Returns
        -------
        bool
            ``True`` if the watchdog triggered and ``flatten_all`` was called.
        """
        triggered = False
        reason: str | None = None

        if (
            self._liveness_deadline_s is not None
            and (now_ts_float - last_heartbeat_ts) > self._liveness_deadline_s
        ):
            triggered = True
            reason = (
                f"daemon stale: last_heartbeat {now_ts_float - last_heartbeat_ts:.1f}s ago "
                f"> deadline {self._liveness_deadline_s}s"
            )

        if self._equity_floor is not None and float(current_equity) < self._equity_floor:
            triggered = True
            reason = (
                f"equity floor breached: {float(current_equity):.2f} < {self._equity_floor:.2f}"
            )

        if triggered:
            _logger.error(
                "Watchdog.check: triggered, calling flatten_all",
                reason=reason,
            )
            flatten_all(
                self._kill_client,
                self._cfg,
                self._ledger,
                strategy_id="watchdog",
                now_ts=str(int(now_ts_float)),
                alert_sink=self._alert_sink,
            )

        return triggered


# ---------------------------------------------------------------------------
# Tier 3 — Nuclear: master-signed agent revocation runbook
# ---------------------------------------------------------------------------


def revoke_runbook() -> str:
    """Return the operator runbook for Tier 3 nuclear kill-switch.

    This function is NEVER executed by the daemon.  It returns a documented
    string for human operators to follow in a catastrophic scenario.

    Returns
    -------
    str
        Step-by-step operator runbook for revoking the trading agent.
    """
    return """
TIER 3 NUCLEAR KILL-SWITCH — Agent Revocation Runbook
======================================================

Purpose: Instantly remove ALL trading authority from the hl-allocator agent
by revoking the agent approval on-chain.  This is faster than rotating keys
and works even if the trading host is unreachable.

CRITICAL SAFETY NOTE:
  Never run two concurrent signers on the same agent/nonce set.
  Hyperliquid uses the nonce to prevent replay; two concurrent signers on the
  same agent key will collide, producing corrupted nonce state and potentially
  orphaned orders on both ends.

Steps
-----
1. From the SECURE HOST (separate from trading daemon; NEVER the trading host):

   a. Open the Hyperliquid account management UI at:
      https://app.hyperliquid.xyz/portfolio  (mainnet)
      https://app.hyperliquid.xyz/portfolio  (testnet)

   b. Navigate to Account → Settings → API Wallets / Agents.

   c. Locate the trading agent wallet address (from your B8 provisioning record).

   d. Click "Revoke" (or equivalent) to de-register the agent.
      This submits a master-signed `account/provisioning` action that
      immediately removes the agent's ability to sign orders.

2. Verify the agent is revoked:
   - Any subsequent order signed by the agent wallet should be rejected
     with an authorization error.
   - Check the account's active agents list; the revoked agent should be absent.

3. After revocation:
   - Stop the daemon process if still running (it will begin rejecting orders
     immediately but may spam the logs).
   - Audit the ledger for any PENDING intents and resolve manually.
   - To re-enable trading, provision a NEW agent wallet via task B8 and restart.

4. DO NOT re-approve the SAME agent key after revocation for this incident —
   provision a fresh key to ensure no nonce-state confusion.

Security contacts: See your ops runbook for on-call escalation.
"""
