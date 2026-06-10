"""
automation.runtime.daemon — Supervised orchestration of the hl-allocator.

Architecture
------------
``Daemon.run_once`` is the TESTABLE unit — all scheduling, audit, alerting, and
state-persistence logic lives here.  ``Daemon.run_forever`` is a THIN injected
loop with no testable logic of its own: the ``clock``, ``sleep``, and ``stop``
callables are injected so the loop body itself is trivially correct.

Design decisions
----------------
- No ``time.sleep`` in tested paths.  All timing is injected.
- ``run_once`` fires once per ``as_of``.  If the cycle fails (``run_cycle``
  raises), ``last_scheduled_as_of`` is STILL updated to prevent infinite
  error-loops.  The next calendar day's cycle will pick up fresh.
- ``boot`` reconciles PENDING intents, emits a DAEMON_RESTART alert, and
  (optionally) checks agent expiry.
- ``backfill_fills`` is a pure function for dedup of WebSocket reconnect
  fill backfill (MF-11).

Kill-switch tiers (operator reference)
---------------------------------------
Tier 1 — In-process file-flag kill (THIS class).
    Operator arms the kill by creating the file at ``kill_flag_path``.
    The daemon polls the flag each tick in ``run_once``; when present it calls
    ``kill_switch.flatten_all`` (reduce-only) and returns ``None`` — halting
    trading until the operator removes the flag (``rm <kill_flag_path>``).
    A disarm → re-arm cycle will trigger a fresh flatten.

Tier 2 — Independent-host ``kill_switch.Watchdog``.
    MUST be deployed as a SEPARATE PROCESS on an INDEPENDENT HOST with its OWN
    dedicated agent key (a different nonce space).  It is NOT instantiated by
    this daemon — that would defeat the independent-host isolation guarantee.
    See ``kill_switch.Watchdog`` for details.

Tier 3 — Master-signed agent revocation (``kill_switch.revoke_runbook()``).
    Operator runbook only; never executed by the daemon.

Note: ``scheduleCancel`` is NOT a kill — it only cancels resting orders and is
hygiene-only (volume-gated).  It does not close positions.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from automation.core.config import Config
from automation.core.redaction import get_logger
from automation.execution import reconciler
from automation.execution.executor import make_cloid
from automation.monitoring.position_health import assess, gather_health_snapshot
from automation.reporting.alerts import Alert, AlertSink, AlertType, Severity
from automation.reporting.audit import AuditLog, cycle_record
from automation.reporting.state import State, save
from automation.runtime import kill_switch
from automation.runtime import rebalance as _rebalance_mod
from automation.runtime.health import Heartbeat
from automation.runtime.intent_ledger import Intent, IntentLedger
from automation.runtime.rebalance import CycleReport
from automation.runtime.scheduler import (
    expected_as_of,
    retry_window_expired,
    should_fire,
    should_retry,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MF-11 reconnect backfill dedup
# ---------------------------------------------------------------------------


def backfill_fills(
    fills: list[dict[str, Any]],
    seen_keys: set[tuple[Any, ...]],
) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    """Deduplicate fills received after a WebSocket reconnect.

    Each fill is keyed by ``(oid, tid)`` (order-id + trade-id) — the minimal
    unique key that identifies a single exchange fill event.  Fills whose key
    already appears in ``seen_keys`` are silently dropped.

    This is a PURE function — it never touches the network or disk.

    Parameters
    ----------
    fills:
        Raw fill dicts as returned by ``client.fills_since``.  Each should
        contain ``"oid"`` (order id) and ``"tid"`` (trade id) fields.
    seen_keys:
        Set of ``(oid, tid)`` tuples already processed in prior batches.

    Returns
    -------
    tuple[list[dict[str, Any]], set[tuple[Any, ...]]]
        ``(new_fills, updated_seen_keys)`` where ``new_fills`` contains only
        fills not previously seen and ``updated_seen_keys`` is the union of the
        input set and the keys from ``new_fills``.
    """
    new_fills: list[dict[str, Any]] = []
    updated = set(seen_keys)

    for fill in fills:
        key = (fill.get("oid"), fill.get("tid"))
        if key in updated:
            continue
        updated.add(key)
        new_fills.append(fill)

    return new_fills, updated


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


@dataclass
class Daemon:
    """Supervised orchestration daemon for the hl-allocator.

    Parameters
    ----------
    cfg:
        Executor configuration.
    client:
        Duck-typed HLClient (or fake) with read + write methods.
    source:
        Allocation source implementing ``AllocationSource`` protocol.
    ledger:
        Crash-safe intent ledger.
    state:
        Mutable persistent state (modified in-place by ``run_once``).
    state_path:
        Path to the JSON state file for atomic persistence.
    audit:
        Append-only JSONL audit log.
    alert_sink:
        Alert sink for all daemon-level alerts.
    strategy_id:
        Opaque strategy identifier (forwarded to cloid generation).
    """

    cfg: Config
    client: Any
    source: Any
    ledger: IntentLedger
    state: State
    state_path: Any
    audit: AuditLog
    alert_sink: AlertSink
    strategy_id: str
    # G3 — Tier-1 in-process kill-switch file-flag.
    # ``None`` → kill disabled (default; backwards-compatible with all existing
    # Daemon construction sites).  When set, ``run_once`` polls the path each
    # tick; if the file exists it calls ``kill_switch.flatten_all`` (EVERY armed
    # tick — idempotent no-op when flat) and returns ``None`` (HALT) until the
    # operator removes the file.
    kill_flag_path: Any = None
    # Chain-fill resolver (Layer 4 belt-and-suspenders for the bounded-retry
    # window).  ``Callable[[Intent], bool]`` returning ``True`` when the order
    # for ``intent`` is confirmed FILLED on-chain.  Forwarded to
    # ``ledger.reconcile_pending`` at boot (so a resumed window can't re-fire a
    # leg whose PENDING actually filled during a crash) and to ``retry_attempt``
    # / ``run_cycle`` (so each cycle resolves orphan PENDINGs against the chain).
    # ``None`` → conservative ``lambda _: False`` (PENDINGs stay PENDING) — the
    # backwards-compatible default for every existing construction site.
    is_filled: Callable[[Intent], bool] | None = None
    # Liveness heartbeat (Task 5).  ``run_once`` beats it once per non-killed tick
    # (incl. idle ticks) so a /healthz probe can distinguish a STUCK daemon (no
    # beats landing → 503) from a merely idle one.  ``None`` → no heartbeat (the
    # backwards-compatible default for --once/--dry-run and every existing
    # construction site; only the FOREVER path wires one in).
    heartbeat: Heartbeat | None = None
    # Arming-edge guard for the daemon's own "armed" CRITICAL alert ONLY.
    # The FLATTEN runs on EVERY armed tick (so a position appearing while armed —
    # or a fast remove+recreate of the flag between ticks — is always closed; see
    # K2/K3).  This flag only edge-gates the alert to avoid per-tick alert spam.
    # Reset to False on a disarmed tick so a re-arm re-emits the arming alert.
    _kill_handled: bool = dataclasses.field(default=False, init=False)

    def _kill_armed(self) -> bool:
        """Return True if the Tier-1 kill flag file currently exists.

        May raise (e.g. permissions / IO error on ``Path.exists()``).  The
        caller (``run_once``) MUST treat any exception as a fail-safe HALT — an
        unreadable kill flag must stop trading loudly, never silently (K1).
        """
        return self.kill_flag_path is not None and Path(self.kill_flag_path).exists()

    def _kill_flatten(self, now_ts: str) -> None:
        """Reduce-only flatten via the Tier-1 kill path; never crashes the loop.

        Wraps ``kill_switch.flatten_all`` (routed to the trading agent / sub) in
        a try/except.  On a flat book this is a cheap no-op (``positions()`` →
        ``{}`` → zero ``market_close`` calls), so it is safe to call on EVERY
        armed tick.  Any exception is caught and surfaced as a KILL_SWITCH
        CRITICAL alert — the daemon loop must never die on a flatten failure.
        """
        try:
            kill_switch.flatten_all(
                self.client,
                self.cfg,
                self.ledger,
                strategy_id="KILL",
                now_ts=now_ts,
                alert_sink=self.alert_sink,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error("daemon._kill_flatten: kill flatten error", error=str(exc))
            self._emit(
                AlertType.KILL_SWITCH,
                Severity.CRITICAL,
                f"kill flatten error: {exc}",
                {"kill_flag_path": str(self.kill_flag_path), "now_ts": now_ts},
            )

    def run_once(
        self,
        now: datetime.datetime,
        now_ts: str,
    ) -> CycleReport | None:
        """Execute at most one rebalance cycle.

        This is the TESTABLE orchestration unit.  ``run_forever`` calls this
        on every tick; tests call it directly with injected ``now``/``now_ts``.

        Execution sequence
        ------------------
        0a. Tier-1 kill-switch guard (G3) — HIGHEST PRIORITY, FAIL-SAFE.
            If reading the flag raises (permissions / IO error), treat it as a
            fail-safe HALT (K1): emit KILL_SWITCH CRITICAL, attempt a flatten,
            and return ``None`` — never let an unreadable flag silently disable
            the kill.  If the flag exists, call ``kill_switch.flatten_all``
            (reduce-only, idempotent no-op when flat) on EVERY armed tick (K2/K3:
            so a position appearing while armed — or a fast remove+recreate of
            the flag between ticks — is always closed) and return ``None`` —
            HALTING trading until the operator removes the flag.  The daemon's
            own "armed" CRITICAL alert is edge-gated by ``_kill_handled`` to
            avoid per-tick alert spam; the FLATTEN itself is NOT gated.  The
            day's ``as_of`` is NOT consumed, so trading resumes cleanly after
            the flag is removed.
        0b. Agent-expiry fail-safe guard (S-12) — if the agent key is EXPIRED,
            emit AGENT_EXPIRY CRITICAL and return ``None`` without consuming
            the day's ``as_of``.  Trading resumes cleanly after the operator
            rotates the agent and updates ``cfg.agent_valid_until_ms``.
        1. ``should_fire`` — decide whether a new cycle is due.
        2. Mark-scheduled FIRST — persist ``state.last_scheduled_as_of`` to disk
           BEFORE the cycle runs.  This is the crash-safe idempotency guarantee
           (F1): a hard process death (SIGKILL/OOM/power loss) mid-cycle then
           never re-fires the same as_of on reboot.  At worst the day is skipped
           (fail-safe) — boot-reconcile recovers in-flight PENDING orders and
           the abs-drift backstop catches up the next day.
        3. ``run_cycle`` — delegate to B7 cycle orchestrator.  If it raises,
           wrap in a failed CycleReport, emit EXEC_ERROR, and continue.
        4. Post-cycle save — persist the remaining State fields written by
           ``run_cycle`` (it mutates the SAME in-memory State object).
        5. Audit — write a compact JSONL record.
        6. Alerts — emit EXEC_ERROR / CYCLE_INCOMPLETE as appropriate.

        Parameters
        ----------
        now:
            Current time.  MUST be tz-aware.
        now_ts:
            ISO timestamp string injected by the caller (also the cycle nonce).

        Returns
        -------
        CycleReport | None
            The CycleReport if a cycle fired, or ``None`` if the scheduler
            decided not to fire (too early, already fired today, or the agent
            key is expired — fail-safe HOLD per S-12).
        """
        # ------------------------------------------------------------------
        # Liveness heartbeat (Task 5) — beat FIRST, before EVERY guard.
        # A liveness probe answers "is the process alive and looping?", NOT
        # "is it trading?".  A KILL-armed, expiry-HOLD, or idle daemon is still
        # alive and looping (it flattens / holds each tick) → it must report
        # HEALTHY.  Beating before the Step-0a kill guard is therefore required:
        # if we beat only after the guard, a KILL-armed daemon would stop
        # beating → /healthz 503 → the supervisor restarts it → it reboots,
        # re-reads the PERSISTED kill flag, HALTs, stops beating again → restart
        # churn fighting the operator's deliberate stop.  A HALT is surfaced via
        # the existing KILL_SWITCH CRITICAL alert (the correct channel), never
        # via the restart-triggering health probe.
        # ------------------------------------------------------------------
        if self.heartbeat is not None:
            self.heartbeat.beat(now_ts)

        # ------------------------------------------------------------------
        # Step 0a: Tier-1 in-process kill-switch guard (G3) — HIGHEST PRIORITY.
        # FAIL-SAFE + act on EVERY armed tick:
        #   - Reading the flag raises (K1) → fail-safe HALT, loudly: emit
        #     KILL_SWITCH CRITICAL, attempt a flatten, return None.  An
        #     unreadable flag must NEVER silently disable the kill.
        #   - Armed → flatten on EVERY tick (K2/K3: closes a position that
        #     appears while armed, and survives a fast remove+recreate of the
        #     flag with no disarmed tick observed).  flatten_all on a flat book
        #     is a cheap no-op.  The daemon's own "armed" CRITICAL alert is
        #     edge-gated by _kill_handled (avoid per-tick spam); the FLATTEN is
        #     NOT gated.  Return None (HALT) — no fire, no mark, no run_cycle.
        #   - Disarmed → reset _kill_handled so a re-arm re-emits the alert.
        # The day's as_of is deliberately NOT consumed so trading resumes
        # cleanly once the operator removes the flag.
        # ------------------------------------------------------------------
        try:
            armed = self._kill_armed()
        except Exception as exc:  # noqa: BLE001 — K1: unreadable flag => fail-safe HALT
            _logger.error(
                "daemon.run_once: kill flag unreadable — fail-safe HALT",
                error=str(exc),
            )
            self._emit(
                AlertType.KILL_SWITCH,
                Severity.CRITICAL,
                f"kill flag unreadable — refusing to trade (fail-safe HALT): {exc}",
                {"kill_flag_path": str(self.kill_flag_path), "now_ts": now_ts},
            )
            self._kill_flatten(now_ts)
            return None  # HALT: never trade with an unreadable kill flag.

        if armed:
            if not self._kill_handled:
                # Arming edge: emit the daemon's "armed" CRITICAL alert ONCE.
                _logger.error(
                    "daemon.run_once: kill flag armed — flattening to cash and HALTING trading",
                    kill_flag_path=str(self.kill_flag_path),
                )
                self._emit(
                    AlertType.KILL_SWITCH,
                    Severity.CRITICAL,
                    "kill flag armed — flattening to cash and HALTING trading",
                    {"kill_flag_path": str(self.kill_flag_path), "now_ts": now_ts},
                )
                self._kill_handled = True
            # K2/K3: flatten on EVERY armed tick (idempotent no-op when flat).
            self._kill_flatten(now_ts)
            return None  # HALT: no fire, no mark, no run_cycle while armed
        self._kill_handled = False  # disarmed → reset so re-arm re-alerts

        # ------------------------------------------------------------------
        # Step 0b: Agent-expiry fail-safe guard (S-12).
        # If the agent key is EXPIRED, refuse to trade and return None WITHOUT
        # consuming the day's as_of — so trading resumes cleanly once the
        # operator rotates the agent key and updates cfg.agent_valid_until_ms.
        # ------------------------------------------------------------------
        # An UNREADABLE expiry value (malformed timestamp, parse error) is also
        # treated as fail-safe HOLD — we never trade with an unknown agent state.
        # Unreachable via pydantic (Config.agent_valid_until_ms is int|None) but
        # symmetric with the boot try/except and defends this trading-control path.
        try:
            expired, days_remaining = self._agent_expiry(now_ts)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "daemon.run_once: agent expiry unreadable — fail-safe HOLD",
                error=str(exc),
            )
            self._emit(
                AlertType.AGENT_EXPIRY,
                Severity.CRITICAL,
                "agent expiry unreadable — refusing to trade (fail-safe HOLD)",
                {"error": str(exc)},
            )
            return None
        if expired:
            _logger.error(
                "daemon.run_once: agent key EXPIRED — fail-safe HOLD",
                days_remaining=days_remaining,
            )
            self._emit(
                AlertType.AGENT_EXPIRY,
                Severity.CRITICAL,
                "agent key EXPIRED — refusing to trade (fail-safe HOLD); rotate the agent",
                {"days_remaining": days_remaining},
            )
            return None

        fire, as_of = should_fire(
            now, self.state.last_scheduled_as_of, self.cfg.rebalance_utc_time
        )
        if not fire:
            # No new daily cycle is due.  If a bounded-retry window is OPEN, this
            # is where we advance it: fire one retry attempt, best-effort commit
            # on expiry/day-roll, or hold until the next interval.  Placed AFTER
            # the kill (0a) and agent-expiry (0b) guards above, so a HALT never
            # lets a retry fire.
            return self._maybe_retry(now, now_ts)

        assert as_of is not None  # guaranteed when fire=True

        # ------------------------------------------------------------------
        # Step 1b (C1): SUPERSEDE a stale retry window for a DIFFERENT as_of.
        # A new daily cycle is firing for ``as_of`` (= as_of_new).  If a retry
        # window is still open for a PRIOR as_of, it MUST be dropped HERE —
        # BEFORE run_cycle commits the fresh baseline — otherwise a later no-fire
        # tick's ``_maybe_retry`` would see a day-roll (expected_as_of != stale
        # retry_as_of) and best-effort-commit the STALE frozen target, clobbering
        # the just-committed fresh baseline (ratchet poisoning) and emitting a
        # spurious UNFILLED alert.  We DROP (not best-effort-commit) the stale
        # window: the new daily cycle's target supersedes it and its
        # delta-from-live reconciliation naturally absorbs any residual prior
        # position.  Same-as_of re-fire is impossible (should_fire idempotency),
        # so ``!= as_of`` is the exact, sufficient condition.
        # ------------------------------------------------------------------
        if self.state.retry_as_of is not None and self.state.retry_as_of != as_of:
            _logger.warning(
                "daemon.run_once: superseding stale retry window — new daily cycle",
                old_as_of=self.state.retry_as_of,
                new_as_of=as_of,
            )
            self._clear_retry()
            save(self.state, self.state_path)

        # ------------------------------------------------------------------
        # Step 2: Mark-scheduled FIRST (F1 — crash-safe idempotency).
        # Persist the idempotency key to DISK before any trade is committed.
        # If the process is hard-killed mid-cycle, on reboot should_fire sees
        # this as_of already scheduled and will NOT re-fire — preventing a
        # double-rebalance.  We accept the fail-safe trade-off: a crash between
        # this save and run_cycle skips the day entirely (the next day's as_of
        # fires; the abs-drift backstop closes the gap).
        #
        # MF-9 defer interaction (B10 F1 preserved):
        # A rate-budget DEFER submits ZERO orders, so re-firing the same as_of
        # is SAFE and DESIRED (retry).  We capture prev_scheduled BEFORE marking
        # so that after a deferred cycle we can RESTORE the idempotency key to
        # its prior value — making should_fire re-fire on the next tick.  This
        # restoration is ONLY done on a deferred cycle (n_submitted=0, nothing
        # landed) and is SAFE because no orders were ever sent to the exchange.
        # A real submit (even partial/error) keeps the as_of consumed (F1 intact).
        # ------------------------------------------------------------------
        prev_scheduled = self.state.last_scheduled_as_of  # capture before marking
        self.state.last_scheduled_as_of = as_of
        save(self.state, self.state_path)

        # ------------------------------------------------------------------
        # Step 3: Run the cycle.
        # ------------------------------------------------------------------
        report: CycleReport
        try:
            report = _rebalance_mod.run_cycle(
                self.cfg,
                self.client,
                self.source,
                self.ledger,
                self.state,
                self.state_path,
                now_ts=now_ts,
                as_of=datetime.date.fromisoformat(as_of),
                is_filled=self.is_filled,
                alert_sink=self.alert_sink,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "daemon.run_once: run_cycle raised unexpectedly",
                as_of=as_of,
                error=str(exc),
            )
            # Emit EXEC_ERROR alert.
            self._emit(
                AlertType.EXEC_ERROR,
                Severity.CRITICAL,
                f"run_cycle raised: {exc}",
                {"as_of": as_of, "error": str(exc)},
            )
            # Build a minimal failed CycleReport so we can audit it.
            report = CycleReport(
                rebalanced=False,
                reason=f"exception: {exc}",
                exec_report=None,
                unresolved_pending=[],
                committed=False,
            )

        # ------------------------------------------------------------------
        # MF-9 defer-retry: if the cycle was deferred (zero submits), RESTORE
        # last_scheduled_as_of to its pre-mark value so should_fire re-fires
        # the same as_of on the next tick (retry).
        #
        # Safety: a deferred cycle submits NOTHING (n_submitted=0), so there
        # is no double-trade risk in re-firing.  The mark-before-cycle (Step 2)
        # still protects the SUBMIT path: the as_of is marked before any order
        # is ever sent, and a crash mid-submit (non-deferred path) still lands
        # the mark on disk — F1 is fully intact.
        #
        # We use getattr to guard against a hypothetical non-CycleReport
        # (e.g. the exception path above that creates a minimal CycleReport —
        # it has no `deferred` field so getattr returns False safely).
        # ------------------------------------------------------------------
        if getattr(report, "deferred", False):
            _logger.info(
                "daemon.run_once: deferred cycle — restoring last_scheduled_as_of "
                "to retry on next tick (MF-9 defer-retry; zero submits → safe)",
                as_of=as_of,
                prev_scheduled=prev_scheduled,
            )
            self.state.last_scheduled_as_of = prev_scheduled
            try:
                save(self.state, self.state_path)
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "daemon.run_once: failed to persist defer-retry state restore",
                    as_of=as_of,
                    error=str(exc),
                )

        # ------------------------------------------------------------------
        # Step 4: Post-cycle save — persist the remaining State fields written
        # by run_cycle (last_rebalanced_target / last_rebalanced_as_of).  This
        # is the SAME in-memory State object, so last_scheduled_as_of is also
        # re-written (no lost update).  run_cycle already saved internally on a
        # clean commit; this is a belt-and-suspenders flush that also covers the
        # exception path where run_cycle never reached its own save.
        # ------------------------------------------------------------------
        try:
            save(self.state, self.state_path)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "daemon.run_once: failed to persist state after cycle",
                as_of=as_of,
                error=str(exc),
            )

        # ------------------------------------------------------------------
        # Step 4b: OPEN a bounded-retry window if the daily cycle left a
        # RETRYABLE residual (a leg errored / partially-filled / was
        # liquidity-deferred) and did NOT commit.  The window freezes the served
        # target and arms _maybe_retry to re-attempt the residual on subsequent
        # ticks (no fire) until it commits cleanly or the window expires.  A
        # deferred (zero-submit) cycle is NOT a residual — it has its own MF-9
        # defer-retry and committed=False but exec_report.n_*==0, so the > 0
        # guard below excludes it.
        # ------------------------------------------------------------------
        if (
            self.cfg.retry.enabled
            and report is not None
            and not report.committed
            and report.exec_report is not None
            and report.target_weights is not None
            and (
                report.exec_report.n_error
                + report.exec_report.n_partial
                + report.exec_report.n_liquidity_deferred
            )
            > 0
        ):
            self.state.retry_as_of = as_of
            self.state.retry_target = dict(report.target_weights)
            self.state.retry_window_start = now_ts
            self.state.retry_last_attempt = now_ts
            self.state.retry_attempt_count = 0
            try:
                save(self.state, self.state_path)
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "daemon.run_once: failed to persist opened retry window",
                    as_of=as_of,
                    error=str(exc),
                )
            _logger.info("daemon.run_once: retry window OPENED", as_of=as_of)

        # ------------------------------------------------------------------
        # Step 5: Audit.
        # ------------------------------------------------------------------
        try:
            self.audit.write(cycle_record(ts=now_ts, as_of=as_of, report=report))
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "daemon.run_once: audit.write failed",
                as_of=as_of,
                error=str(exc),
            )

        # ------------------------------------------------------------------
        # Step 6: Alerts from the report.
        # ------------------------------------------------------------------
        er = report.exec_report
        if er is not None:
            if er.aborted:
                self._emit(
                    AlertType.CYCLE_INCOMPLETE,
                    Severity.CRITICAL,
                    f"Cycle aborted (pre-flight cap breach): {er.abort_reason}",
                    {"as_of": as_of, "abort_reason": er.abort_reason},
                )
            elif er.n_error > 0 or er.n_partial > 0:
                self._emit(
                    AlertType.EXEC_ERROR,
                    Severity.WARNING,
                    (
                        f"Cycle completed with errors: n_error={er.n_error} "
                        f"n_partial={er.n_partial}"
                    ),
                    {
                        "as_of": as_of,
                        "n_error": er.n_error,
                        "n_partial": er.n_partial,
                    },
                )

        if report.rebalanced and not report.committed:
            self._emit(
                AlertType.CYCLE_INCOMPLETE,
                Severity.WARNING,
                "Cycle rebalanced but state was NOT committed (baseline poisoning risk)",
                {"as_of": as_of},
            )

        # ------------------------------------------------------------------
        # Step 7: Post-cycle position-health monitor (PH-4).
        # Runs AFTER trading completes (state saved, audit written, cycle alerts
        # emitted), so it observes the FINAL positions.  Gated on
        # cfg.health.enabled and FULLY wrapped — a monitoring failure must NEVER
        # abort or affect the trading cycle (run_once still returns the report).
        # ------------------------------------------------------------------
        if self.cfg.health.enabled:
            try:
                snap = gather_health_snapshot(self.client)
                for f in assess(snap, self.cfg):
                    self._emit(f.alert_type, f.severity, f.message, f.context)
            except Exception as exc:  # noqa: BLE001 — monitoring must NEVER affect trading
                _logger.warning(
                    "daemon: health monitor failed (continuing)", error=str(exc)
                )

        return report

    def boot(self, now_ts: str) -> None:
        """Boot-time reconciliation and alerting.

        Steps
        -----
        1. Reconcile any PENDING intents from a prior crash.
        2. Emit a DAEMON_RESTART alert.
        3. Check agent expiry (if ``cfg`` or ``state`` carries
           ``agent_valid_until_ms``); alert with AGENT_EXPIRY if < 14 days
           remain.  Skipped with a TODO log if the field is absent.
        4. Log nonce/last-processed context (informational only).

        Parameters
        ----------
        now_ts:
            ISO timestamp of this boot event.
        """
        # Step 1: Reconcile orphan PENDING intents via the CHAIN resolver
        # (Layer 4 belt-and-suspenders).  Using ``self.is_filled`` — not the
        # conservative ``lambda _: False`` — guarantees a resumed bounded-retry
        # window cannot re-fire a leg whose PENDING order actually FILLED during
        # a crash: the orphan is marked DONE here before any window reopens.
        # When ``is_filled`` is ``None`` we fall back to the conservative
        # resolver (PENDINGs stay PENDING) — the backwards-compatible default.
        resolver: Callable[[Intent], bool] = (
            self.is_filled if self.is_filled is not None else (lambda _: False)
        )
        try:
            unresolved = self.ledger.reconcile_pending(resolver)
            if unresolved:
                _logger.warning(
                    "daemon.boot: unresolved PENDING intents after reconciliation",
                    n_unresolved=len(unresolved),
                    cloids=[i.cloid for i in unresolved],
                )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "daemon.boot: ledger.reconcile_pending raised",
                error=str(exc),
            )

        # Step 1b: Sweep orphan open orders (IOC-only ⇒ none expected).
        # A crash mid-submit or a resting-status anomaly can leave stray orders
        # on the book.  Cancel them best-effort; never block boot on failure.
        try:
            reconciler.sweep_open_orders(
                self.client,
                now_ts=now_ts,
                alert_sink=self.alert_sink,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "daemon.boot: sweep_open_orders raised (skipped, boot continues)",
                error=str(exc),
            )

        # Step 2: DAEMON_RESTART alert.
        self._emit(
            AlertType.DAEMON_RESTART,
            Severity.INFO,
            "Daemon (re)started",
            {"now_ts": now_ts, "last_scheduled_as_of": self.state.last_scheduled_as_of},
        )

        # Step 3: Agent expiry check via shared helper.
        # Routes through _agent_expiry so boot and run_once share the same logic.
        #
        # ``expired`` defaults to True (fail-safe): if the expiry value is
        # UNREADABLE (the try below raises) we must NOT sign any boot action — an
        # unknown agent state is treated as expired, mirroring run_once's
        # unreadable-expiry HOLD.  An UNSET field (days_remaining is None) leaves
        # expired=False below — no configured expiry means "not expired", so the
        # leverage step still runs (additive, backward-compatible).
        expired: bool = True
        try:
            expired, days_remaining = self._agent_expiry(now_ts)
            if days_remaining is None:
                # Field not set in cfg or state — log TODO and skip.
                _logger.info(
                    "daemon.boot: agent_valid_until_ms not set in cfg/state — "
                    "TODO: populate via B8 provisioning record to enable expiry alerts"
                )
            elif expired:
                self._emit(
                    AlertType.AGENT_EXPIRY,
                    Severity.CRITICAL,
                    "agent key EXPIRED — refusing to trade (fail-safe HOLD); rotate the agent",
                    {"days_remaining": days_remaining},
                )
                _logger.error(
                    "daemon.boot: agent key EXPIRED",
                    days_remaining=days_remaining,
                )
            elif days_remaining < 14:
                self._emit(
                    AlertType.AGENT_EXPIRY,
                    Severity.WARNING,
                    f"Agent key expires in {days_remaining:.1f} days — renew soon",
                    {"days_remaining": round(days_remaining, 1)},
                )
                _logger.warning(
                    "daemon.boot: agent expiry warning",
                    days_remaining=round(days_remaining, 1),
                )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "daemon.boot: agent expiry check failed (skipped)",
                error=str(exc),
            )

        # Step 4: Set per-coin leverage (HAARP is a 1x ISOLATED basket).
        # The venue default is 10x cross on testnet; without this the positions
        # would open leveraged.  Isolate per-coin failures so one bad coin (e.g.
        # not listed, or already at a different leverage with an open position)
        # never aborts boot.  Leverage is set while typically FLAT; an already-open
        # position keeps its open-time leverage (HL only applies the new leverage to
        # subsequent opens) — boot runs before any cycle, so this is the right moment.
        #
        # Gated on ``not expired``: update_leverage is a SIGNED action, and we must
        # never sign on an expired (or unreadable-expiry) agent — same invariant
        # run_once upholds by returning None when expired.
        if self.cfg.set_leverage_on_boot and not expired:
            # HL refuses to switch an OPEN position's leverage TYPE (cross↔isolated):
            # update_leverage on such a position errors out.  So for each tradeable
            # coin we reconcile against the HELD leverage:
            #   * TYPE mismatch on an open position → cannot fix in place.  When
            #     enforce_leverage_on_drift is True, FLATTEN it (reduce-only) so the
            #     next cycle reopens it at the boot-set default; then fall through to
            #     update_leverage (position now flat → sets the default).  When False,
            #     emit a LEVERAGE_DRIFT alert and SKIP update_leverage (it would fail
            #     on the still-open position).
            #   * VALUE-only mismatch (same type) is handled implicitly: the position
            #     stays open and update_leverage corrects its value in place (HL allows
            #     value change within the same mode).
            #   * No held position (flat) → update_leverage sets the default.
            held = self.client.positions_leverage()  # {coin: {"type","value"}}
            target_type = "cross" if self.cfg.leverage_is_cross else "isolated"
            coins = self.cfg.tradeable_coins or list(self.cfg.universe_perp_map.keys())
            for coin in coins:
                cur = held.get(coin)
                if cur is not None and cur.get("type") != target_type:
                    # TYPE mismatch on an OPEN position — cannot update in place.
                    if self.cfg.enforce_leverage_on_drift:
                        try:
                            self._flatten_for_leverage(coin, now_ts)  # reduce-only close
                        except Exception as exc:  # noqa: BLE001
                            _logger.warning(
                                "daemon.boot: leverage flatten failed (continuing)",
                                coin=coin,
                                error=str(exc),
                            )
                        self._emit(
                            AlertType.LEVERAGE_DRIFT,
                            Severity.WARNING,
                            "held position leverage type mismatch — flattened to reopen at target",
                            {"coin": coin, "had": cur, "target_type": target_type},
                        )
                        # fall through: position now flat → update_leverage below sets the default
                    else:
                        self._emit(
                            AlertType.LEVERAGE_DRIFT,
                            Severity.WARNING,
                            "held position leverage type mismatch — NOT enforced (alert only)",
                            {"coin": coin, "had": cur, "target_type": target_type},
                        )
                        continue  # leave it; update_leverage would fail on the still-open position
                try:
                    self.client.update_leverage(
                        coin, self.cfg.target_leverage, is_cross=self.cfg.leverage_is_cross
                    )
                except Exception as exc:  # noqa: BLE001 — one coin's failure must not abort boot
                    _logger.warning(
                        "daemon.boot: update_leverage failed (continuing)",
                        coin=coin,
                        error=str(exc),
                    )
            _logger.info(
                "daemon.boot: leverage reconciled",
                n_coins=len(coins),
                leverage=self.cfg.target_leverage,
                is_cross=self.cfg.leverage_is_cross,
            )

    def _flatten_for_leverage(self, coin: str, now_ts: str) -> None:
        """Reduce-only close of ``coin``'s position so it reopens at the target leverage.

        Reuses the ``client.market_close`` reduce-only primitive (same one
        ``kill_switch.flatten_all`` uses) with the configured order slippage and
        a deterministic cloid.  ``market_close`` is reduce-only by construction —
        it can NEVER open or flip a position.  Called from the boot leverage
        reconcile when a HELD position's leverage TYPE mismatches the target
        (HL refuses an in-place cross↔isolated switch on an open position).
        """
        cloid = make_cloid(self.cfg.strategy_id, "boot-lev", coin, 0, now_ts)
        self.client.market_close(coin, self.cfg.caps.max_order_slippage, cloid)

    def shutdown(self) -> None:
        """Flush the audit log and emit a shutdown log line.

        Also releases the exclusive nonce-file lock held by the client (if any)
        so a clean restart can re-acquire it immediately.

        WebSocket close is deferred to B11 (testnet integration).
        """
        _logger.info("daemon.shutdown: flushing audit log and stopping")
        # AuditLog already fsync's on every write; nothing extra to flush.
        # WS close: TODO for B11.
        # Release the nonce-file lock so a clean restart doesn't have to wait
        # for the OS to collect the process (or for a lock timeout).
        if hasattr(self.client, "close"):
            with contextlib.suppress(Exception):
                self.client.close()

    def run_forever(
        self,
        *,
        clock: Any,
        sleep: Any,
        stop: Any,
        interval: float = 60.0,
    ) -> None:
        """THIN supervised loop — NOT unit-tested (no testable logic).

        All meaningful logic is in ``run_once``.  This method only provides the
        tick loop; ``clock``, ``sleep``, and ``stop`` are injected so the loop
        is trivially verifiable by inspection.

        Parameters
        ----------
        clock:
            Zero-argument callable returning a tz-aware ``datetime``.
        sleep:
            Callable accepting a float seconds argument.
        stop:
            Zero-argument callable returning ``True`` when the loop should exit.
        interval:
            Seconds between ticks.  Default 60.
        """
        while not stop():
            now = clock()
            now_ts = now.isoformat()
            try:
                self.run_once(now, now_ts)
            except Exception as exc:  # noqa: BLE001
                # run_once should not raise (it catches internally), but as a
                # belt-and-suspenders guard we log and continue.
                _logger.error(
                    "daemon.run_forever: run_once raised unexpectedly (continuing)",
                    error=str(exc),
                )
            sleep(interval)

    # ------------------------------------------------------------------
    # Bounded intra-cycle retry — window lifecycle
    # ------------------------------------------------------------------

    def _maybe_retry(self, now: datetime.datetime, now_ts: str) -> CycleReport | None:
        """Advance the bounded-retry window on a no-fire tick.

        Called from ``run_once`` when ``should_fire`` declined to fire a new
        daily cycle.  Decision order:

        1. No window open (or retry disabled) → ``None`` (nothing to do).
        2. The window's frozen ``retry_as_of`` no longer matches the current
           day's expected ``as_of`` (day-roll), OR the window has expired
           (>= ``window_seconds`` since ``retry_window_start``) → BEST-EFFORT
           commit the frozen target to baseline, emit UNFILLED, clear, ``None``.
        3. ``should_retry`` says it is too early (interval not yet elapsed) →
           ``None`` (hold; window untouched).
        4. Otherwise → fire ONE ``retry_attempt`` toward the frozen target.  On
           a CLEAN attempt (committed) → clear the window.

        Returns the retry ``CycleReport`` when an attempt fired, else ``None``.
        """
        if self.state.retry_as_of is None:
            return None
        if not self.cfg.retry.enabled:
            # I1: the feature was DISABLED while a window was still open.  Don't
            # leak the frozen target as stale state forever — DRAIN the window
            # once (best-effort commit + clear), honouring the spec's
            # "best-effort commit on give-up" promise.
            self._best_effort_commit(now_ts)
            return None
        r = self.cfg.retry
        # Day-roll or expiry → best-effort commit and close the window.
        if (
            expected_as_of(now, self.cfg.rebalance_utc_time) != self.state.retry_as_of
            or retry_window_expired(now, self.state.retry_window_start, r.window_seconds)
        ):
            self._best_effort_commit(now_ts)
            return None
        # Too early to retry again → hold.
        if not should_retry(
            now=now,
            retry_as_of=self.state.retry_as_of,
            retry_window_start=self.state.retry_window_start,
            retry_last_attempt=self.state.retry_last_attempt,
            interval_seconds=r.interval_seconds,
            window_seconds=r.window_seconds,
            rebalance_utc_time=self.cfg.rebalance_utc_time,
        ):
            return None
        # Fire ONE retry attempt.  Stamp last_attempt + bump count FIRST and
        # persist — so a crash mid-attempt does not re-fire on the same interval
        # tick (the executor's structural delta-from-live anti-double-order is
        # the economic guard; this is the timing-idempotency guard).
        self.state.retry_last_attempt = now_ts
        self.state.retry_attempt_count += 1
        save(self.state, self.state_path)
        report = _rebalance_mod.retry_attempt(
            self.cfg,
            self.client,
            self.ledger,
            self.state,
            self.state_path,
            retry_target=self.state.retry_target or {},
            retry_as_of=self.state.retry_as_of,
            now_ts=now_ts,
            alert_sink=self.alert_sink,
            is_filled=self.is_filled,
        )
        if report.committed:
            _logger.info(
                "daemon._maybe_retry: retry CLEAN — window cleared",
                as_of=self.state.retry_as_of,
                attempts=self.state.retry_attempt_count,
            )
            self._clear_retry()
            save(self.state, self.state_path)
        return report

    def _best_effort_commit(self, now_ts: str) -> None:
        """Commit the frozen retry target to baseline, emit UNFILLED, clear.

        Called when the retry window expires (or the day rolls) with an
        unfilled residual.  Committing the FROZEN target to
        ``last_rebalanced_target`` prevents baseline poisoning: the next daily
        cycle's drift is measured against what we actually TRIED to hold, not a
        stale pre-retry baseline.  An UNFILLED WARNING surfaces the residual to
        the operator.
        """
        tgt = self.state.retry_target or {}
        as_of = self.state.retry_as_of
        attempts = self.state.retry_attempt_count
        self.state.last_rebalanced_target = dict(tgt)
        self.state.last_rebalanced_as_of = as_of
        self._clear_retry()
        save(self.state, self.state_path)
        _logger.warning(
            "daemon: retry window EXPIRED — best-effort commit",
            as_of=as_of,
            attempts=attempts,
        )
        self._emit(
            AlertType.UNFILLED,
            Severity.WARNING,
            "retry window expired with unfilled residual; best-effort committed",
            {"as_of": as_of, "attempts": attempts},
        )

    def _clear_retry(self) -> None:
        """Reset all bounded-retry window fields to IDLE (in-memory only)."""
        self.state.retry_as_of = None
        self.state.retry_target = None
        self.state.retry_window_start = None
        self.state.retry_last_attempt = None
        self.state.retry_attempt_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _agent_expiry(self, now_ts: str) -> tuple[bool, float | None]:
        """Compute agent-key expiry status from the configured timestamp.

        Resolves ``agent_valid_until_ms`` from ``self.cfg.agent_valid_until_ms``
        (primary) then ``getattr(self.state, "agent_valid_until_ms", None)``
        (legacy fallback).  If neither is set, returns ``(False, None)``
        (unknown — don't block).

        Parameters
        ----------
        now_ts:
            ISO timestamp string for "now" (may be naive or UTC-aware;
            naive is treated as UTC).

        Returns
        -------
        tuple[bool, float | None]
            ``(expired, days_remaining)`` where:

            * ``expired`` — ``True`` if ``days_remaining <= 0``.
            * ``days_remaining`` — fractional days until expiry, or ``None``
              when ``agent_valid_until_ms`` is not configured.
        """
        agent_valid_until_ms: int | None = getattr(self.cfg, "agent_valid_until_ms", None)
        if agent_valid_until_ms is None:
            agent_valid_until_ms = getattr(self.state, "agent_valid_until_ms", None)

        if agent_valid_until_ms is None:
            return (False, None)

        # Parse now_ts; treat naive timestamps as UTC.
        now_dt = datetime.datetime.fromisoformat(now_ts.replace("Z", "+00:00"))
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=datetime.UTC)

        days_remaining = (agent_valid_until_ms / 1000.0 - now_dt.timestamp()) / 86400.0
        expired = days_remaining <= 0
        return (expired, days_remaining)

    def _emit(
        self,
        alert_type: AlertType,
        severity: Severity,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emit an alert via ``alert_sink``, swallowing all errors."""
        alert = Alert(
            type=alert_type,
            severity=severity,
            message=message,
            context=context or {},
        )
        try:
            self.alert_sink.emit(alert)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "daemon._emit: alert_sink.emit failed (swallowed)",
                alert_type=alert_type.value,
                error=str(exc),
            )
