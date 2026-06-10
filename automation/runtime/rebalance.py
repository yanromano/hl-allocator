"""
automation.runtime.rebalance — One-cycle live rebalance orchestrator.

This module ties together B5 (allocation source), B6 (sizer), B7 (executor),
and the intent ledger + state persistence into a single, auditable cycle.

Critical invariants (S-2, MF-7)
---------------------------------
- The PENDING intent is written to the ledger BEFORE any order is submitted
  (enforced by executor.execute — not re-enforced here).
- ``state.last_rebalanced_target`` is updated to ``ta.weights`` (the FROZEN
  NEW target, NOT achieved weights) ONLY when a rebalance fired.  Hold cycles
  leave the baseline unchanged so the ratchet retains its memory.
- State is persisted atomically via ``reporting.state.save``.
- ``state.last_successful_signal_ts`` is updated after every cycle in which the
  signal was both fetched AND passed ``validate_target_allocation``.  This
  timestamp drives the N-3 staleness clock: when it is absent or too old,
  ``run_cycle`` HOLDS (within window) or de-risks to CASH (beyond window).

B10 hooks (TODO)
-----------------
- Audit JSONL append for every cycle (for post-hoc reconciliation).
- Circuit-breaker: halt after N consecutive errors.
These are intentionally absent from B7 to keep this module focused.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automation.allocation.base import (
    AllocationRejected,
    AllocationSource,
    TargetAllocation,
    validate_target_allocation,
)
from automation.core.config import Config
from automation.core.redaction import get_logger
from automation.execution import reconciler, sizer
from automation.execution.executor import ExecReport, execute
from automation.reporting.alerts import Alert, AlertSink, AlertType, Severity
from automation.reporting.state import State, save
from automation.runtime.intent_ledger import Intent, IntentLedger
from automation.runtime.kill_switch import flatten_all

_logger = get_logger(__name__)


@dataclass(frozen=True)
class CycleReport:
    """Summary of one live rebalance cycle."""

    rebalanced: bool
    """``True`` if the size plan decided a rebalance was needed."""

    reason: str
    """Human-readable reason from the size plan (e.g. ``"bootstrap"``, ``"hold"``)."""

    exec_report: ExecReport | None
    """Execution report (``None`` when rebalance=False or validation failed)."""

    unresolved_pending: list[Intent] = field(default_factory=list)
    """Intents that remained PENDING after boot-time reconciliation."""

    committed: bool = False
    """``True`` if state was successfully persisted after a live rebalance."""

    signal_ok: bool = True
    """``False`` when the signal fetch or validation failed this cycle."""

    derisked: bool = False
    """``True`` when the staleness clock expired and flatten_all was called."""

    derisk_incomplete: bool = False
    """``True`` when ``derisked`` and ``flatten_all`` did NOT fully close every
    position (at least one coin returned a non-``"filled"`` status — error /
    partial / resting).  Always ``False`` when ``derisked`` is ``False``."""

    max_divergence: float = 0.0
    """Maximum absolute deviation (as a fraction, not percentage points) between
    achieved weights and target weights, computed post-execute.  Zero on hold
    cycles or when the post-snapshot failed.  Used for observability only —
    never affects cycle execution."""

    n_unallocated: int = 0
    """Number of target coins that were silently unallocated by the sizer
    (``below_min_notional`` or ``zeroed_by_cap``).  Zero when no coins were
    skipped for these reasons."""

    deferred: bool = False
    """``True`` when the cycle was deferred by the rate-budget pre-flight (MF-9).
    Zero orders were submitted; the daemon must retry on the next tick instead of
    consuming the day's ``as_of``."""

    data_age_days: int = 0
    """FORENSIC STAMP (observability only) — how stale the served HAARP bar was
    on the GOOD-signal path, in whole days: ``(now.date() - ta.as_of).days``.

    NO ALERT is derived from this field.  On the SUCCESS path the daemon passes
    ``as_of = scheduler.expected_as_of`` and ``HaarpAllocationSource`` enforces a
    STRICT ``raw.as_of == as_of``, so the served age is essentially always exactly
    ``1`` (the normal closed-daily-bar lag).  A "pre-stale" alert here would never
    fire on the success path — a never-firing alert is noise — so this is recorded
    purely for the post-hoc audit trail, never to gate trading.

    Convention on the NO-good-signal / de-risk path: left at the SENTINEL ``-1``
    ("no signal served this cycle, no age to measure") to distinguish it from a
    genuine ``0``-day-old good signal.  Only the good-signal rebalanced and hold
    returns carry a real (>=0) age."""

    target_weights: dict[str, float] | None = None
    """The target weights this cycle attempted (served allocation's weights), or
    ``None`` when the signal fetch/validation failed.  Used by the daemon to stash
    ``retry_target`` when opening a bounded-retry window."""

    n_liquidity_deferred: int = 0
    """Legs the executor skipped for insufficient opposing depth (retryable)."""


def _emit_alert(
    alert_sink: AlertSink | None,
    alert_type: AlertType,
    severity: Severity,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Emit an alert via ``alert_sink``, swallowing all errors.  No-op when sink is None."""
    if alert_sink is None:
        return
    alert = Alert(
        type=alert_type,
        severity=severity,
        message=message,
        context=context or {},
    )
    try:
        alert_sink.emit(alert)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "rebalance._emit_alert: alert_sink.emit failed (swallowed)",
            alert_type=alert_type.value,
            error=str(exc),
        )


def _handle_no_good_signal(
    *,
    cfg: Config,
    client: Any,
    ledger: IntentLedger,
    state: State,
    state_path: str | Path,
    now_ts: str,
    alert_sink: AlertSink | None,
    alert_type: AlertType,
    reason_prefix: str,
    unresolved: list[Intent],
) -> CycleReport:
    """Decide HOLD vs DE-RISK TO CASH when no good signal is available this cycle.

    A "good signal" is one that was both fetched AND passed ``validate_target_allocation``.

    Staleness anchor (policy)
    -------------------------
    The clock is anchored to the most meaningful "we last knew the book was
    intentional" timestamp::

        anchor = last_successful_signal_ts or first_cycle_ts or now_ts

    - If a good signal has ever arrived → anchor on it (original N-3 behaviour).
    - Else (NEVER had a good signal) → anchor on ``first_cycle_ts`` (this
      deployment's first boot).  This treats never-had-signal as STALE while still
      honouring the window: a fresh deploy whose very first signal fails has
      ``anchor ≈ now`` → ``age ≈ 0`` → HOLD (grace = the staleness window).  Only
      after ``max_signal_staleness_days`` elapse since first boot WITHOUT ever
      getting a good signal does it de-risk to cash (flattening any orphan /
      pre-existing positions).  A flat book makes ``flatten_all`` a no-op.

    Decision logic
    --------------
    - ``age_days <= cfg.max_signal_staleness_days`` → within window; HOLD + WARNING.
    - ``age_days > cfg.max_signal_staleness_days`` → beyond window; CRITICAL alert +
      ``flatten_all`` (reduce-only market_close), reset baseline to {}, persist.
    """
    last = state.last_successful_signal_ts
    # Unified anchor: prefer the last good signal; else this deployment's first
    # boot; else (defensive) now_ts → age 0 → HOLD.  first_cycle_ts is set at the
    # top of run_cycle, so it is non-None here in normal operation.
    anchor = last or state.first_cycle_ts or now_ts

    # H1: tz-normalize BOTH timestamps before subtracting.  ``now_ts`` is always
    # written tz-aware (daemon passes ``now.isoformat()``), but a legacy or
    # hand-edited state file could carry a NAIVE anchor — subtracting aware−naive
    # raises TypeError.  Coerce naive → UTC for both (mirrors ``daemon.boot``).
    now_dt = datetime.datetime.fromisoformat(now_ts)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.UTC)
    anchor_dt = datetime.datetime.fromisoformat(anchor)
    if anchor_dt.tzinfo is None:
        anchor_dt = anchor_dt.replace(tzinfo=datetime.UTC)
    age_days = (now_dt - anchor_dt).total_seconds() / 86400.0

    if age_days > cfg.max_signal_staleness_days:
        # N-3: signal is stale beyond the window → DE-RISK TO CASH.
        msg = (
            f"{reason_prefix} — signal stale {age_days:.1f}d "
            f"> {cfg.max_signal_staleness_days}d — DE-RISKING TO CASH"
        )
        _logger.error("rebalance._handle_no_good_signal: %s", msg)
        _emit_alert(
            alert_sink,
            alert_type,
            Severity.CRITICAL,
            msg,
            {
                "now_ts": now_ts,
                "anchor_ts": anchor,
                "last_successful_signal_ts": last,
                "first_cycle_ts": state.first_cycle_ts,
                "age_days": round(age_days, 2),
                "max_signal_staleness_days": cfg.max_signal_staleness_days,
            },
        )
        # Reduce-only flatten — idempotent if already flat.
        # TODO (D5/spot): flatten_all is PERP-ONLY — it reduce-only market_closes
        # perp positions and does NOT sell spot holdings.  When cfg.spot_routing is
        # non-empty, a de-risk-to-cash leaves the routed SPOT balances untouched.
        # Spot de-risk = SELL the spot balance (SPOT has no reduce-only market_close);
        # deferred to a later increment.  For now de-risk flattens perp only.
        kill_results = flatten_all(
            client,
            cfg,
            ledger,
            strategy_id="DERISK",
            now_ts=now_ts,
            alert_sink=alert_sink,
        )
        # H3: surface a PARTIAL kill.  ``flatten_all`` only emits a KILL_SWITCH
        # alert via alert_sink — when alert_sink is None a coin that failed to
        # close is otherwise visible only in logs.  Inspect the per-coin results:
        # any status other than "filled" (error / partial / resting) means the
        # de-risk did NOT fully flatten.  Flag it on the report + audit record so
        # the incomplete kill is observable even with no alert sink.
        derisk_incomplete = any(
            r.get("status") != "filled" for r in kill_results
        )
        if derisk_incomplete:
            _logger.error(
                "rebalance._handle_no_good_signal: de-risk kill INCOMPLETE — "
                "one or more positions did NOT fully close",
                outcomes=[r.get("status") for r in kill_results],
            )
        # Reset frozen ratchet baseline to CASH so the next good signal bootstraps
        # a fresh rebalance instead of comparing against a stale target.
        state.last_rebalanced_target = {}
        state.last_rebalanced_as_of = now_ts
        save(state, state_path)
        return CycleReport(
            rebalanced=False,
            reason="derisk_to_cash",
            exec_report=None,
            unresolved_pending=unresolved,
            committed=True,
            signal_ok=False,
            derisked=True,
            derisk_incomplete=derisk_incomplete,
            data_age_days=-1,  # no good signal served → no age to measure (sentinel)
        )

    # Within the staleness window — HOLD (cache last allocation).
    msg = (
        f"{reason_prefix} — signal not fresh "
        f"({age_days:.1f}d ≤ {cfg.max_signal_staleness_days}d) — holding"
    )
    _logger.warning("rebalance._handle_no_good_signal: %s", msg)
    _emit_alert(
        alert_sink,
        alert_type,
        Severity.WARNING,
        msg,
        {
            "now_ts": now_ts,
            "anchor_ts": anchor,
            "last_successful_signal_ts": last,
            "first_cycle_ts": state.first_cycle_ts,
            "age_days": round(age_days, 2),
            "max_signal_staleness_days": cfg.max_signal_staleness_days,
        },
    )
    return CycleReport(
        rebalanced=False,
        reason="hold_stale_within_window",
        exec_report=None,
        unresolved_pending=unresolved,
        committed=False,
        signal_ok=False,
        derisked=False,
        data_age_days=-1,  # no good signal served → no age to measure (sentinel)
    )


def run_cycle(
    cfg: Config,
    client: Any,
    source: AllocationSource,
    ledger: IntentLedger,
    state: State,
    state_path: str | Path,
    *,
    now_ts: str,
    as_of: datetime.date | None = None,
    is_filled: Callable[[Intent], bool] | None = None,
    alert_sink: AlertSink | None = None,
) -> CycleReport:
    """Execute one live rebalance cycle.

    Steps
    -----
    0. Boot-time reconciliation of any PENDING intents from a prior crash.
    1. Fetch an on-chain snapshot (failure propagates — cannot act blind).
    2. Retrieve the target allocation.  On fetch failure → staleness logic.
    3. Validate the target allocation.  On validation failure → staleness logic.
    4. Record freshness: update ``state.last_successful_signal_ts`` + persist.
    5. Compute the size plan.
    6. Execute orders (only if plan.rebalance is True).
    7. Commit the frozen new target to persistent state (only if rebalanced clean).

    Parameters
    ----------
    cfg:
        Executor configuration.
    client:
        Duck-typed ``HLClient`` (or fake) with read + write methods.
    source:
        Allocation source implementing the ``AllocationSource`` protocol.
    ledger:
        Intent ledger for crash-safe PENDING → DONE tracking.
    state:
        Current persistent executor state object (mutable — updated in-place
        before being persisted via ``save``).
    state_path:
        Path to the state JSON file.  Used by ``save`` for atomic persistence.
    now_ts:
        ISO timestamp injected by the caller (deterministic for tests).
    as_of:
        Optional reference date forwarded to ``source.get_target_allocation``.
    is_filled:
        Optional resolver callable passed to ``ledger.reconcile_pending``.
        In production this queries the chain by cloid; the default is a
        conservative ``lambda intent: False`` that leaves intents PENDING.
    alert_sink:
        Optional alert sink.  Receives ``STALE_SIGNAL`` / ``INVALID_SIGNAL``
        alerts when the signal fetch or validation fails.  Never crashes when
        ``None`` (all emit calls are guarded).

    Returns
    -------
    CycleReport
        Summary of the cycle, including whether a rebalance fired and whether
        state was committed.
    """
    # Step 0: Boot-time reconciliation of stale PENDING intents.
    resolver: Callable[[Intent], bool] = is_filled if is_filled is not None else (lambda _: False)
    unresolved = ledger.reconcile_pending(resolver)
    if unresolved:
        _logger.warning(
            "run_cycle: unresolved PENDING intents after reconciliation",
            n_unresolved=len(unresolved),
            cloids=[i.cloid for i in unresolved],
        )

    # Boot anchor (policy): record when THIS deployment first ran, once.  This is
    # the staleness anchor used by _handle_no_good_signal when no good signal has
    # EVER arrived — so a fresh deploy whose first signal fails holds for the
    # window (anchored to now), and only de-risks orphan positions after the
    # window elapses since first boot.  Written once and never updated.
    if state.first_cycle_ts is None:
        state.first_cycle_ts = now_ts
        save(state, state_path)

    # D5: hybrid spot+perp venue routing.  When non-empty, coins in spot_routing
    # trade on HL SPOT (no funding); the rest on perp.  Threaded into BOTH snapshots
    # (so equity spans both pools) AND execute() (so spot coins route to
    # submit_spot_ioc).  Empty/absent ⇒ None ⇒ byte-identical perp-only cycle.
    #
    # COMPOSITION-ROOT NOTE (B11): when cfg.spot_routing is non-empty the client MUST
    # be spot-enabled (HLClient.for_trading(..., enable_spot=True)).  The snapshot's
    # spot reads raise otherwise (D2 fail-loud, no silent perp-only fallback).  The
    # bootstrap constructs the spot-enabled client; this module only consumes it.
    spot_routing = cfg.spot_routing or None

    # Step 1: Snapshot (failure PROPAGATES — we cannot act without reading positions).
    snap = reconciler.snapshot(client, spot_routing=spot_routing)

    # Steps 2-3: Fetch + validate.  Both failure modes route to staleness logic.
    whitelist = set(cfg.universe_perp_map) if cfg.universe_perp_map else None
    client_id = cfg.signal_source.client_id or ""

    ta = None
    try:
        ta = source.get_target_allocation(as_of)
        # Temporarily compute whitelist from TA if universe_perp_map is empty.
        effective_whitelist = whitelist if whitelist is not None else set(ta.weights.keys())
        validate_target_allocation(
            ta,
            max_per_asset=cfg.max_per_asset,
            whitelist=effective_whitelist,
            client_id=client_id,
        )
    except AllocationRejected as exc:
        # Present-but-UNTRUSTED signal (S-11): do NOT update freshness clock.
        _logger.warning(
            "run_cycle: AllocationRejected — signal present but untrusted",
            error=str(exc),
        )
        return _handle_no_good_signal(
            cfg=cfg,
            client=client,
            ledger=ledger,
            state=state,
            state_path=state_path,
            now_ts=now_ts,
            alert_sink=alert_sink,
            alert_type=AlertType.INVALID_SIGNAL,
            reason_prefix=f"invalid signal: {exc}",
            unresolved=unresolved,
        )
    except Exception as exc:
        # Signal unavailable (network error, SignalRejected, etc.).
        _logger.warning(
            "run_cycle: signal fetch failed — treating as unavailable",
            error=str(exc),
        )
        return _handle_no_good_signal(
            cfg=cfg,
            client=client,
            ledger=ledger,
            state=state,
            state_path=state_path,
            now_ts=now_ts,
            alert_sink=alert_sink,
            alert_type=AlertType.STALE_SIGNAL,
            reason_prefix=f"signal unavailable: {exc}",
            unresolved=unresolved,
        )

    # Step 4: Record freshness — good signal (fetched + validated).
    # Persist even on a hold so the staleness clock advances correctly.
    state.last_successful_signal_ts = now_ts
    save(state, state_path)

    # The served target weights — carried on EVERY CycleReport return from here
    # on (the signal-failure early returns above keep the None default).  The
    # daemon stashes this as ``retry_target`` when opening a bounded-retry window.
    served_weights = dict(ta.weights)

    # FORENSIC STAMP (observability only, NO ALERT): how stale the served bar was,
    # in whole days = (now.date() - ta.as_of).days.  On the success path the daemon
    # passes as_of = scheduler.expected_as_of and HaarpAllocationSource enforces a
    # STRICT raw.as_of == as_of, so this is essentially always exactly 1 (the normal
    # closed-daily-bar lag) — a "pre-stale" alert would never fire here, so we only
    # record the age for the audit trail.  ``now_ts`` is ISO (possibly naive → coerce
    # to UTC, mirroring _handle_no_good_signal); ``ta.as_of`` is a datetime.date.
    now_dt_good = datetime.datetime.fromisoformat(now_ts)
    if now_dt_good.tzinfo is None:
        now_dt_good = now_dt_good.replace(tzinfo=datetime.UTC)
    data_age_days = (now_dt_good.date() - ta.as_of).days

    # Recompute effective_whitelist for sizer (ta is guaranteed non-None here).
    effective_whitelist = whitelist if whitelist is not None else set(ta.weights.keys())

    # Step 5: Size plan.
    sp = sizer.plan(ta, snap, state, cfg)

    # G4a: MIN_NOTIONAL_UNALLOCATED — coins the strategy WANTS that were silently
    # skipped by the sizer (below_min_notional or zeroed_by_cap).  Emit BEFORE the
    # hold/execute branch so the operator knows even on a hold cycle that a coin
    # it expected to be allocated has been dropped.
    unalloc = [
        s for s in sp.skipped
        if s.get("reason") in ("below_min_notional", "zeroed_by_cap")
    ]
    if unalloc:
        _emit_alert(
            alert_sink,
            AlertType.MIN_NOTIONAL_UNALLOCATED,
            Severity.WARNING,
            f"{len(unalloc)} target coin(s) unallocated (below min-notional / zeroed by cap)",
            {
                "coins": [s["coin"] for s in unalloc],
                "reasons": [s["reason"] for s in unalloc],
            },
        )

    # Step 6: Execute (only when rebalance=True).
    if not sp.rebalance:
        _logger.info(
            "run_cycle: holding — no rebalance",
            reason=sp.reason,
        )
        return CycleReport(
            rebalanced=False,
            reason=sp.reason,
            exec_report=None,
            unresolved_pending=unresolved,
            committed=False,
            signal_ok=True,
            derisked=False,
            n_unallocated=len(unalloc),
            data_age_days=data_age_days,
            target_weights=served_weights,
            n_liquidity_deferred=0,
        )

    exec_report = execute(
        sp,
        client,
        cfg,
        ledger,
        strategy_id=ta.strategy_id,
        as_of=ta.as_of.isoformat(),
        now_ts=now_ts,
        spot_routing=spot_routing,
    )

    # MF-9: rate-budget defer — zero orders submitted, retry on next tick.
    # Skip commit, skip DIVERGENCE snapshot (nothing was traded), emit THROTTLE.
    if exec_report.deferred:
        _emit_alert(
            alert_sink,
            AlertType.THROTTLE,
            Severity.WARNING,
            f"cycle DEFERRED — {exec_report.defer_reason}; will retry next tick",
            {
                "defer_reason": exec_report.defer_reason,
                "as_of": ta.as_of.isoformat(),
                "n_orders": len(sp.orders),
            },
        )
        _logger.warning(
            "run_cycle: cycle DEFERRED by rate-budget pre-flight — "
            "baseline stays frozen, will retry next tick",
            defer_reason=exec_report.defer_reason,
        )
        return CycleReport(
            rebalanced=True,
            reason=sp.reason,
            exec_report=exec_report,
            unresolved_pending=unresolved,
            committed=False,
            signal_ok=True,
            derisked=False,
            n_unallocated=len(unalloc),
            deferred=True,
            data_age_days=data_age_days,
            target_weights=served_weights,
            n_liquidity_deferred=exec_report.n_liquidity_deferred,
        )

    # MF-9 mid-cycle: THROTTLE alert when any coin was rate-limited (non-deferred path).
    if exec_report.n_throttled > 0:
        _emit_alert(
            alert_sink,
            AlertType.THROTTLE,
            Severity.WARNING,
            f"{exec_report.n_throttled} coin(s) rate-limited mid-cycle",
            {
                "n_throttled": exec_report.n_throttled,
                "as_of": ta.as_of.isoformat(),
            },
        )

    # Step 7: Commit the FROZEN new target to persistent state (S-2) — ONLY when
    # execution was CLEAN.  Committing on a hard-failed or partial execution would
    # poison the ratchet: the baseline would record a target we never achieved, so
    # the next cycle (max_delta=0 vs that target) would never re-attempt the unfilled
    # legs (F-1).  Gate the commit on:
    #   - not aborted (no pre-flight CapBreach), AND
    #   - n_error == 0 (no rejected/errored coins), AND
    #   - n_partial == 0 (no partially-filled coins).
    # Intentionally-SKIPPED coins (below_min_notional / zeroed_by_cap) live in
    # sp.skipped and never reach exec_report — they do NOT block the commit.
    # When NOT committing, the baseline stays frozen so the next cycle re-attempts.
    clean = (
        not exec_report.aborted
        and exec_report.n_error == 0
        and exec_report.n_partial == 0
        and exec_report.n_liquidity_deferred == 0
    )

    committed = False
    if clean:
        try:
            state.last_rebalanced_target = dict(ta.weights)
            state.last_rebalanced_as_of = ta.as_of.isoformat()
            save(state, state_path)
            committed = True
            _logger.info(
                "run_cycle: state committed (clean execution)",
                as_of=ta.as_of.isoformat(),
                n_weights=len(ta.weights),
            )
        except Exception as exc:
            # B10: circuit-breaker / hard-alert goes here.
            _logger.error(
                "run_cycle: FAILED to commit state — next cycle may spuriously rebalance",
                error=str(exc),
            )
    else:
        # B10: hard-alert on dirty execution goes here.
        _logger.warning(
            "run_cycle: execution NOT clean — baseline left FROZEN, will re-attempt next cycle",
            aborted=exec_report.aborted,
            abort_reason=exec_report.abort_reason,
            n_error=exec_report.n_error,
            n_partial=exec_report.n_partial,
        )

    # G4a: DIVERGENCE — compare achieved post-execute weights against the target.
    # This is informational: runs regardless of commit success so the magnitude is
    # always surfaced.  The try/except ensures a snapshot failure never crashes the
    # cycle (a failed snapshot after execution is non-fatal; EXEC_ERROR / CYCLE_INCOMPLETE
    # from the daemon already captures execution problems).
    max_dev = 0.0
    try:
        # D5: the divergence snapshot MUST also span the hybrid book — otherwise the
        # spot pool is dropped from the denominator and achieved spot weights are
        # inflated, understating (or spuriously inflating) the computed divergence.
        post_snap = reconciler.snapshot(client, spot_routing=spot_routing)
        achieved = reconciler.achieved_weights(post_snap)
        target_by_perp = {
            cfg.universe_perp_map.get(t, t): w for t, w in ta.weights.items()
        }
        perps = set(target_by_perp) | set(achieved)
        max_dev = max(
            (abs(target_by_perp.get(p, 0.0) - achieved.get(p, 0.0)) for p in perps),
            default=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "run_cycle: post-execute divergence snapshot failed (skipped)",
            error=str(exc),
        )

    threshold_frac = cfg.divergence_alert_pct / 100.0
    if max_dev > threshold_frac:
        sev = (
            Severity.CRITICAL
            if max_dev > 2 * threshold_frac
            else Severity.WARNING
        )
        _emit_alert(
            alert_sink,
            AlertType.DIVERGENCE,
            sev,
            (
                f"post-rebalance divergence {max_dev:.4f} > {threshold_frac:.4f} "
                "(achieved vs target)"
            ),
            {
                "max_divergence": round(max_dev, 6),
                "as_of": ta.as_of.isoformat(),
            },
        )

    return CycleReport(
        rebalanced=True,
        reason=sp.reason,
        exec_report=exec_report,
        unresolved_pending=unresolved,
        committed=committed,
        signal_ok=True,
        derisked=False,
        max_divergence=max_dev,
        n_unallocated=len(unalloc),
        data_age_days=data_age_days,
        target_weights=served_weights,
        n_liquidity_deferred=exec_report.n_liquidity_deferred,
    )


class _FrozenSource:
    """An ``AllocationSource`` serving a fixed, pre-computed target.

    Used by ``retry_attempt`` to drive ``run_cycle`` toward the frozen
    ``retry_target`` WITHOUT re-fetching the signal.  The served allocation is
    all-risk (``cash = 1 - sum(weights)``), echoing the strategy_id/audience the
    executor expects so ``validate_target_allocation`` passes the audience guard.
    """

    def __init__(
        self,
        weights: dict[str, float],
        as_of: str,
        *,
        client_id: str,
        strategy_id: str = "retry",
        model_rev: str = "retry",
    ) -> None:
        self._weights = dict(weights)
        self._as_of = datetime.date.fromisoformat(as_of)
        self._client_id = client_id
        self._strategy_id = strategy_id
        self._model_rev = model_rev

    def get_target_allocation(
        self, as_of: datetime.date | None = None
    ) -> TargetAllocation:
        return TargetAllocation(
            as_of=self._as_of,
            weights=dict(self._weights),
            cash=1.0 - sum(self._weights.values()),
            strategy_id=self._strategy_id,
            model_rev=self._model_rev,
            audience=self._client_id,
        )


def retry_attempt(
    cfg: Config,
    client: Any,
    ledger: IntentLedger,
    state: State,
    state_path: str | Path,
    *,
    retry_target: dict[str, float],
    retry_as_of: str,
    now_ts: str,
    alert_sink: AlertSink | None = None,
    is_filled: Callable[[Intent], bool] | None = None,
) -> CycleReport:
    """Execute ONE bounded-retry attempt toward the frozen ``retry_target``.

    Reuses ``run_cycle`` wholesale via ``_FrozenSource`` — identical execution
    path (reconcile → size → execute → commit-if-clean).  Anti-double-order is
    structural: the sizer computes the delta from LIVE positions, so an
    already-filled leg yields 0 orders.

    Parameters
    ----------
    cfg, client, ledger, state, state_path, now_ts, alert_sink, is_filled:
        Forwarded verbatim to ``run_cycle``.
    retry_target:
        The frozen target weights to retry toward (the served allocation whose
        residual leg was liquidity-deferred on the daily cycle).
    retry_as_of:
        ISO date string of the frozen allocation.  Becomes the ``ta.as_of`` of
        the synthetic ``TargetAllocation`` and the ``as_of`` reference forwarded
        to ``run_cycle``.

    Returns
    -------
    CycleReport
        Same shape as a normal ``run_cycle`` report.
    """
    source = _FrozenSource(
        retry_target,
        retry_as_of,
        client_id=cfg.signal_source.client_id or "",
    )
    return run_cycle(
        cfg,
        client,
        source,
        ledger,
        state,
        state_path,
        now_ts=now_ts,
        as_of=datetime.date.fromisoformat(retry_as_of),
        is_filled=is_filled,
        alert_sink=alert_sink,
    )
