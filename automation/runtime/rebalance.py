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
from automation.allocation.sleeves import (
    SleeveOutcome,
    combine_sleeves,
    resolve_outcome,
)
from automation.core.config import Config, SignalSourceConfig
from automation.core.redaction import get_logger
from automation.execution import reconciler, sizer
from automation.execution.executor import ExecReport, execute
from automation.reporting.alerts import Alert, AlertSink, AlertType, Severity
from automation.reporting.state import State, save
from automation.runtime.intent_ledger import Intent, IntentLedger
from automation.runtime.kill_switch import flatten_all

#: The synthetic audience / client_id stamped on the COMBINED TargetAllocation.
#: The per-sleeve audience guard already fired inside each sleeve's own
#: ``validate_target_allocation`` (or the RemoteSignalSource's MF-17 check) during
#: the fetch loop; the combined target is an internal construct that no longer
#: carries any single sleeve's audience, so it validates against this constant.
_COMBINED_AUDIENCE = "combined"

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

    sleeves: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-sleeve attribution for this cycle (CRASH Phase 1, spec §4.5).  Maps each
    sleeve ``name`` → ``{"as_of": str | None, "weights": dict[str, float],
    "outcome": "fresh" | "hold" | "stale"}``.  ``weights`` are the sleeve-relative
    SIGNED weights that fed the combination (served weights for FRESH; the frozen
    ``last_weights`` for HOLD; ``{}`` for STALE).  Written into the audit record so
    each cycle's combined book can be attributed back to its sleeves WITHOUT
    per-sleeve position bookkeeping (positions are fungible per-coin on HL).  Empty
    ``{}`` on the no-good-signal / de-risk early returns (no combination ran)."""


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


@dataclass(frozen=True)
class _SleeveFetch:
    """The outcome of fetching + validating ONE sleeve in the per-sleeve loop.

    A FRESH fetch (``served_ok=True``) carries the validated ``TargetAllocation``;
    a failed fetch carries the exception and whether it was an ``AllocationRejected``
    (present-but-untrusted → ``INVALID_SIGNAL``) versus any other error
    (unavailable → ``STALE_SIGNAL``).  The classification drives BOTH the per-sleeve
    HOLD/STALE alert and — when EVERY sleeve fails — the legacy no-good-signal
    routing (so the singular path's INVALID_SIGNAL / STALE_SIGNAL alerts and reason
    strings are reproduced byte-for-byte).
    """

    sleeve: SignalSourceConfig
    served_ok: bool
    ta: TargetAllocation | None
    exc: Exception | None
    is_rejected: bool


def _fetch_sleeve(
    *,
    cfg: Config,
    sleeve: SignalSourceConfig,
    source: AllocationSource,
    as_of: datetime.date | None,
) -> _SleeveFetch:
    """Fetch + validate ONE sleeve's target allocation (spec §4.1).

    Mirrors the legacy single-source fetch+validate block exactly so the singular
    ("default") sleeve reproduces today's behaviour:
      * fetch ``source.get_target_allocation(as_of)``;
      * validate with THIS sleeve's ``client_id`` / ``allow_short`` and the global
        whitelist (RemoteSignalSource also self-validates with the same args — the
        re-validation here is a no-op for it but is the ONLY gate for the local
        ``equal_weight`` / test sources);
      * enforce the ``cfg.pinned_model_rev`` pin (remote NOGATE-TEST defense).

    Never raises: any fetch/validation failure is captured into the returned
    ``_SleeveFetch`` (``served_ok=False``) so the caller can apply the per-sleeve
    HOLD/STALE staleness policy uniformly.  ``AllocationRejected`` is flagged
    (``is_rejected=True``) to preserve the present-but-untrusted → INVALID_SIGNAL
    routing.
    """
    whitelist = set(cfg.universe_perp_map) if cfg.universe_perp_map else None
    client_id = sleeve.client_id or ""
    try:
        ta = source.get_target_allocation(as_of)
        effective_whitelist = (
            whitelist if whitelist is not None else set(ta.weights.keys())
        )
        validate_target_allocation(
            ta,
            max_per_asset=cfg.max_per_asset,
            whitelist=effective_whitelist,
            client_id=client_id,
            allow_short=sleeve.allow_short,
        )
        # Model-rev pin for REMOTE sources (the local HaarpAllocationSource enforces
        # its own pin at fetch; remote payloads had NO rev defense until this check).
        # Mismatch routes through the AllocationRejected path (S-11: present-but-
        # untrusted — the freshness clock does NOT advance).
        if cfg.pinned_model_rev is not None and ta.model_rev != cfg.pinned_model_rev:
            raise AllocationRejected(
                f"model_rev pin mismatch: served '{ta.model_rev}' != pinned "
                f"'{cfg.pinned_model_rev}'"
            )
    except AllocationRejected as exc:
        _logger.warning(
            "run_cycle: sleeve AllocationRejected — present but untrusted",
            sleeve=sleeve.name,
            error=str(exc),
        )
        return _SleeveFetch(
            sleeve=sleeve, served_ok=False, ta=None, exc=exc, is_rejected=True
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "run_cycle: sleeve fetch failed — treating as unavailable",
            sleeve=sleeve.name,
            error=str(exc),
        )
        return _SleeveFetch(
            sleeve=sleeve, served_ok=False, ta=None, exc=exc, is_rejected=False
        )
    return _SleeveFetch(
        sleeve=sleeve, served_ok=True, ta=ta, exc=None, is_rejected=False
    )


def _within_staleness_window(
    *,
    cfg: Config,
    sleeve: SignalSourceConfig,
    state: State,
    now_ts: str,
) -> bool:
    """Whether *sleeve* is still within its per-sleeve staleness window.

    The anchor is the sleeve's own ``last_success_ts`` from the ledger; on the
    FIRST cycle (no ledger entry yet) it falls back to ``state.first_cycle_ts``
    (this deployment's boot), so a fresh deploy whose very first sleeve fetch
    fails HOLDS within the window rather than de-risking immediately — mirroring
    the global no-good-signal anchor logic.  The window length is
    ``cfg.effective_staleness_days(sleeve)`` (per-sleeve override, else global).
    """
    rec = state.sleeves.get(sleeve.name) or {}
    anchor: str = rec.get("last_success_ts") or state.first_cycle_ts or now_ts
    now_dt = datetime.datetime.fromisoformat(now_ts)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.UTC)
    anchor_dt = datetime.datetime.fromisoformat(anchor)
    if anchor_dt.tzinfo is None:
        anchor_dt = anchor_dt.replace(tzinfo=datetime.UTC)
    age_days = (now_dt - anchor_dt).total_seconds() / 86400.0
    return bool(age_days <= cfg.effective_staleness_days(sleeve))


def _normalize_sources(
    sources: AllocationSource | dict[str, AllocationSource],
) -> dict[str, AllocationSource]:
    """Accept the legacy single source OR the multi-sleeve dict; return the dict.

    The plumbing change keeps ``run_cycle``'s third positional parameter
    byte-compatible: every existing caller (and ``test_rebalance.py``) passes a
    bare ``AllocationSource`` there, which is wrapped into the lone ``"default"``
    sleeve — the SAME name the singular ``signal_source`` config normalizes to via
    ``cfg.sleeves()``.  A new daemon passes the ``build_sources(cfg)`` dict directly.
    """
    if isinstance(sources, dict):
        return sources
    return {"default": sources}


def _detect_liquidations(
    *,
    cfg: Config,
    state: State,
    snap: Any,
    now_ts: str,
    state_path: str | Path,
    alert_sink: AlertSink | None,
) -> None:
    """Detect isolated liquidations and arm a per-coin re-entry cooldown (M3).

    HEURISTIC (spec §5, red-team M3)
    --------------------------------
    A coin is treated as LIQUIDATED when ALL of:

      * it carries a NONZERO weight in ``state.last_rebalanced_target`` (the FROZEN
        baseline — the last target we actually committed), AND
      * its live position in ``snap.positions`` is ABSENT or zero (the book is flat
        for it), AND
      * it is not already under a cooldown (idempotent — don't re-arm / re-alert a
        coin we already flagged on a prior cycle while it stays flat).

    WHY this is the honest "our exit" signal
    -----------------------------------------
    The committed baseline is the SOLE record of what we last *intentionally* held.
    When WE close a coin, ``run_cycle`` commits the new target — which records that
    coin at ``0.0`` (or drops it) — so a coin we deliberately flattened has a
    baseline weight of ``0`` and is NOT flagged.  Only a coin whose baseline is
    still NONZERO yet has vanished from the book was closed by SOMETHING ELSE — an
    isolated liquidation (the daemon submits no order between the prior commit and
    this snapshot, so the engine itself cannot have produced the flat).

    FALSE-POSITIVE BOUND
    --------------------
    Detection runs at the TOP of the cycle, before any order for THIS cycle fires,
    so an order we are about to place can never masquerade as a liquidation.  The
    ONLY residual false positive is a coin whose close we submitted on a PRIOR
    cycle that executed DIRTY (so the baseline stayed frozen at the old nonzero
    target) yet actually filled — leaving a nonzero baseline against a flat book.
    That is a narrow operator-intervention / partial-failure edge, and the
    consequence is conservative-safe: arming a (default 1-day) SAME-DIRECTION
    cooldown only blocks RE-OPENING the same side — it never forces a trade and
    never blocks an exit or an opposite-direction flip.  The cost of a false
    positive is at most a one-day delayed same-direction re-entry; the cost of a
    false negative is feeding fresh margin into an active squeeze.  We bias toward
    the cheaper error.

    Side effects: mutates ``state.liq_cooldowns`` and persists, emits one
    ``POSITION_LIQUIDATED`` CRITICAL per newly-detected coin.  Pure-flat books and
    all-zero baselines are no-ops.
    """
    baseline = state.last_rebalanced_target
    if not baseline:
        return
    today_iso = datetime.datetime.fromisoformat(now_ts).date().isoformat()
    newly_flagged: list[str] = []
    for ticker, w in baseline.items():
        if w == 0.0:
            continue
        if ticker in state.liq_cooldowns:
            # Already under cooldown — don't re-arm or re-alert while it stays flat.
            continue
        perp = cfg.universe_perp_map.get(ticker, ticker)
        live = snap.positions.get(perp)
        live_f = float(live) if live is not None else 0.0
        if live_f == 0.0:
            # Baseline says we hold it; the book says we don't, and we issued no
            # order to flatten it → liquidated.
            state.liq_cooldowns[ticker] = today_iso
            newly_flagged.append(ticker)
            _emit_alert(
                alert_sink,
                AlertType.POSITION_LIQUIDATED,
                Severity.CRITICAL,
                f"position {ticker!r} vanished from the book with no exit order of "
                "ours — apparent liquidation; arming re-entry cooldown",
                {
                    "coin": ticker,
                    "baseline_weight": w,
                    "cooldown_date": today_iso,
                    "liquidation_cooldown_days": cfg.liquidation_cooldown_days,
                },
            )
    if newly_flagged:
        save(state, state_path)


def _cooldown_blocked_set(
    *,
    cfg: Config,
    state: State,
    combined_weights: dict[str, float],
    now_ts: str,
    state_path: str | Path,
) -> frozenset[str]:
    """Compute the SAME-DIRECTION cooldown skip set; expire stale cooldowns (M3).

    For each coin in ``state.liq_cooldowns``:

      * EXPIRED (``today − cooldown_date >= cfg.liquidation_cooldown_days``) →
        dropped from the map (and persisted) so it trades normally and the entry
        clears.  A zero-day cooldown therefore expires the very next calendar day.
      * ACTIVE → blocked ONLY when the NEW combined target re-enters in the SAME
        direction as the liquidated position.  The liquidated direction is the sign
        of the coin in ``state.last_rebalanced_target`` (the committed baseline,
        which preserves the last-acted direction across the cooldown — a
        same-direction re-entry is skipped-but-committed, so the baseline sign is
        sticky).  An OPPOSITE-direction re-entry (the squeeze flipped the thesis)
        is intentionally NOT blocked; a target that gave the coin up (sign 0 / no
        entry) has nothing to block either.

    Returns a TICKER-space frozenset the sizer drops from the traded set.  Side
    effect: persists ``state`` when any expired cooldown is cleared.
    """
    if not state.liq_cooldowns:
        return frozenset()
    today = datetime.datetime.fromisoformat(now_ts).date()
    baseline = state.last_rebalanced_target or {}
    blocked: set[str] = set()
    expired: list[str] = []
    for ticker, date_iso in list(state.liq_cooldowns.items()):
        age_days = (today - datetime.date.fromisoformat(date_iso)).days
        if age_days >= cfg.liquidation_cooldown_days:
            expired.append(ticker)
            continue
        liq_dir = _sign(baseline.get(ticker, 0.0))
        new_dir = _sign(combined_weights.get(ticker, 0.0))
        if new_dir != 0 and new_dir == liq_dir:
            blocked.add(ticker)
    if expired:
        for ticker in expired:
            del state.liq_cooldowns[ticker]
        save(state, state_path)
    return frozenset(blocked)


def _sign(x: float) -> int:
    """Sign of *x* as an int in ``{-1, 0, 1}`` (0 for exactly zero / NaN-safe ==)."""
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def run_cycle(
    cfg: Config,
    client: Any,
    sources: AllocationSource | dict[str, AllocationSource],
    ledger: IntentLedger,
    state: State,
    state_path: str | Path,
    *,
    now_ts: str,
    as_of: datetime.date | None = None,
    is_filled: Callable[[Intent], bool] | None = None,
    alert_sink: AlertSink | None = None,
    sleeve_configs: list[SignalSourceConfig] | None = None,
) -> CycleReport:
    """Execute one live rebalance cycle.

    Steps
    -----
    0. Boot-time reconciliation of any PENDING intents from a prior crash.
    1. Fetch an on-chain snapshot (failure propagates — cannot act blind).
    2. Per-sleeve fetch + validate + status resolution (FRESH / HOLD / STALE).
    3. Combine sleeves → ONE synthetic combined TargetAllocation + frozen coins.
    4. Record freshness + per-sleeve ledger writes.
    5. Compute the size plan (combined target, allow_short, frozen coins).
    6. Execute orders (only if plan.rebalance is True).
    7. Commit the frozen new COMBINED target to persistent state (if clean).

    ``sleeve_configs`` overrides ``cfg.sleeves()`` for the per-sleeve loop — the
    bounded-retry path (``retry_attempt``) passes a SINGLE synthetic combined
    sleeve so the frozen (possibly SIGNED) combined target re-drives through the
    same machinery without re-deriving per-source sleeves.  ``None`` ⇒ the normal
    ``cfg.sleeves()`` (the deployed path).

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

    # Step 1b: Liquidation detection (CRASH Phase 1, spec §5 / red-team M3).  Runs
    # BEFORE the per-sleeve fetch + plan so a newly-armed cooldown gates THIS cycle's
    # re-entry: a coin nonzero in the committed baseline but absent from the live
    # book (with no exit order of ours) was liquidated → arm a same-direction
    # re-entry cooldown + emit CRITICAL.  No-op on a flat book or all-zero baseline.
    _detect_liquidations(
        cfg=cfg,
        state=state,
        snap=snap,
        now_ts=now_ts,
        state_path=state_path,
        alert_sink=alert_sink,
    )

    # Normalize the source argument: the legacy single-source positional binds to a
    # one-element {"default": source} dict (byte-compatible with every existing
    # caller and test_rebalance.py); a new daemon passes the multi-sleeve dict.
    sources_by_name = _normalize_sources(sources)

    # The sleeve set to iterate: cfg.sleeves() (deployed path) unless the retry
    # path overrode it with a single synthetic combined sleeve.
    sleeves_list = sleeve_configs if sleeve_configs is not None else cfg.sleeves()

    # ----------------------------------------------------------------------------
    # Steps 2-3: PER-SLEEVE fetch + validate + status resolution (spec §4.1, §4.3).
    #
    # Iterate cfg.sleeves() in DETERMINISTIC config order.  For each sleeve: fetch +
    # validate (the sleeve's own client_id / allow_short / model-rev pin), then
    # resolve its status (FRESH / HOLD / STALE) under the per-sleeve staleness
    # policy.  A HOLD sleeve is FROZEN: we rebuild its outcome from the ledger's
    # last_weights so combine_sleeves freezes exactly those coins (resolve_outcome
    # returns {} for a hold — the daemon supplies the frozen weights).  A STALE
    # sleeve contributes {} and unwinds via the combined delta.
    #
    # For the singular ("default") sleeve this reproduces the legacy single-source
    # fetch behaviour exactly (one sleeve, budget 1.0, allow_short False, no frozen
    # coins) so the deployed long-only path stays byte-compatible.
    # ----------------------------------------------------------------------------
    outcomes: list[SleeveOutcome] = []
    sleeve_audit: dict[str, dict[str, Any]] = {}
    new_holds: dict[str, int] = {}
    fresh_served: dict[str, dict[str, float]] = {}   # name → served weights
    fresh_as_of: dict[str, datetime.date] = {}       # name → served as_of
    fresh_strategy_ids: list[str] = []               # FRESH sleeves' strategy ids
    any_short = False
    failures: list[_SleeveFetch] = []

    for sleeve in sleeves_list:
        source = sources_by_name.get(sleeve.name)
        if source is None:
            # A configured sleeve has no built source — fail-closed as unavailable
            # (treated like a fetch failure so the staleness policy applies rather
            # than crashing the whole cycle for the OTHER sleeves).
            failures.append(
                _SleeveFetch(
                    sleeve=sleeve,
                    served_ok=False,
                    ta=None,
                    exc=KeyError(f"no source built for sleeve {sleeve.name!r}"),
                    is_rejected=False,
                )
            )
            fetch = failures[-1]
        else:
            fetch = _fetch_sleeve(
                cfg=cfg, sleeve=sleeve, source=source, as_of=as_of
            )
            if not fetch.served_ok:
                failures.append(fetch)

        prior = state.sleeves.get(sleeve.name) or {}
        prior_holds = int(prior.get("consecutive_holds", 0))
        within = _within_staleness_window(
            cfg=cfg, sleeve=sleeve, state=state, now_ts=now_ts
        )
        served_w = dict(fetch.ta.weights) if (fetch.served_ok and fetch.ta) else {}
        outcome, holds = resolve_outcome(
            served_ok=fetch.served_ok,
            weights=served_w,
            allow_short=sleeve.allow_short,
            budget=sleeve.budget,
            name=sleeve.name,
            consecutive_holds=prior_holds,
            within_staleness_window=within,
        )
        new_holds[sleeve.name] = holds
        any_short = any_short or sleeve.allow_short

        if outcome.status == "fresh" and fetch.ta is not None:
            fresh_served[sleeve.name] = served_w
            fresh_as_of[sleeve.name] = fetch.ta.as_of
            fresh_strategy_ids.append(fetch.ta.strategy_id)
            sleeve_audit[sleeve.name] = {
                "as_of": fetch.ta.as_of.isoformat(),
                "weights": dict(served_w),
                "outcome": "fresh",
            }
        elif outcome.status == "hold":
            # FREEZE-NOTIONAL: resolve_outcome returns {} for a hold; supply the
            # ledger's last_weights so combine_sleeves freezes exactly those coins.
            frozen_w = dict(prior.get("last_weights") or {})
            outcome = SleeveOutcome(
                name=sleeve.name,
                status="hold",
                weights=frozen_w,
                budget=sleeve.budget,
                allow_short=sleeve.allow_short,
            )
            sleeve_audit[sleeve.name] = {
                "as_of": prior.get("last_as_of"),
                "weights": frozen_w,
                "outcome": "hold",
            }
        else:  # stale
            sleeve_audit[sleeve.name] = {
                "as_of": prior.get("last_as_of"),
                "weights": {},
                "outcome": "stale",
            }
        outcomes.append(outcome)

    # ----------------------------------------------------------------------------
    # ALL sleeves HOLD/STALE → no fresh signal this cycle → the existing
    # no-good-signal path (SP3 recheck machinery unchanged).  Reproduce the
    # singular path's INVALID_SIGNAL vs STALE_SIGNAL routing: if the FIRST failing
    # sleeve was an AllocationRejected (present-but-untrusted) use INVALID_SIGNAL,
    # else STALE_SIGNAL.  The per-sleeve ledger (consecutive_holds) is persisted
    # FIRST so a sustained dark short still escalates to STALE on the next cycle.
    # ----------------------------------------------------------------------------
    if not fresh_served:
        for name, holds in new_holds.items():
            rec = state.sleeves.get(name)
            if rec is not None:
                rec["consecutive_holds"] = holds
        save(state, state_path)
        first_fail = failures[0] if failures else None
        if first_fail is not None and first_fail.is_rejected:
            alert_type = AlertType.INVALID_SIGNAL
            reason_prefix = f"invalid signal: {first_fail.exc}"
        else:
            alert_type = AlertType.STALE_SIGNAL
            exc_txt = first_fail.exc if first_fail is not None else "no fresh sleeve"
            reason_prefix = f"signal unavailable: {exc_txt}"
        return _handle_no_good_signal(
            cfg=cfg,
            client=client,
            ledger=ledger,
            state=state,
            state_path=state_path,
            now_ts=now_ts,
            alert_sink=alert_sink,
            alert_type=alert_type,
            reason_prefix=reason_prefix,
            unresolved=unresolved,
        )

    # ----------------------------------------------------------------------------
    # Step 3b: Per-sleeve HOLD / DE-RISK alerts (spec §4.3, red-team L1).
    # ----------------------------------------------------------------------------
    for sleeve in sleeves_list:
        if sleeve_audit[sleeve.name]["outcome"] == "hold":
            _emit_alert(
                alert_sink,
                AlertType.SLEEVE_HOLD,
                Severity.WARNING,
                f"sleeve {sleeve.name!r} HELD — coins frozen this cycle",
                {
                    "sleeve": sleeve.name,
                    "consecutive_holds": new_holds[sleeve.name],
                    "frozen_coins": sorted(sleeve_audit[sleeve.name]["weights"].keys()),
                },
            )

    # Emit SLEEVE_DERISKED only when a sleeve that PREVIOUSLY held inventory goes
    # stale — a never-served sleeve going "stale" carries no positions to unwind,
    # so the CRITICAL would be noise.  We approximate "had inventory" via the
    # ledger's last_weights (the stale sleeve's prior committed exposure).
    for sleeve in sleeves_list:
        if sleeve_audit[sleeve.name]["outcome"] != "stale":
            continue
        prior = state.sleeves.get(sleeve.name) or {}
        had_inventory = bool(prior.get("last_weights"))
        if had_inventory:
            _emit_alert(
                alert_sink,
                AlertType.SLEEVE_DERISKED,
                Severity.CRITICAL,
                f"sleeve {sleeve.name!r} STALE — de-risking its coins to cash",
                {
                    "sleeve": sleeve.name,
                    "consecutive_holds": new_holds[sleeve.name],
                    "unwound_coins": sorted((prior.get("last_weights") or {}).keys()),
                },
            )

    # ----------------------------------------------------------------------------
    # Step 3c: COMBINE → one synthetic combined TargetAllocation (spec §4.3, §4.4).
    # ----------------------------------------------------------------------------
    combined = combine_sleeves(outcomes)
    combined_weights = combined.weights
    frozen_coins = combined.frozen_coins

    # cash = 1 − Σ|combined| — the UNENCUMBERED fraction of the TRADED target.
    # NOTE on accounting: frozen sleeves are NOT in ``combined_weights`` so they do
    # NOT reduce this cash figure here; their budget is held by their LIVE positions
    # (carried untouched) and is folded into the sizer's gross-cap math via the
    # frozen-coin live |notional|.  So "cash" is the unencumbered slice of the
    # traded book, not the account-wide cash.
    combined_gross = sum(abs(w) for w in combined_weights.values())
    combined_cash = 1.0 - combined_gross

    # The combined as_of: all sleeves share the 00:00 UTC daily close calendar, so
    # use the explicit cycle ``as_of`` when provided, else the (consistent) served
    # as_of from the fresh sleeves.  ``ta.as_of`` drives data_age_days + the
    # executor's as_of stamp + the audit record.
    combined_as_of = as_of if as_of is not None else next(iter(fresh_as_of.values()))

    # Synthetic strategy_id: "+"-join the FRESH sleeves' strategy ids (config
    # order).  For the singular path this is exactly the lone sleeve's strategy_id
    # (e.g. "haarp"), so the executor's cloid prefix is byte-compatible.
    combined_strategy_id = "+".join(fresh_strategy_ids)

    ta = TargetAllocation(
        as_of=combined_as_of,
        weights=combined_weights,
        cash=combined_cash,
        strategy_id=combined_strategy_id,
        model_rev="combined",
        audience=_COMBINED_AUDIENCE,
    )

    # Validate the COMBINED target with allow_short = any sleeve allows shorts and
    # the COMBINED gross rule (Σ|w| ≤ 1).  Frozen coins are absent from the combined
    # weights so they are not validated here (they carry no new target this cycle).
    # The whitelist is the global universe (or the combined weights' coins when the
    # universe is empty, mirroring the per-sleeve fallback).
    whitelist = set(cfg.universe_perp_map) if cfg.universe_perp_map else None
    effective_whitelist = (
        whitelist if whitelist is not None else set(combined_weights.keys())
    )
    validate_target_allocation(
        ta,
        max_per_asset=cfg.max_per_asset,
        whitelist=effective_whitelist,
        client_id=_COMBINED_AUDIENCE,
        allow_short=any_short,
    )

    # Step 4: Record freshness + per-sleeve ledger writes.  At least one sleeve
    # served fresh this cycle → advance the global staleness clock (kept in sync
    # for the legacy single-source path) and persist each sleeve's ledger:
    #   FRESH → {last_weights, last_as_of, last_success_ts: now, consecutive_holds:0}
    #   HOLD  → keep last_weights, bump consecutive_holds (no last_success_ts move)
    #   STALE → keep last_weights (de-risk is via the combined delta, not the ledger)
    #
    # Skip the write for any sleeve NOT in the real config (C5-review cleanup): the
    # bounded-retry path drives the cycle with a SINGLE synthetic ``"default"``
    # sleeve (``sleeve_configs=[retry_sleeve]``) that does not correspond to any
    # configured source.  Writing ``state.sleeves["default"]`` for it on a
    # multi-sleeve deployment would inject a phantom ledger record that no real
    # sleeve owns (it would never be read by the per-sleeve staleness clock, but it
    # pollutes the persisted state and any audit of the sleeve map).  The singular
    # config DOES name its lone sleeve ``"default"``, so that path is unaffected.
    configured_sleeve_names = {s.name for s in cfg.sleeves()}
    state.last_successful_signal_ts = now_ts
    for sleeve in sleeves_list:
        name = sleeve.name
        if name not in configured_sleeve_names:
            continue
        status = sleeve_audit[name]["outcome"]
        rec = state.sleeves.get(name)
        if status == "fresh":
            state.sleeves[name] = {
                "last_weights": dict(fresh_served[name]),
                "last_as_of": fresh_as_of[name].isoformat(),
                "last_success_ts": now_ts,
                "consecutive_holds": 0,
            }
        elif rec is not None:
            # HOLD / STALE: keep last_weights; sync the consecutive_holds counter.
            rec["consecutive_holds"] = new_holds[name]
    save(state, state_path)

    # The combined SIGNED weights — carried on EVERY CycleReport return from here.
    # The daemon stashes this as ``retry_target`` when opening a bounded-retry
    # window (the retry freezes the COMBINED target, spec §4.5).
    served_weights = dict(combined_weights)

    # FORENSIC STAMP (observability only, NO ALERT): how stale the served bar was,
    # in whole days = (now.date() - ta.as_of).days.  ``now_ts`` is ISO (possibly
    # naive → coerce to UTC, mirroring _handle_no_good_signal).
    now_dt_good = datetime.datetime.fromisoformat(now_ts)
    if now_dt_good.tzinfo is None:
        now_dt_good = now_dt_good.replace(tzinfo=datetime.UTC)
    data_age_days = (now_dt_good.date() - ta.as_of).days

    # Step 5: Size plan — ONE combined target, allow_short if any sleeve allows it,
    # excluding the frozen coins (carried untouched; their live notional still
    # counts toward the gross cap inside the sizer) and any SAME-DIRECTION
    # cooldown-blocked coins (red-team M3): a coin whose unexpired liquidation
    # cooldown matches the new target's direction is dropped from the traded set so
    # we never re-open into a squeeze.  Expired cooldowns are cleared here (so the
    # entry trades normally and the map self-prunes); opposite-direction re-entries
    # pass through (the flip is a new thesis).
    cooldown_blocked = _cooldown_blocked_set(
        cfg=cfg,
        state=state,
        combined_weights=combined_weights,
        now_ts=now_ts,
        state_path=state_path,
    )
    sp = sizer.plan(
        ta,
        snap,
        state,
        cfg,
        allow_short=any_short,
        frozen_coins=frozen_coins,
        cooldown_blocked=cooldown_blocked,
    )

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
            sleeves=sleeve_audit,
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
            sleeves=sleeve_audit,
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
        # EXCLUDE frozen coins from the divergence set (spec §4.3): a held sleeve's
        # live position appears in ``achieved`` but is intentionally absent from the
        # combined target — counting it would report the full frozen weight as
        # "divergence" every cycle the sleeve is dark, which is noise, not drift.
        frozen_perps_div = {
            cfg.universe_perp_map.get(c, c) for c in frozen_coins
        }
        perps = (set(target_by_perp) | set(achieved)) - frozen_perps_div
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
        sleeves=sleeve_audit,
    )


class _FrozenSource:
    """An ``AllocationSource`` serving a fixed, pre-computed target.

    Used by ``retry_attempt`` to drive ``run_cycle`` toward the frozen
    ``retry_target`` WITHOUT re-fetching the signal.  The served allocation
    echoes the strategy_id/audience the executor expects so
    ``validate_target_allocation`` passes the audience guard.

    cash = ``1 − Σ|w|`` (red-team H1): the retry target is the COMBINED target,
    which may carry SIGNED (short) weights.  The naive ``1 − Σw`` would yield
    ``cash > 1`` for a net-short target (e.g. a single ``−0.4`` short → cash
    ``1.4``), and the allow_short unity check (``Σ|w| + cash == 1``) would then
    REJECT the retry — silently breaking the bounded-retry recovery for any
    cycle whose combined book was net-short.  Summing magnitudes makes the
    served allocation a valid signed target for both long-only and signed retries.
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
            cash=1.0 - sum(abs(w) for w in self._weights.values()),
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

    The retry freezes the COMBINED target (spec §4.5), so it bypasses the
    per-sleeve fetch loop: it drives ``run_cycle`` with a SINGLE synthetic
    ``"default"`` sleeve (``sleeve_configs=[...]``) whose ``allow_short`` is the
    OR of every real sleeve's flag — so a signed (net-short) combined retry
    validates and sizes correctly.  Its ``client_id`` matches the
    ``_FrozenSource`` audience so the per-sleeve audience guard passes.

    Parameters
    ----------
    cfg, client, ledger, state, state_path, now_ts, alert_sink, is_filled:
        Forwarded verbatim to ``run_cycle``.
    retry_target:
        The frozen COMBINED target weights to retry toward (the served combined
        allocation whose residual leg was liquidity-deferred on the daily cycle).
    retry_as_of:
        ISO date string of the frozen allocation.  Becomes the ``ta.as_of`` of
        the synthetic ``TargetAllocation`` and the ``as_of`` reference forwarded
        to ``run_cycle``.

    Returns
    -------
    CycleReport
        Same shape as a normal ``run_cycle`` report.
    """
    any_short = any(s.allow_short for s in cfg.sleeves())
    source = _FrozenSource(
        retry_target,
        retry_as_of,
        client_id=_COMBINED_AUDIENCE,
    )
    # A single synthetic combined sleeve — drives the per-sleeve loop with the
    # frozen combined target as the lone "fresh" sleeve (no real per-source fetch).
    retry_sleeve = SignalSourceConfig(
        type="remote",
        name="default",
        budget=1.0,
        allow_short=any_short,
        client_id=_COMBINED_AUDIENCE,
        # url/root_key_fingerprint_ref are only validated at source-BUILD time; the
        # _FrozenSource is injected directly, so these are never dereferenced here.
        url="https://retry.local",
        root_key_fingerprint_ref="RETRY",
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
        sleeve_configs=[retry_sleeve],
    )
