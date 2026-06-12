"""
automation.execution.sizer — Weight-to-size planning with frozen-target ratchet.

This module is PURE AND SIDE-EFFECT-FREE.  It takes the current target allocation,
the on-chain snapshot, the persistent state, and the config, and returns a
deterministic SizePlan.  It NEVER submits orders, NEVER mutates state.

Ratchet design (MF-5, MF-8, S-2)
----------------------------------
The sizer uses state.last_rebalanced_target (the FROZEN last-submitted target
weights) as its ratchet baseline.  It compares the NEW target against this
frozen baseline, NOT against achieved/live weights (those are only used for the
abs-drift backstop).

Commit rule: state.last_rebalanced_target is updated to target.weights ONLY AFTER
successful execution (in B7/B10).  The sizer only reads state, never writes it.

Constants
---------
MIN_NOTIONAL_USD : Decimal
    Minimum order value in USD.  Hyperliquid rejects orders smaller than $10.
    Deltas below this threshold are skipped and recorded in SizePlan.skipped
    with reason "below_min_notional".  This is intentional: the achieved weight
    at that coin persists (not the intended weight) — see S-5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypedDict

from automation.allocation.base import TargetAllocation
from automation.core.config import Config
from automation.core.redaction import get_logger
from automation.core.rounding import floor_size
from automation.execution.reconciler import Snapshot, achieved_weights
from automation.reporting.state import State

_logger = get_logger(__name__)

# Hyperliquid minimum order notional (USD).  Orders below this value are rejected
# by the exchange — skip and record rather than wasting API calls.
MIN_NOTIONAL_USD: Decimal = Decimal("10")


class SkipEntry(TypedDict):
    """A single entry in ``SizePlan.skipped``.

    Keys
    ----
    coin:
        Perp name of the excluded coin.
    reason:
        One of ``"below_min_notional"``, ``"zeroed_by_cap"``, or
        ``"unpriceable"``.
    delta_notional:
        USD notional of the would-be delta (0.0 for unpriceable coins).
    """

    coin: str
    reason: str
    delta_notional: float


class SizerError(Exception):
    """Raised when a hard money-safety invariant is violated in the sizer.

    Currently raised only when the post-sizing gross notional exceeds the
    safety-buffer cap (Σnotional ≤ E·(1−safety_buffer)).  This is a HARD guard
    that must hold regardless of ``python -O`` (``assert`` is stripped under -O,
    so we raise explicitly instead — F5).
    """


@dataclass(frozen=True)
class OrderPlan:
    """A single planned order.

    Fields
    ------
    coin:
        Perp name (after universe_perp_map translation).
    target_size:
        Target position size, floored toward zero.  SIGNED when the plan was
        built with ``allow_short=True`` — negative = short.  ≥ 0 in long-only
        plans (the default).
    delta_size:
        Signed size change: ``target_size - current_position``.
    delta_notional:
        USD notional of the delta: ``delta_size * mark_price``.
    side:
        ``"buy"`` if delta_size > 0, ``"sell"`` otherwise.
    is_close_leg:
        ``True`` only for the CLOSE leg of a zero-crossing flip (red-team H4).
        A flipping coin emits TWO OrderPlans: a close-leg
        (``target_size=0.0``, ``delta=-pos``, ``is_close_leg=True``) followed
        by an open-leg (the final signed target).  The executor routes a close
        leg through the reduce-only full-exit path and the cap pre-flight
        counts the coin's end-state notional ONCE (the open-leg's).  Default
        ``False`` keeps every existing construction valid.
    """

    coin: str
    target_size: float
    delta_size: float
    delta_notional: float
    side: Literal["buy", "sell"]
    is_close_leg: bool = False


@dataclass(frozen=True)
class SizePlan:
    """The full sizing plan for one rebalance cycle.

    Fields
    ------
    rebalance:
        ``True`` if a rebalance should be executed; ``False`` for hold (all
        orders and flatten lists are empty when rebalance=False).
    reason:
        Human-readable reason string (e.g. ``"bootstrap"``, ``"hold"``,
        ``"max_delta=0.0400>=0.0300"``).
    orders:
        Planned OrderPlan objects, sorted by abs(delta_notional) DESCENDING
        (per coin: by the coin's largest leg, the close-leg of a flip always
        directly before its open-leg).  Only populated when rebalance=True.
    skipped:
        List of dicts with keys ``{"coin", "reason", "delta_notional"}`` for
        coins excluded from orders (e.g. below_min_notional, unpriceable).
    flatten:
        Perp names with current szi < 0 (short positions) that must be closed
        before the new longs are entered (long-only guard).  Populated only
        when ``allow_short=False`` — a sleeve that allows shorts OWNS its
        short positions and manages them through delta orders instead.
    target_sizes:
        Final target sizes by perp name (post-flooring, post-scale-down).
    achieved_weights:
        Live weight vector from achieved_weights(snap) — used for reporting.
    scaled_down:
        ``True`` if target notional was scaled down to fit within
        ``equity × (1 - safety_buffer)``.
    scale_factor:
        The scale factor applied (1.0 if no scale-down occurred).
    gross_notional:
        Sum of ``|target_size × mark|`` across all planned coins (post-scale).
        Magnitude-based: shorts ADD to gross exposure, they never cancel
        longs (red-team C1).
    equity:
        Equity value used for sizing (float for convenience in reports).
    """

    rebalance: bool
    reason: str
    orders: list[OrderPlan]
    skipped: list[SkipEntry]
    flatten: list[str]
    target_sizes: dict[str, float]
    achieved_weights: dict[str, float]
    scaled_down: bool
    scale_factor: float
    gross_notional: float
    equity: float


def _map_perp(ticker: str, universe_perp_map: dict[str, str]) -> str:
    """Map a strategy ticker to a perp name via universe_perp_map (identity fallback)."""
    return universe_perp_map.get(ticker, ticker)


def _floor_size_signed(raw_size: float, sz_decimals: int) -> float:
    """Floor a SIGNED size TOWARD zero — magnitude shrinks, direction is kept.

    C2 (red-team CRITICAL): ``floor_size`` uses ``math.floor``, which on a
    NEGATIVE size rounds AWAY from zero (``floor(-1.0001) = -2.0``) — flooring
    a short target would GROW its exposure, the exact opposite of rounding.py's
    "flooring can only shrink Σ(size × price)" guarantee.  Floor the magnitude,
    then reapply the sign (``copysign``).  For longs this is bit-identical to
    the plain ``floor_size`` call it replaces.

    ``floor_size`` is referenced via the module global (not re-imported) so the
    guard tests can monkeypatch it.
    """
    if raw_size == 0.0:
        return 0.0
    return math.copysign(floor_size(abs(raw_size), sz_decimals), raw_size)


def plan(
    target: TargetAllocation,
    snap: Snapshot,
    state: State,
    cfg: Config,
    *,
    allow_short: bool = False,
    frozen_coins: frozenset[str] = frozenset(),
    cooldown_blocked: frozenset[str] = frozenset(),
) -> SizePlan:
    """Compute the full size plan for one rebalance cycle.

    This function is PURE — it does not mutate state, does not submit orders,
    and has no side effects.  The caller (B7/B10) is responsible for updating
    state.last_rebalanced_target = target.weights AFTER successful execution.

    Parameters
    ----------
    target:
        Verified target allocation from the signal source.
    snap:
        On-chain snapshot from reconciler.snapshot().
    state:
        Current persistent executor state (read-only here).
    cfg:
        Executor configuration.
    allow_short:
        When True, negative target weights are planned as SHORT positions —
        target sizes stay SIGNED end-to-end (sell-to-open, buy-to-close fall
        out of the sign-aware delta math) and the long-only flatten guard is
        disabled.  Default False preserves the long-only contract: every held
        short is flagged in ``SizePlan.flatten``.
    frozen_coins:
        Coins belonging to HELD sleeves (CRASH Phase 1, spec §4.3, red-team H3 /
        C5 sizer note).  A frozen coin's live position is CARRIED UNTOUCHED this
        cycle: it produces NO OrderPlan and is EXCLUDED from the ratchet
        threshold / abs-drift comparison (its stale position must never trip — or
        damp — a rebalance for the TRADED book).  But its live ``|notional|``
        STILL consumes margin, so it IS counted into the gross-cap math (the
        scale-down cap and the post-sizing hard guard) — otherwise a frozen
        sleeve's exposure would be invisible to the Σ|notional| ≤ E·(1−buffer)
        invariant and the account could be silently over-leveraged.  Default
        ``frozenset()`` preserves every existing (singular / long-only) cycle
        byte-for-byte: no coin is frozen, so every branch below is a no-op.
    cooldown_blocked:
        Coins under a post-liquidation re-entry cooldown whose NEW target is in the
        SAME direction as the position that was liquidated (CRASH Phase 1, spec
        §5 / red-team M3).  ``run_cycle`` owns the direction reasoning — a coin only
        lands here when its new combined-target sign MATCHES the liquidated
        direction; an OPPOSITE-direction re-entry (the squeeze flipped the thesis)
        is intentionally ABSENT so it trades normally.  A blocked coin produces NO
        OrderPlan and is recorded in ``SizePlan.skipped`` with reason ``"cooldown"``
        so the operator sees the deliberate non-entry.  Unlike ``frozen_coins`` a
        cooldown coin was LIQUIDATED → flat, so there is NO live position: no
        gross-cap contribution and no exit to suppress — it is simply dropped from
        the sizing set before any size is computed.  Coins are addressed in TICKER
        space (the baseline / target key space); the perp mapping below translates
        them so the drop lines up with ``target_w_by_perp``.  Default
        ``frozenset()`` is a no-op for every existing cycle.

    Returns
    -------
    SizePlan
        Deterministic plan for the current cycle.
    """
    # -----------------------------------------------------------------------
    # Step 1: Map strategy tickers to perp names
    # -----------------------------------------------------------------------
    # COOLDOWN DROP (red-team M3): a coin under a same-direction post-liquidation
    # cooldown is removed from the target BEFORE sizing so it can produce no order
    # and never enters the gross/threshold math.  It is recorded as a deliberate
    # skip (reason "cooldown") for operator visibility.  Drop in TICKER space (the
    # target's key space) so the entry is excluded regardless of ticker→perp
    # aliasing; the live book is flat for these coins (they were liquidated), so
    # there is nothing else to carry.
    cooldown_skipped: list[SkipEntry] = []
    effective_weights: dict[str, float] = dict(target.weights)
    for ticker in cooldown_blocked:
        if ticker in effective_weights:
            cooldown_skipped.append({
                "coin": _map_perp(ticker, cfg.universe_perp_map),
                "reason": "cooldown",
                "delta_notional": 0.0,
            })
            del effective_weights[ticker]

    target_w_by_perp: dict[str, float] = {
        _map_perp(ticker, cfg.universe_perp_map): w
        for ticker, w in effective_weights.items()
    }

    # Frozen coins are addressed in PERP space (the combination layer keys them by
    # the same symbol the snapshot uses).  Map through universe_perp_map so the
    # exclusion below lines up with snap.positions / target_w_by_perp regardless of
    # ticker→perp aliasing.  Frozen coins are guaranteed disjoint from the traded
    # target (the combination layer raises on a frozen-coin conflict), so this set
    # never intersects target_w_by_perp.
    frozen_perps: set[str] = {
        _map_perp(c, cfg.universe_perp_map) for c in frozen_coins
    }

    # -----------------------------------------------------------------------
    # Step 2: Compute achieved weights from live positions
    # -----------------------------------------------------------------------
    achieved = achieved_weights(snap)

    # -----------------------------------------------------------------------
    # Step 3: RATCHET — all-or-nothing threshold check
    # -----------------------------------------------------------------------
    threshold = cfg.threshold_pct / 100.0
    backstop = 2.0 * threshold

    if state.last_rebalanced_target is None:
        # Cycle-0 bootstrap: first ever rebalance, no frozen baseline yet.
        rebalance = True
        reason = "bootstrap"
        _logger.info("sizer.plan: cycle-0 bootstrap — rebalance=True")
    else:
        prev = state.last_rebalanced_target
        # Compare in TICKER space (both target.weights and prev are in ticker space).
        # EXCLUDE frozen coins (red-team H3): a held sleeve's coins are carried
        # untouched this cycle, so their stale baseline entry must NOT count toward
        # the threshold — otherwise a frozen short sitting in ``prev`` (the combined
        # baseline records signed weights) but absent from the new traded target
        # would register a spurious ``max_delta`` and force a rebalance of the
        # TRADED book on the dark sleeve's behalf.  Frozen coins are disjoint from
        # the new target, so excluding them only drops stale-baseline noise.
        # Compare against ``effective_weights`` (the cooldown-dropped target): a
        # coin under a same-direction cooldown is removed from the traded set, so
        # its baseline-vs-target delta must NOT count toward the threshold —
        # otherwise the liquidated short still sitting in ``prev`` (e.g. −0.3) vs the
        # dropped re-entry would register a large ``max_delta`` and force a rebalance
        # of the rest of the book on behalf of a coin we are deliberately NOT
        # re-entering.  Exclude both frozen and cooldown-blocked tickers.
        all_tickers = (
            (set(effective_weights.keys()) | set(prev.keys()))
            - frozen_coins
            - cooldown_blocked
        )
        # ``default=0.0`` guards the all-cash → all-cash transition: when both the
        # previous frozen target and the new target are empty (100% cash, e.g. HAARP
        # fully risk-off for >1 day), ``all_tickers`` is empty and a bare ``max()``
        # would raise ValueError and crash the daemon.  No tickers ⇒ no change ⇒ HOLD.
        max_delta = max(
            (abs(effective_weights.get(t, 0.0) - prev.get(t, 0.0)) for t in all_tickers),
            default=0.0,
        )

        # Abs-drift backstop: compare perp-space achieved against perp-space target.
        # Map prev tickers to perps for the drift check.
        prev_by_perp = {
            _map_perp(t, cfg.universe_perp_map): w for t, w in prev.items()
        }
        # F3: the drift union MUST include `achieved` (currently-held perps) so a
        # coin held on-chain that is in NEITHER the new nor the previous target is
        # still drift-checked.  Without it a stray position whose target weight is
        # 0 (and max_delta=0) never triggers a rebalance and is stranded forever
        # (Step-7 exits only run when a rebalance fires).
        # F4: `achieved` and `target_w_by_perp` MUST both be PERP-keyed for this
        # comparison to be meaningful — achieved_weights() keys by perp name (its
        # input is snap.positions, which is perp-keyed).  If a future change keyed
        # achieved by ticker, target_w(p) − achieved(p) would equal target_w for
        # every perp, spuriously forcing a full rebalance every cycle.
        # EXCLUDE frozen perps from the drift union too (red-team H3): a frozen
        # sleeve's live position appears in ``achieved`` but its target is
        # intentionally absent — without the exclusion ``target_w(p) − achieved(p)``
        # would equal the full live weight and trip the abs-drift backstop, forcing
        # the traded book to rebalance because a DARK sleeve is holding inventory.
        # EXCLUDE cooldown perps too (red-team M3): a liquidated coin's OLD baseline
        # weight lives in ``prev_by_perp`` (e.g. −0.3) while the book is now flat
        # (``achieved`` has no entry) — ``target_w(p) − achieved(p)`` would equal the
        # full stale weight and trip the backstop, forcing a rebalance just to honor
        # a coin we are deliberately holding OUT during the cooldown.
        cooldown_perps: set[str] = {
            _map_perp(c, cfg.universe_perp_map) for c in cooldown_blocked
        }
        all_perps = (
            set(target_w_by_perp.keys())
            | set(prev_by_perp.keys())
            | set(achieved.keys())
        ) - frozen_perps - cooldown_perps
        abs_drift = max(
            abs(target_w_by_perp.get(p, 0.0) - achieved.get(p, 0.0))
            for p in all_perps
        ) if all_perps else 0.0

        if max_delta >= threshold:
            rebalance = True
            reason = f"max_delta={max_delta:.4f}>={threshold:.4f}"
        elif abs_drift >= backstop:
            rebalance = True
            reason = f"abs_drift_backstop={abs_drift:.4f}>={backstop:.4f}"
        else:
            rebalance = False
            reason = "hold"

        _logger.info(
            "sizer.plan: ratchet",
            max_delta=round(max_delta, 4),
            abs_drift=round(abs_drift, 4),
            threshold=threshold,
            backstop=backstop,
            rebalance=rebalance,
            reason=reason,
        )

    # -----------------------------------------------------------------------
    # Early return: HOLD (all-or-nothing)
    # -----------------------------------------------------------------------
    if not rebalance:
        # Surface any cooldown drops even on a HOLD: a same-direction re-entry that
        # the cooldown suppressed must remain visible to the operator (it is the
        # WHOLE reason a coin the strategy wanted produced no order this cycle).
        return SizePlan(
            rebalance=False,
            reason=reason,
            orders=[],
            skipped=cooldown_skipped,
            flatten=[],
            target_sizes={},
            achieved_weights=achieved,
            scaled_down=False,
            scale_factor=1.0,
            gross_notional=0.0,
            equity=float(snap.equity),
        )

    # -----------------------------------------------------------------------
    # Step 4: SIZE — compute raw target sizes
    # -----------------------------------------------------------------------
    E = float(snap.equity)
    skipped: list[SkipEntry] = []
    raw_target_sizes: dict[str, float] = {}  # perp → floored size
    perp_prices: dict[str, float] = {}       # perp → mark price (for delta math)
    intended_notional: dict[str, float] = {}  # perp → unscaled intended USD notional

    for perp, w in target_w_by_perp.items():
        mark_dec = snap.marks.get(perp)
        if mark_dec is None or float(mark_dec) <= 0.0:
            _logger.warning(
                "sizer.plan: unpriceable coin — excluding from sizing",
                coin=perp,
                weight=w,
            )
            skipped.append({
                "coin": perp,
                "reason": "unpriceable",
                "delta_notional": 0.0,
            })
            continue

        px = float(mark_dec)
        perp_prices[perp] = px
        szd = snap.sz_decimals.get(perp, 0)
        # target_notional and raw_size are SIGNED (negative = short target).
        # The sign is the position's direction and must survive into the order;
        # every CAP computation below works on the magnitude instead (C1).
        target_notional = w * E
        intended_notional[perp] = target_notional
        raw_size = target_notional / px
        floored_size = _floor_size_signed(raw_size, szd)
        raw_target_sizes[perp] = floored_size

    # -----------------------------------------------------------------------
    # Step 5: Σnotional ≤ E×(1 − safety_buffer) invariant (MF-8)
    # -----------------------------------------------------------------------
    cap = E * (1.0 - cfg.safety_buffer)
    # FROZEN-coin live exposure (CRASH Phase 1, spec §4.3 / C5 sizer note): a held
    # sleeve's coins are not in the traded target, but their live positions still
    # consume margin THIS cycle.  Sum their |notional| once (live size × mark) and
    # add it to BOTH gross accumulations below so the Σ|notional| ≤ E·(1−buffer)
    # invariant accounts for the frozen book — otherwise a dark sleeve's exposure
    # is invisible to the cap and the traded sleeve could be sized into an
    # over-leveraged combined book.  A frozen coin with no mark (unpriceable) is
    # silently 0 here — it cannot be sized anyway and the divergence/health layer
    # surfaces an unpriceable held coin separately.
    frozen_live_notional = 0.0
    for fp in frozen_perps:
        pos = float(snap.positions.get(fp, Decimal("0")))
        mark_dec = snap.marks.get(fp)
        if pos != 0.0 and mark_dec is not None and float(mark_dec) > 0.0:
            frozen_live_notional += abs(pos * float(mark_dec))
    # C1 (red-team CRITICAL): gross exposure is the sum of MAGNITUDES, never the
    # signed sum.  Long and short notionals both consume margin — at 1x-isolated
    # each position's margin equals its |notional| — but in a signed sum they
    # CANCEL: an all-short book has NEGATIVE "gross" and would never trip this
    # cap, and a +0.5/−0.5 book has signed gross ≈ 0 while true exposure is
    # 1.0·E.  Either way the MF-8 invariant (Σ|notional| ≤ E·(1−safety_buffer))
    # would be silently unenforced and the account over-leveraged.  Every gross
    # accumulation in this function (here, and post_gross below) must therefore
    # be abs()-based.
    #
    # TRADED gross = the scalable portion (the sleeves we are actively trading).
    # FROZEN gross = the frozen book's live |notional|, which is UNSCALABLE this
    # cycle (we emit no orders for frozen coins).  The combined book is
    # traded + frozen.  When the combined book exceeds the cap, only the TRADED
    # portion can be scaled — and it must be scaled to fit the cap MINUS the
    # frozen notional (scaling by cap/combined would leave traded·(cap/combined) +
    # frozen, which still exceeds the cap because frozen is fixed).  If the frozen
    # book ALONE already meets or exceeds the cap, ``headroom`` is 0 → the traded
    # book scales to size 0 (we add nothing to an already-over-cap book).  But the
    # frozen exposure ITSELF still violates the invariant, so ``post_gross`` (=
    # 0 + frozen_live_notional) > cap and the HARD guard below RAISES SizerError —
    # this is fail-loud-safe, NOT a silent clamp: the daemon catches it as an
    # EXEC_ERROR and halts the cycle rather than trading into an over-leveraged
    # frozen book.  Reducing a frozen sleeve's own exposure is impossible here (we
    # emit no orders for it); the operator / de-risk path must address it.
    traded_gross = sum(
        abs(raw_target_sizes[p] * perp_prices[p]) for p in raw_target_sizes
    )
    gross = traded_gross + frozen_live_notional

    scaled_down = False
    scale_factor = 1.0
    target_sizes: dict[str, float] = {}

    if gross > cap and traded_gross > 0.0:
        # Scale the TRADED book to fit the headroom left after the unscalable
        # frozen book.  ``headroom`` can be <= 0 if the frozen book alone exceeds
        # the cap → scale_factor clamps to 0 (drop the traded legs entirely).
        headroom = max(cap - frozen_live_notional, 0.0)
        scale_factor = headroom / traded_gross
        scaled_down = True
        _logger.info(
            "sizer.plan: scaling down to fit safety buffer",
            gross=round(gross, 2),
            traded_gross=round(traded_gross, 2),
            frozen_live_notional=round(frozen_live_notional, 2),
            cap=round(cap, 2),
            scale_factor=round(scale_factor, 6),
        )
        for perp, raw_size in raw_target_sizes.items():
            px = perp_prices[perp]
            szd = snap.sz_decimals.get(perp, 0)
            # scale_factor ∈ [0, 1) so multiplying the SIGNED notional shrinks
            # its magnitude while preserving direction — the cap math above is
            # what must (and does) use abs(); the size itself stays signed.
            scaled_notional = raw_size * px * scale_factor
            target_sizes[perp] = _floor_size_signed(scaled_notional / px, szd)
    else:
        target_sizes = dict(raw_target_sizes)

    # F1: detect any coin the strategy WANTED (intended_notional > 0) that floored
    # to size 0 — whether from scale-down OR because a single lot already exceeds
    # the cap.  Such a coin would silently vanish (no order, no skip, no alert) and
    # the Σnotional invariant would still pass (0 ≤ cap).  MF-8 is scale-TO-FIT,
    # not silently-drop: record it in `skipped` with reason "zeroed_by_cap" and
    # emit a WARNING so the wanted coin never disappears without a trace.
    zeroed_by_cap: set[str] = set()
    for perp, want_notional in intended_notional.items():
        # abs() because want_notional is SIGNED (negative = short intent, B3):
        # a bare `> 0.0` can never fire for a wanted SHORT erased by the cap —
        # it would be silently misfiled under below_min_notional with no F1
        # warning.  Explicit-zero targets (want_notional == 0.0, i.e. "close
        # this coin") remain exempt: nothing was wanted, nothing was erased.
        if abs(want_notional) > 0.0 and target_sizes.get(perp, 0.0) == 0.0:
            zeroed_by_cap.add(perp)
            _logger.warning(
                "sizer.plan: coin zeroed by cap/scale-down — wanted but no order",
                coin=perp,
                intended_notional=round(want_notional, 2),
                scale_factor=round(scale_factor, 6),
                cap=round(cap, 2),
            )
            skipped.append({
                "coin": perp,
                "reason": "zeroed_by_cap",
                "delta_notional": want_notional,
            })

    # Recompute gross post-flooring and enforce the money invariant as a HARD
    # guard (F5).  `assert` is stripped under ``python -O``, so we raise
    # SizerError explicitly.  A relative tolerance is used (robust at large
    # equity) instead of an absolute 1e-6.
    # Like `gross` above, this MUST sum magnitudes (C1): a signed sum lets a
    # short-heavy book pass the guard with true exposure far above the cap.  The
    # frozen book's live |notional| is folded in (it consumes margin this cycle).
    post_gross = (
        sum(abs(target_sizes[p] * perp_prices[p]) for p in target_sizes)
        + frozen_live_notional
    )
    if post_gross > cap * (1.0 + 1e-9):
        raise SizerError(
            f"Σ|notional| {post_gross:.4f} exceeds cap {cap:.4f} after sizing — "
            "money invariant violated (flooring/scale-down should only shrink)"
        )

    # -----------------------------------------------------------------------
    # Step 6: Long-only guard — flag any existing short positions to flatten
    # -----------------------------------------------------------------------
    # Conditional on allow_short (B3).  For pure-long deployments this stays
    # on as a RESIDUE SWEEPER: any short on the book is foreign (manual
    # intervention, partial fill of an old close) and must be flattened before
    # longs are entered.  Sleeves with shorts OWN their positions — sweeping
    # them here would close the sleeve's own shorts every cycle, fighting the
    # delta orders built in Step 7.
    flatten: list[str] = []
    if not allow_short:
        flatten = [
            perp
            for perp, szi in snap.positions.items()
            if float(szi) < 0.0 and perp not in frozen_perps
        ]

    # -----------------------------------------------------------------------
    # Step 7: Build delta orders
    # -----------------------------------------------------------------------
    # Union of target perps and currently-held perps (so we can plan exits too).
    # EXCLUDE frozen perps (red-team H3): a held sleeve's coins are carried
    # untouched — they must produce NO order this cycle (neither a delta toward a
    # target they have, since they have none, NOR a close-to-zero exit, which is
    # exactly what including a held position with no target would generate).  The
    # frozen position simply persists; its sleeve will re-target it once its signal
    # is fresh again.
    all_perps_to_plan = (
        set(target_sizes.keys()) | set(snap.positions.keys())
    ) - frozen_perps

    orders: list[OrderPlan] = []
    order_skipped: list[SkipEntry] = []

    for perp in all_perps_to_plan:
        ts = target_sizes.get(perp, 0.0)
        pos = float(snap.positions.get(perp, Decimal("0")))

        # Price lookup — for currently-held coins not in target, we need a mark.
        if perp in perp_prices:
            px = perp_prices[perp]
        else:
            mark_dec = snap.marks.get(perp)
            if mark_dec is None or float(mark_dec) <= 0.0:
                _logger.warning(
                    "sizer.plan: no mark for held coin not in target — skipping delta",
                    coin=perp,
                )
                order_skipped.append({
                    "coin": perp,
                    "reason": "unpriceable",
                    "delta_notional": 0.0,
                })
                continue
            px = float(mark_dec)

        # -------------------------------------------------------------------
        # FLIP (zero-crossing, red-team H4): target and position both nonzero
        # with OPPOSITE signs → NEVER cross zero in one IOC.  Emit TWO legs,
        # close-leg first:
        #   close-leg: target 0.0, delta = −pos (the full close) — routed by
        #              the executor through reduce-only ``market_close``, which
        #              structurally cannot overshoot past zero.
        #   open-leg:  target = final signed target, delta = target − 0 — a
        #              plain IOC anchored at flat.
        # WHY two legs: a single crossing IOC would "work" on HL but couples
        # the close and open margin semantics into one fill — a partial fill
        # can strand the book anywhere between the old and new position, and
        # the reduce-only safety property (auditable can't-over-close) is
        # impossible for a crossing order.  Recovery is STRUCTURAL: if the
        # close fills and the open fails/defers, the next plan() is anchored
        # to the live (now flat) position and re-emits the open delta alone.
        # Flips only arise with allow_short sleeves (a long-only target can't
        # oppose a long position), but the detection is pure sign math — it
        # guards foreign shorts under long-only configs too.
        if ts * pos < 0.0:
            legs: list[tuple[float, float, bool]] = [
                (0.0, -pos, True),   # close-leg: reduce the position to flat
                (ts, ts, False),     # open-leg: open the final signed target
            ]
        else:
            legs = [(ts, ts - pos, False)]

        for leg_target, leg_delta, is_close_leg in legs:
            delta_notional = leg_delta * px

            if abs(delta_notional) < float(MIN_NOTIONAL_USD):
                # A coin already recorded as zeroed_by_cap with no position to
                # exit would land here with delta≈0 — don't double-record it as
                # below_min_notional; its zeroed_by_cap entry already explains it.
                if perp not in zeroed_by_cap:
                    order_skipped.append({
                        "coin": perp,
                        "reason": "below_min_notional",
                        "delta_notional": delta_notional,
                    })
                continue

            side: Literal["buy", "sell"] = "buy" if leg_delta > 0 else "sell"
            orders.append(OrderPlan(
                coin=perp,
                target_size=leg_target,
                delta_size=leg_delta,
                delta_notional=delta_notional,
                side=side,
                is_close_leg=is_close_leg,
            ))

    # Sort largest-first by the COIN's biggest leg notional, with the close
    # leg of a flip ALWAYS before its open leg.  A plain abs(delta_notional)
    # sort would put the open leg first whenever the new |target| exceeds the
    # old |position| — the open IOC would then fire while the opposite
    # position is still on (double-sided margin, and the close would no longer
    # be a clean reduce-only of the original size).  Keying both legs of a
    # coin by the pair's max notional preserves the global size ordering while
    # the is_close_leg tie-break pins close-before-open within the coin.
    pair_key: dict[str, float] = {}
    for o in orders:
        pair_key[o.coin] = max(pair_key.get(o.coin, 0.0), abs(o.delta_notional))
    orders.sort(key=lambda o: (pair_key[o.coin], o.is_close_leg), reverse=True)

    all_skipped: list[SkipEntry] = cooldown_skipped + skipped + order_skipped

    _logger.info(
        "sizer.plan: complete",
        rebalance=rebalance,
        reason=reason,
        n_orders=len(orders),
        n_skipped=len(all_skipped),
        n_flatten=len(flatten),
        scaled_down=scaled_down,
        scale_factor=round(scale_factor, 6),
        post_gross=round(post_gross, 2),
        equity=round(E, 2),
    )

    return SizePlan(
        rebalance=True,
        reason=reason,
        orders=orders,
        skipped=all_skipped,
        flatten=flatten,
        target_sizes=target_sizes,
        achieved_weights=achieved,
        scaled_down=scaled_down,
        scale_factor=scale_factor,
        gross_notional=post_gross,
        equity=E,
    )
