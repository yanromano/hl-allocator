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
        Target absolute position size (floored, ≥ 0 for long-only).
    delta_size:
        Signed size change: ``target_size - current_position``.
    delta_notional:
        USD notional of the delta: ``delta_size * mark_price``.
    side:
        ``"buy"`` if delta_size > 0, ``"sell"`` otherwise.
    """

    coin: str
    target_size: float
    delta_size: float
    delta_notional: float
    side: Literal["buy", "sell"]


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
        Planned OrderPlan objects, sorted by abs(delta_notional) DESCENDING.
        Only populated when rebalance=True.
    skipped:
        List of dicts with keys ``{"coin", "reason", "delta_notional"}`` for
        coins excluded from orders (e.g. below_min_notional, unpriceable).
    flatten:
        Perp names with current szi < 0 (short positions) that must be closed
        before the new longs are entered (long-only guard).
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
        Sum of ``target_size × mark`` across all planned coins (post-scale).
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


def plan(
    target: TargetAllocation,
    snap: Snapshot,
    state: State,
    cfg: Config,
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

    Returns
    -------
    SizePlan
        Deterministic plan for the current cycle.
    """
    # -----------------------------------------------------------------------
    # Step 1: Map strategy tickers to perp names
    # -----------------------------------------------------------------------
    target_w_by_perp: dict[str, float] = {
        _map_perp(ticker, cfg.universe_perp_map): w
        for ticker, w in target.weights.items()
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
        all_tickers = set(target.weights.keys()) | set(prev.keys())
        # ``default=0.0`` guards the all-cash → all-cash transition: when both the
        # previous frozen target and the new target are empty (100% cash, e.g. HAARP
        # fully risk-off for >1 day), ``all_tickers`` is empty and a bare ``max()``
        # would raise ValueError and crash the daemon.  No tickers ⇒ no change ⇒ HOLD.
        max_delta = max(
            (abs(target.weights.get(t, 0.0) - prev.get(t, 0.0)) for t in all_tickers),
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
        all_perps = (
            set(target_w_by_perp.keys())
            | set(prev_by_perp.keys())
            | set(achieved.keys())
        )
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
        return SizePlan(
            rebalance=False,
            reason=reason,
            orders=[],
            skipped=[],
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
        target_notional = w * E
        intended_notional[perp] = target_notional
        raw_size = target_notional / px
        floored_size = floor_size(raw_size, szd)
        raw_target_sizes[perp] = floored_size

    # -----------------------------------------------------------------------
    # Step 5: Σnotional ≤ E×(1 − safety_buffer) invariant (MF-8)
    # -----------------------------------------------------------------------
    cap = E * (1.0 - cfg.safety_buffer)
    gross = sum(raw_target_sizes[p] * perp_prices[p] for p in raw_target_sizes)

    scaled_down = False
    scale_factor = 1.0
    target_sizes: dict[str, float] = {}

    if gross > cap and gross > 0.0:
        scale_factor = cap / gross
        scaled_down = True
        _logger.info(
            "sizer.plan: scaling down to fit safety buffer",
            gross=round(gross, 2),
            cap=round(cap, 2),
            scale_factor=round(scale_factor, 6),
        )
        for perp, raw_size in raw_target_sizes.items():
            px = perp_prices[perp]
            szd = snap.sz_decimals.get(perp, 0)
            scaled_notional = raw_size * px * scale_factor
            target_sizes[perp] = floor_size(scaled_notional / px, szd)
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
        if want_notional > 0.0 and target_sizes.get(perp, 0.0) == 0.0:
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
    post_gross = sum(target_sizes[p] * perp_prices[p] for p in target_sizes)
    if post_gross > cap * (1.0 + 1e-9):
        raise SizerError(
            f"Σnotional {post_gross:.4f} exceeds cap {cap:.4f} after sizing — "
            "money invariant violated (flooring/scale-down should only shrink)"
        )

    # -----------------------------------------------------------------------
    # Step 6: Long-only guard — flag any existing short positions to flatten
    # -----------------------------------------------------------------------
    flatten: list[str] = [
        perp
        for perp, szi in snap.positions.items()
        if float(szi) < 0.0
    ]

    # -----------------------------------------------------------------------
    # Step 7: Build delta orders
    # -----------------------------------------------------------------------
    # Union of target perps and currently-held perps (so we can plan exits too).
    all_perps_to_plan = set(target_sizes.keys()) | set(snap.positions.keys())

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

        delta_size = ts - pos
        delta_notional = delta_size * px

        if abs(delta_notional) < float(MIN_NOTIONAL_USD):
            # A coin already recorded as zeroed_by_cap with no position to exit
            # would land here with delta≈0 — don't double-record it as
            # below_min_notional; its zeroed_by_cap entry already explains it.
            if perp not in zeroed_by_cap:
                order_skipped.append({
                    "coin": perp,
                    "reason": "below_min_notional",
                    "delta_notional": delta_notional,
                })
            continue

        side: Literal["buy", "sell"] = "buy" if delta_size > 0 else "sell"
        orders.append(OrderPlan(
            coin=perp,
            target_size=ts,
            delta_size=delta_size,
            delta_notional=delta_notional,
            side=side,
        ))

    # Sort by abs(delta_notional) descending (largest first).
    orders.sort(key=lambda o: abs(o.delta_notional), reverse=True)

    all_skipped: list[SkipEntry] = skipped + order_skipped

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
