"""
automation.allocation.sleeves — PURE sleeve combination + status resolution.

This module is the heart of CRASH Phase 1's multi-sleeve execution (spec §4.3).
It is intentionally IO-free and clock-free: it has NO knowledge of the exchange,
the persisted state, the wall clock, or the config loader.  It only transforms
already-resolved per-sleeve outcomes into one combined signed target plus the
set of frozen coins, and it resolves a single sleeve's status from a fetch
result and the caller's staleness/hold accounting.

Why a pure layer?
-----------------
The money-path combination math (freeze-NOTIONAL holds, short-sleeve fast
de-risk, signed netting across overlapping universes) is the most consequential
arithmetic in the daemon.  Isolating it here lets us test every branch
exhaustively without standing up a fake exchange or clock, and it keeps the
``run_cycle`` rewrite (Task C5) to plumbing: fetch each sleeve, call
``resolve_outcome`` to get a status, then call ``combine_sleeves`` once to get a
single synthetic ``TargetAllocation`` + the frozen-coin set that the sizer must
exclude from the delta.

Spec mapping (§4.3 "Combination")
---------------------------------
    combined_w[coin] = Σ_s  budget_s × effective_w_s[coin]      (signed)
    where effective_w_s = FRESH → w_s
                          HOLD  → FROZEN-NOTIONAL (coins frozen, NOT re-weighted)
                          STALE → {}              (sleeve de-risks to cash)

- FRESH: contributes ``budget_s × w`` per coin into the signed combined sum.
- HOLD:  its coins enter ``frozen_coins`` and do NOT appear in combined weights.
  Freeze-NOTIONAL (red-team H3): re-weighting ``last_weights`` against fresh
  equity every cycle would actively trade a sleeve whose signal is dark.  Instead
  the daemon EXCLUDES frozen coins from the combined delta, so the live positions
  carry untouched.  "Sleeve freeze never invents exposure" is literal: the frozen
  coin's contribution is absent from the combined weights dict, not zeroed.
- STALE: contributes ``{}`` and freezes nothing; its prior coins are simply
  absent from the combined weights → target 0 → the daemon closes them.
- A coin both frozen (HOLD A) and traded (FRESH B) → ``ValueError`` (fail-closed):
  one coin cannot be both carried-untouched and re-targeted in the same cycle.
  HAARP×CRASH universes are disjoint in production so this never fires, but the
  math must NOT silently pick a winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SleeveStatus = Literal["fresh", "hold", "stale"]


@dataclass(frozen=True)
class SleeveOutcome:
    """One sleeve's resolved contribution to this cycle (spec §4.3).

    Attributes
    ----------
    name:
        Stable sleeve identifier (config order is preserved by the caller).
    status:
        ``"fresh"`` — served and validated this cycle; ``weights`` are served.
        ``"hold"``  — fetch/validation failed within the staleness window; the
        daemon passes the LAST committed ``last_weights`` here so combine knows
        which coins to freeze (their KEYS drive ``frozen_coins``).
        ``"stale"`` — staleness window exceeded (or short-sleeve fast de-risk);
        ``weights`` is ``{}`` and the sleeve unwinds via the combined delta.
    weights:
        Signed, sleeve-relative weights (negative entries for short sleeves).
        For ``fresh`` these are the served weights; for ``hold`` they are the
        frozen coin set carried by the daemon; for ``stale`` they are ``{}``.
    budget:
        The sleeve's fraction of total equity (∈ [0, 1]); FRESH weights are
        scaled by this before being summed into the combined target.
    allow_short:
        Whether this sleeve may hold signed/short weights (carried through for
        the combined validation step in the daemon).
    """

    name: str
    status: SleeveStatus
    weights: dict[str, float]
    budget: float
    allow_short: bool


@dataclass(frozen=True)
class CombinedTarget:
    """The single synthetic target produced by combining all sleeves.

    Attributes
    ----------
    weights:
        Signed, budget-scaled, per-coin netted combined weights.  Only FRESH
        sleeves contribute.  A coin that nets to exactly 0.0 across sleeves is
        retained at 0.0 (it was actively targeted; the daemon will close it).
    frozen_coins:
        Coins belonging to HOLD sleeves — EXCLUDED from the combined delta
        (freeze-NOTIONAL).  The daemon must NOT emit orders for these coins and
        must exclude their LIVE notional from threshold/divergence math while
        still counting it toward the gross cap.  These coins are guaranteed
        disjoint from ``weights.keys()`` (a conflict raises in ``combine_sleeves``).
    """

    weights: dict[str, float]
    frozen_coins: frozenset[str] = field(default_factory=frozenset)


def combine_sleeves(outcomes: list[SleeveOutcome]) -> CombinedTarget:
    """Combine per-sleeve outcomes into one signed budget-scaled target (spec §4.3).

    Steps
    -----
    1. FRESH sleeves: accumulate ``budget × w`` per coin into a signed running
       sum (overlapping coins net naturally).
    2. HOLD sleeves: collect their coin KEYS into ``frozen_coins`` — they are NOT
       weighted (freeze-NOTIONAL).
    3. STALE sleeves: ignored entirely ({} weights, freeze nothing).
    4. Fail-closed: a coin that is BOTH frozen (HOLD) and traded (FRESH) raises
       ``ValueError`` — cannot freeze and trade one coin in the same cycle.

    Parameters
    ----------
    outcomes:
        The per-sleeve outcomes in config order (order does not affect the
        signed sum; it only affects nothing observable here).

    Returns
    -------
    CombinedTarget
        Signed combined weights + the frozen-coin set.

    Raises
    ------
    ValueError
        When a single coin is both frozen (HOLD sleeve) and freshly traded
        (FRESH sleeve) — ``"frozen-coin conflict: {coin}"``.
    """
    combined: dict[str, float] = {}
    frozen_coins: set[str] = set()

    # Pass 1 — accumulate FRESH contributions and collect frozen coins.
    for outcome in outcomes:
        if outcome.status == "fresh":
            for coin, w in outcome.weights.items():
                # Signed, budget-scaled accumulation; overlaps net naturally.
                combined[coin] = combined.get(coin, 0.0) + outcome.budget * w
        elif outcome.status == "hold":
            # Freeze-NOTIONAL: the coin's live position carries untouched; we
            # record only its identity so the daemon excludes it from the delta.
            frozen_coins.update(outcome.weights.keys())
        # STALE: contributes nothing and freezes nothing (deliberately skipped).

    # Pass 2 — fail-closed on any coin that is BOTH frozen and traded.
    # A single coin cannot be simultaneously carried-untouched (HOLD) and
    # re-targeted (FRESH).  Disjoint universes mean this never fires in prod, but
    # the combination must refuse rather than silently honour one side.
    conflict = frozen_coins.intersection(combined.keys())
    if conflict:
        # Deterministic message: report the first conflicting coin (sorted).
        coin = sorted(conflict)[0]
        raise ValueError(f"frozen-coin conflict: {coin}")

    return CombinedTarget(weights=combined, frozen_coins=frozenset(frozen_coins))


def resolve_outcome(
    *,
    served_ok: bool,
    weights: dict[str, float],
    allow_short: bool,
    budget: float,
    name: str,
    consecutive_holds: int,
    within_staleness_window: bool,
) -> tuple[SleeveOutcome, int]:
    """Resolve a single sleeve's status from its fetch result + staleness policy.

    Policy (spec §4.3, red-team H2 — short sleeves get a tighter dark-signal
    policy because holding a dark short for 2 days is the wrong fail-safe):

    - ``served_ok=True``  → ``("fresh", weights)``; ``consecutive_holds`` resets
      to 0.  A fresh serve always wins, even if the prior staleness window had
      elapsed.
    - ``served_ok=False`` AND within the per-sleeve staleness window:
        * a SHORT sleeve (``allow_short``) whose hold count would reach 2
          (``consecutive_holds + 1 >= 2``) → ``("stale", {})`` — fast de-risk
          after the SECOND consecutive daily HOLD rather than re-freezing.
        * otherwise → ``("hold", {})``; ``consecutive_holds`` increments by 1.
          (Long sleeves keep the freeze behaviour; a short sleeve's FIRST hold
          also freezes — only the second consecutive hold de-risks.)
    - ``served_ok=False`` AND past the window → ``("stale", {})`` regardless of
      side.

    ``resolve_outcome`` is pure: it does NOT read the ledger.  HOLD/STALE return
    EMPTY weights here — the daemon (C5) rebuilds a HOLD ``SleeveOutcome`` with
    ``last_weights`` from the ledger before calling ``combine_sleeves`` (that is
    what supplies the frozen coin keys).

    Parameters
    ----------
    served_ok:
        Whether the sleeve fetched AND validated a target this cycle.
    weights:
        The served signed weights (only used when ``served_ok``).
    allow_short:
        Whether the sleeve permits short/signed weights (gates fast de-risk).
    budget:
        The sleeve's equity budget (carried through onto the outcome).
    name:
        Sleeve identifier (carried through onto the outcome).
    consecutive_holds:
        The sleeve's consecutive-HOLD count BEFORE this cycle (from the ledger).
    within_staleness_window:
        Whether the sleeve is still within its per-sleeve staleness window
        (the caller computes this from ``last_success_ts`` and the per-sleeve
        ``max_signal_staleness_days``).

    Returns
    -------
    tuple[SleeveOutcome, int]
        The resolved outcome and the NEW ``consecutive_holds`` value to persist.
    """
    if served_ok:
        outcome = SleeveOutcome(
            name=name,
            status="fresh",
            weights=dict(weights),
            budget=budget,
            allow_short=allow_short,
        )
        return outcome, 0

    if within_staleness_window:
        new_holds = consecutive_holds + 1
        # Short sleeve fast de-risk: after the SECOND consecutive daily HOLD a
        # short sleeve de-risks (STALE) rather than re-freezing (red-team H2).
        if allow_short and new_holds >= 2:
            outcome = SleeveOutcome(
                name=name,
                status="stale",
                weights={},
                budget=budget,
                allow_short=allow_short,
            )
            return outcome, new_holds
        # Otherwise freeze (HOLD): long sleeves always, short sleeves on the
        # first hold.  resolve_outcome returns EMPTY weights; the daemon supplies
        # last_weights for the freeze.
        outcome = SleeveOutcome(
            name=name,
            status="hold",
            weights={},
            budget=budget,
            allow_short=allow_short,
        )
        return outcome, new_holds

    # Past the per-sleeve staleness window → de-risk to cash regardless of side.
    outcome = SleeveOutcome(
        name=name,
        status="stale",
        weights={},
        budget=budget,
        allow_short=allow_short,
    )
    return outcome, consecutive_holds


__all__ = [
    "CombinedTarget",
    "SleeveOutcome",
    "SleeveStatus",
    "combine_sleeves",
    "resolve_outcome",
]
