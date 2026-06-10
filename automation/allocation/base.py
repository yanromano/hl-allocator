"""
automation.allocation.base — AllocationSource open/closed boundary.

Key design decisions
--------------------
- ``TargetAllocation`` is a pydantic v2 model whose fields match the signed-signal
  payload schema (spec §5.2).  B5 (RemoteSignalSource) will construct it from the
  verified payload; all consumers must treat it as *trusted for alpha, not safety*.

- ``validate_target_allocation`` is the local safety gate called by the executor
  BEFORE acting on any allocation (spec §5.1).  It RAISES ``AllocationRejected``
  on every violation — it NEVER silently clamps values (spec S-11).  Clamping
  would allow a corrupted signal to reach the exchange with a false sense of safety.

- ``AllocationSource`` is a ``typing.Protocol`` so that the executor core can type-
  check any source (HAARP remote, equal-weight, back-test replay) without coupling
  to any concrete implementation.
"""

from __future__ import annotations

import datetime
import math
import typing

import pydantic

from automation.core.redaction import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TargetAllocation(pydantic.BaseModel):
    """Verified target-allocation vector.  Matches the signed-signal payload (spec §5.2).

    Fields
    ------
    as_of:
        The date for which this allocation is valid (UTC calendar date).
    weights:
        Coin → weight mapping.  All values must be in [0, 1]; long-only.
        An empty dict is valid (all-cash).
    cash:
        Uninvested fraction in [0, 1].  Must satisfy
        ``abs(sum(weights.values()) + cash - 1.0) <= tol``.
    strategy_id:
        Opaque identifier for the producing strategy (e.g. ``"haarp-v3"``).
    model_rev:
        Model / parameter revision string (e.g. ``"2026-06-01"``).
    schema_version:
        Payload schema version; defaults to 1.  Guards forward-compatibility.
    audience:
        Client-scoped identifier.  The executor must validate this equals its own
        ``client_id`` before trusting the allocation (spec §5.1, anti-replay).
    """

    as_of: datetime.date
    weights: dict[str, float]
    cash: float
    strategy_id: str
    model_rev: str
    schema_version: int = 1
    audience: str

    model_config = pydantic.ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class AllocationSource(typing.Protocol):
    """Protocol for any allocation source (HAARP remote, equal-weight, replay, …).

    The executor core depends on this Protocol, not on any concrete class, so
    sources can be swapped at configuration time without changing executor logic.
    """

    def get_target_allocation(
        self, as_of: datetime.date | None = None
    ) -> TargetAllocation:
        """Return the current target allocation.

        Parameters
        ----------
        as_of:
            Optional reference date.  If ``None``, the source uses the current
            calendar date or its own internal clock.

        Returns
        -------
        TargetAllocation
            The allocation vector.  The caller is expected to pass it through
            ``validate_target_allocation`` before acting.
        """
        ...


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class AllocationRejected(Exception):
    """Raised by ``validate_target_allocation`` when the allocation is unsafe.

    NEVER catch this silently — it signals that the alpha signal is corrupt or
    misrouted.  The daemon must hard-alert (spec S-11) and halt execution for
    the affected portfolio until human review.
    """


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_target_allocation(
    ta: TargetAllocation,
    *,
    max_per_asset: float,
    whitelist: set[str],
    client_id: str,
    tol: float = 1e-6,
) -> None:
    """Local safety gate: validates a ``TargetAllocation`` before any trade is placed.

    This function is the **abort-not-clamp** safety rail (spec §5.1, S-11).  On
    ANY violation it logs a redacted warning and raises ``AllocationRejected``.
    It NEVER silently adjusts weights — clamping would mask signal corruption.

    Parameters
    ----------
    ta:
        The allocation to validate.
    max_per_asset:
        Maximum fractional weight allowed for any single coin.  If any coin
        exceeds this, the signal is considered untrusted and execution halts.
    whitelist:
        Set of allowed coin symbols.  Any coin outside this set is rejected.
    client_id:
        The executor's own client identifier.  Must match ``ta.audience``
        (anti-replay / audience mismatch guard).
    tol:
        Floating-point tolerance for the unity-sum check.  Defaults to 1e-6.

    Raises
    ------
    AllocationRejected
        On any of the following violations:
        - any weight is NaN or infinite
        - any weight is negative (long-only contract)
        - ``sum(weights) > 1.0 + tol`` (over-allocated / implicit leverage)
        - ``abs(sum(weights) + cash - 1.0) > tol`` (does not sum to unity)
        - any weight exceeds ``max_per_asset`` (hard safety rail, spec S-11)
        - any coin is not in ``whitelist``
        - ``ta.audience != client_id`` (anti-replay guard)
    """
    # ---- 1. Audience check (anti-replay, do first — cheapest) -------------
    if ta.audience != client_id:
        _logger.warning(
            "validate_target_allocation: audience mismatch",
            audience=ta.audience,
            expected_client_id=client_id,
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"Allocation audience {ta.audience!r} does not match client_id {client_id!r}"
        )

    # ---- 2. NaN / inf check (before arithmetic to avoid contamination) -----
    # CRITICAL (F11): ``cash`` MUST be included in the finiteness guard.  If
    # ``cash`` is NaN, the unity-sum check ``abs(weight_sum + cash - 1.0) > tol``
    # evaluates to ``nan > tol == False`` and a fully-signed payload with
    # ``cash=NaN`` (even empty weights) would PASS and reach the sizer.  Guard
    # both weights AND cash here so every AllocationSource is protected.
    bad_values = {
        coin: w
        for coin, w in ta.weights.items()
        if not math.isfinite(w)
    }
    if bad_values:
        _logger.warning(
            "validate_target_allocation: non-finite weights detected",
            coins=list(bad_values.keys()),
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"Non-finite weights for coins: {list(bad_values.keys())}"
        )

    if not math.isfinite(ta.cash):
        _logger.warning(
            "validate_target_allocation: non-finite cash detected (F11)",
            cash=str(ta.cash),
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(f"Non-finite cash value: {ta.cash}")

    # ---- 3. Long-only: no negative weights ---------------------------------
    negative = {coin: w for coin, w in ta.weights.items() if w < 0.0}
    if negative:
        _logger.warning(
            "validate_target_allocation: negative weights (long-only contract violated)",
            coins=list(negative.keys()),
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"Negative weights violate long-only contract: {list(negative.keys())}"
        )

    # ---- 4. Whitelist check ------------------------------------------------
    unknown = {coin for coin in ta.weights if coin not in whitelist}
    if unknown:
        _logger.warning(
            "validate_target_allocation: coins outside whitelist",
            unknown_coins=sorted(unknown),
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"Coins not in whitelist: {sorted(unknown)}"
        )

    # ---- 5. Per-asset cap (SAFETY RAIL — abort, never clamp) ---------------
    breaching = {coin: w for coin, w in ta.weights.items() if w > max_per_asset}
    if breaching:
        _logger.warning(
            "validate_target_allocation: per-asset cap breached — ABORTING (spec S-11)",
            breaching_coins={coin: round(w, 6) for coin, w in breaching.items()},
            max_per_asset=max_per_asset,
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"Per-asset cap ({max_per_asset}) breached for: "
            f"{[(c, round(w, 6)) for c, w in breaching.items()]}. "
            "Signal is untrusted — executor hard-alerts (spec S-11)."
        )

    # ---- 6. Over-allocation check (sum(weights) alone) ---------------------
    weight_sum = sum(ta.weights.values())
    if weight_sum > 1.0 + tol:
        _logger.warning(
            "validate_target_allocation: weights sum exceeds 1.0",
            weight_sum=round(weight_sum, 9),
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"weights sum {weight_sum:.9f} exceeds 1.0 + tol ({1.0 + tol:.9f})"
        )

    # ---- 7. Unity sum: abs(sum(weights) + cash - 1.0) > tol ---------------
    total = weight_sum + ta.cash
    if abs(total - 1.0) > tol:
        _logger.warning(
            "validate_target_allocation: sum(weights) + cash does not equal 1.0",
            total=round(total, 9),
            weight_sum=round(weight_sum, 9),
            cash=round(ta.cash, 9),
            strategy_id=ta.strategy_id,
        )
        raise AllocationRejected(
            f"sum(weights) + cash = {total:.9f}, expected 1.0 ± {tol}"
        )
