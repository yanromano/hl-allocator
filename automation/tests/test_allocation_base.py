"""
Tests for automation.allocation.base and automation.allocation.examples.equal_weight.

Covers:
- validate_target_allocation: passes a valid allocation; rejects sum>1; rejects negative
  weight; rejects weight > max_per_asset (RAISES, does NOT clamp); rejects unknown coin;
  rejects audience mismatch; rejects NaN weight; rejects sum+cash != 1.
- EqualWeightSource: produces an allocation that passes validation for a matching
  whitelist / client_id / max_per_asset.
"""

from __future__ import annotations

import datetime
import math

import pytest

from automation.allocation.base import (
    AllocationRejected,
    AllocationSource,
    TargetAllocation,
    validate_target_allocation,
)
from automation.allocation.examples.equal_weight import EqualWeightSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = datetime.date(2026, 6, 1)


def _make_ta(
    weights: dict[str, float] | None = None,
    cash: float = 0.0,
    audience: str = "client-1",
    strategy_id: str = "test-strategy",
    model_rev: str = "v1",
    as_of: datetime.date = _TODAY,
    schema_version: int = 1,
) -> TargetAllocation:
    if weights is None:
        weights = {"BTC": 0.5, "ETH": 0.5}
    return TargetAllocation(
        as_of=as_of,
        weights=weights,
        cash=cash,
        strategy_id=strategy_id,
        model_rev=model_rev,
        schema_version=schema_version,
        audience=audience,
    )


_DEFAULT_WHITELIST: set[str] = {"BTC", "ETH", "SOL"}
_DEFAULT_CLIENT = "client-1"
_DEFAULT_MAX_PER_ASSET = 0.6


def _validate_default(ta: TargetAllocation, **kwargs: object) -> None:
    validate_target_allocation(
        ta,
        max_per_asset=_DEFAULT_MAX_PER_ASSET,
        whitelist=_DEFAULT_WHITELIST,
        client_id=_DEFAULT_CLIENT,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_allocation_passes() -> None:
    """A well-formed allocation with correct sum, whitelist, and audience passes."""
    ta = _make_ta(weights={"BTC": 0.5, "ETH": 0.5}, cash=0.0)
    _validate_default(ta)  # must not raise


def test_valid_allocation_with_cash_passes() -> None:
    """An allocation with partial cash (weights + cash == 1.0) passes."""
    ta = _make_ta(weights={"BTC": 0.3, "ETH": 0.3}, cash=0.4)
    _validate_default(ta)


def test_valid_single_asset_passes() -> None:
    """A single-asset allocation with matching cash passes."""
    ta = _make_ta(weights={"BTC": 0.6}, cash=0.4)
    _validate_default(ta)


def test_valid_zero_weights_passes() -> None:
    """All-cash allocation (empty weights dict, cash=1.0) passes."""
    ta = _make_ta(weights={}, cash=1.0)
    _validate_default(ta)


# ---------------------------------------------------------------------------
# Rejection: sum > 1
# ---------------------------------------------------------------------------


def test_rejects_weights_sum_exceeds_one() -> None:
    """weights summing > 1.0 raises AllocationRejected."""
    # weights sum to 1.2 — clearly over-allocated
    ta = _make_ta(weights={"BTC": 0.6, "ETH": 0.6}, cash=0.0)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_weights_plus_cash_exceeds_one() -> None:
    """weights + cash > 1.0 (leverage) raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": 0.5, "ETH": 0.3}, cash=0.4)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


# ---------------------------------------------------------------------------
# Rejection: weights + cash != 1.0 (under-allocated by more than tol)
# ---------------------------------------------------------------------------


def test_rejects_sum_plus_cash_below_one() -> None:
    """sum(weights) + cash < 1.0 by more than tol raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": 0.3, "ETH": 0.3}, cash=0.1)  # total = 0.7
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


# ---------------------------------------------------------------------------
# Rejection: negative weight
# ---------------------------------------------------------------------------


def test_rejects_negative_weight() -> None:
    """A negative weight (short) raises AllocationRejected (long-only contract)."""
    ta = _make_ta(weights={"BTC": 0.6, "ETH": -0.1}, cash=0.5)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


# ---------------------------------------------------------------------------
# Rejection: weight > max_per_asset (SAFETY RAIL — must NOT clamp)
# ---------------------------------------------------------------------------


def test_rejects_weight_exceeds_max_per_asset() -> None:
    """A single weight > max_per_asset raises AllocationRejected — never clamps (spec S-11)."""
    ta = _make_ta(weights={"BTC": 0.65, "ETH": 0.35}, cash=0.0)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)  # max_per_asset=0.6, BTC=0.65 → reject


def test_max_per_asset_breach_raises_not_clamps() -> None:
    """Verifies clamping did NOT happen: the allocation object is unchanged after rejection."""
    ta = _make_ta(weights={"BTC": 0.7, "ETH": 0.3}, cash=0.0)
    original_btc = ta.weights["BTC"]
    with pytest.raises(AllocationRejected):
        _validate_default(ta)
    # The weight must be UNCHANGED — validate_target_allocation must not mutate the model
    assert ta.weights["BTC"] == original_btc


def test_weight_exactly_at_max_per_asset_passes() -> None:
    """A weight exactly equal to max_per_asset is NOT rejected (boundary)."""
    ta = _make_ta(weights={"BTC": 0.6, "ETH": 0.4}, cash=0.0)
    _validate_default(ta)  # max_per_asset=0.6, BTC=0.6 → OK


# ---------------------------------------------------------------------------
# Rejection: coin not in whitelist
# ---------------------------------------------------------------------------


def test_rejects_coin_not_in_whitelist() -> None:
    """A coin outside the whitelist raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": 0.5, "DOGE": 0.5}, cash=0.0)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_all_coins_not_in_whitelist() -> None:
    """Every coin outside the whitelist raises AllocationRejected."""
    ta = _make_ta(weights={"XMR": 0.5, "ZEC": 0.5}, cash=0.0)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


# ---------------------------------------------------------------------------
# Rejection: audience mismatch
# ---------------------------------------------------------------------------


def test_rejects_wrong_audience() -> None:
    """audience != client_id raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": 0.5, "ETH": 0.5}, cash=0.0, audience="other-client")
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_empty_audience() -> None:
    """An empty audience string raises AllocationRejected when client_id is non-empty."""
    ta = _make_ta(weights={"BTC": 0.5, "ETH": 0.5}, cash=0.0, audience="")
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


# ---------------------------------------------------------------------------
# Rejection: NaN / inf weight
# ---------------------------------------------------------------------------


def test_rejects_nan_weight() -> None:
    """A NaN weight raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": float("nan"), "ETH": 0.5}, cash=0.5)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_inf_weight() -> None:
    """An infinite weight raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": float("inf"), "ETH": 0.0}, cash=0.0)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_neg_inf_weight() -> None:
    """A negative infinite weight raises AllocationRejected."""
    ta = _make_ta(weights={"BTC": float("-inf"), "ETH": 1.0}, cash=0.0)
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_nan_cash() -> None:
    """A NaN cash raises AllocationRejected (F11).

    Without the cash finiteness guard, the unity-sum check
    ``abs(weight_sum + cash - 1.0) > tol`` evaluates ``nan > tol == False`` and
    a payload with cash=NaN would silently PASS.
    """
    ta = _make_ta(weights={}, cash=float("nan"))
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


def test_rejects_inf_cash() -> None:
    """An infinite cash raises AllocationRejected (F11)."""
    ta = _make_ta(weights={"BTC": 0.5}, cash=float("inf"))
    with pytest.raises(AllocationRejected):
        _validate_default(ta)


# ---------------------------------------------------------------------------
# TargetAllocation model: field defaults and construction
# ---------------------------------------------------------------------------


def test_target_allocation_defaults() -> None:
    """schema_version defaults to 1 when omitted."""
    ta = TargetAllocation(
        as_of=_TODAY,
        weights={"BTC": 1.0},
        cash=0.0,
        strategy_id="s",
        model_rev="r",
        audience="c",
    )
    assert ta.schema_version == 1


def test_target_allocation_is_pydantic_model() -> None:
    """TargetAllocation fields are accessible and correctly typed."""
    ta = _make_ta()
    assert isinstance(ta.weights, dict)
    assert isinstance(ta.cash, float)
    assert isinstance(ta.as_of, datetime.date)


# ---------------------------------------------------------------------------
# AllocationSource Protocol (structural check)
# ---------------------------------------------------------------------------


def test_allocation_source_protocol_satisfied() -> None:
    """EqualWeightSource satisfies the AllocationSource Protocol structurally."""
    source: AllocationSource = EqualWeightSource(
        coins=["BTC", "ETH"], client_id="client-1"
    )
    ta = source.get_target_allocation()
    assert isinstance(ta, TargetAllocation)


# ---------------------------------------------------------------------------
# EqualWeightSource: happy-path validation
# ---------------------------------------------------------------------------


def test_equal_weight_source_validates() -> None:
    """EqualWeightSource(["BTC","ETH"], client_id="c1") passes validate_target_allocation."""
    source = EqualWeightSource(coins=["BTC", "ETH"], client_id="c1")
    ta = source.get_target_allocation()
    validate_target_allocation(
        ta,
        max_per_asset=0.6,
        whitelist={"BTC", "ETH"},
        client_id="c1",
    )


def test_equal_weight_source_correct_weights() -> None:
    """Equal weight of 0.5 each for two coins with invested_fraction=1.0."""
    source = EqualWeightSource(coins=["BTC", "ETH"], client_id="c1")
    ta = source.get_target_allocation()
    assert math.isclose(ta.weights["BTC"], 0.5)
    assert math.isclose(ta.weights["ETH"], 0.5)
    assert math.isclose(ta.cash, 0.0)


def test_equal_weight_source_partial_investment() -> None:
    """invested_fraction=0.6 → 0.3 per coin, cash=0.4."""
    source = EqualWeightSource(
        coins=["BTC", "ETH"], client_id="c1", invested_fraction=0.6
    )
    ta = source.get_target_allocation()
    assert math.isclose(ta.weights["BTC"], 0.3)
    assert math.isclose(ta.weights["ETH"], 0.3)
    assert math.isclose(ta.cash, 0.4)


def test_equal_weight_source_sum_one() -> None:
    """sum(weights) + cash == 1.0 for any invested_fraction."""
    for frac in [0.0, 0.25, 0.5, 0.8, 1.0]:
        source = EqualWeightSource(coins=["BTC", "ETH", "SOL"], client_id="c1", invested_fraction=frac)
        ta = source.get_target_allocation()
        total = sum(ta.weights.values()) + ta.cash
        assert math.isclose(total, 1.0, abs_tol=1e-9), f"frac={frac}: total={total}"


def test_equal_weight_source_audience_matches_client_id() -> None:
    """EqualWeightSource sets audience == client_id."""
    source = EqualWeightSource(coins=["BTC", "ETH"], client_id="my-client")
    ta = source.get_target_allocation()
    assert ta.audience == "my-client"


def test_equal_weight_source_as_of_override() -> None:
    """get_target_allocation(as_of=...) uses the provided date."""
    source = EqualWeightSource(coins=["BTC"], client_id="c1")
    specific_date = datetime.date(2026, 1, 15)
    ta = source.get_target_allocation(as_of=specific_date)
    assert ta.as_of == specific_date


def test_equal_weight_source_strategy_id() -> None:
    """Custom strategy_id and model_rev are passed through to the TargetAllocation."""
    source = EqualWeightSource(
        coins=["BTC"],
        client_id="c1",
        strategy_id="custom-strat",
        model_rev="v42",
    )
    ta = source.get_target_allocation()
    assert ta.strategy_id == "custom-strat"
    assert ta.model_rev == "v42"


def test_equal_weight_source_three_coins() -> None:
    """Equal weight for 3 coins: each should be invested_fraction/3."""
    source = EqualWeightSource(coins=["BTC", "ETH", "SOL"], client_id="c1")
    ta = source.get_target_allocation()
    expected = 1.0 / 3.0
    for coin in ["BTC", "ETH", "SOL"]:
        assert math.isclose(ta.weights[coin], expected, rel_tol=1e-9)
    assert math.isclose(ta.cash, 0.0, abs_tol=1e-9)


def test_equal_weight_source_validates_three_coins() -> None:
    """3-coin equal-weight validates with max_per_asset=0.4."""
    source = EqualWeightSource(coins=["BTC", "ETH", "SOL"], client_id="c1")
    ta = source.get_target_allocation()
    validate_target_allocation(
        ta,
        max_per_asset=0.4,
        whitelist={"BTC", "ETH", "SOL"},
        client_id="c1",
    )
