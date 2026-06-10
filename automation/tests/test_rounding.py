"""
Tests for automation.core.rounding.

Verifies the known cases from the spec:
- round_price(123456.7, 2) → 123457.0  (≥1e5 → int)
- round_price(2.34567, 4)  → 2.35       (5-sig + ≤ 2 dp)
- floor_size(0.123999, 3)  → 0.123      (floors, not rounds to 0.124)
- floor_size(1.9999, 0)    → 1.0        (floors to integer)

Plus additional edge-case coverage.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from automation.core.rounding import floor_size, round_price

# ---------------------------------------------------------------------------
# round_price — spec examples
# ---------------------------------------------------------------------------


def test_round_price_large_integer() -> None:
    """≥ 1e5 → integer (round to nearest int)."""
    result = round_price(123456.7, 2)
    assert result == 123457.0
    assert isinstance(result, float)


def test_round_price_large_exact() -> None:
    """An already-integer value at ≥ 1e5 stays unchanged."""
    assert round_price(100000.0, 2) == 100000.0


def test_round_price_large_round_down() -> None:
    """Large price rounds down when fractional part < 0.5."""
    assert round_price(123456.2, 2) == 123456.0


def test_round_price_5sig_2dp() -> None:
    """5 significant figures with ≤ 2 decimal places (sz_decimals=4 on perp)."""
    result = round_price(2.34567, 4)
    # max_dec = 6 - 4 = 2; 5-sig of 2.34567 = 2.3457, rounded to 2dp → 2.35
    assert result == 2.35


def test_round_price_5sig_4dp() -> None:
    """Price < 1e5, sz_decimals=2, max_dec=4; 5 sig-figs and ≤ 4 dp."""
    # 5-sig of 1.23456 = 1.2346; clamped to 4dp → 1.2346
    result = round_price(1.23456, 2)
    assert result == pytest.approx(1.2346, abs=1e-9)


def test_round_price_spot_extra_decimals() -> None:
    """Spot markets allow 8 - sz_decimals decimal places."""
    # is_spot=True, sz_decimals=2 → max_dec=6
    # 5-sig of 1.234567 = 1.2346; clamped to 6dp → 1.2346
    result = round_price(1.234567, 2, is_spot=True)
    assert result == pytest.approx(1.2346, abs=1e-9)


def test_round_price_accepts_decimal() -> None:
    """round_price must accept Decimal inputs."""
    result = round_price(Decimal("2.34567"), 4)
    assert result == 2.35


def test_round_price_returns_float() -> None:
    """Return type is always float."""
    assert isinstance(round_price(1.5, 2), float)
    assert isinstance(round_price(200000.0, 2), float)


def test_round_price_zero() -> None:
    """Price of 0 should return 0.0."""
    assert round_price(0.0, 2) == 0.0


def test_round_price_just_below_1e5() -> None:
    """Price just below 1e5 uses the 5-sig path, not the int path."""
    # 99999.9 < 1e5 → 5-sig: "99999.9"[:5sig] = 1.0000e5 = 100000
    result = round_price(99999.9, 0)
    # 5-sig of 99999.9 = 1.0000e5 = 100000.0
    assert result == 100000.0


def test_round_price_exactly_1e5() -> None:
    """Price exactly at 1e5 uses the integer path."""
    assert round_price(100000.0, 0) == 100000.0


def test_round_price_small_price() -> None:
    """Small prices (< 1) work correctly with 5 sig-figs.

    Pipeline: 0.000123456 → 5-sig = 0.00012346 → round(0.00012346, 6) = 0.000123
    (The 7th decimal is 4, so rounding to 6 dp gives 0.000123, not 0.00012346.)
    """
    result = round_price(0.000123456, 0)
    assert result == pytest.approx(0.000123, abs=1e-12)


# ---------------------------------------------------------------------------
# floor_size — spec examples
# ---------------------------------------------------------------------------


def test_floor_size_spec_example_3dp() -> None:
    """floor_size(0.123999, 3) must return 0.123 (floor, not round to 0.124)."""
    result = floor_size(0.123999, 3)
    assert result == 0.123


def test_floor_size_spec_example_0dp() -> None:
    """floor_size(1.9999, 0) must return 1.0 (floor to integer)."""
    result = floor_size(1.9999, 0)
    assert result == 1.0
    assert isinstance(result, float)


def test_floor_size_exact_value() -> None:
    """An already-exact value should be unchanged."""
    assert floor_size(0.123, 3) == 0.123


def test_floor_size_accepts_decimal() -> None:
    """floor_size must accept Decimal inputs."""
    result = floor_size(Decimal("0.123999"), 3)
    assert result == 0.123


def test_floor_size_2dp() -> None:
    """floor_size always floors, never rounds: 1.235 → 1.23 (not 1.24) with 2dp."""
    assert floor_size(1.235, 2) == 1.23


def test_floor_size_1dp() -> None:
    """floor_size with 1 decimal place."""
    assert floor_size(9.99, 1) == 9.9


def test_floor_size_large_integer() -> None:
    """Large integers with 0 decimals stay as-is."""
    assert floor_size(1000.0, 0) == 1000.0


def test_floor_size_returns_float() -> None:
    """Return type is always float."""
    assert isinstance(floor_size(1.5, 2), float)


def test_floor_size_zero() -> None:
    """floor_size of 0 is 0.0."""
    assert floor_size(0.0, 4) == 0.0


def test_floor_size_negative_floors_toward_negative_infinity() -> None:
    """floor_size of negative size floors toward negative infinity (math.floor semantics)."""
    # -0.1 floored to 0dp = -1.0
    assert floor_size(-0.1, 0) == -1.0


def test_floor_size_does_not_round_up() -> None:
    """Critical: must never round UP (safety property for position sizing)."""
    # 5.999 with 2dp: rounding would give 6.00; floor gives 5.99
    assert floor_size(5.999, 2) == 5.99
    # 0.9995 with 3dp: rounding would give 1.000; floor gives 0.999
    assert floor_size(0.9995, 3) == 0.999
