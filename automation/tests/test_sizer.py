"""
Tests for automation.execution.sizer — weight→size plan + frozen-target ratchet.

All expected values are HAND-COMPUTED — no tautologies.

Coverage:
- bootstrap (state.last_rebalanced_target=None) → rebalance=True, reason="bootstrap"
- HOLD: prev==target, achieved≈target → rebalance=False, orders=[]
- threshold fire: max_delta >= 0.03 → rebalance=True
- abs_drift backstop: prev==target but live drift >=0.06 → rebalance=True
- floor sizing: hand-verify specific BTC size
- Σnotional scale-down: post-scale gross ≤ cap
- min-notional skip: tiny delta → in skipped not orders
- long-only flatten: szi<0 → in flatten list
- unpriceable coin: missing mark → in skipped, no crash
- universe_perp_map translation: ticker→perp respected for price/pos lookups
- all-or-nothing: rebalance=False → orders=[], flatten=[], target_sizes={}
- sizer is pure: does not mutate state
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from automation.allocation.base import TargetAllocation
from automation.core.config import CapsConfig, Config, SignalSourceConfig
from automation.execution.reconciler import Snapshot
from automation.execution.sizer import MIN_NOTIONAL_USD, SizerError, plan
from automation.reporting.state import State

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _cfg(
    threshold_pct: float = 3.0,
    safety_buffer: float = 0.03,
    universe_perp_map: dict[str, str] | None = None,
    max_per_asset: float = 0.6,
) -> Config:
    return Config(
        env="testnet",
        subaccount_address="0x0000000000000000000000000000000000000000",
        signal_source=SignalSourceConfig(type="equal_weight"),
        rebalance_utc_time="00:15",
        threshold_pct=threshold_pct,
        max_per_asset=max_per_asset,
        safety_buffer=safety_buffer,
        universe_perp_map=universe_perp_map or {},
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.01,
        ),
    )


def _target(
    weights: dict[str, float],
    cash: float | None = None,
    as_of: str = "2026-06-01",
) -> TargetAllocation:
    if cash is None:
        cash = 1.0 - sum(weights.values())
    return TargetAllocation(
        as_of=datetime.date.fromisoformat(as_of),
        weights=weights,
        cash=cash,
        strategy_id="test",
        model_rev="test",
        audience="test",
    )


def _snap(
    equity: float = 100_000.0,
    positions: dict[str, float] | None = None,
    marks: dict[str, float] | None = None,
    sz_decimals: dict[str, int] | None = None,
) -> Snapshot:
    return Snapshot(
        equity=Decimal(str(equity)),
        positions={k: Decimal(str(v)) for k, v in (positions or {}).items()},
        marks={k: Decimal(str(v)) for k, v in (marks or {}).items()},
        sz_decimals=sz_decimals or {},
    )


def _state(
    last_target: dict[str, float] | None = None,
    as_of: str | None = None,
) -> State:
    return State(
        last_rebalanced_target=last_target,
        last_rebalanced_as_of=as_of,
    )


# ---------------------------------------------------------------------------
# 1. Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_bootstrap_triggers_rebalance(self) -> None:
        """state.last_rebalanced_target=None → rebalance=True, reason='bootstrap'."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        snap = _snap(
            equity=100_000,
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = _state(last_target=None)
        cfg = _cfg()

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        assert sp.reason == "bootstrap"
        assert len(sp.orders) > 0

    def test_bootstrap_does_not_mutate_state(self) -> None:
        """The sizer is pure — state.last_rebalanced_target stays None after plan()."""
        ta = _target({"BTC": 1.0})
        snap = _snap(marks={"BTC": 50_000.0}, sz_decimals={"BTC": 3})
        state = _state(last_target=None)

        plan(ta, snap, state, _cfg())

        assert state.last_rebalanced_target is None


# ---------------------------------------------------------------------------
# 2. HOLD (all-or-nothing)
# ---------------------------------------------------------------------------


class TestHold:
    def test_hold_when_no_change_no_drift(self) -> None:
        """prev==target weights AND achieved≈target → rebalance=False."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        # achieved close to target: BTC 0.5*60000/100000=0.3 not quite 0.5
        # To make abs_drift < backstop (0.06), set up so achieved ≈ target.
        # BTC: szi × mark / equity = target_w
        # BTC w=0.5 → szi = 0.5 * 100000 / 60000 = 0.8333
        # ETH w=0.5 → szi = 0.5 * 100000 / 2000 = 25.0
        snap = _snap(
            equity=100_000,
            positions={"BTC": 0.8333, "ETH": 25.0},
            marks={"BTC": 60_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        state = _state(last_target={"BTC": 0.5, "ETH": 0.5})
        cfg = _cfg(threshold_pct=3.0)

        sp = plan(_target(weights), snap, state, cfg)

        assert sp.rebalance is False
        assert sp.reason == "hold"
        assert sp.orders == []
        assert sp.flatten == []
        assert sp.target_sizes == {}

    def test_hold_gross_notional_zero(self) -> None:
        """HOLD plan returns gross_notional=0.0."""
        weights = {"BTC": 0.5}
        snap = _snap(
            equity=100_000,
            positions={"BTC": 0.8333},
            marks={"BTC": 60_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target={"BTC": 0.5})
        sp = plan(_target(weights), snap, state, _cfg())

        assert sp.gross_notional == 0.0
        assert sp.scale_factor == 1.0
        assert sp.scaled_down is False


# ---------------------------------------------------------------------------
# 3. Threshold fire
# ---------------------------------------------------------------------------


class TestThresholdFire:
    def test_threshold_fire_on_btc_delta(self) -> None:
        """prev BTC=0.50, target BTC=0.54 (Δ=0.04 ≥ 0.03) → rebalance=True."""
        ta = _target({"BTC": 0.54, "ETH": 0.46})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = _state(last_target={"BTC": 0.50, "ETH": 0.46})
        cfg = _cfg(threshold_pct=3.0)

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        assert "max_delta" in sp.reason
        assert "0.0400" in sp.reason

    def test_threshold_fire_sizes_correctly(self) -> None:
        """Hand-verify sizes when threshold fires.

        E=100_000, BTC w=0.54, px=50_000, szDecimals=3:
          target_notional = 0.54 * 100_000 = 54_000
          raw_size = 54_000 / 50_000 = 1.08
          floored = floor(1.08 * 1000) / 1000 = floor(1080) / 1000 = 1.080

        ETH w=0.46, px=2_000, szDecimals=2:
          target_notional = 0.46 * 100_000 = 46_000
          raw_size = 46_000 / 2_000 = 23.0
          floored = floor(23.0 * 100) / 100 = 23.00

        Σnotional = 1.080 * 50_000 + 23.00 * 2_000 = 54_000 + 46_000 = 100_000
        cap = 100_000 * (1 - 0.03) = 97_000  → needs scale-down!
        scale = 97_000 / 100_000 = 0.97
        BTC_scaled_notional = 54_000 * 0.97 = 52_380 → size = floor(52_380/50_000 * 1000)/1000
                            = floor(1.0476 * 1000)/1000 = floor(1047.6)/1000 = 1.047
        ETH_scaled_notional = 46_000 * 0.97 = 44_620 → size = floor(44_620/2_000 * 100)/100
                            = floor(22.31 * 100)/100 = floor(2231)/100 = 22.31
        post_gross = 1.047 * 50_000 + 22.31 * 2_000 = 52_350 + 44_620 = 96_970 ≤ 97_000 ✓
        """
        ta = _target({"BTC": 0.54, "ETH": 0.46})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = _state(last_target={"BTC": 0.50, "ETH": 0.46})

        sp = plan(ta, snap, state, _cfg(safety_buffer=0.03))

        assert sp.scaled_down is True
        assert abs(sp.scale_factor - 0.97) < 1e-9

        btc_ts = sp.target_sizes.get("BTC", 0.0)
        eth_ts = sp.target_sizes.get("ETH", 0.0)
        assert abs(btc_ts - 1.047) < 1e-6, f"BTC size={btc_ts}"
        assert abs(eth_ts - 22.31) < 1e-6, f"ETH size={eth_ts}"

        # Gross notional ≤ cap
        assert sp.gross_notional <= 100_000 * (1 - 0.03) + 1e-4

    def test_below_threshold_no_rebalance(self) -> None:
        """max_delta=0.02 < 0.03 and no drift → HOLD."""
        ta = _target({"BTC": 0.52, "ETH": 0.48})
        # achieved ≈ target (szi ≈ target/px)
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.04, "ETH": 24.0},   # ≈ 0.52 and 0.48 weights
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        # prev was 0.50/0.50 → delta=0.02 < threshold
        state = _state(last_target={"BTC": 0.50, "ETH": 0.50})

        sp = plan(ta, snap, state, _cfg(threshold_pct=3.0))

        # abs_drift: BTC = |0.52 - 1.04*50000/100000| = |0.52-0.52| = 0
        #            ETH = |0.48 - 24.0*2000/100000| = |0.48-0.48| = 0
        assert sp.rebalance is False


# ---------------------------------------------------------------------------
# 4. Abs-drift backstop
# ---------------------------------------------------------------------------


class TestAbsDriftBackstop:
    def test_backstop_fires_when_achieved_drifts(self) -> None:
        """prev==target (max_delta=0) but achieved drifted 0.07 → rebalance=True."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        # prev == target → max_delta=0.0
        # achieved: BTC pos=0.3/50000*100000=0.3 → drift vs target=0.5 is 0.2 >> backstop=0.06
        snap = _snap(
            equity=100_000,
            positions={"BTC": 0.3},   # achieved_w = 0.3*50000/100000 = 0.15
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = _state(last_target={"BTC": 0.5, "ETH": 0.5})

        sp = plan(_target(weights), snap, state, _cfg(threshold_pct=3.0))

        # BTC drift = |0.5 - 0.15| = 0.35 >> backstop 0.06
        assert sp.rebalance is True
        assert "backstop" in sp.reason

    def test_backstop_does_not_fire_when_drift_small(self) -> None:
        """Drift = 0.04 < backstop (0.06) and max_delta=0 → HOLD."""
        weights = {"BTC": 0.5}
        # achieved BTC = 0.46 (drift=0.04 < 0.06)
        # szi = 0.46 * 100000 / 60000 = 0.7667
        snap = _snap(
            equity=100_000,
            positions={"BTC": 0.7667},
            marks={"BTC": 60_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target={"BTC": 0.5})

        sp = plan(_target(weights), snap, state, _cfg(threshold_pct=3.0))
        # achieved_w = 0.7667 * 60000 / 100000 = 0.46002 → drift ≈ 0.04 < 0.06
        assert sp.rebalance is False


# ---------------------------------------------------------------------------
# 5. Floor sizing
# ---------------------------------------------------------------------------


class TestFloorSizing:
    def test_floor_sizing_hand_verified(self) -> None:
        """E=10000, w=0.5, px=63333, szDecimals=3 → floored=0.078.

        target_notional = 0.5 * 10000 = 5000
        raw = 5000 / 63333 = 0.078947...
        floored = floor(0.078947 * 1000) / 1000 = floor(78.947) / 1000 = 78/1000 = 0.078
        """
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=10_000,
            positions={},
            marks={"BTC": 63_333.0},
            sz_decimals={"BTC": 3},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.0)  # no scale-down for isolation

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        # Account for safety_buffer=0: gross == E, cap=E → no scale-down
        # But gross = 0.078 * 63333 = 4939.974 < 10000 = cap → no scale
        btc_ts = sp.target_sizes.get("BTC", -1.0)
        assert abs(btc_ts - 0.078) < 1e-9, f"Expected 0.078, got {btc_ts}"

    def test_floor_never_rounds_up(self) -> None:
        """Size is always FLOORED, never rounded to nearest."""
        # 1.9999 with szDecimals=0 → floor(1.9999) = 1 (not 2)
        ta = _target({"BTC": 1.0})
        # E=10000, px=5000.25, raw = 10000/5000.25 = 1.9999...
        snap = _snap(
            equity=10_000,
            positions={},
            marks={"BTC": 5_000.25},
            sz_decimals={"BTC": 0},
        )
        state = _state(last_target=None)

        sp = plan(ta, snap, state, _cfg(safety_buffer=0.0))

        btc_ts = sp.target_sizes.get("BTC", -1.0)
        assert btc_ts == 1.0, f"Expected 1.0 (floored), got {btc_ts}"


# ---------------------------------------------------------------------------
# 6. Σnotional scale-down
# ---------------------------------------------------------------------------


class TestSigmaNotionalScaleDown:
    def test_scale_down_fires_when_gross_exceeds_cap(self) -> None:
        """Two coins summing to ~1.0 weight, safety_buffer=0.03 → scaled_down=True."""
        # BTC: w=0.5, px=50000, E=100000 → raw notional=50000
        # ETH: w=0.5, px=2000, szDec=1 → raw notional=50000
        # total raw = 100_000, cap = 97_000 → needs scale-down
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)

        assert sp.scaled_down is True
        assert sp.scale_factor < 1.0

    def test_post_scale_gross_lte_cap(self) -> None:
        """After scale-down, gross_notional ≤ equity * (1 - safety_buffer)."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        cfg = _cfg(safety_buffer=0.03)
        state = _state(last_target=None)

        sp = plan(ta, snap, state, cfg)

        cap = 100_000 * (1 - 0.03)
        assert sp.gross_notional <= cap + 1e-4, (
            f"gross {sp.gross_notional:.4f} exceeds cap {cap:.4f}"
        )

    def test_no_scale_when_gross_within_cap(self) -> None:
        """Small weight → gross < cap → scaled_down=False."""
        ta = _target({"BTC": 0.3})  # only 30% invested
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)

        # gross = 0.3 * 100000 (approx, after floor) = ~30000 < 97000
        assert sp.scaled_down is False
        assert sp.scale_factor == 1.0


# ---------------------------------------------------------------------------
# 7. Min-notional skip
# ---------------------------------------------------------------------------


class TestMinNotionalSkip:
    def test_tiny_delta_skipped_not_in_orders(self) -> None:
        """A delta < $10 is in skipped with reason 'below_min_notional', not in orders."""
        # Existing BTC position at target_size already.
        # BTC: target_size = floor(0.5 * 100000 / 50000) = floor(1.0) = 1.0
        # Current pos = 0.9999 → delta_size = 1.0 - 0.9999 = 0.0001
        # delta_notional = 0.0001 * 50000 = $5 < $10 → skip
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=100_000,
            positions={"BTC": 0.9999},
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.0)

        sp = plan(ta, snap, state, cfg)

        btc_orders = [o for o in sp.orders if o.coin == "BTC"]
        assert btc_orders == [], f"Expected BTC to be skipped, got orders: {sp.orders}"

        btc_skipped = [s for s in sp.skipped if s["coin"] == "BTC"]
        assert len(btc_skipped) == 1
        assert btc_skipped[0]["reason"] == "below_min_notional"

    def test_min_notional_constant_is_10(self) -> None:
        """MIN_NOTIONAL_USD is exactly Decimal('10')."""
        assert Decimal("10") == MIN_NOTIONAL_USD


# ---------------------------------------------------------------------------
# 8. Long-only flatten
# ---------------------------------------------------------------------------


class TestLongOnlyFlatten:
    def test_short_position_goes_in_flatten(self) -> None:
        """A coin with szi < 0 is added to flatten list."""
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.0, "ETH": -5.0},  # ETH is short
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        state = _state(last_target=None)

        sp = plan(ta, snap, state, _cfg())

        assert "ETH" in sp.flatten

    def test_long_position_not_in_flatten(self) -> None:
        """A coin with szi > 0 is NOT added to flatten."""
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.0},
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target=None)

        sp = plan(ta, snap, state, _cfg())
        assert "BTC" not in sp.flatten


# ---------------------------------------------------------------------------
# 9. Unpriceable coin
# ---------------------------------------------------------------------------


class TestUnpriceable:
    def test_unpriceable_target_coin_skipped(self) -> None:
        """A target coin missing from marks is in skipped with reason 'unpriceable'."""
        ta = _target({"BTC": 0.5, "GHOST": 0.5})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0},  # GHOST has no mark
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target=None)

        sp = plan(ta, snap, state, _cfg())

        assert sp.rebalance is True
        ghost_orders = [o for o in sp.orders if o.coin == "GHOST"]
        assert ghost_orders == []
        ghost_skipped = [s for s in sp.skipped if s["coin"] == "GHOST"]
        assert len(ghost_skipped) == 1
        assert ghost_skipped[0]["reason"] == "unpriceable"

    def test_unpriceable_does_not_crash(self) -> None:
        """Unpriceable coin in target does not raise any exception."""
        ta = _target({"MISSINGCOIN": 1.0})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={},  # no marks at all
            sz_decimals={},
        )
        state = _state(last_target=None)

        # Must not raise
        sp = plan(ta, snap, state, _cfg())
        assert sp.rebalance is True
        assert sp.orders == []


# ---------------------------------------------------------------------------
# 10. universe_perp_map translation
# ---------------------------------------------------------------------------


class TestUniversePerpMap:
    def test_perp_map_respected_for_price_lookup(self) -> None:
        """Ticker 'BTC' mapped to perp 'BTC-PERP'; price/pos lookups use 'BTC-PERP'."""
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC-PERP": 50_000.0},   # key is the PERP name
            sz_decimals={"BTC-PERP": 3},
        )
        state = _state(last_target=None)
        cfg = _cfg(universe_perp_map={"BTC": "BTC-PERP"}, safety_buffer=0.0)

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        # Order should use the perp name
        order_coins = {o.coin for o in sp.orders}
        assert "BTC-PERP" in order_coins
        assert "BTC" not in order_coins

    def test_perp_map_identity_fallback(self) -> None:
        """When a ticker has no entry in perp_map, it is used as-is."""
        ta = _target({"SOL": 0.5})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"SOL": 150.0},
            sz_decimals={"SOL": 1},
        )
        state = _state(last_target=None)
        cfg = _cfg(universe_perp_map={})  # no mapping

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        order_coins = {o.coin for o in sp.orders}
        assert "SOL" in order_coins

    def test_perp_map_does_not_affect_ticker_space_ratchet(self) -> None:
        """Ratchet comparison uses ticker space; perp_map only affects sizing."""
        # prev target in ticker space: BTC=0.50
        # new target in ticker space: BTC=0.54 → delta=0.04 ≥ 0.03
        ta = _target({"BTC": 0.54})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC-PERP": 50_000.0},
            sz_decimals={"BTC-PERP": 3},
        )
        state = _state(last_target={"BTC": 0.50})
        cfg = _cfg(
            universe_perp_map={"BTC": "BTC-PERP"},
            threshold_pct=3.0,
        )

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        assert "max_delta" in sp.reason


# ---------------------------------------------------------------------------
# 11. All-or-nothing: HOLD means zero orders
# ---------------------------------------------------------------------------


class TestAllOrNothing:
    def test_hold_yields_exactly_zero_orders(self) -> None:
        """When rebalance=False, orders list is empty (all-or-nothing spec).

        Deterministic by construction (F6: no self-skip): prev==target so
        max_delta=0, and achieved exactly equals target so abs_drift=0 < backstop.
            achieved BTC = szi*mark/equity = 1.0*50000/100000 = 0.5 == target 0.5
        Therefore rebalance MUST be False — the test can never pass vacuously.
        """
        weights = {"BTC": 0.5}
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.0},
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target={"BTC": 0.5})

        sp = plan(_target(weights), snap, state, _cfg(threshold_pct=3.0))

        # Assert the HOLD precondition explicitly (no skip-on-rebalance escape).
        assert sp.rebalance is False, f"expected HOLD, got rebalance ({sp.reason})"
        assert sp.orders == []
        assert sp.flatten == []
        assert sp.target_sizes == {}

    def test_orders_sorted_by_notional_descending(self) -> None:
        """orders are sorted by abs(delta_notional) descending (largest first)."""
        # BTC: delta_notional ≈ large; ETH: delta_notional ≈ small
        ta = _target({"BTC": 0.7, "ETH": 0.3})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.0)

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        assert len(sp.orders) == 2
        # BTC delta_notional > ETH delta_notional
        assert abs(sp.orders[0].delta_notional) >= abs(sp.orders[1].delta_notional)
        assert sp.orders[0].coin == "BTC"


# ---------------------------------------------------------------------------
# F1 — scale-down / single-lot must never silently zero a wanted coin
# ---------------------------------------------------------------------------


class TestZeroedByCap:
    def test_single_lot_exceeds_cap_records_zeroed_by_cap(self) -> None:
        """Repro from the review: E=100k, single coin w=1.0, px=98000, szd=0.

        raw_size = floor(100000/98000, 0) = floor(1.0204) = 1.0  → notional 98000
        cap = 100000 * (1 - 0.03) = 97000 → gross(98000) > cap → scale-down
        scale = 97000/98000 = 0.98979...
        scaled_notional = 98000 * 0.98979 = 97000 → size = floor(97000/98000) = 0
        The wanted coin floors to 0 → MUST appear in skipped as 'zeroed_by_cap',
        NOT silently vanish.
        """
        ta = _target({"BTC": 1.0})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 98_000.0},
            sz_decimals={"BTC": 0},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        # No order placed (size floored to 0) ...
        assert [o for o in sp.orders if o.coin == "BTC"] == []
        # ... but it MUST be recorded, not silent.
        zeroed = [s for s in sp.skipped if s["reason"] == "zeroed_by_cap"]
        assert len(zeroed) == 1
        assert zeroed[0]["coin"] == "BTC"
        # intended notional recorded = w * E = 100000
        assert abs(zeroed[0]["delta_notional"] - 100_000.0) < 1e-6

    def test_scale_down_zeroes_small_coin_records_it(self) -> None:
        """Scale-down floors a small coin to 0 → recorded as zeroed_by_cap.

        Big coin BTC w=0.95 px=50000 szd=3 → raw notional 95000.
        Small coin SMALL w=0.05 px=9000 szd=0 → raw_size floor(0.05*100000/9000,0)
            = floor(0.5556) = 0 → already 0 BEFORE scale even.  intended>0 → recorded.
        """
        ta = _target({"BTC": 0.95, "SMALL": 0.05})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "SMALL": 9_000.0},
            sz_decimals={"BTC": 3, "SMALL": 0},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)

        assert sp.rebalance is True
        small_zeroed = [
            s for s in sp.skipped
            if s["coin"] == "SMALL" and s["reason"] == "zeroed_by_cap"
        ]
        assert len(small_zeroed) == 1
        # SMALL never appears as an order
        assert [o for o in sp.orders if o.coin == "SMALL"] == []

    def test_zeroed_coin_not_double_recorded(self) -> None:
        """A coin zeroed by cap (no position) is NOT also tagged below_min_notional."""
        ta = _target({"BTC": 1.0})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 98_000.0},
            sz_decimals={"BTC": 0},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)

        btc_entries = [s for s in sp.skipped if s["coin"] == "BTC"]
        assert len(btc_entries) == 1, f"expected exactly 1 skip entry, got {btc_entries}"
        assert btc_entries[0]["reason"] == "zeroed_by_cap"


# ---------------------------------------------------------------------------
# F3 — abs-drift backstop must see stray held coins absent from both targets
# ---------------------------------------------------------------------------


class TestStrayHoldingBackstop:
    def test_stray_held_coin_fires_backstop_and_plans_sell(self) -> None:
        """prev==target={BTC:0.5}; on-chain holds BTC + stray ETH at 0.9 weight.

        max_delta = 0 (target unchanged).  Stray ETH is in NEITHER target.
        Without F3, ETH is never drift-checked → rebalance=False → ETH stranded.
        With F3, achieved is in the drift union → ETH drift 0.9 ≥ backstop 0.06 →
        rebalance fires → Step 7 plans the sell-to-zero for ETH.
            ETH achieved = 45.0 * 2000 / 100000 = 0.90
        """
        weights = {"BTC": 0.5}
        # BTC achieved exactly on target so it alone wouldn't fire; ETH is stray.
        # BTC szi = 0.5*100000/50000 = 1.0 ; ETH szi = 0.9*100000/2000 = 45.0
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.0, "ETH": 45.0},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        state = _state(last_target={"BTC": 0.5})
        cfg = _cfg(threshold_pct=3.0)

        sp = plan(_target(weights), snap, state, cfg)

        assert sp.rebalance is True
        assert "backstop" in sp.reason
        # Step 7 must plan a SELL of the stray ETH down to zero.
        eth_orders = [o for o in sp.orders if o.coin == "ETH"]
        assert len(eth_orders) == 1
        assert eth_orders[0].side == "sell"
        assert eth_orders[0].target_size == 0.0
        # delta_size = 0 - 45.0 = -45.0
        assert abs(eth_orders[0].delta_size - (-45.0)) < 1e-9

    def test_no_stray_no_spurious_rebalance(self) -> None:
        """Sanity: when achieved exactly matches target and prev==target, HOLD.

        Confirms the F3 union addition does NOT cause a spurious rebalance when
        there is no stray holding (achieved keys ⊆ target keys, drift 0).
        """
        weights = {"BTC": 0.5}
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.0},   # achieved 0.5 == target
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 4},
        )
        state = _state(last_target={"BTC": 0.5})
        sp = plan(_target(weights), snap, state, _cfg(threshold_pct=3.0))
        assert sp.rebalance is False

    def test_stray_coin_with_perp_map_fires_backstop(self) -> None:
        """F3 + perp_map: stray held PERP not in target still drift-checked.

        target ticker BTC → perp BTC-PERP.  On-chain also holds SOL-PERP (stray,
        not in target).  achieved is perp-keyed, so the union picks up SOL-PERP
        and the backstop fires.
        """
        weights = {"BTC": 0.5}
        snap = _snap(
            equity=100_000,
            positions={"BTC-PERP": 1.0, "SOL-PERP": 300.0},
            marks={"BTC-PERP": 50_000.0, "SOL-PERP": 150.0},
            sz_decimals={"BTC-PERP": 4, "SOL-PERP": 1},
        )
        state = _state(last_target={"BTC": 0.5})
        cfg = _cfg(universe_perp_map={"BTC": "BTC-PERP"}, threshold_pct=3.0)

        sp = plan(_target(weights), snap, state, cfg)

        # SOL-PERP achieved = 300 * 150 / 100000 = 0.45 ≥ backstop 0.06
        assert sp.rebalance is True
        assert "backstop" in sp.reason
        sol_orders = [o for o in sp.orders if o.coin == "SOL-PERP"]
        assert len(sol_orders) == 1
        assert sol_orders[0].side == "sell"


# ---------------------------------------------------------------------------
# F5 — money invariant is a hard guard (raise), not a strippable assert
# ---------------------------------------------------------------------------


class TestMoneyInvariantHardGuard:
    def test_sizer_error_is_distinct_exception(self) -> None:
        """SizerError is a dedicated Exception subclass (importable)."""
        assert issubclass(SizerError, Exception)

    def test_invariant_holds_under_normal_sizing(self) -> None:
        """Normal sizing never raises SizerError; post_gross ≤ cap (relative tol)."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 4, "ETH": 2},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)  # must not raise
        cap = 100_000 * (1 - 0.03)
        assert sp.gross_notional <= cap * (1 + 1e-9)


class TestAllCashTransition:
    """Regression: all-cash → all-cash must HOLD, never crash on an empty max().

    Found by the golden execution-fidelity sim (2026-06-03): when HAARP is fully
    risk-off (weights={}) for a 2nd consecutive day, prev=={} and new target=={},
    so the ticker union is empty and a bare ``max()`` raised ValueError — the live
    daemon would crash. The fix is ``default=0.0`` in the max_delta computation.
    """

    def test_all_cash_to_all_cash_holds_no_crash(self) -> None:
        ta = _target({}, cash=1.0)
        snap = _snap(equity=10_000, marks={"BTC": 60_000.0}, sz_decimals={"BTC": 5})
        state = _state(last_target={})  # previous frozen target is also all-cash

        sp = plan(ta, snap, state, _cfg())  # must NOT raise

        assert sp.rebalance is False
        assert sp.orders == []

    def test_all_cash_to_allocation_rebalances(self) -> None:
        """all-cash prev → a real target still triggers a rebalance (max_delta > 0)."""
        ta = _target({"BTC": 0.5}, cash=0.5)
        snap = _snap(equity=10_000, marks={"BTC": 60_000.0}, sz_decimals={"BTC": 5})
        state = _state(last_target={})

        sp = plan(ta, snap, state, _cfg())

        assert sp.rebalance is True

    def test_allocation_to_all_cash_rebalances_to_flat(self) -> None:
        """A held book → all-cash target must rebalance (exit), not be stranded."""
        ta = _target({}, cash=1.0)
        snap = _snap(
            equity=10_000,
            positions={"BTC": 0.08},
            marks={"BTC": 60_000.0},
            sz_decimals={"BTC": 5},
        )
        state = _state(last_target={"BTC": 0.5})

        sp = plan(ta, snap, state, _cfg())

        assert sp.rebalance is True
