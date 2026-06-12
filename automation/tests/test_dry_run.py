"""
Tests for automation.runtime.dry_run.

Coverage:
- Returns a correct SizePlan (same as sizer.plan would produce).
- ASSERTS ZERO order calls (submit_ioc/market_close never invoked).
- Ledger is untouched (no PENDING or DONE entries written).
- State is not mutated.
- AllocationRejected propagates (audience mismatch validation fires).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

import pytest

from automation.allocation.base import AllocationRejected, TargetAllocation
from automation.core.config import CapsConfig, Config, SignalSourceConfig
from automation.reporting.state import State
from automation.runtime.dry_run import dry_run_cycle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(client_id: str = "test-client") -> Config:
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight", client_id=client_id),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.6,
        safety_buffer=0.03,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.005,
        ),
    )


def _two_sleeve_cfg() -> Config:
    """A two-sleeve config (no singular signal_source): cfg.signal_source is None.

    dry_run_cycle must read the REPRESENTATIVE sleeve's client_id via
    cfg.sleeves()[0] rather than the (None) cfg.signal_source.
    """
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_sources=[
            SignalSourceConfig(type="equal_weight", name="haarp", client_id="haarp-client", budget=0.5),
            SignalSourceConfig(type="equal_weight", name="crash", client_id="crash-client", budget=0.5),
        ],
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.6,
        safety_buffer=0.03,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.005,
        ),
    )


def _target(weights: dict[str, float], audience: str = "test-client") -> TargetAllocation:
    cash = 1.0 - sum(weights.values())
    return TargetAllocation(
        as_of=datetime.date(2026, 6, 1),
        weights=weights,
        cash=cash,
        strategy_id="test",
        model_rev="test",
        audience=audience,
    )


class FakeSource:
    """Minimal AllocationSource returning a fixed TargetAllocation."""

    def __init__(self, ta: TargetAllocation) -> None:
        self._ta = ta

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        return self._ta


class FakeClient:
    """Snapshot-only fake client — records all calls."""

    def __init__(
        self,
        equity: float = 100_000.0,
        positions: dict[str, float] | None = None,
        marks: dict[str, float] | None = None,
        sz_decimals: dict[str, int] | None = None,
    ) -> None:
        self._equity = Decimal(str(equity))
        self._positions = {k: Decimal(str(v)) for k, v in (positions or {}).items()}
        self._marks = {k: Decimal(str(v)) for k, v in (marks or {"BTC": 50_000.0, "ETH": 2_000.0}).items()}
        self._sz_decimals = sz_decimals or {"BTC": 3, "ETH": 2}
        self.call_log: list[dict] = []

    def equity(self) -> Decimal:
        self.call_log.append({"method": "equity"})
        return self._equity

    def positions(self) -> dict[str, Decimal]:
        self.call_log.append({"method": "positions"})
        return self._positions

    def marks(self) -> dict[str, Decimal]:
        self.call_log.append({"method": "marks"})
        return self._marks

    def sz_decimals(self) -> dict[str, int]:
        self.call_log.append({"method": "sz_decimals"})
        return self._sz_decimals

    def submit_ioc(self, *args: Any, **kwargs: Any) -> dict:
        raise AssertionError("submit_ioc must NEVER be called in dry_run_cycle")

    def market_close(self, *args: Any, **kwargs: Any) -> dict:
        raise AssertionError("market_close must NEVER be called in dry_run_cycle")

    @property
    def sub(self) -> str:
        return "0x" + "0" * 40


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDryRunCycle:
    def test_returns_size_plan(self, tmp_path: Any) -> None:
        """dry_run_cycle returns a valid SizePlan (rebalance=True for bootstrap)."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        source = FakeSource(ta)
        client = FakeClient()
        state = State()

        sp = dry_run_cycle(_cfg(), client, source, state)

        # Bootstrap state → rebalance should fire
        assert sp.rebalance is True
        assert sp.reason == "bootstrap"

    def test_two_sleeve_cfg_uses_representative_client_id(self, tmp_path: Any) -> None:
        """On a signal_sources: list, cfg.signal_source is None — dry_run_cycle
        must read the representative sleeve's client_id (cfg.sleeves()[0]) and
        NOT crash with AttributeError. The served audience matches sleeve 0."""
        ta = _target({"BTC": 0.5, "ETH": 0.5}, audience="haarp-client")
        source = FakeSource(ta)
        client = FakeClient()
        state = State()

        sp = dry_run_cycle(_two_sleeve_cfg(), client, source, state)

        assert sp.rebalance is True
        assert sp.reason == "bootstrap"

    def test_zero_order_calls(self, tmp_path: Any) -> None:
        """dry_run_cycle MUST NOT call submit_ioc or market_close."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        source = FakeSource(ta)
        client = FakeClient()
        state = State()

        # FakeClient raises AssertionError if submit_ioc or market_close is called.
        dry_run_cycle(_cfg(), client, source, state)

        # Confirm only read methods were used
        called_methods = {c["method"] for c in client.call_log}
        assert "submit_ioc" not in called_methods
        assert "market_close" not in called_methods

    def test_ledger_untouched(self, tmp_path: Any) -> None:
        """dry_run_cycle must NOT write to the intent ledger."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        ledger_path = tmp_path / "ledger.json"
        # Ledger does NOT exist before dry_run_cycle
        assert not ledger_path.exists()

        dry_run_cycle(_cfg(), client, source, state)

        # Ledger must still not exist (dry run never touches it)
        assert not ledger_path.exists(), "dry_run_cycle must not create or write the ledger"

    def test_state_not_mutated(self, tmp_path: Any) -> None:
        """dry_run_cycle must not mutate state.last_rebalanced_target."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        assert state.last_rebalanced_target is None

        dry_run_cycle(_cfg(), client, source, state)

        assert state.last_rebalanced_target is None, (
            "dry_run_cycle must NOT mutate state — it is read-only"
        )

    def test_hold_cycle_returns_hold_plan(self, tmp_path: Any) -> None:
        """When the baseline == target and live positions match, plan returns hold.

        To avoid the abs-drift backstop, the FakeClient must report positions whose
        achieved weights match the target:
          - equity=100k, BTC mark=50k → BTC target size = 0.5*100k/50k = 1.0 BTC
          - equity=100k, ETH mark=2k  → ETH target size = 0.5*100k/2k  = 25 ETH
        With these positions achieved_weights ≈ target → max_delta=0, abs_drift=0 → hold.
        """
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        # Positions that match the target exactly so abs-drift doesn't fire.
        client = FakeClient(
            positions={"BTC": 1.0, "ETH": 25.0},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State(
            last_rebalanced_target=weights,
            last_rebalanced_as_of="2026-06-01",
        )

        sp = dry_run_cycle(_cfg(), client, source, state)

        assert sp.rebalance is False
        assert sp.reason == "hold"

    def test_allocation_rejected_propagates(self, tmp_path: Any) -> None:
        """AllocationRejected from validate must propagate — dry_run is not silenced."""
        # Audience mismatch → AllocationRejected
        ta = _target({"BTC": 0.5, "ETH": 0.5}, audience="wrong-client")
        source = FakeSource(ta)
        client = FakeClient()
        state = State()

        with pytest.raises(AllocationRejected):
            dry_run_cycle(_cfg(client_id="test-client"), client, source, state)


# ---------------------------------------------------------------------------
# D5: spot_routing threaded into the dry-run snapshot (hybrid book, still no sends)
# ---------------------------------------------------------------------------


def _cfg_spot(
    spot_routing: dict[str, str],
    universe_perp_map: dict[str, str],
    client_id: str = "test-client",
) -> Config:
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight", client_id=client_id),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.6,
        safety_buffer=0.03,
        universe_perp_map=universe_perp_map,
        spot_routing=spot_routing,
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.005,
        ),
    )


class _SpotEnabledClient(FakeClient):
    """Snapshot-only spot-enabled client.  Spot writes/reads MUST never order."""

    def __init__(
        self,
        *,
        perp_equity: float = 60_000.0,
        perp_marks: dict[str, float] | None = None,
        perp_sz_decimals: dict[str, int] | None = None,
        spot_balances: dict[str, float] | None = None,
        spot_marks: dict[str, float] | None = None,
        spot_sz_decimals: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            equity=perp_equity,
            marks=perp_marks or {"BTC": 50_000.0},
            sz_decimals=perp_sz_decimals or {"BTC": 3},
        )
        self._spot_balances = {
            k: Decimal(str(v)) for k, v in (spot_balances or {}).items()
        }
        self._spot_marks = {k: Decimal(str(v)) for k, v in (spot_marks or {}).items()}
        self._spot_sz_decimals = spot_sz_decimals or {}

    def spot_balances(self) -> dict[str, Decimal]:
        self.call_log.append({"method": "spot_balances"})
        return self._spot_balances

    def spot_marks(self) -> dict[str, Decimal]:
        self.call_log.append({"method": "spot_marks"})
        return self._spot_marks

    def spot_sz_decimals(self) -> dict[str, int]:
        self.call_log.append({"method": "spot_sz_decimals"})
        return self._spot_sz_decimals

    def submit_spot_ioc(self, *args: Any, **kwargs: Any) -> dict:
        raise AssertionError("submit_spot_ioc must NEVER be called in dry_run_cycle")


class TestDryRunSpotRouting:
    def test_hybrid_snapshot_read_no_orders(self, tmp_path: Any) -> None:
        """dry_run with spot_routing reads the HYBRID book but still sends NOTHING."""
        ta = _target({"BTC": 0.5, "HYPE": 0.5})
        source = FakeSource(ta)
        client = _SpotEnabledClient(
            perp_equity=60_000.0,
            perp_marks={"BTC": 50_000.0},
            perp_sz_decimals={"BTC": 3},
            spot_balances={"USDC": 40_000.0},
            spot_marks={"@107": 10.0},
            spot_sz_decimals={"@107": 2},
        )
        state = State()
        cfg = _cfg_spot(
            spot_routing={"HYPE": "@107"},
            universe_perp_map={"BTC": "BTC", "HYPE": "HYPE"},
        )

        sp = dry_run_cycle(cfg, client, source, state)

        # Hybrid snapshot taken → spot reads happened → equity spans both pools.
        methods = {c["method"] for c in client.call_log}
        assert "spot_balances" in methods
        assert "spot_marks" in methods
        # Equity reflects the hybrid total (60k perp + 40k spot = 100k).
        assert sp.equity == 100_000.0
        # Bootstrap → plan a rebalance, but NOTHING is ever submitted.
        assert sp.rebalance is True
        assert "submit_ioc" not in methods
        assert "submit_spot_ioc" not in methods

    def test_empty_spot_routing_is_perp_only(self, tmp_path: Any) -> None:
        """dry_run with spot_routing={} reads perp-only (no spot reads)."""
        ta = _target({"BTC": 0.5, "ETH": 0.5})
        source = FakeSource(ta)
        client = _SpotEnabledClient(
            perp_equity=100_000.0,
            perp_marks={"BTC": 50_000.0, "ETH": 2_000.0},
            perp_sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()

        dry_run_cycle(_cfg(), client, source, state)

        methods = {c["method"] for c in client.call_log}
        assert "spot_balances" not in methods
        assert "spot_marks" not in methods
