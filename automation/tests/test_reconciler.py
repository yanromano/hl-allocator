"""
Tests for automation.execution.reconciler — on-chain snapshot and achieved weights.

Coverage:
- snapshot() packs all four client method results correctly
- achieved_weights() math hand-verified
- negative szi → negative weight
- equity=0 → empty dict (no ZeroDivisionError)
- coin with position but no mark → skipped (warning, no crash)
- multiple positions
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from automation.execution.reconciler import Snapshot, achieved_weights, snapshot, sweep_open_orders
from automation.reporting.alerts import Alert, AlertType, Severity

# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class FakeClient:
    """Minimal duck-typed stand-in for HLClient."""

    def __init__(
        self,
        equity: Decimal,
        positions: dict[str, Decimal],
        marks: dict[str, Decimal],
        sz_decimals: dict[str, int],
    ) -> None:
        self._equity = equity
        self._positions = positions
        self._marks = marks
        self._sz_decimals = sz_decimals

    def equity(self) -> Decimal:
        return self._equity

    def positions(self) -> dict[str, Decimal]:
        return self._positions

    def marks(self) -> dict[str, Decimal]:
        return self._marks

    def sz_decimals(self) -> dict[str, int]:
        return self._sz_decimals


class FakeHybridClient:
    """Spot-capable duck-typed stand-in: BOTH perp and spot reads.

    Perp reads mirror :class:`FakeClient`.  Spot reads mirror the D1 HLClient
    spot methods: ``spot_balances`` keyed by coin (incl. a ``USDC`` free-cash
    entry), ``spot_marks`` and ``spot_sz_decimals`` keyed by spot PAIR name.

    If constructed with ``spot_enabled=False``, the three spot reads raise
    ``RuntimeError`` exactly like an HLClient that was not opted into spot —
    used to prove the snapshot does NOT silently fall back to perp-only.
    """

    def __init__(
        self,
        equity: Decimal,
        positions: dict[str, Decimal],
        marks: dict[str, Decimal],
        sz_decimals: dict[str, int],
        spot_balances: dict[str, Decimal],
        spot_marks: dict[str, Decimal],
        spot_sz_decimals: dict[str, int],
        spot_enabled: bool = True,
    ) -> None:
        self._equity = equity
        self._positions = positions
        self._marks = marks
        self._sz_decimals = sz_decimals
        self._spot_balances = spot_balances
        self._spot_marks = spot_marks
        self._spot_sz_decimals = spot_sz_decimals
        self._spot_enabled = spot_enabled

    # Perp reads
    def equity(self) -> Decimal:
        return self._equity

    def positions(self) -> dict[str, Decimal]:
        return self._positions

    def marks(self) -> dict[str, Decimal]:
        return self._marks

    def sz_decimals(self) -> dict[str, int]:
        return self._sz_decimals

    # Spot reads (raise unless spot-enabled, matching D1 HLClient)
    def _require_spot(self) -> None:
        if not self._spot_enabled:
            raise RuntimeError("spot reads require enable_spot=True")

    def spot_balances(self) -> dict[str, Decimal]:
        self._require_spot()
        return self._spot_balances

    def spot_marks(self) -> dict[str, Decimal]:
        self._require_spot()
        return self._spot_marks

    def spot_sz_decimals(self) -> dict[str, int]:
        self._require_spot()
        return self._spot_sz_decimals


# ---------------------------------------------------------------------------
# Tests: snapshot()
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_packs_correctly(self) -> None:
        """snapshot() calls all four methods and packs them into Snapshot."""
        client = FakeClient(
            equity=Decimal("100000"),
            positions={"BTC": Decimal("0.5")},
            marks={"BTC": Decimal("60000"), "ETH": Decimal("2000")},
            sz_decimals={"BTC": 5, "ETH": 4},
        )
        snap = snapshot(client)

        assert snap.equity == Decimal("100000")
        assert snap.positions == {"BTC": Decimal("0.5")}
        assert snap.marks["BTC"] == Decimal("60000")
        assert snap.marks["ETH"] == Decimal("2000")
        assert snap.sz_decimals["BTC"] == 5
        assert snap.sz_decimals["ETH"] == 4

    def test_snapshot_empty_positions(self) -> None:
        """snapshot() works with no open positions."""
        client = FakeClient(
            equity=Decimal("50000"),
            positions={},
            marks={"BTC": Decimal("70000")},
            sz_decimals={"BTC": 5},
        )
        snap = snapshot(client)
        assert snap.equity == Decimal("50000")
        assert snap.positions == {}

    def test_snapshot_is_frozen(self) -> None:
        """Snapshot is a frozen dataclass — mutation raises."""
        client = FakeClient(
            equity=Decimal("10000"),
            positions={},
            marks={},
            sz_decimals={},
        )
        snap = snapshot(client)
        import dataclasses

        assert dataclasses.is_dataclass(snap)
        # Confirm frozen (attempt setattr should raise FrozenInstanceError)
        try:
            snap.equity = Decimal("0")  # type: ignore[misc]
            raised = False
        except Exception:
            raised = True
        assert raised, "Snapshot should be frozen"


# ---------------------------------------------------------------------------
# Tests: achieved_weights()
# ---------------------------------------------------------------------------


class TestAchievedWeights:
    def test_basic_weight_computation(self) -> None:
        """Hand-verified: szi=0.5, mark=60000, equity=60000 → weight=0.5."""
        snap = Snapshot(
            equity=Decimal("60000"),
            positions={"BTC": Decimal("0.5")},
            marks={"BTC": Decimal("60000")},
            sz_decimals={},
        )
        weights = achieved_weights(snap)
        assert abs(weights["BTC"] - 0.5) < 1e-9

    def test_negative_szi_gives_negative_weight(self) -> None:
        """Short positions (szi < 0) produce negative weights."""
        snap = Snapshot(
            equity=Decimal("10000"),
            positions={"ETH": Decimal("-2.0")},
            marks={"ETH": Decimal("2000")},
            sz_decimals={},
        )
        weights = achieved_weights(snap)
        # weight = -2.0 * 2000 / 10000 = -0.4
        assert abs(weights["ETH"] - (-0.4)) < 1e-9

    def test_equity_zero_returns_empty(self) -> None:
        """equity=0 returns empty dict without raising ZeroDivisionError."""
        snap = Snapshot(
            equity=Decimal("0"),
            positions={"BTC": Decimal("1.0")},
            marks={"BTC": Decimal("50000")},
            sz_decimals={},
        )
        weights = achieved_weights(snap)
        assert weights == {}

    def test_missing_mark_skips_coin(self) -> None:
        """A coin with a position but no mark is skipped (no crash)."""
        snap = Snapshot(
            equity=Decimal("100000"),
            positions={"BTC": Decimal("0.5"), "UNKNOWNCOIN": Decimal("100.0")},
            marks={"BTC": Decimal("60000")},  # UNKNOWNCOIN has no mark
            sz_decimals={},
        )
        weights = achieved_weights(snap)
        # UNKNOWNCOIN skipped; BTC computed correctly
        assert "UNKNOWNCOIN" not in weights
        assert abs(weights["BTC"] - (0.5 * 60000 / 100000)) < 1e-9

    def test_multiple_positions(self) -> None:
        """Multiple positions computed correctly and independently."""
        # BTC: 0.5 * 60000 / 100000 = 0.30
        # ETH: 10.0 * 2000 / 100000 = 0.20
        snap = Snapshot(
            equity=Decimal("100000"),
            positions={
                "BTC": Decimal("0.5"),
                "ETH": Decimal("10.0"),
            },
            marks={
                "BTC": Decimal("60000"),
                "ETH": Decimal("2000"),
            },
            sz_decimals={},
        )
        weights = achieved_weights(snap)
        assert abs(weights["BTC"] - 0.30) < 1e-9
        assert abs(weights["ETH"] - 0.20) < 1e-9

    def test_no_positions_returns_empty(self) -> None:
        """Empty positions dict → empty weights."""
        snap = Snapshot(
            equity=Decimal("50000"),
            positions={},
            marks={"BTC": Decimal("70000")},
            sz_decimals={},
        )
        weights = achieved_weights(snap)
        assert weights == {}


# ---------------------------------------------------------------------------
# Tests: snapshot() hybrid spot+perp venue routing (D2 — hole #2 fix)
# ---------------------------------------------------------------------------


class TestHybridSnapshot:
    """D2: snapshot() must span both pools when spot_routing is given."""

    def _hybrid_client(self, spot_enabled: bool = True) -> FakeHybridClient:
        # Perp pool: ETH + BTC held as perp positions.
        # Spot pool: HYPE held (30 tokens) + 2000 free USDC. HYPE pair = @107.
        return FakeHybridClient(
            equity=Decimal("100000"),  # perp accountValue
            positions={"ETH": Decimal("5.0"), "BTC": Decimal("0.5")},
            marks={"ETH": Decimal("2000"), "BTC": Decimal("60000")},
            sz_decimals={"ETH": 4, "BTC": 5},
            spot_balances={"HYPE": Decimal("30"), "USDC": Decimal("2000")},
            spot_marks={"@107": Decimal("35")},  # HYPE/USDC spot mark
            spot_sz_decimals={"@107": 2},
            spot_enabled=spot_enabled,
        )

    def test_spot_routing_none_identical_to_perp_only(self) -> None:
        """spot_routing=None ⇒ byte-identical to today's pure-perp snapshot.

        Regression: the spot reads must NOT be consulted; positions/marks/equity
        match the perp pool exactly.
        """
        client = self._hybrid_client()
        snap = snapshot(client, spot_routing=None)

        assert snap.equity == Decimal("100000")  # perp only, NO spot pool added
        assert snap.positions == {"ETH": Decimal("5.0"), "BTC": Decimal("0.5")}
        assert snap.marks == {"ETH": Decimal("2000"), "BTC": Decimal("60000")}
        assert snap.sz_decimals == {"ETH": 4, "BTC": 5}
        # HYPE (spot) must be absent — spot endpoints were never read.
        assert "HYPE" not in snap.positions
        assert "HYPE" not in snap.marks

    def test_spot_routing_empty_dict_identical_to_perp_only(self) -> None:
        """spot_routing={} is the same no-op as None."""
        client = self._hybrid_client()
        snap_none = snapshot(client, spot_routing=None)
        snap_empty = snapshot(client, spot_routing={})
        assert snap_empty.equity == snap_none.equity == Decimal("100000")
        assert snap_empty.positions == snap_none.positions
        assert snap_empty.marks == snap_none.marks
        assert snap_empty.sz_decimals == snap_none.sz_decimals

    def test_spot_routing_unused_when_spot_disabled_and_none(self) -> None:
        """A spot-disabled client with spot_routing=None must NOT raise.

        Proves the pure-perp path never touches spot reads (which would raise).
        """
        client = self._hybrid_client(spot_enabled=False)
        snap = snapshot(client, spot_routing=None)
        assert snap.equity == Decimal("100000")

    def test_hybrid_combined_snapshot_and_equity(self) -> None:
        """spot_routing={"HYPE":"@107"} ⇒ hybrid combined snapshot.

        Hand-computed:
          perp accountValue       = 100000
          free spot USDC          =   2000
          HYPE holding value      = 30 * 35 = 1050
          ----------------------------------------
          total equity            = 103050

        HYPE position = spot balance (30), HYPE mark = spot mark via @107 (35).
        Perp coins (ETH/BTC) come straight from the perp reads.
        """
        client = self._hybrid_client()
        snap = snapshot(client, spot_routing={"HYPE": "@107"})

        # Total equity spans both pools.
        assert snap.equity == Decimal("103050")

        # HYPE is now a (spot) position = owned token qty.
        assert snap.positions["HYPE"] == Decimal("30")
        # HYPE mark resolved via the @107 spot pair.
        assert snap.marks["HYPE"] == Decimal("35")
        # HYPE sz_decimals resolved via the @107 spot pair.
        assert snap.sz_decimals["HYPE"] == 2

        # Perp coins unchanged, keyed by coin ticker (same shape as before).
        assert snap.positions["ETH"] == Decimal("5.0")
        assert snap.positions["BTC"] == Decimal("0.5")
        assert snap.marks["ETH"] == Decimal("2000")
        assert snap.marks["BTC"] == Decimal("60000")
        assert snap.sz_decimals["ETH"] == 4
        assert snap.sz_decimals["BTC"] == 5

    def test_hybrid_achieved_weights_spot_coin_nonnegative(self) -> None:
        """achieved_weights on the hybrid snapshot yields the right spot weight.

        HYPE achieved weight = 30 * 35 / 103050 = 1050 / 103050 ≈ 0.0101892...
        and is NON-NEGATIVE (spot position >= 0 ⇒ long-only weight).
        """
        client = self._hybrid_client()
        snap = snapshot(client, spot_routing={"HYPE": "@107"})
        weights = achieved_weights(snap)

        expected_hype = (30.0 * 35.0) / 103050.0
        assert abs(weights["HYPE"] - expected_hype) < 1e-12
        assert weights["HYPE"] >= 0.0  # spot is long-only ⇒ non-negative

        # Perp weights still computed against the (now larger) total equity.
        assert abs(weights["ETH"] - (5.0 * 2000.0 / 103050.0)) < 1e-12
        assert abs(weights["BTC"] - (0.5 * 60000.0 / 103050.0)) < 1e-12

    def test_hybrid_spot_coin_unpriceable_omitted(self) -> None:
        """A routed spot coin whose mark is missing is omitted (no crash).

        The HYPE pair @999 has no spot mark ⇒ HYPE has no mark in the snapshot
        and is skipped by achieved_weights. Its holding is also excluded from the
        spot pool value (cannot price it).
        """
        client = FakeHybridClient(
            equity=Decimal("100000"),
            positions={"BTC": Decimal("0.5")},
            marks={"BTC": Decimal("60000")},
            sz_decimals={"BTC": 5},
            spot_balances={"HYPE": Decimal("30"), "USDC": Decimal("2000")},
            spot_marks={},  # no mark for HYPE's pair
            spot_sz_decimals={},
            spot_enabled=True,
        )
        snap = snapshot(client, spot_routing={"HYPE": "@999"})

        # HYPE position is still recorded (balance known) but has NO mark.
        assert snap.positions["HYPE"] == Decimal("30")
        assert "HYPE" not in snap.marks
        # Equity = perp 100000 + free USDC 2000 (+ 0 unpriceable HYPE) = 102000.
        assert snap.equity == Decimal("102000")

        # achieved_weights skips the unpriceable HYPE, no crash.
        weights = achieved_weights(snap)
        assert "HYPE" not in weights
        assert abs(weights["BTC"] - (0.5 * 60000.0 / 102000.0)) < 1e-12

    def test_hybrid_spot_not_enabled_raises(self) -> None:
        """spot_routing set but client spot-disabled ⇒ snapshot RAISES.

        Must NOT silently return an understated perp-only equity (that is the
        very bug this increment closes).
        """
        client = self._hybrid_client(spot_enabled=False)
        with pytest.raises(RuntimeError, match="enable_spot"):
            snapshot(client, spot_routing={"HYPE": "@107"})

    def test_hybrid_spot_coin_flat_balance_zero_position(self) -> None:
        """A routed coin with NO spot balance ⇒ position 0, no value added."""
        client = FakeHybridClient(
            equity=Decimal("100000"),
            positions={"BTC": Decimal("0.5")},
            marks={"BTC": Decimal("60000")},
            sz_decimals={"BTC": 5},
            spot_balances={"USDC": Decimal("2000")},  # no HYPE held
            spot_marks={"@107": Decimal("35")},
            spot_sz_decimals={"@107": 2},
            spot_enabled=True,
        )
        snap = snapshot(client, spot_routing={"HYPE": "@107"})
        assert snap.positions["HYPE"] == Decimal("0")
        # equity = perp 100000 + free USDC 2000 + 0*35 = 102000
        assert snap.equity == Decimal("102000")


# ---------------------------------------------------------------------------
# Tests: sweep_open_orders()
# ---------------------------------------------------------------------------


class _FakeSweepClient:
    """Fake client for sweep_open_orders tests."""

    def __init__(self, open_orders: list[dict[str, Any]]) -> None:
        self._orders = open_orders
        self.cancel_by_cloid_calls: list[tuple[str, str]] = []
        self.cancel_oid_calls: list[tuple[str, int]] = []

    def open_orders(self) -> list[dict[str, Any]]:
        return list(self._orders)

    def cancel_by_cloid(self, coin: str, cloid: str) -> dict[str, Any]:
        self.cancel_by_cloid_calls.append((coin, cloid))
        return {}

    def cancel_oid(self, coin: str, oid: int) -> dict[str, Any]:
        self.cancel_oid_calls.append((coin, oid))
        return {}


class _CapturingAlertSink:
    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


class TestSweepOpenOrders:
    def test_empty_open_orders_returns_empty_no_alert(self) -> None:
        """No open orders → returns [], no alert emitted."""
        client = _FakeSweepClient([])
        sink = _CapturingAlertSink()

        result = sweep_open_orders(client, now_ts="2026-06-01T00:00:00Z", alert_sink=sink)

        assert result == []
        assert len(sink.received) == 0
        assert len(client.cancel_by_cloid_calls) == 0
        assert len(client.cancel_oid_calls) == 0

    def test_orphan_with_cloid_cancels_by_cloid(self) -> None:
        """An order with a cloid is cancelled via cancel_by_cloid."""
        orders = [{"coin": "BTC", "oid": 1, "sz": "0.5", "side": "B", "cloid": "0x" + "ab" * 16}]
        client = _FakeSweepClient(orders)
        sink = _CapturingAlertSink()

        result = sweep_open_orders(client, now_ts="T", alert_sink=sink)

        assert len(result) == 1
        assert len(client.cancel_by_cloid_calls) == 1
        coin, cloid = client.cancel_by_cloid_calls[0]
        assert coin == "BTC"
        assert cloid == "0x" + "ab" * 16

    def test_orphan_without_cloid_cancels_by_oid(self) -> None:
        """An order without a cloid is cancelled via cancel_oid."""
        orders = [{"coin": "ETH", "oid": 999, "sz": "1.0", "side": "B"}]
        client = _FakeSweepClient(orders)
        sink = _CapturingAlertSink()

        result = sweep_open_orders(client, now_ts="T", alert_sink=sink)

        assert len(result) == 1
        assert len(client.cancel_oid_calls) == 1
        coin, oid = client.cancel_oid_calls[0]
        assert coin == "ETH"
        assert oid == 999

    def test_emits_chain_ledger_divergence_warning(self) -> None:
        """When orphan orders are found, CHAIN_LEDGER_DIVERGENCE WARNING is emitted."""
        orders = [{"coin": "BTC", "oid": 1, "sz": "0.5", "side": "B"}]
        client = _FakeSweepClient(orders)
        sink = _CapturingAlertSink()

        sweep_open_orders(client, now_ts="T", alert_sink=sink)

        assert len(sink.received) == 1
        alert = sink.received[0]
        assert alert.type == AlertType.CHAIN_LEDGER_DIVERGENCE
        assert alert.severity == Severity.WARNING
        assert "orphan" in alert.message.lower()

    def test_multiple_orders_all_cancelled(self) -> None:
        """Multiple orphan orders are each cancelled individually."""
        orders = [
            {"coin": "BTC", "oid": 1, "sz": "0.5", "side": "B", "cloid": "0x" + "aa" * 16},
            {"coin": "ETH", "oid": 2, "sz": "1.0", "side": "B"},
        ]
        client = _FakeSweepClient(orders)
        sink = _CapturingAlertSink()

        result = sweep_open_orders(client, now_ts="T", alert_sink=sink)

        assert len(result) == 2
        assert len(client.cancel_by_cloid_calls) == 1  # BTC has cloid
        assert len(client.cancel_oid_calls) == 1        # ETH has only oid

    def test_alert_sink_none_does_not_crash(self) -> None:
        """alert_sink=None → no crash, orders still cancelled."""
        orders = [{"coin": "BTC", "oid": 1, "sz": "0.5", "side": "B"}]
        client = _FakeSweepClient(orders)

        result = sweep_open_orders(client, now_ts="T", alert_sink=None)

        assert len(result) == 1
        assert len(client.cancel_oid_calls) == 1

    def test_cancel_failure_does_not_crash(self) -> None:
        """If a cancel call raises, the sweep continues for remaining orders."""

        class _FailingCancelClient(_FakeSweepClient):
            def cancel_oid(self, coin: str, oid: int) -> dict[str, Any]:
                raise RuntimeError("exchange down")

        orders = [
            {"coin": "BTC", "oid": 1, "sz": "0.5", "side": "B"},
            {"coin": "ETH", "oid": 2, "sz": "1.0", "side": "B"},
        ]
        client = _FailingCancelClient(orders)

        # Must not raise despite cancel failures
        result = sweep_open_orders(client, now_ts="T", alert_sink=None)
        assert len(result) == 2

    def test_returns_found_orders_list(self) -> None:
        """The return value is the list of raw order dicts that were found."""
        orders = [{"coin": "SOL", "oid": 7, "sz": "10.0", "side": "B"}]
        client = _FakeSweepClient(orders)

        result = sweep_open_orders(client, now_ts="T", alert_sink=None)

        assert result == orders
