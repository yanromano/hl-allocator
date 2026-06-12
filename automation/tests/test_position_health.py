"""Tests for automation.monitoring.position_health (PH-2 core).

Exhaustive coverage of the PURE checks (breach / no-breach / boundary for each),
the ``assess`` aggregator, and the thin read-only snapshot adapter.

The checks are pure functions over a :class:`HealthSnapshot`; only the adapter
touches an HLClient.  These tests construct snapshots directly (no IO) for the
check coverage and use a ``FakeHLClient`` only for the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from automation.monitoring.position_health import (
    HealthSnapshot,
    PositionHealth,
    assess,
    check_collateral,
    check_funding,
    check_margin,
    check_venue,
    gather_health_snapshot,
)
from automation.reporting.alerts import AlertType, Severity
from automation.tests._fakes import FakeHLClient

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _pos(
    coin: str,
    *,
    position_value: float = 1000.0,
    unrealized_pnl: float = 0.0,
    liquidation_px: float | None = None,
    mark: float = 100.0,
    funding_since_open: float = 0.0,
    leverage_type: str | None = "cross",
    leverage_value: int = 3,
) -> PositionHealth:
    return PositionHealth(
        coin=coin,
        position_value=position_value,
        unrealized_pnl=unrealized_pnl,
        liquidation_px=liquidation_px,
        mark=mark,
        funding_since_open=funding_since_open,
        leverage_type=leverage_type,
        leverage_value=leverage_value,
    )


def _snap(
    *,
    equity: float = 1000.0,
    maint_margin: float = 0.0,
    positions: list[PositionHealth] | None = None,
    spot_usdc: float = 0.0,
    venue_coins: dict[str, dict] | None = None,
) -> HealthSnapshot:
    return HealthSnapshot(
        equity=equity,
        maint_margin=maint_margin,
        positions=positions or [],
        spot_usdc=spot_usdc,
        venue_coins=venue_coins if venue_coins is not None else {},
    )


# ---------------------------------------------------------------------------
# check_funding
# ---------------------------------------------------------------------------


def test_funding_over_threshold_emits_one_finding_sorted_per_coin() -> None:
    snap = _snap(
        equity=1000.0,
        positions=[
            _pos("BTC", funding_since_open=10.0),
            _pos("ETH", funding_since_open=50.0),  # the worst offender
        ],
    )
    out = check_funding(snap, drag_pct_warn=5.0)
    assert len(out) == 1
    f = out[0]
    assert f.alert_type is AlertType.FUNDING_DRAG
    assert f.severity is Severity.WARNING
    assert f.context["total_funding"] == 60.0
    assert f.context["pct_of_equity"] == 6.0
    # per_coin sorted DESC by funding — worst offender first
    assert list(f.context["per_coin"].items()) == [("ETH", 50.0), ("BTC", 10.0)]


def test_funding_under_threshold_no_finding() -> None:
    snap = _snap(equity=1000.0, positions=[_pos("BTC", funding_since_open=30.0)])
    assert check_funding(snap, drag_pct_warn=5.0) == []  # 3% < 5%


def test_funding_at_threshold_boundary_no_finding() -> None:
    # exactly 5% — check is strict `>` so the boundary does NOT breach
    snap = _snap(equity=1000.0, positions=[_pos("BTC", funding_since_open=50.0)])
    assert check_funding(snap, drag_pct_warn=5.0) == []


def test_funding_equity_non_positive_guard() -> None:
    snap = _snap(equity=0.0, positions=[_pos("BTC", funding_since_open=100.0)])
    assert check_funding(snap, drag_pct_warn=5.0) == []
    snap_neg = _snap(equity=-10.0, positions=[_pos("BTC", funding_since_open=100.0)])
    assert check_funding(snap_neg, drag_pct_warn=5.0) == []


# ---------------------------------------------------------------------------
# check_margin
# ---------------------------------------------------------------------------


def test_margin_maint_frac_over_threshold() -> None:
    snap = _snap(equity=1000.0, maint_margin=600.0)  # 0.6 > 0.5
    out = check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0)
    assert len(out) == 1
    f = out[0]
    assert f.alert_type is AlertType.MARGIN_RATIO_LOW
    assert f.severity is Severity.WARNING
    assert f.context["maint_margin"] == 600.0
    assert f.context["equity"] == 1000.0
    assert f.context["maint_frac"] == 0.6


def test_margin_maint_frac_under_threshold() -> None:
    snap = _snap(equity=1000.0, maint_margin=300.0)  # 0.3 < 0.5
    assert check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0) == []


def test_margin_liq_distance_within_warning() -> None:
    # mark 100, liq 95 -> 5% < 10% -> WARNING (not < 5% so not CRITICAL)
    snap = _snap(positions=[_pos("BTC", mark=100.0, liquidation_px=95.0)])
    out = check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0)
    assert len(out) == 1
    f = out[0]
    assert f.alert_type is AlertType.MARGIN_RATIO_LOW
    assert f.severity is Severity.WARNING
    assert f.context["coin"] == "BTC"
    assert f.context["mark"] == 100.0
    assert f.context["liquidation_px"] == 95.0
    assert f.context["distance_pct"] == 5.0


def test_margin_liq_distance_within_critical() -> None:
    # mark 100, liq 97 -> 3% < 5% (= warn/2) -> CRITICAL
    snap = _snap(positions=[_pos("BTC", mark=100.0, liquidation_px=97.0)])
    out = check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0)
    assert len(out) == 1
    assert out[0].severity is Severity.CRITICAL
    assert out[0].context["distance_pct"] == 3.0


def test_margin_liq_distance_outside_no_finding() -> None:
    # mark 100, liq 80 -> 20% > 10% -> none
    snap = _snap(positions=[_pos("BTC", mark=100.0, liquidation_px=80.0)])
    assert check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0) == []


def test_margin_liq_distance_boundary_no_finding() -> None:
    # exactly 10% — strict `<` so the boundary does NOT breach
    snap = _snap(positions=[_pos("BTC", mark=100.0, liquidation_px=90.0)])
    assert check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0) == []


def test_margin_liquidation_px_none_skipped() -> None:
    snap = _snap(positions=[_pos("BTC", mark=100.0, liquidation_px=None)])
    assert check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0) == []


def test_margin_mark_non_positive_skipped() -> None:
    snap = _snap(positions=[_pos("BTC", mark=0.0, liquidation_px=95.0)])
    assert check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0) == []


def test_margin_combines_maint_and_liq_findings() -> None:
    snap = _snap(
        equity=1000.0,
        maint_margin=600.0,
        positions=[_pos("BTC", mark=100.0, liquidation_px=97.0)],
    )
    out = check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# check_venue
# ---------------------------------------------------------------------------


def test_venue_held_coin_absent_emits_finding() -> None:
    snap = _snap(positions=[_pos("BTC")], venue_coins={})  # BTC not listed
    out = check_venue(snap)
    assert len(out) == 1
    f = out[0]
    assert f.alert_type is AlertType.COIN_VENUE_DEGRADED
    assert f.severity is Severity.WARNING
    assert f.context["coin"] == "BTC"
    assert f.context["reason"] == "absent_from_venue"


def test_venue_held_coin_delisted_emits_finding() -> None:
    snap = _snap(
        positions=[_pos("BTC")],
        venue_coins={"BTC": {"delisted": True, "only_isolated": False, "max_leverage": 50}},
    )
    out = check_venue(snap)
    assert len(out) == 1
    assert out[0].alert_type is AlertType.COIN_VENUE_DEGRADED
    assert out[0].context["reason"] == "delisted"


def test_venue_healthy_held_coin_no_finding() -> None:
    snap = _snap(
        positions=[_pos("BTC")],
        venue_coins={"BTC": {"delisted": False, "only_isolated": False, "max_leverage": 50}},
    )
    assert check_venue(snap) == []


# ---------------------------------------------------------------------------
# check_collateral
# ---------------------------------------------------------------------------


def test_collateral_over_threshold_emits_finding() -> None:
    snap = _snap(spot_usdc=5.0)
    out = check_collateral(snap, spot_usdc_warn=1.0)
    assert len(out) == 1
    f = out[0]
    assert f.alert_type is AlertType.COLLATERAL_IN_SPOT
    assert f.severity is Severity.WARNING
    assert f.context["spot_usdc"] == 5.0
    assert f.context["threshold"] == 1.0


def test_collateral_under_threshold_no_finding() -> None:
    assert check_collateral(_snap(spot_usdc=0.5), spot_usdc_warn=1.0) == []


def test_collateral_at_threshold_boundary_no_finding() -> None:
    # exactly 1.0 — strict `>` so boundary does NOT breach
    assert check_collateral(_snap(spot_usdc=1.0), spot_usdc_warn=1.0) == []


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HealthCfg:
    funding_drag_pct: float = 5.0
    margin_maint_frac_warn: float = 0.5
    liq_distance_pct_warn: float = 10.0
    spot_collateral_warn_usd: float = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class _Cfg:
    health: _HealthCfg = _HealthCfg()


def test_assess_multiple_breaches_returns_all() -> None:
    snap = _snap(
        equity=1000.0,
        maint_margin=600.0,  # margin maint breach
        positions=[
            _pos("BTC", mark=100.0, liquidation_px=97.0, funding_since_open=60.0),  # funding + liq
        ],
        spot_usdc=5.0,  # collateral breach
        venue_coins={},  # BTC absent -> venue breach
    )
    out = assess(snap, _Cfg())
    types = {f.alert_type for f in out}
    assert AlertType.FUNDING_DRAG in types
    assert AlertType.MARGIN_RATIO_LOW in types
    assert AlertType.COIN_VENUE_DEGRADED in types
    assert AlertType.COLLATERAL_IN_SPOT in types
    # funding(1) + margin maint(1) + margin liq(1) + venue(1) + collateral(1)
    assert len(out) == 5


def test_assess_clean_snapshot_returns_empty() -> None:
    snap = _snap(
        equity=1000.0,
        maint_margin=100.0,  # 0.1 < 0.5
        positions=[
            _pos("BTC", mark=100.0, liquidation_px=50.0, funding_since_open=1.0),
        ],
        spot_usdc=0.1,
        venue_coins={"BTC": {"delisted": False, "only_isolated": False, "max_leverage": 50}},
    )
    assert assess(snap, _Cfg()) == []


def test_assess_does_not_gate_on_enabled() -> None:
    # assess is pure over thresholds — caller gates on enabled, not assess.
    snap = _snap(equity=1000.0, spot_usdc=5.0)
    out = assess(snap, _Cfg(health=_HealthCfg(enabled=False)))
    assert any(f.alert_type is AlertType.COLLATERAL_IN_SPOT for f in out)


# ---------------------------------------------------------------------------
# adapter — gather_health_snapshot (the only IO)
# ---------------------------------------------------------------------------


def test_gather_health_snapshot_assembles_from_client() -> None:
    client = FakeHLClient(
        account_snapshot={
            "equity": 1234.5,
            "maint_margin": 67.8,
            "positions": {
                "BTC": {
                    "szi": 0.5,
                    "position_value": 30000.0,
                    "unrealized_pnl": 250.0,
                    "liquidation_px": 40000.0,
                    "funding_since_open": -12.0,
                    "leverage_type": "cross",
                    "leverage_value": 3,
                },
                "ETH": {
                    "szi": -2.0,
                    "position_value": 6000.0,
                    "unrealized_pnl": -40.0,
                    "liquidation_px": None,
                    "funding_since_open": 4.0,
                    "leverage_type": "isolated",
                    "leverage_value": 5,
                },
            },
        },
        marks={"BTC": Decimal("60000"), "ETH": Decimal("3000")},
        venue_meta={
            "BTC": {"delisted": False, "only_isolated": False, "max_leverage": 50},
            "ETH": {"delisted": False, "only_isolated": False, "max_leverage": 25},
        },
        spot_balances={"USDC": Decimal("9.99"), "BTC": Decimal("0.01")},
    )

    snap = gather_health_snapshot(client)

    assert snap.equity == 1234.5
    assert snap.maint_margin == 67.8
    assert snap.spot_usdc == 9.99
    assert snap.venue_coins == {
        "BTC": {"delisted": False, "only_isolated": False, "max_leverage": 50},
        "ETH": {"delisted": False, "only_isolated": False, "max_leverage": 25},
    }

    by_coin = {p.coin: p for p in snap.positions}
    assert set(by_coin) == {"BTC", "ETH"}

    btc = by_coin["BTC"]
    assert btc.mark == 60000.0  # joined from marks()
    assert btc.position_value == 30000.0
    assert btc.unrealized_pnl == 250.0
    assert btc.liquidation_px == 40000.0
    assert btc.funding_since_open == -12.0
    assert btc.leverage_type == "cross"
    assert btc.leverage_value == 3

    eth = by_coin["ETH"]
    assert eth.mark == 3000.0
    assert eth.liquidation_px is None  # preserved as None
    assert eth.leverage_type == "isolated"


def test_gather_health_snapshot_missing_mark_defaults_zero() -> None:
    client = FakeHLClient(
        account_snapshot={
            "equity": 100.0,
            "maint_margin": 1.0,
            "positions": {
                "SOL": {
                    "szi": 10.0,
                    "position_value": 1500.0,
                    "unrealized_pnl": 0.0,
                    "liquidation_px": None,
                    "funding_since_open": 0.0,
                    "leverage_type": "cross",
                    "leverage_value": 2,
                }
            },
        },
        marks={},  # no mark for SOL
        venue_meta={},
        spot_balances={},
    )
    snap = gather_health_snapshot(client)
    assert snap.positions[0].mark == 0.0
    assert snap.spot_usdc == 0.0


# ---------------------------------------------------------------------------
# B5 — direction-blind health checks for short positions
# ---------------------------------------------------------------------------


def test_liq_distance_alert_for_short() -> None:
    """A SHORT position fires MARGIN_RATIO_LOW when its liquidation price is ABOVE
    the mark price and within the warning threshold.

    For a short: liq_px > mark (liquidation triggers on an adverse UP move).
    The formula ``|mark − liq| / mark * 100`` is symmetric — the absolute value
    makes it direction-blind.  This test pins that the check fires correctly for
    the short geometry (liq_px > mark) so any accidental removal of ``abs()``
    would be caught immediately.

    Spec §5 (plan task B5): ``position_health.check_margin``'s symmetric formula
    is correct for shorts; add the test.

    Setup: mark=100, liq_px=105 (5% above mark, within 10% warn threshold).
    """
    snap = _snap(positions=[_pos("ETH", mark=100.0, liquidation_px=105.0)])
    out = check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0)

    assert len(out) == 1, (
        "short position with liq_px within warn distance must produce exactly one finding"
    )
    f = out[0]
    assert f.alert_type is AlertType.MARGIN_RATIO_LOW
    assert f.severity is Severity.WARNING  # 5% < 10%, but 5% >= 10%/2=5% → WARNING (not CRITICAL)
    assert f.context["coin"] == "ETH"
    assert f.context["mark"] == 100.0
    assert f.context["liquidation_px"] == 105.0
    # Symmetric distance: |100 − 105| / 100 * 100 = 5.0%
    assert f.context["distance_pct"] == pytest.approx(5.0)


def test_liq_distance_short_critical_threshold() -> None:
    """Short position with liq_px 3% above mark (< 5% = 10%/2) → CRITICAL severity.

    Mirrors the existing long-side CRITICAL test but with the short geometry
    (liq above mark).  Pins the severity escalation path for shorts.
    """
    snap = _snap(positions=[_pos("BTC", mark=100.0, liquidation_px=103.0)])
    out = check_margin(snap, maint_frac_warn=0.5, liq_distance_pct_warn=10.0)

    assert len(out) == 1
    assert out[0].severity is Severity.CRITICAL
    assert out[0].context["distance_pct"] == pytest.approx(3.0)


def test_funding_drag_short_earning_no_alert() -> None:
    """A SHORT position with NEGATIVE ``funding_since_open`` (earning funding) must
    NOT emit a FUNDING_DRAG alert, even when the magnitude is large.

    Spec §5 / red-team M2 (plan task B5): ``check_funding`` sums the SIGNED
    ``funding_since_open`` values with no ``abs()`` anywhere.  A short that is
    earning funding has a negative value, which REDUCES the total funding sum and
    therefore reduces (not increases) the drag percentage.  This test pins that
    the code is correct: negative funding on a short → total drag ≤ 0 → no alert.

    Setup: one long position paying $10 in funding drag, one short position
    earning −$60 in funding (net sum = −$50, i.e. the account is NET earning).
    With equity = $1000 and warn threshold = 5%, the net −5% must not fire.
    """
    snap = _snap(
        equity=1000.0,
        positions=[
            _pos("BTC", funding_since_open=10.0),   # long paying drag
            _pos("ETH", funding_since_open=-60.0),  # short earning funding
        ],
    )
    # Net sum = −50, pct = −5% → strictly below 5% warn threshold → no alert.
    out = check_funding(snap, drag_pct_warn=5.0)
    assert out == [], (
        "short position earning funding (negative funding_since_open) must not "
        "trigger FUNDING_DRAG — signed sum already correct (red-team M2)"
    )
