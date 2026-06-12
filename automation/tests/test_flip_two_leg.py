"""
Tests for FLIP handling — zero-crossing as a close+open two-leg (Task B4,
red-team H4).  All expected values are HAND-COMPUTED — no tautologies.

Why two legs (spec §4.4)
------------------------
A position flipping sign (long→short or short→long) must NEVER cross zero in
one IOC.  The sizer emits TWO OrderPlans for a flipping coin, close-leg FIRST:

- close-leg: ``target_size=0.0``, ``delta=-pos`` (the full close),
  ``is_close_leg=True`` — routed through the executor's reduce-only
  ``market_close`` path, which structurally cannot overshoot past zero.
- open-leg: ``target_size=<final signed target>``, ``delta=target-0`` — a
  plain IOC anchored at flat.

A single crossing IOC would "work" on HL but couples the close and open margin
semantics into one fill: a partial fill could strand the book anywhere between
the old position and the new one (including an unintended residual in the OLD
direction), and reduce-only — the auditable can't-over-close property — is
impossible for a crossing order.  Two legs keep recovery STRUCTURAL: if the
close fills and the open fails/defers, the next plan() is anchored to the live
(now flat) position and re-emits the open delta alone.

Cap semantics (red-team H4)
---------------------------
- END-STATE notional per coin is counted ONCE (the open-leg's |target|·px;
  the close-leg ends at 0 and must not add a second position value).
- TURNOVER counts BOTH legs' |delta| — both IOCs really trade.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from automation.core.config import CapsConfig, Config, RetryConfig, SignalSourceConfig
from automation.execution.executor import execute
from automation.execution.sizer import SizePlan, plan
from automation.runtime.intent_ledger import IntentLedger
from automation.tests._fakes import FakeHLClient
from automation.tests.test_sizer import _snap, _state, _target

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _cfg(
    *,
    safety_buffer: float = 0.03,
    max_notional_per_coin: float = 1_000_000,
    max_total_notional: float = 10_000_000,
    max_turnover_per_cycle: float = 5_000_000,
    liquidity_safety: float = 0.0,
) -> Config:
    """Config factory with sizer fields AND tunable executor caps.

    ``test_sizer._cfg`` hardcodes the caps; the cap-dedupe tests here need to
    pin exact breach boundaries, so we build the Config directly.
    """
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight"),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.6,
        safety_buffer=safety_buffer,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=max_notional_per_coin,
            max_total_notional=max_total_notional,
            max_turnover_per_cycle=max_turnover_per_cycle,
            max_order_slippage=0.01,
        ),
        retry=RetryConfig(liquidity_safety=liquidity_safety),
    )


# ---------------------------------------------------------------------------
# Sizer: flip detection emits two legs, close first
# ---------------------------------------------------------------------------


class TestFlipTwoLegPlanning:
    def test_flip_long_to_short_emits_two_plans(self) -> None:
        """pos +5, target −3 → [close(sell 5, target 0), open(sell 3, target −3)].

        E=100_000, TON px=6_000, szd=1, w=−0.18 → target size −3.0 (dyadic-
        exact, floor no-op).  Held +5.0 → ts·pos < 0 → FLIP:
            close-leg: target 0.0, delta −5.0, notional −30_000, SELL, close
            open-leg:  target −3.0, delta −3.0, notional −18_000, SELL
        Ordering pinned: close-leg FIRST.
        """
        ta = _target({"TON": -0.18}, cash=0.82)
        snap = _snap(
            equity=100_000,
            positions={"TON": 5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        sp = plan(ta, snap, _state(last_target=None), _cfg(), allow_short=True)

        assert sp.rebalance is True
        assert len(sp.orders) == 2, f"expected close+open legs, got {sp.orders}"
        close, opn = sp.orders
        # Close leg: full reduce to flat, flagged.
        assert close.coin == "TON"
        assert close.is_close_leg is True
        assert close.target_size == 0.0
        assert close.delta_size == -5.0
        assert abs(close.delta_notional - (-30_000.0)) < 1e-9
        assert close.side == "sell"
        # Open leg: the final signed target, anchored at flat.
        assert opn.coin == "TON"
        assert opn.is_close_leg is False
        assert opn.target_size == -3.0
        assert opn.delta_size == -3.0
        assert abs(opn.delta_notional - (-18_000.0)) < 1e-9
        assert opn.side == "sell"
        # The committed end-state is the open-leg's target.
        assert sp.target_sizes["TON"] == -3.0

    def test_flip_short_to_long_mirror(self) -> None:
        """pos −5, target +3 → [close(buy 5, target 0), open(buy 3, target +3)]."""
        ta = _target({"TON": 0.18}, cash=0.82)
        snap = _snap(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        sp = plan(ta, snap, _state(last_target=None), _cfg(), allow_short=True)

        assert len(sp.orders) == 2, f"expected close+open legs, got {sp.orders}"
        close, opn = sp.orders
        assert close.is_close_leg is True
        assert close.target_size == 0.0
        assert close.delta_size == 5.0  # buy-to-close the short
        assert close.side == "buy"
        assert opn.is_close_leg is False
        assert opn.target_size == 3.0
        assert opn.delta_size == 3.0
        assert opn.side == "buy"
        assert sp.target_sizes["TON"] == 3.0

    def test_no_flip_same_direction_single_plan(self) -> None:
        """pos +5 → target +8 (same direction) → ONE plan, no close leg.

        E=100_000, TON px=6_000, szd=1, w=+0.48 → target +8.0; held +5.0 →
        delta +3.0.  ts·pos > 0 → NOT a flip.
        """
        ta = _target({"TON": 0.48}, cash=0.52)
        snap = _snap(
            equity=100_000,
            positions={"TON": 5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        sp = plan(ta, snap, _state(last_target=None), _cfg(), allow_short=True)

        assert len(sp.orders) == 1, f"same-direction adjust must be ONE leg: {sp.orders}"
        o = sp.orders[0]
        assert o.is_close_leg is False
        assert o.target_size == 8.0
        assert o.delta_size == 3.0
        assert o.side == "buy"


# ---------------------------------------------------------------------------
# Executor: cap pre-flight dedupe (end-state once, turnover both legs)
# ---------------------------------------------------------------------------


def _flip_plan_and_client(
    cfg: Config,
) -> tuple[SizePlan, FakeHLClient]:
    """pos +5, target −3 @ px 1_000 (E=100_000, w=−0.03, szd=1).

    Legs: close sell 5 (turnover 5_000, end-state 0) + open sell 3
    (turnover 3_000, end-state 3_000).  Total turnover 8_000; end-state
    counted once = 3_000.
    """
    ta = _target({"TON": -0.03}, cash=0.97)
    snap = _snap(
        equity=100_000,
        positions={"TON": 5.0},
        marks={"TON": 1_000.0},
        sz_decimals={"TON": 1},
    )
    sp = plan(ta, snap, _state(last_target=None), cfg, allow_short=True)
    client = FakeHLClient(
        equity=100_000,
        positions={"TON": 5.0},
        marks={"TON": 1_000.0},
        sz_decimals={"TON": 1},
    )
    return sp, client


class TestFlipCapDedupe:
    def test_flip_cap_end_state_not_double_counted(self, tmp_path: Path) -> None:
        """End-state notional counted ONCE per flipping coin → caps PASS.

        max_total_notional=4_000: single-counted end-state (open leg only,
        3_000) passes; any double count (close 5·1_000 + open 3_000 = 8_000,
        or open twice = 6_000) would breach.  The cycle must NOT abort and
        BOTH legs must reach the exchange (close via market_close, open via
        submit_ioc).
        """
        cfg = _cfg(
            max_total_notional=4_000,
            max_notional_per_coin=1_000_000,
            max_turnover_per_cycle=1_000_000,
        )
        sp, client = _flip_plan_and_client(cfg)
        assert len(sp.orders) == 2  # precondition: the flip emitted two legs

        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        assert report.aborted is False, (
            f"cap pre-flight double-counted the flip end-state: {report.abort_reason}"
        )
        assert len(client.market_close_calls) == 1  # close leg, reduce-only
        assert len(client.submit_ioc_calls) == 1    # open leg
        assert client.submit_ioc_calls[0]["is_buy"] is False

    def test_flip_turnover_counts_both_legs(self, tmp_path: Path) -> None:
        """Turnover counts BOTH legs' |delta| → caps REJECT.

        max_turnover_per_cycle=6_000: either leg alone fits (close 5_000,
        open 3_000) but the legs really both trade — 8_000 > 6_000 must
        abort the cycle BEFORE any submit.
        """
        cfg = _cfg(
            max_total_notional=1_000_000,
            max_notional_per_coin=1_000_000,
            max_turnover_per_cycle=6_000,
        )
        sp, client = _flip_plan_and_client(cfg)
        assert len(sp.orders) == 2  # precondition: the flip emitted two legs

        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        assert report.aborted is True, (
            "turnover cap must sum BOTH flip legs (8_000 > 6_000)"
        )
        assert report.abort_reason is not None
        assert "turnover" in report.abort_reason.lower()
        assert client.market_close_calls == []
        assert client.submit_ioc_calls == []


# ---------------------------------------------------------------------------
# Executor: side-independent full-exit routing
# ---------------------------------------------------------------------------


class TestFullExitSideIndependent:
    def test_full_exit_of_short_routes_market_close(self, tmp_path: Path) -> None:
        """pos −5, target 0 → full exit is a BUY and must use market_close.

        The old gate (``target==0 AND side=='sell'``) routed a fully-closing
        SHORT (a buy) to the residual-retry IOC path, losing the reduce-only
        property.  Full exit = ``target_size == 0.0``, side-independent.
        """
        ta = _target({"TON": 0.0}, cash=1.0)
        snap = _snap(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        cfg = _cfg()
        sp = plan(ta, snap, _state(last_target=None), cfg, allow_short=True)
        assert len(sp.orders) == 1
        assert sp.orders[0].side == "buy"  # closing a short is a BUY

        client = FakeHLClient(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        assert len(client.market_close_calls) == 1, (
            "full exit of a SHORT must route through reduce-only market_close"
        )
        assert client.market_close_calls[0]["coin"] == "TON"
        assert client.submit_ioc_calls == [], (
            "a full exit must never take the residual-retry IOC path"
        )
        assert report.n_filled == 1
        assert client.positions() == {}  # flat


# ---------------------------------------------------------------------------
# Executor: close fills + open defers → structural recovery
# ---------------------------------------------------------------------------


class TestFlipCloseFillsOpenDefers:
    def test_flip_close_fills_open_defers_recovers(self, tmp_path: Path) -> None:
        """Close leg fills, open leg liquidity-defers; a fresh plan() against
        the now-flat book yields the open delta ALONE (structural recovery).

        pos +2, target −10 @ px 1_000 (E=100_000, w=−0.1, szd=1).  Both legs
        are SELLS (a long→short flip only ever sells) and consult the SAME
        bid book; bids carry 5.0 within the band (liquidity_safety=1.0):
            close needs 2.0 ≤ 5.0 → fires (market_close, fills)
            open  needs 10.0 > 5.0 → DEFERRED (no IOC)
        NOTE the open leg's |notional| (10_000) EXCEEDS the close leg's
        (2_000) — this also pins the ordering invariant: the close leg must
        run first even when the plain size-descending sort would not put it
        first.
        """
        ta = _target({"TON": -0.1}, cash=0.9)
        snap = _snap(
            equity=100_000,
            positions={"TON": 2.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        cfg = _cfg(liquidity_safety=1.0)
        sp = plan(ta, snap, _state(last_target=None), cfg, allow_short=True)

        assert len(sp.orders) == 2
        assert sp.orders[0].is_close_leg is True, (
            "close leg must precede the open leg even when the open leg's "
            "notional is larger (sort must not reorder a flip pair)"
        )

        client = FakeHLClient(
            equity=100_000,
            positions={"TON": 2.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
            # mark 1_000, slippage 0.01 → sells reach bids ≥ 990; depth 5.0.
            l2={"TON": {"bids": [(995.0, 5.0)], "asks": []}},
        )
        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        # Close filled (reduce-only), open deferred — the cycle REPORTS it.
        assert report.n_filled == 1
        assert report.n_liquidity_deferred == 1
        assert report.n_error == 0
        assert len(client.market_close_calls) == 1
        assert client.submit_ioc_calls == []  # open never submitted
        deferred = [r for r in report.results if r.liquidity_deferred]
        assert len(deferred) == 1
        assert deferred[0].coin == "TON"

        # STRUCTURAL RECOVERY: the book is now flat; a fresh plan() against
        # live positions re-emits the open delta alone — no flip, no close leg.
        assert client.positions() == {}  # close leg flattened the +2
        snap2 = _snap(
            equity=100_000,
            positions={k: float(v) for k, v in client.positions().items()},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        sp2 = plan(ta, snap2, _state(last_target=None), cfg, allow_short=True)
        assert len(sp2.orders) == 1, f"recovery plan must be ONE leg: {sp2.orders}"
        o = sp2.orders[0]
        assert o.is_close_leg is False
        assert o.target_size == -10.0
        assert o.delta_size == -10.0  # anchored at flat: the open delta alone
        assert o.side == "sell"


# ---------------------------------------------------------------------------
# Executor: both legs fill → distinct cloids (MF-6 across legs of one coin)
# ---------------------------------------------------------------------------


class TestFlipBothLegsFill:
    def test_flip_both_legs_fill_distinct_cloids(self, tmp_path: Path) -> None:
        """Happy path: close fills then open fills; the two legs share a coin,
        an as_of, and a now_ts — their cloids MUST still differ (MF-6: every
        order attempt is distinguishable on-chain and in the ledger).

        Under naive keying both legs derive ``make_cloid(..., coin, 0, now_ts)``
        → identical cloids → the exchange sees a duplicate and the ledger's
        second PENDING overwrites the first.
        """
        cfg = _cfg()
        sp, client = _flip_plan_and_client(cfg)  # pos +5 → target −3 @ 1_000
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = execute(
            sp, client, cfg, ledger,
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        assert report.n_filled == 2
        assert report.n_error == 0
        # End state: short −3 (close flattened +5, open sold 3).
        assert float(client.positions()["TON"]) == -3.0
        close_res, open_res = report.results
        assert close_res.cloids and open_res.cloids
        assert close_res.cloids[0] != open_res.cloids[0], (
            "flip legs share coin+as_of+now_ts — cloids collided (MF-6)"
        )
        assert ledger.pending() == []  # both intents resolved DONE

    def test_executor_short_to_long_flip_both_fill(self, tmp_path: Path) -> None:
        """Executor-level mirror: short→long flip, BOTH legs are BUYS.

        pos −5, target +3 @ px 1_000 (E=100_000, w=+0.03, szd=1).  Pins the
        buy-side close routing DIRECTLY: the close leg (a BUY — closing a
        short) must go through reduce-only ``market_close``, the open leg
        through a plain ``submit_ioc`` buy.  End state +3 — never more long
        than the target, never a doubled close.
        """
        ta = _target({"TON": 0.03}, cash=0.97)
        snap = _snap(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        cfg = _cfg()
        sp = plan(ta, snap, _state(last_target=None), cfg, allow_short=True)
        assert len(sp.orders) == 2
        assert sp.orders[0].is_close_leg is True
        assert sp.orders[0].side == "buy"  # buy-to-close the short

        client = FakeHLClient(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        assert report.n_filled == 2
        assert report.n_error == 0
        # Close leg routed reduce-only despite being a BUY.
        assert len(client.market_close_calls) == 1
        assert client.market_close_calls[0]["coin"] == "TON"
        # Open leg: exactly one plain IOC buy of 3.
        assert len(client.submit_ioc_calls) == 1
        assert client.submit_ioc_calls[0]["is_buy"] is True
        assert client.submit_ioc_calls[0]["sz"] == 3.0
        # End state: long +3 (close flattened −5, open bought 3).
        assert float(client.positions()["TON"]) == 3.0


# ---------------------------------------------------------------------------
# Executor: pair gating — an incomplete close leg suppresses its open leg
# ---------------------------------------------------------------------------
#
# The open leg is anchored at FLAT (delta = target − 0).  If the close leg did
# not FULLY fill, the book is NOT flat when the open leg's turn comes: firing
# it would create transient wrong-direction exposure up to |pos|, and on a
# partial close the open IOC itself crosses zero (spec §7: zero-crossing is
# ALWAYS two-legged with a reduce-only close).  The executor must suppress the
# open leg and record it via the defer channel so the cycle is non-clean
# (commit gate frozen, retry window arms) and the next attempt re-plans from
# live positions.
# ---------------------------------------------------------------------------


def _filled_resp(total_sz: float) -> dict[str, Any]:
    """A parse_order_response-compatible fill of *total_sz* (local twin of the
    private ``_fakes._ok_resp`` — same shape, kept typed here)."""
    return {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"filled": {"oid": 1, "totalSz": str(total_sz), "avgPx": "1000"}}
                ]
            }
        },
    }


class _PartialCloseClient(FakeHLClient):
    """FakeHLClient whose ``market_close`` PARTIALLY fills a fixed size.

    The real ``market_close`` is reduce-only and can under-fill on a thin
    book; the base fake always fully flattens.  This variant reduces the
    position by ``close_fill`` (sign-aware) and reports that as the fill so
    the executor sees ``status="partial"`` on the close leg.
    """

    def __init__(self, *, close_fill: float, **kw: Any) -> None:
        super().__init__(**kw)
        self._close_fill = close_fill

    def market_close(self, coin: str, slippage: float, cloid: str) -> dict[str, Any]:
        self.market_close_calls.append(
            {"coin": coin, "size": self._close_fill, "cloid": cloid}
        )
        prev = self._positions.get(coin, Decimal("0"))
        step = Decimal(str(self._close_fill if prev > 0 else -self._close_fill))
        self._positions[coin] = prev - step  # reduce toward zero, never past
        return _filled_resp(self._close_fill)


class TestFlipPairGating:
    def test_flip_close_defers_open_suppressed(self, tmp_path: Path) -> None:
        """Close leg liquidity-defers → the open leg must NOT be submitted.

        pos +5, target −3 @ px 1_000, liquidity_safety=1.0.  Both legs are
        SELLS against the SAME bid book; bids carry 4.0 within the band:
            close needs 5.0 > 4.0 → DEFERRED (no market_close)
            open  needs 3.0 ≤ 4.0 → the LIQUIDITY gate would let it FIRE —
        only the pair gate stops it.  Firing it would sell 3 against a +5
        book the strategy ordered CLOSED (end +2, target −3, and worse on
        thinner books).  Both legs land in the defer accounting → the cycle
        is non-clean → the commit gate stays frozen and the retry window
        arms; the next attempt re-plans from live (+5) positions.
        """
        ta = _target({"TON": -0.03}, cash=0.97)
        snap = _snap(
            equity=100_000,
            positions={"TON": 5.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        cfg = _cfg(liquidity_safety=1.0)
        sp = plan(ta, snap, _state(last_target=None), cfg, allow_short=True)
        assert len(sp.orders) == 2
        assert sp.orders[0].is_close_leg is True

        client = FakeHLClient(
            equity=100_000,
            positions={"TON": 5.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
            # mark 1_000, slippage 0.01 → sells reach bids ≥ 990; depth 4.0.
            l2={"TON": {"bids": [(995.0, 4.0)], "asks": []}},
        )
        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        # NEITHER leg reached the exchange.
        assert client.market_close_calls == []  # close deferred by liquidity
        assert client.submit_ioc_calls == [], (
            "open leg fired while its close leg was deferred — pair gating "
            "missing (transient wrong-direction exposure)"
        )
        # Both legs recorded as deferred → cycle non-clean → commit frozen.
        assert report.n_liquidity_deferred == 2
        assert report.n_filled == 0
        assert report.n_error == 0
        # Book untouched — recovery is a clean re-plan next attempt.
        assert float(client.positions()["TON"]) == 5.0

    def test_flip_close_partial_open_suppressed(self, tmp_path: Path) -> None:
        """Close leg PARTIALLY fills → open suppressed; zero never crossed.

        pos +5, target −3 @ px 1_000, gate OFF.  market_close fills only 4.0
        of the 5.0 close (residual +1).  Without pair gating the open IOC
        (sell 3, anchored at flat) fires against the +1 residual → position
        −2: the open order itself CROSSED ZERO (spec §7 violation).  With
        gating: open suppressed, position stays +1 (same sign as before),
        n_partial=1 + the suppressed leg in the defer count keep the cycle
        non-clean.
        """
        ta = _target({"TON": -0.03}, cash=0.97)
        snap = _snap(
            equity=100_000,
            positions={"TON": 5.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        cfg = _cfg()
        sp = plan(ta, snap, _state(last_target=None), cfg, allow_short=True)
        assert len(sp.orders) == 2

        client = _PartialCloseClient(
            close_fill=4.0,
            equity=100_000,
            positions={"TON": 5.0},
            marks={"TON": 1_000.0},
            sz_decimals={"TON": 1},
        )
        report = execute(
            sp, client, cfg, IntentLedger(tmp_path / "ledger.json"),
            strategy_id="t", as_of="2026-06-10", now_ts="T",
        )

        assert len(client.market_close_calls) == 1
        assert client.submit_ioc_calls == [], (
            "open leg fired after a PARTIAL close — the open IOC crossed zero"
        )
        # No order crossed zero: residual position keeps its ORIGINAL sign.
        end_pos = float(client.positions()["TON"])
        assert end_pos == 1.0, f"expected +1 residual, got {end_pos}"
        assert end_pos > 0.0, "position flipped sign — an order crossed zero"
        # Accounting: partial close + suppressed open → non-clean cycle.
        assert report.n_partial == 1
        assert report.n_liquidity_deferred == 1  # the suppressed open leg
        assert report.n_filled == 0
