"""
Tests for automation.execution.executor.

Coverage:
- make_cloid: deterministic, attempt_seq changes id, format "0x"+32hex.
- full-fill: single order fully fills → CoinResult status "filled", ledger
  has cloid DONE, slippage == cfg.caps.max_order_slippage, is_buy matches side,
  sz == abs(delta).
- ordering invariant: PENDING intent written to ledger BEFORE submit_ioc is
  called (fake client captures ledger.pending() snapshot at call-time and the
  test asserts the cloid was PENDING at that moment).
- partial fill then fill: attempt 0 fills half → new cloid → residual filled
  → status "filled", attempts==2, two distinct cloids.
- error: OrderRejected → coin status "error", retry loop does NOT spin
  (max one error attempt), other coins still processed.
- full exit (target_size=0, side sell): market_close called, NOT submit_ioc.
- plan.rebalance=False → execute returns n_submitted=0, NO client order calls.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from automation.core.config import CapsConfig, Config, RetryConfig, SignalSourceConfig
from automation.core.safety import OrderRejected
from automation.execution.executor import execute, make_cloid
from automation.execution.sizer import OrderPlan, SizePlan
from automation.runtime.intent_ledger import IntentLedger
from automation.tests._fakes import FakeHLClient

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _cfg(
    slippage: float = 0.005,
    max_notional_per_coin: float = 1_000_000,
    max_total_notional: float = 10_000_000,
    max_turnover_per_cycle: float = 5_000_000,
    liquidity_safety: float = 0.0,
) -> Config:
    # liquidity_safety defaults to 0.0 here (gate OFF) so the existing executor
    # tests — which build clients WITHOUT an l2 book — keep firing and filling.
    # Only the dedicated liquidity tests below pass liquidity_safety > 0.
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight"),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.6,
        safety_buffer=0.03,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=max_notional_per_coin,
            max_total_notional=max_total_notional,
            max_turnover_per_cycle=max_turnover_per_cycle,
            max_order_slippage=slippage,
        ),
        retry=RetryConfig(liquidity_safety=liquidity_safety),
    )


def _ok_resp(total_sz: float) -> dict[str, Any]:
    """Build a valid parse_order_response-compatible dict for a filled order."""
    return {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"filled": {"oid": 1, "totalSz": str(total_sz), "avgPx": "50000"}}
                ]
            }
        },
    }


def _order_plan(
    coin: str,
    target_size: float,
    delta_size: float,
    side: str = "buy",
    delta_notional: float = 1000.0,
) -> OrderPlan:
    return OrderPlan(
        coin=coin,
        target_size=target_size,
        delta_size=delta_size,
        delta_notional=delta_notional,
        side=side,  # type: ignore[arg-type]
    )


def _plan_with(orders: list[OrderPlan], rebalance: bool = True) -> SizePlan:
    return SizePlan(
        rebalance=rebalance,
        reason="bootstrap" if rebalance else "hold",
        orders=orders,
        skipped=[],
        flatten=[],
        target_sizes={},
        achieved_weights={},
        scaled_down=False,
        scale_factor=1.0,
        gross_notional=0.0,
        equity=100_000.0,
    )


# ---------------------------------------------------------------------------
# FakeClient: records every call and its arguments.
# The `ledger` reference is injected so submit_ioc can snapshot pending().
# ---------------------------------------------------------------------------


class FakeClient:
    """Fake client that records calls and serves canned responses.

    Attributes
    ----------
    call_log : list[dict]
        Each element has keys ``{"method", "args", "kwargs", "pending_snapshot"}``
        where ``pending_snapshot`` is the set of cloids that were PENDING in the
        ledger at the moment the method was called (used to prove write-before-submit).
    """

    def __init__(
        self,
        ledger: IntentLedger,
        submit_responses: list[dict[str, Any]] | None = None,
        market_close_response: dict[str, Any] | None = None,
        query_response: dict[str, Any] | None = None,
        raise_on_submit: Exception | None = None,
        spot_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ledger = ledger
        # Queue of responses for submit_ioc; repeated last item when exhausted.
        self._submit_queue: list[dict[str, Any]] = list(
            submit_responses or [_ok_resp(0.0)]
        )
        # Queue of responses for submit_spot_ioc (D4 spot venue).
        self._spot_queue: list[dict[str, Any]] = list(
            spot_responses or [_ok_resp(0.0)]
        )
        self._market_close_resp = market_close_response or _ok_resp(0.0)
        self._query_resp = query_response or {}
        self._raise_on_submit = raise_on_submit
        self.call_log: list[dict[str, Any]] = []

    def _snapshot_pending_cloids(self) -> set[str]:
        return {i.cloid for i in self._ledger.pending()}

    def submit_ioc(
        self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str
    ) -> dict[str, Any]:
        pending_now = self._snapshot_pending_cloids()
        self.call_log.append(
            {
                "method": "submit_ioc",
                "coin": coin,
                "is_buy": is_buy,
                "sz": sz,
                "slippage": slippage,
                "cloid": cloid,
                "pending_snapshot": pending_now,
            }
        )
        if self._raise_on_submit is not None:
            raise self._raise_on_submit
        resp = self._submit_queue.pop(0) if self._submit_queue else _ok_resp(0.0)
        return resp

    def submit_spot_ioc(
        self, pair: str, is_buy: bool, sz: float, slippage: float, cloid: str
    ) -> dict[str, Any]:
        pending_now = self._snapshot_pending_cloids()
        self.call_log.append(
            {
                "method": "submit_spot_ioc",
                "pair": pair,
                "is_buy": is_buy,
                "sz": sz,
                "slippage": slippage,
                "cloid": cloid,
                "pending_snapshot": pending_now,
            }
        )
        if self._raise_on_submit is not None:
            raise self._raise_on_submit
        resp = self._spot_queue.pop(0) if self._spot_queue else _ok_resp(0.0)
        return resp

    def market_close(
        self, coin: str, slippage: float, cloid: str
    ) -> dict[str, Any]:
        pending_now = self._snapshot_pending_cloids()
        self.call_log.append(
            {
                "method": "market_close",
                "coin": coin,
                "slippage": slippage,
                "cloid": cloid,
                "pending_snapshot": pending_now,
            }
        )
        if self._raise_on_submit is not None:
            raise self._raise_on_submit
        return self._market_close_resp

    def query_by_cloid(self, cloid: str) -> dict[str, Any]:
        self.call_log.append({"method": "query_by_cloid", "cloid": cloid})
        return self._query_resp

    # Snapshot methods (needed if reconciler is called — not used in executor tests)
    def equity(self) -> Decimal:
        return Decimal("100000")

    def positions(self) -> dict[str, Decimal]:
        return {}

    def marks(self) -> dict[str, Decimal]:
        return {}

    def sz_decimals(self) -> dict[str, int]:
        return {}

    @property
    def sub(self) -> str:
        return "0x" + "0" * 40


# ---------------------------------------------------------------------------
# make_cloid tests
# ---------------------------------------------------------------------------


_NOW = "2026-06-01T00:00:00Z"


class TestMakeCloid:
    def test_format_is_0x_plus_32_hex(self) -> None:
        cloid = make_cloid("haarp", "2026-06-01", "BTC", 0, _NOW)
        assert cloid.startswith("0x")
        assert len(cloid) == 34  # "0x" + 32 hex chars
        assert all(c in "0123456789abcdef" for c in cloid[2:])

    def test_deterministic_same_inputs(self) -> None:
        a = make_cloid("haarp", "2026-06-01", "BTC", 0, _NOW)
        b = make_cloid("haarp", "2026-06-01", "BTC", 0, _NOW)
        assert a == b

    def test_attempt_seq_changes_cloid(self) -> None:
        c0 = make_cloid("haarp", "2026-06-01", "BTC", 0, _NOW)
        c1 = make_cloid("haarp", "2026-06-01", "BTC", 1, _NOW)
        c2 = make_cloid("haarp", "2026-06-01", "BTC", 2, _NOW)
        assert c0 != c1
        assert c1 != c2
        assert c0 != c2

    def test_different_coin_different_cloid(self) -> None:
        btc = make_cloid("haarp", "2026-06-01", "BTC", 0, _NOW)
        eth = make_cloid("haarp", "2026-06-01", "ETH", 0, _NOW)
        assert btc != eth

    def test_different_strategy_different_cloid(self) -> None:
        a = make_cloid("haarp", "2026-06-01", "BTC", 0, _NOW)
        b = make_cloid("rsps", "2026-06-01", "BTC", 0, _NOW)
        assert a != b

    def test_now_ts_nonce_changes_cloid_same_as_of(self) -> None:
        """F-7: re-running the same as_of with a NEW now_ts produces a FRESH cloid."""
        run1 = make_cloid("haarp", "2026-06-01", "BTC", 0, "2026-06-01T00:00:00Z")
        run2 = make_cloid("haarp", "2026-06-01", "BTC", 0, "2026-06-01T12:00:00Z")
        assert run1 != run2

    def test_two_coins_same_cycle_distinct(self) -> None:
        """F-7: two coins in ONE cycle (same now_ts) still get distinct cloids."""
        now = "2026-06-01T00:00:00Z"
        btc = make_cloid("haarp", "2026-06-01", "BTC", 0, now)
        eth = make_cloid("haarp", "2026-06-01", "ETH", 0, now)
        assert btc != eth


# ---------------------------------------------------------------------------
# execute() — plan.rebalance=False
# ---------------------------------------------------------------------------


class TestExecuteHold:
    def test_hold_returns_empty_report_no_calls(self, tmp_path: Any) -> None:
        """plan.rebalance=False → n_submitted=0, no client order calls."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with(orders=[], rebalance=False)
        cfg = _cfg()
        client = FakeClient(ledger)

        report = execute(
            plan, client, cfg, ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="2026-06-01T00:00:00Z",
        )

        assert report.n_submitted == 0
        assert report.n_filled == 0
        assert report.n_error == 0
        assert report.results == []
        order_calls = [c for c in client.call_log if c["method"] in ("submit_ioc", "market_close")]
        assert len(order_calls) == 0


# ---------------------------------------------------------------------------
# execute() — full fill
# ---------------------------------------------------------------------------


class TestExecuteFullFill:
    def test_full_fill_status_and_ledger(self, tmp_path: Any) -> None:
        """Single buy order fully fills → status 'filled', ledger DONE."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", target_size=0.5, delta_size=delta, side="buy")])
        cfg = _cfg(slippage=0.005)
        client = FakeClient(ledger, submit_responses=[_ok_resp(delta)])

        report = execute(
            plan, client, cfg, ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
        )

        assert len(report.results) == 1
        r = report.results[0]
        assert r.status == "filled"
        assert r.coin == "BTC"
        assert r.attempts == 1
        assert abs(r.filled_size - delta) < 1e-9

        # Ledger: no PENDING entries remain
        assert ledger.pending() == []

    def test_full_fill_slippage_is_tight(self, tmp_path: Any) -> None:
        """submit_ioc must use cfg.caps.max_order_slippage (TIGHT), not SDK default."""
        slippage = 0.003
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        cfg = _cfg(slippage=slippage)
        client = FakeClient(ledger, submit_responses=[_ok_resp(0.5)])

        execute(plan, client, cfg, ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(submit_calls) == 1
        assert submit_calls[0]["slippage"] == slippage

    def test_full_fill_is_buy_matches_side(self, tmp_path: Any) -> None:
        """is_buy=True for 'buy' side, is_buy=False for 'sell' (non-zero target)."""
        ledger_b = IntentLedger(tmp_path / "ledger_b.json")
        plan_b = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        cfg = _cfg()
        client_b = FakeClient(ledger_b, submit_responses=[_ok_resp(0.5)])
        execute(plan_b, client_b, cfg, ledger_b, strategy_id="x", as_of="2026-06-01", now_ts="T")
        assert client_b.call_log[0]["is_buy"] is True

        ledger_s = IntentLedger(tmp_path / "ledger_s.json")
        plan_s = _plan_with([_order_plan("BTC", 0.2, -0.3, "sell")])
        client_s = FakeClient(ledger_s, submit_responses=[_ok_resp(0.3)])
        execute(plan_s, client_s, cfg, ledger_s, strategy_id="x", as_of="2026-06-01", now_ts="T")
        assert client_s.call_log[0]["is_buy"] is False

    def test_full_fill_sz_matches_abs_delta(self, tmp_path: Any) -> None:
        """sz passed to submit_ioc == abs(delta_size)."""
        delta = 1.234
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("ETH", 1.5, delta, "buy")])
        cfg = _cfg()
        client = FakeClient(ledger, submit_responses=[_ok_resp(delta)])

        execute(plan, client, cfg, ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert abs(submit_calls[0]["sz"] - delta) < 1e-9


# ---------------------------------------------------------------------------
# Ordering invariant: PENDING written BEFORE submit
# ---------------------------------------------------------------------------


class TestPendingBeforeSubmit:
    def test_cloid_was_pending_when_submit_called(self, tmp_path: Any) -> None:
        """The ledger.record_pending(cloid) must happen BEFORE client.submit_ioc(cloid).

        The FakeClient captures ledger.pending() snapshot at the moment submit_ioc
        is called.  We assert the cloid was in that snapshot.
        """
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        cfg = _cfg()
        client = FakeClient(ledger, submit_responses=[_ok_resp(delta)])

        execute(plan, client, cfg, ledger, strategy_id="haarp", as_of="2026-06-01", now_ts="T")

        # Find the submit_ioc call and check its pending snapshot
        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(submit_calls) == 1
        cloid_used = submit_calls[0]["cloid"]
        pending_at_call_time = submit_calls[0]["pending_snapshot"]
        assert cloid_used in pending_at_call_time, (
            f"ORDERING VIOLATION: cloid {cloid_used!r} was NOT in ledger.pending() "
            f"at the time submit_ioc was called.  pending_snapshot={pending_at_call_time}"
        )

    def test_cloid_not_pending_after_execution(self, tmp_path: Any) -> None:
        """After execute() returns, no intents remain PENDING."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(ledger, submit_responses=[_ok_resp(0.5)])

        execute(plan, client, _cfg(), ledger, strategy_id="haarp", as_of="2026-06-01", now_ts="T")

        assert ledger.pending() == []


# ---------------------------------------------------------------------------
# Partial fill + retry
# ---------------------------------------------------------------------------


class TestPartialFillRetry:
    def test_partial_then_full_fill(self, tmp_path: Any) -> None:
        """Attempt 0 fills half → attempt 1 gets new cloid → fills remainder."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        delta = 1.0
        half = 0.5

        # First call fills half, second call fills the rest.
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        client = FakeClient(
            ledger,
            submit_responses=[_ok_resp(half), _ok_resp(half)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
        )

        assert len(report.results) == 1
        r = report.results[0]
        assert r.status == "filled"
        assert r.attempts == 2

        # Two distinct cloids
        assert len(r.cloids) == 2
        assert r.cloids[0] != r.cloids[1]

        # No pending intents remain
        assert ledger.pending() == []

    def test_partial_retry_has_new_cloid_for_each_attempt(self, tmp_path: Any) -> None:
        """Each retry uses a NEW cloid (MF-6) based on attempt_seq."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("SOL", 10.0, 10.0, "buy")])
        # 3 attempts, each fills 1/3
        client = FakeClient(
            ledger,
            submit_responses=[_ok_resp(10.0 / 3), _ok_resp(10.0 / 3), _ok_resp(10.0 / 3 + 0.1)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        r = report.results[0]
        # All cloids must be distinct
        assert len(set(r.cloids)) == len(r.cloids), "Retry cloids must be distinct"

    def test_residual_submitted_in_retry(self, tmp_path: Any) -> None:
        """The sz passed to the second attempt == residual after partial fill."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        delta = 1.0
        first_fill = 0.3
        expected_residual = round(delta - first_fill, 9)

        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        client = FakeClient(
            ledger,
            submit_responses=[_ok_resp(first_fill), _ok_resp(expected_residual)],
        )

        execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(submit_calls) == 2
        second_sz = submit_calls[1]["sz"]
        assert abs(second_sz - expected_residual) < 1e-6


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_order_rejected_sets_error_status(self, tmp_path: Any) -> None:
        """OrderRejected → CoinResult status='error', retry loop does NOT spin."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(
            ledger,
            raise_on_submit=OrderRejected("Insufficient margin"),
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        assert len(report.results) == 1
        r = report.results[0]
        assert r.status == "error"
        assert r.error is not None
        # Only ONE attempt — no spin on deterministic rejection
        assert r.attempts == 1
        assert report.n_error == 1

    def test_error_on_one_coin_does_not_abort_other(self, tmp_path: Any) -> None:
        """A single-coin error must not abort the cycle — other coins still process."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        orders = [
            _order_plan("BTC", 0.5, 0.5, "buy", 5000.0),
            _order_plan("ETH", 2.0, 2.0, "buy", 4000.0),
        ]
        plan = _plan_with(orders)

        # submit_ioc raises on first call (BTC), succeeds on second (ETH)
        call_count = {"n": 0}

        class SelectiveFakeClient(FakeClient):
            def submit_ioc(self, coin, is_buy, sz, slippage, cloid):
                self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid,
                                      "is_buy": is_buy, "sz": sz, "slippage": slippage,
                                      "pending_snapshot": self._snapshot_pending_cloids()})
                call_count["n"] += 1
                if coin == "BTC":
                    raise OrderRejected("BTC rejected")
                return _ok_resp(2.0)

        client = SelectiveFakeClient(ledger)

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        assert len(report.results) == 2
        btc_r = next(r for r in report.results if r.coin == "BTC")
        eth_r = next(r for r in report.results if r.coin == "ETH")
        assert btc_r.status == "error"
        assert eth_r.status == "filled"

    def test_unexpected_exception_is_isolated(self, tmp_path: Any) -> None:
        """RuntimeError from submit_ioc → error status, no spin, other coins ok."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(
            ledger,
            raise_on_submit=RuntimeError("network timeout"),
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        r = report.results[0]
        assert r.status == "error"
        assert r.attempts == 1  # no spin

    def test_unexpected_exception_leaves_intent_pending(self, tmp_path: Any) -> None:
        """F-4: an UNEXPECTED exception leaves the cloid PENDING (order MIGHT have landed).

        Unlike a clean OrderRejected (nothing landed → mark_done), a network-class
        exception means the submit's fate is unknown.  The intent must stay PENDING
        so the next boot's reconcile_pending checks the chain by cloid.
        """
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(
            ledger,
            raise_on_submit=RuntimeError("network timeout"),
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        r = report.results[0]
        cloid = r.cloids[0]
        pending_cloids = {i.cloid for i in ledger.pending()}
        assert cloid in pending_cloids, (
            "F-4: unexpected-exception intent must be left PENDING for boot reconcile"
        )

    def test_order_rejected_marks_intent_done(self, tmp_path: Any) -> None:
        """F-4 contrast: a clean OrderRejected DOES mark the intent DONE (nothing landed)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(
            ledger,
            raise_on_submit=OrderRejected("Insufficient margin"),
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        cloid = report.results[0].cloids[0]
        pending_cloids = {i.cloid for i in ledger.pending()}
        assert cloid not in pending_cloids, (
            "A clean OrderRejected (nothing landed) should mark the intent DONE"
        )


# ---------------------------------------------------------------------------
# Full exit (market_close)
# ---------------------------------------------------------------------------


class TestFullExit:
    def test_full_exit_uses_market_close(self, tmp_path: Any) -> None:
        """target_size=0, side='sell' → market_close called, NOT submit_ioc."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # target_size=0.0 and side="sell" triggers the full-exit path
        exit_op = OrderPlan(
            coin="BTC",
            target_size=0.0,
            delta_size=-0.5,
            delta_notional=-25000.0,
            side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, market_close_response=_ok_resp(0.5))

        execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        ioc_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(mc_calls) == 1, "market_close should be called for full exits"
        assert len(ioc_calls) == 0, "submit_ioc must NOT be called for full exits"
        assert mc_calls[0]["coin"] == "BTC"

    def test_full_exit_pending_before_market_close(self, tmp_path: Any) -> None:
        """PENDING intent written before market_close (same MF-7 invariant)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="ETH",
            target_size=0.0,
            delta_size=-2.0,
            delta_notional=-4000.0,
            side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, market_close_response=_ok_resp(2.0))

        execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(mc_calls) == 1
        cloid_used = mc_calls[0]["cloid"]
        pending_at_call_time = mc_calls[0]["pending_snapshot"]
        assert cloid_used in pending_at_call_time, (
            f"ORDERING VIOLATION: cloid {cloid_used!r} was NOT pending "
            f"when market_close was called.  snapshot={pending_at_call_time}"
        )

    def test_partial_sell_uses_submit_ioc_not_market_close(self, tmp_path: Any) -> None:
        """A non-zero target_size sell (partial reduction) uses submit_ioc."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        partial_sell = OrderPlan(
            coin="BTC",
            target_size=0.2,   # non-zero — partial reduction, not full exit
            delta_size=-0.3,
            delta_notional=-15000.0,
            side="sell",
        )
        plan = _plan_with([partial_sell])
        client = FakeClient(ledger, submit_responses=[_ok_resp(0.3)])

        execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        ioc_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(ioc_calls) == 1
        assert len(mc_calls) == 0

    def test_full_exit_single_shot_full_fill(self, tmp_path: Any) -> None:
        """F-2: full-fill close → exactly ONE market_close call, status 'filled'."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-0.5,
            delta_notional=-25000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, market_close_response=_ok_resp(0.5))

        report = execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(mc_calls) == 1, "market_close must be SINGLE-shot"
        r = report.results[0]
        assert r.status == "filled"
        assert r.attempts == 1
        assert len(r.cloids) == 1

    def test_full_exit_partial_fill_single_shot_no_double_count(self, tmp_path: Any) -> None:
        """F-2: a partially-filled close does ONE market_close call, filled ≤ requested.

        The OLD bug drove the size-residual loop, calling market_close repeatedly
        and summing fills across attempts (so filled could exceed requested).  Now
        a partial close is a single shot: filled_size == the one response's fill,
        status == 'partial', exactly ONE market_close call.
        """
        ledger = IntentLedger(tmp_path / "ledger.json")
        requested = 0.5
        partial = 0.3  # close only filled 0.3 of the 0.5 position
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-requested,
            delta_notional=-25000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, market_close_response=_ok_resp(partial))

        report = execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(mc_calls) == 1, "market_close must be SINGLE-shot, never retried"
        r = report.results[0]
        assert r.status == "partial"
        # filled must NOT exceed requested (no double-counting across attempts)
        assert r.filled_size <= requested + 1e-9
        assert abs(r.filled_size - partial) < 1e-9
        assert r.attempts == 1

    def test_full_exit_zero_fill_is_error_single_shot(self, tmp_path: Any) -> None:
        """F-2: a close that fills nothing → status 'error', still ONE market_close call."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-0.5,
            delta_notional=-25000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, market_close_response=_ok_resp(0.0))

        report = execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(mc_calls) == 1
        assert report.results[0].status == "error"

    def test_full_exit_order_rejected_marks_done(self, tmp_path: Any) -> None:
        """F-4: a clean OrderRejected on close marks the intent DONE (nothing landed)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-0.5,
            delta_notional=-25000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, raise_on_submit=OrderRejected("no position"))

        report = execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        cloid = report.results[0].cloids[0]
        assert cloid not in {i.cloid for i in ledger.pending()}
        assert report.results[0].status == "error"

    def test_full_exit_unexpected_exception_leaves_pending(self, tmp_path: Any) -> None:
        """F-4: an unexpected exception on close leaves the intent PENDING (might have landed)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-0.5,
            delta_notional=-25000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, raise_on_submit=RuntimeError("network timeout"))

        report = execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        cloid = report.results[0].cloids[0]
        assert cloid in {i.cloid for i in ledger.pending()}, (
            "F-4: unexpected-exception close must leave the intent PENDING"
        )
        assert report.results[0].status == "error"


# ---------------------------------------------------------------------------
# F-5: pre-flight cap enforcement
# ---------------------------------------------------------------------------


class TestCapEnforcement:
    def test_total_notional_breach_aborts_zero_submits(self, tmp_path: Any) -> None:
        """An order set breaching max_total_notional → abort, ZERO submits."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # price = abs(delta_notional/delta_size) = 50000/1.0 = 50000.
        # end-state notional = target_size * price.
        # BTC: 1.0 * 50000 = 50000; ETH: 1.0 * 50000 = 50000; total = 100000.
        orders = [
            _order_plan("BTC", target_size=1.0, delta_size=1.0, side="buy", delta_notional=50_000.0),
            _order_plan("ETH", target_size=1.0, delta_size=1.0, side="buy", delta_notional=50_000.0),
        ]
        plan = _plan_with(orders)
        # cap total to 75k → 100k end-state breaches it.
        cfg = _cfg(max_total_notional=75_000, max_notional_per_coin=1_000_000)
        client = FakeClient(ledger, submit_responses=[_ok_resp(1.0), _ok_resp(1.0)])

        report = execute(plan, client, cfg, ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        assert report.aborted is True
        assert report.n_submitted == 0
        assert report.results == []
        order_calls = [c for c in client.call_log if c["method"] in ("submit_ioc", "market_close")]
        assert len(order_calls) == 0, "NO order may be submitted after a CapBreach"
        # Ledger must be untouched (no PENDING written before the abort)
        assert ledger.pending() == []

    def test_per_coin_notional_breach_aborts(self, tmp_path: Any) -> None:
        """A single coin breaching max_notional_per_coin → abort, zero submits."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # BTC end-state notional = 2.0 * 50000 = 100000.
        orders = [
            _order_plan("BTC", target_size=2.0, delta_size=2.0, side="buy", delta_notional=100_000.0),
        ]
        plan = _plan_with(orders)
        cfg = _cfg(max_notional_per_coin=50_000)  # 100k breaches it
        client = FakeClient(ledger, submit_responses=[_ok_resp(2.0)])

        report = execute(plan, client, cfg, ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        assert report.aborted is True
        assert report.n_submitted == 0
        assert ledger.pending() == []

    def test_turnover_breach_aborts(self, tmp_path: Any) -> None:
        """Total turnover (sum abs delta_notional) breaching cap → abort."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        orders = [
            _order_plan("BTC", target_size=0.5, delta_size=0.5, side="buy", delta_notional=40_000.0),
            _order_plan("ETH", target_size=10.0, delta_size=10.0, side="buy", delta_notional=40_000.0),
        ]
        plan = _plan_with(orders)
        # turnover = 80k; cap to 50k. (per-coin / total end-state are well within.)
        cfg = _cfg(
            max_turnover_per_cycle=50_000,
            max_notional_per_coin=10_000_000,
            max_total_notional=100_000_000,
        )
        client = FakeClient(ledger, submit_responses=[_ok_resp(0.5), _ok_resp(10.0)])

        report = execute(plan, client, cfg, ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        assert report.aborted is True
        assert report.n_submitted == 0

    def test_within_caps_executes_normally(self, tmp_path: Any) -> None:
        """An order set within all caps executes (no abort)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        orders = [
            _order_plan("BTC", target_size=0.5, delta_size=0.5, side="buy", delta_notional=25_000.0),
        ]
        plan = _plan_with(orders)
        cfg = _cfg()  # generous defaults
        client = FakeClient(ledger, submit_responses=[_ok_resp(0.5)])

        report = execute(plan, client, cfg, ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        assert report.aborted is False
        assert report.n_submitted == 1
        assert report.results[0].status == "filled"


# ---------------------------------------------------------------------------
# F-3 / F-6: report accounting
# ---------------------------------------------------------------------------


class TestReportAccounting:
    def test_n_partial_counted(self, tmp_path: Any) -> None:
        """F-3: a coin that fills partially across all attempts → n_partial == 1."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 1.0, 1.0, "buy")])
        # Each of 3 attempts fills only 0.1 → never reaches residual≈0 → partial.
        client = FakeClient(
            ledger,
            submit_responses=[_ok_resp(0.1), _ok_resp(0.1), _ok_resp(0.1)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T", max_attempts=3,
        )

        assert report.results[0].status == "partial"
        assert report.n_partial == 1
        assert report.n_filled == 0
        assert report.n_error == 0

    def test_n_submitted_no_overcount_on_exhaustion(self, tmp_path: Any) -> None:
        """F-6: 3 attempts that all partially fill → n_submitted == 3 (not 4)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 1.0, 1.0, "buy")])
        client = FakeClient(
            ledger,
            submit_responses=[_ok_resp(0.1), _ok_resp(0.1), _ok_resp(0.1)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T", max_attempts=3,
        )

        # Exactly 3 real submits — attempts must equal the number of submit calls.
        assert report.results[0].attempts == 3
        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(submit_calls) == 3
        assert report.n_submitted == 3

    def test_full_fill_first_attempt_n_submitted_is_one(self, tmp_path: Any) -> None:
        """F-6: a clean single-attempt full fill reports attempts==1, n_submitted==1."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(ledger, submit_responses=[_ok_resp(0.5)])

        report = execute(plan, client, _cfg(), ledger, strategy_id="x", as_of="2026-06-01", now_ts="T")

        assert report.results[0].attempts == 1
        assert report.n_submitted == 1


# ---------------------------------------------------------------------------
# Resting-status anomaly (peer-issue-2)
# ---------------------------------------------------------------------------


def _resting_resp(cloid: str | None = None) -> dict[str, Any]:
    """Build a parse_order_response-compatible dict with a resting status."""
    resting: dict[str, Any] = {"oid": 42, "sz": "0.5"}
    if cloid is not None:
        resting["cloid"] = cloid
    return {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [{"resting": resting}]
            }
        },
    }


class FakeClientWithCancel(FakeClient):
    """Extends FakeClient with cancel_by_cloid / cancel_oid recording.

    cancel_behavior controls what cancel_by_cloid does:
      - None  → succeed (return {})
      - Exception → raise that exception
    """

    def __init__(
        self,
        ledger: IntentLedger,
        submit_responses: list[dict[str, Any]] | None = None,
        cancel_behavior: Exception | None = None,
        spot_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            ledger, submit_responses=submit_responses, spot_responses=spot_responses
        )
        self._cancel_behavior = cancel_behavior
        self.cancel_by_cloid_calls: list[tuple[str, str]] = []
        self.cancel_oid_calls: list[tuple[str, int]] = []

    def cancel_by_cloid(self, coin: str, cloid: str) -> dict[str, Any]:
        self.cancel_by_cloid_calls.append((coin, cloid))
        self.call_log.append({"method": "cancel_by_cloid", "coin": coin, "cloid": cloid})
        if self._cancel_behavior is not None:
            raise self._cancel_behavior
        return {}

    def cancel_oid(self, coin: str, oid: int) -> dict[str, Any]:
        self.cancel_oid_calls.append((coin, oid))
        self.call_log.append({"method": "cancel_oid", "coin": coin, "oid": oid})
        return {}

    def open_orders(self) -> list[dict[str, Any]]:
        return []


class TestRestingAnomalyHandling:
    """peer-issue-2 — IOC resting-status anomaly detection and cancel-before-retry."""

    def test_resting_response_triggers_cancel_by_cloid(self, tmp_path: Any) -> None:
        """A resting status → cancel_by_cloid is called for that cloid."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        # attempt 0 returns resting; attempt 1 fills fully
        client = FakeClientWithCancel(
            ledger,
            submit_responses=[_resting_resp(), _ok_resp(delta)],
        )

        execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        # cancel_by_cloid must have been called once (for the resting cloid)
        assert len(client.cancel_by_cloid_calls) == 1, (
            f"Expected 1 cancel_by_cloid call, got {client.cancel_by_cloid_calls}"
        )
        # The coin must match
        cancelled_coin = client.cancel_by_cloid_calls[0][0]
        assert cancelled_coin == "BTC"

    def test_resting_not_counted_as_filled(self, tmp_path: Any) -> None:
        """Resting size is NOT counted as filled; the retry re-submits the full residual."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        # attempt 0 rests; attempt 1 fills the full residual
        client = FakeClientWithCancel(
            ledger,
            submit_responses=[_resting_resp(), _ok_resp(delta)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        r = report.results[0]
        # Total filled must equal the actual fill (from attempt 1), not the resting sz
        assert abs(r.filled_size - delta) < 1e-9, (
            f"Resting size was wrongly counted as filled: filled_size={r.filled_size}"
        )

    def test_resting_anomaly_sets_flag_on_coin_result(self, tmp_path: Any) -> None:
        """CoinResult.resting_anomaly=True when a resting status was seen."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        client = FakeClientWithCancel(
            ledger,
            submit_responses=[_resting_resp(), _ok_resp(delta)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        assert report.results[0].resting_anomaly is True

    def test_resting_anomaly_tallied_on_exec_report(self, tmp_path: Any) -> None:
        """ExecReport.n_resting_anomaly == 1 when one coin had a resting status."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        client = FakeClientWithCancel(
            ledger,
            submit_responses=[_resting_resp(), _ok_resp(delta)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        assert report.n_resting_anomaly == 1

    def test_retry_proceeds_after_resting_cancel(self, tmp_path: Any) -> None:
        """After the resting cancel, the retry loop continues and can fill."""
        delta = 1.0
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        # attempt 0 rests; attempt 1 fully fills
        client = FakeClientWithCancel(
            ledger,
            submit_responses=[_resting_resp(), _ok_resp(delta)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        r = report.results[0]
        assert r.status == "filled"
        assert r.attempts == 2  # two submit calls: resting + fill

    def test_cancel_failure_aborts_retry_no_resubmit(self, tmp_path: Any) -> None:
        """CRITICAL (double-exposure guard): if cancel_by_cloid FAILS, the executor
        must NOT re-submit the residual — the resting order is still on the book,
        so a re-submit would double exposure.  The coin retry aborts with
        status='error', resting_anomaly=True, and the error message names the
        double-exposure guard.  Proven by asserting submit_ioc is NOT called again
        after the resting attempt."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        # attempt 0 rests; cancel RAISES.  A second _ok_resp is queued so that IF
        # the executor wrongly re-submits, it would "fill" (and the test would
        # catch the extra submit_ioc call / the wrong filled_size).
        client = FakeClientWithCancel(
            ledger,
            submit_responses=[_resting_resp(), _ok_resp(delta)],
            cancel_behavior=RuntimeError("network error on cancel"),
        )

        # Count submit_ioc calls BEFORE and AFTER to prove no re-submit.
        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        # Exactly ONE submit (the resting attempt).  The cancel failed → NO
        # second submit_ioc (no re-submit on top of the still-resting order).
        assert len(submit_calls) == 1, (
            f"DOUBLE-EXPOSURE BUG: executor re-submitted after a cancel failure "
            f"(submit_ioc called {len(submit_calls)} times; expected 1)"
        )

        r = report.results[0]
        assert r.status == "error", "cancel failure must abort the coin (status=error)"
        assert r.resting_anomaly is True
        assert r.attempts == 1  # only the one resting submit
        # Residual left UNFILLED (under-fill is the safe failure mode).
        assert r.filled_size < 1e-9, (
            f"resting size must NOT count as filled: filled_size={r.filled_size}"
        )
        # Error message names the double-exposure guard.
        assert r.error is not None
        assert "double-exposure guard" in r.error
        assert report.n_resting_anomaly == 1

    def test_no_resting_no_anomaly_flag(self, tmp_path: Any) -> None:
        """Normal fill → resting_anomaly=False, n_resting_anomaly=0 (additive default)."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        client = FakeClientWithCancel(ledger, submit_responses=[_ok_resp(delta)])

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
        )

        assert report.results[0].resting_anomaly is False
        assert report.n_resting_anomaly == 0


# ---------------------------------------------------------------------------
# MF-9: rate-budget pre-flight (G4b)
# ---------------------------------------------------------------------------


class FakeClientWithRateLimit(FakeClientWithCancel):
    """Extends FakeClientWithCancel with a configurable user_rate_limit() response.

    Parameters
    ----------
    rate_limit_response:
        Dict returned by user_rate_limit().  If None, user_rate_limit() raises
        RuntimeError (simulates a query failure).
    """

    def __init__(
        self,
        ledger: IntentLedger,
        submit_responses: list[dict[str, Any]] | None = None,
        rate_limit_response: dict[str, Any] | None = None,
        *,
        rate_limit_raises: bool = False,
        raise_rate_limited_on_submit: bool = False,
    ) -> None:
        super().__init__(ledger, submit_responses=submit_responses)
        self._rl_response = rate_limit_response
        self._rl_raises = rate_limit_raises
        self._raise_rate_limited = raise_rate_limited_on_submit

    def user_rate_limit(self) -> dict[str, Any]:
        if self._rl_raises:
            raise RuntimeError("rate limit query failed")
        if self._rl_response is None:
            raise RuntimeError("no rate limit response configured")
        return self._rl_response

    def submit_ioc(
        self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str
    ) -> dict[str, Any]:
        if self._raise_rate_limited:
            from automation.core.safety import RateLimited  # noqa: PLC0415
            self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid,
                                  "is_buy": is_buy, "sz": sz, "slippage": slippage,
                                  "pending_snapshot": self._snapshot_pending_cloids()})
            raise RateLimited("rate limited mid-cycle")
        return super().submit_ioc(coin, is_buy, sz, slippage, cloid)


def _rl_response(cap: int, used: int) -> dict[str, Any]:
    """Build a user_rate_limit() response dict."""
    return {"cumVlm": "0", "nRequestsCap": cap, "nRequestsUsed": used}


class TestRateBudgetPreFlight:
    """MF-9: rate-budget pre-flight — defer-not-strand logic."""

    def test_rate_defer_when_budget_insufficient(self, tmp_path: Any) -> None:
        """user_rate_limit returns remaining < est_actions → deferred=True, n_submitted=0,
        NO submit_ioc calls."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # 2 orders, max_attempts=3 → est = 2*(3+1)=8; remaining=5 < 8 → defer
        plan = _plan_with([
            _order_plan("BTC", 0.5, 0.5, "buy"),
            _order_plan("ETH", 2.0, 2.0, "buy"),
        ])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[_ok_resp(0.5), _ok_resp(2.0)],
            rate_limit_response=_rl_response(cap=100, used=95),  # remaining=5, est=8
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        assert report.deferred is True, "should be deferred when budget < est_actions"
        assert report.n_submitted == 0, "no orders must be submitted on a defer"
        assert report.defer_reason is not None
        assert "rate budget" in report.defer_reason
        submit_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(submit_calls) == 0, "submit_ioc must NOT be called after a rate defer"

    def test_rate_ok_executes_normally(self, tmp_path: Any) -> None:
        """remaining >= est_actions → no defer, executes normally."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # 1 order, max_attempts=3 → est=4; remaining=10 ≥ 4 → no defer
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[_ok_resp(0.5)],
            rate_limit_response=_rl_response(cap=100, used=90),  # remaining=10
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        assert report.deferred is False
        assert report.n_submitted == 1
        assert report.results[0].status == "filled"

    def test_rate_query_failure_does_not_defer(self, tmp_path: Any) -> None:
        """user_rate_limit() raises → NO defer (best-effort); proceeds to submit."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[_ok_resp(0.5)],
            rate_limit_raises=True,  # query fails
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        # Must NOT defer — failed query should not block trading
        assert report.deferred is False
        assert report.n_submitted == 1

    def test_rate_defer_aborted_and_deferred_are_additive_defaults(self, tmp_path: Any) -> None:
        """ExecReport.deferred/defer_reason/n_throttled exist with additive defaults."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[_ok_resp(0.5)],
            rate_limit_response=_rl_response(cap=100, used=0),  # plenty of budget
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        # Additive defaults: new fields must exist and not break normal reports
        assert report.deferred is False
        assert report.defer_reason is None
        assert report.n_throttled == 0

    def test_rate_exact_boundary_remaining_equals_est_no_defer(self, tmp_path: Any) -> None:
        """Boundary: remaining == est_actions → no defer (sufficient)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # 1 order, max_attempts=3 → est=(1)*(3+1)=4; remaining=4 → no defer
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[_ok_resp(0.5)],
            rate_limit_response=_rl_response(cap=100, used=96),  # remaining=4=est
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        assert report.deferred is False
        assert report.n_submitted == 1

    def test_rate_one_below_boundary_defers(self, tmp_path: Any) -> None:
        """Boundary: remaining == est_actions - 1 → defer."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # 1 order, max_attempts=3 → est=4; remaining=3 < 4 → defer
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[_ok_resp(0.5)],
            rate_limit_response=_rl_response(cap=100, used=97),  # remaining=3=est-1
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
        )

        assert report.deferred is True
        assert report.n_submitted == 0

    def test_empty_orders_plan_never_defers(self, tmp_path: Any) -> None:
        """M2: a rebalance=True plan with ZERO orders (est_actions==0) → no defer,
        even with a negative remaining budget (no orders ⇒ nothing to rate-limit)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        # rebalance=True but no orders → est_actions == 0
        plan = _plan_with(orders=[], rebalance=True)
        # used > cap → remaining is NEGATIVE; the guard must still NOT defer.
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[],
            rate_limit_response=_rl_response(cap=100, used=200),  # remaining = -100
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        assert report.deferred is False, "an empty-orders plan must never be deferred"
        assert report.defer_reason is None
        assert report.n_submitted == 0  # nothing to submit anyway


class TestRateLimitedMidCycle:
    """MF-9: RateLimited raised during submit → throttled=True on CoinResult."""

    def test_rate_limited_mid_cycle_throttled_flag(self, tmp_path: Any) -> None:
        """A coin's submit raises RateLimited → throttled=True on that CoinResult."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClientWithRateLimit(
            ledger,
            submit_responses=[],
            rate_limit_response=_rl_response(cap=100, used=0),  # budget ok → no defer
            raise_rate_limited_on_submit=True,
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        assert len(report.results) == 1
        r = report.results[0]
        assert r.throttled is True, "RateLimited mid-cycle must set throttled=True"
        assert r.status == "error"
        assert report.n_throttled == 1

    def test_rate_limited_other_coins_still_processed(self, tmp_path: Any) -> None:
        """RateLimited on one coin → that coin is error/throttled, other coins still run."""
        from automation.core.safety import RateLimited  # noqa: PLC0415

        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([
            _order_plan("BTC", 0.5, 0.5, "buy", 25_000.0),
            _order_plan("ETH", 2.0, 2.0, "buy", 4_000.0),
        ])

        call_count: dict[str, int] = {"n": 0}

        class SelectiveRateLimited(FakeClient):
            def submit_ioc(self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str) -> dict[str, Any]:
                self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid,
                                      "is_buy": is_buy, "sz": sz, "slippage": slippage,
                                      "pending_snapshot": self._snapshot_pending_cloids()})
                call_count["n"] += 1
                if coin == "BTC":
                    raise RateLimited("throttled")
                return _ok_resp(2.0)

            def user_rate_limit(self) -> dict[str, Any]:
                return {"cumVlm": "0", "nRequestsCap": 100, "nRequestsUsed": 0}

        client = SelectiveRateLimited(ledger)

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        assert len(report.results) == 2
        btc_r = next(r for r in report.results if r.coin == "BTC")
        eth_r = next(r for r in report.results if r.coin == "ETH")
        assert btc_r.throttled is True
        assert btc_r.status == "error"
        assert eth_r.throttled is False
        assert eth_r.status == "filled"
        assert report.n_throttled == 1

    def test_no_throttle_on_order_rejected(self, tmp_path: Any) -> None:
        """OrderRejected does NOT set throttled=True (only RateLimited does)."""
        from automation.core.safety import OrderRejected  # noqa: PLC0415

        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", 0.5, 0.5, "buy")])
        client = FakeClient(ledger, raise_on_submit=OrderRejected("margin"))

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        r = report.results[0]
        assert r.throttled is False
        assert r.status == "error"
        assert report.n_throttled == 0

    def test_full_exit_rate_limited_sets_throttled(self, tmp_path: Any) -> None:
        """M1: a full-exit (target_size=0, side sell) whose market_close raises
        RateLimited → that CoinResult has throttled=True, status='error',
        ExecReport.n_throttled >= 1 (still per-coin isolated)."""
        from automation.core.safety import RateLimited  # noqa: PLC0415

        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-0.5,
            delta_notional=-25_000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, raise_on_submit=RateLimited("close throttled"))

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        # market_close (not submit_ioc) was the call path
        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(mc_calls) == 1
        r = report.results[0]
        assert r.throttled is True, "full-exit RateLimited must set throttled=True"
        assert r.status == "error"  # still status=error, isolated
        assert report.n_throttled >= 1
        # The intent is marked DONE (clean rejection — nothing landed)
        assert r.cloids[0] not in {i.cloid for i in ledger.pending()}

    def test_full_exit_order_rejected_not_throttled(self, tmp_path: Any) -> None:
        """M1 contrast: a full-exit OrderRejected does NOT set throttled=True."""
        from automation.core.safety import OrderRejected  # noqa: PLC0415

        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="BTC", target_size=0.0, delta_size=-0.5,
            delta_notional=-25_000.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, raise_on_submit=OrderRejected("no position"))

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        r = report.results[0]
        assert r.throttled is False
        assert r.status == "error"
        assert report.n_throttled == 0


# ---------------------------------------------------------------------------
# D4: hybrid spot+perp venue routing (spot_routing map)
# ---------------------------------------------------------------------------


class TestSpotRoutingNoneIsPerp:
    """spot_routing=None / omitted → byte-identical perp path; NO spot calls."""

    def test_default_spot_routing_none_routes_to_perp(self, tmp_path: Any) -> None:
        """Omitting spot_routing entirely keeps the perp path (regression)."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", target_size=delta, delta_size=delta, side="buy")])
        client = FakeClient(ledger, submit_responses=[_ok_resp(delta)])

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
        )

        ioc_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(ioc_calls) == 1, "no spot_routing → coin must go to perp submit_ioc"
        assert len(spot_calls) == 0, "no spot_routing → submit_spot_ioc must NOT be called"
        assert report.results[0].status == "filled"

    def test_explicit_spot_routing_none_routes_to_perp(self, tmp_path: Any) -> None:
        """spot_routing=None explicitly → perp path, no spot calls."""
        delta = 0.5
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", delta, delta, "buy")])
        client = FakeClient(ledger, submit_responses=[_ok_resp(delta)])

        execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing=None,
        )

        assert [c["method"] for c in client.call_log if c["method"] == "submit_spot_ioc"] == []
        assert len([c for c in client.call_log if c["method"] == "submit_ioc"]) == 1

    def test_coin_absent_from_map_routes_to_perp(self, tmp_path: Any) -> None:
        """A coin NOT in spot_routing → perp, even though the map is non-empty."""
        delta = 0.4
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("BTC", delta, delta, "buy")])
        client = FakeClient(ledger, submit_responses=[_ok_resp(delta)])

        execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},  # BTC absent → perp
        )

        assert len([c for c in client.call_log if c["method"] == "submit_ioc"]) == 1
        assert len([c for c in client.call_log if c["method"] == "submit_spot_ioc"]) == 0


class TestSpotBuy:
    """A spot coin BUY routes to submit_spot_ioc(pair, True, ...)."""

    def test_spot_buy_calls_submit_spot_ioc(self, tmp_path: Any) -> None:
        delta = 3.0
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", target_size=delta, delta_size=delta, side="buy")])
        slippage = 0.004
        cfg = _cfg(slippage=slippage)
        client = FakeClient(ledger, spot_responses=[_ok_resp(delta)])

        report = execute(
            plan, client, cfg, ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        ioc_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(spot_calls) == 1, "spot buy must call submit_spot_ioc"
        assert len(ioc_calls) == 0, "spot buy must NOT call submit_ioc"
        call = spot_calls[0]
        assert call["pair"] == "@107"
        assert call["is_buy"] is True
        assert abs(call["sz"] - delta) < 1e-9
        assert call["slippage"] == slippage
        assert call["cloid"].startswith("0x")
        # CoinResult filled
        r = report.results[0]
        assert r.coin == "HYPE"
        assert r.status == "filled"
        assert abs(r.filled_size - delta) < 1e-9

    def test_spot_buy_pending_written_before_submit(self, tmp_path: Any) -> None:
        """MF-7: the PENDING intent is in the ledger snapshot at submit_spot_ioc time."""
        delta = 3.0
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", delta, delta, "buy")])
        client = FakeClient(ledger, spot_responses=[_ok_resp(delta)])

        execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(spot_calls) == 1
        cloid_used = spot_calls[0]["cloid"]
        assert cloid_used in spot_calls[0]["pending_snapshot"], (
            "ORDERING VIOLATION: cloid was not PENDING when submit_spot_ioc was called"
        )
        # No pending left after execution.
        assert ledger.pending() == []

    def test_spot_buy_partial_then_fill_residual_retry(self, tmp_path: Any) -> None:
        """Residual retry works on spot: attempt 0 fills half, attempt 1 fills rest."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        delta = 4.0
        half = 2.0
        plan = _plan_with([_order_plan("HYPE", delta, delta, "buy")])
        client = FakeClient(ledger, spot_responses=[_ok_resp(half), _ok_resp(half)])

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(spot_calls) == 2, "residual retry should re-submit on spot"
        # second submit sz == residual after first partial fill
        assert abs(spot_calls[1]["sz"] - (delta - half)) < 1e-6
        assert spot_calls[0]["cloid"] != spot_calls[1]["cloid"]
        r = report.results[0]
        assert r.status == "filled"
        assert r.attempts == 2


class TestSpotPartialReduce:
    """A spot partial reduce (sell delta, target>0) is submit_spot_ioc(is_buy=False)."""

    def test_spot_partial_reduce_is_spot_sell_not_market_close(self, tmp_path: Any) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        partial = OrderPlan(
            coin="HYPE",
            target_size=1.0,   # non-zero → partial reduction (NOT full exit)
            delta_size=-2.0,
            delta_notional=-200.0,
            side="sell",
        )
        plan = _plan_with([partial])
        client = FakeClient(ledger, spot_responses=[_ok_resp(2.0)])

        execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        ioc_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(spot_calls) == 1, "spot partial reduce must use submit_spot_ioc"
        assert len(mc_calls) == 0, "spot reduce must NOT call market_close"
        assert len(ioc_calls) == 0, "spot reduce must NOT call perp submit_ioc"
        call = spot_calls[0]
        assert call["pair"] == "@107"
        assert call["is_buy"] is False, "spot reduce is a SELL (is_buy=False)"
        assert abs(call["sz"] - 2.0) < 1e-9  # abs(delta_size)


class TestSpotFullExit:
    """A spot full exit (target_size==0, sell) is submit_spot_ioc(is_buy=False), single shot."""

    def test_spot_full_exit_is_spot_sell_single_shot(self, tmp_path: Any) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="HYPE",
            target_size=0.0,    # full exit
            delta_size=-5.0,
            delta_notional=-500.0,
            side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, spot_responses=[_ok_resp(5.0)])

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        mc_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(spot_calls) == 1, "spot full exit must use submit_spot_ioc (single shot)"
        assert len(mc_calls) == 0, "spot full exit must NOT call market_close"
        call = spot_calls[0]
        assert call["pair"] == "@107"
        assert call["is_buy"] is False, "spot exit is a SELL"
        assert abs(call["sz"] - 5.0) < 1e-9  # abs(delta_size)
        r = report.results[0]
        assert r.status == "filled"
        assert r.attempts == 1, "spot full exit is single-shot"
        assert len(r.cloids) == 1

    def test_spot_full_exit_pending_before_submit(self, tmp_path: Any) -> None:
        """MF-7: PENDING written before the spot full-exit submit."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        exit_op = OrderPlan(
            coin="HYPE", target_size=0.0, delta_size=-5.0,
            delta_notional=-500.0, side="sell",
        )
        plan = _plan_with([exit_op])
        client = FakeClient(ledger, spot_responses=[_ok_resp(5.0)])

        execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(spot_calls) == 1
        cloid_used = spot_calls[0]["cloid"]
        assert cloid_used in spot_calls[0]["pending_snapshot"]
        assert ledger.pending() == []


class TestSpotPerpMixedBook:
    """A mixed plan routes each coin to its own venue with per-coin isolation."""

    def test_mixed_hype_spot_btc_perp(self, tmp_path: Any) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        orders = [
            _order_plan("HYPE", target_size=3.0, delta_size=3.0, side="buy", delta_notional=300.0),
            _order_plan("BTC", target_size=0.5, delta_size=0.5, side="buy", delta_notional=25_000.0),
        ]
        plan = _plan_with(orders)
        client = FakeClient(
            ledger,
            submit_responses=[_ok_resp(0.5)],   # BTC perp
            spot_responses=[_ok_resp(3.0)],     # HYPE spot
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        ioc_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(spot_calls) == 1, "HYPE must route to spot"
        assert spot_calls[0]["pair"] == "@107"
        assert len(ioc_calls) == 1, "BTC must route to perp"
        assert ioc_calls[0]["coin"] == "BTC"
        # both filled
        assert len(report.results) == 2
        hype_r = next(r for r in report.results if r.coin == "HYPE")
        btc_r = next(r for r in report.results if r.coin == "BTC")
        assert hype_r.status == "filled"
        assert btc_r.status == "filled"

    def test_mixed_per_coin_isolation_spot_error_perp_ok(self, tmp_path: Any) -> None:
        """A spot-coin OrderRejected does NOT abort the perp coin (per-coin isolation)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        orders = [
            _order_plan("HYPE", 3.0, 3.0, "buy", 300.0),
            _order_plan("BTC", 0.5, 0.5, "buy", 25_000.0),
        ]
        plan = _plan_with(orders)

        class SelectiveSpotReject(FakeClient):
            def submit_spot_ioc(self, pair, is_buy, sz, slippage, cloid):  # type: ignore[no-untyped-def]
                self.call_log.append({"method": "submit_spot_ioc", "pair": pair, "cloid": cloid,
                                      "is_buy": is_buy, "sz": sz, "slippage": slippage,
                                      "pending_snapshot": self._snapshot_pending_cloids()})
                raise OrderRejected("HYPE spot rejected")

            def submit_ioc(self, coin, is_buy, sz, slippage, cloid):  # type: ignore[no-untyped-def]
                self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid,
                                      "is_buy": is_buy, "sz": sz, "slippage": slippage,
                                      "pending_snapshot": self._snapshot_pending_cloids()})
                return _ok_resp(0.5)

        client = SelectiveSpotReject(ledger)

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="x", as_of="2026-06-01", now_ts="T",
            spot_routing={"HYPE": "@107"},
        )

        assert len(report.results) == 2
        hype_r = next(r for r in report.results if r.coin == "HYPE")
        btc_r = next(r for r in report.results if r.coin == "BTC")
        assert hype_r.status == "error", "spot rejection → error"
        assert btc_r.status == "filled", "perp coin still processed (isolation intact)"


class TestSpotRestingAnomaly:
    """A spot submit returning a resting status → cancel_by_cloid on the spot pair."""

    def test_spot_resting_triggers_cancel_then_fills(self, tmp_path: Any) -> None:
        delta = 3.0
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", delta, delta, "buy")])
        # attempt 0 rests on spot; attempt 1 fills the full residual
        client = FakeClientWithCancel(
            ledger,
            spot_responses=[_resting_resp(), _ok_resp(delta)],
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
            spot_routing={"HYPE": "@107"},
        )

        # cancel must have been called exactly once, targeting the SPOT PAIR.
        assert len(client.cancel_by_cloid_calls) == 1, (
            f"expected 1 cancel, got {client.cancel_by_cloid_calls}"
        )
        cancelled_target = client.cancel_by_cloid_calls[0][0]
        assert cancelled_target == "@107", "spot resting cancel must target the spot pair"
        # no double-submit beyond the resting + clean retry
        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(spot_calls) == 2
        r = report.results[0]
        assert r.status == "filled"
        assert r.resting_anomaly is True
        # resting size NOT counted as filled
        assert abs(r.filled_size - delta) < 1e-9
        assert report.n_resting_anomaly == 1

    def test_spot_resting_cancel_failure_aborts_no_resubmit(self, tmp_path: Any) -> None:
        """Spot double-exposure guard: cancel failure → NO re-submit on spot."""
        delta = 3.0
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", delta, delta, "buy")])
        client = FakeClientWithCancel(
            ledger,
            spot_responses=[_resting_resp(), _ok_resp(delta)],
            cancel_behavior=RuntimeError("network error on spot cancel"),
        )

        report = execute(
            plan, client, _cfg(), ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
            max_attempts=3,
            spot_routing={"HYPE": "@107"},
        )

        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(spot_calls) == 1, (
            "DOUBLE-EXPOSURE BUG: re-submitted on spot after a cancel failure"
        )
        r = report.results[0]
        assert r.status == "error"
        assert r.resting_anomaly is True
        assert r.filled_size < 1e-9
        assert r.error is not None and "double-exposure guard" in r.error
        assert report.n_resting_anomaly == 1


# ---------------------------------------------------------------------------
# Pre-trade liquidity gate (Task 5): defer a leg when the OPPOSING L2 side has
# insufficient reachable depth within the slippage band.
#
# These tests use FakeHLClient (from _fakes) — a full duck-typed HL client that
# exposes l2_book(coin) + marks() — and set retry.liquidity_safety > 0 so the
# gate is ARMED (the other executor tests keep it at 0.0 = OFF).
# ---------------------------------------------------------------------------


class TestLiquidityGate:
    def test_executor_defers_leg_with_empty_opposing_book(self, tmp_path: Any) -> None:
        """HYPE buy, mark 64.85, asks EMPTY → leg deferred (not error, not filled)."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", target_size=2.41, delta_size=2.41, side="buy")])
        cfg = _cfg(slippage=0.01, liquidity_safety=1.0)
        client = FakeHLClient(
            marks={"HYPE": 64.85},
            l2={"HYPE": {"bids": [(64.7, 100.0)], "asks": []}},
        )

        report = execute(
            plan, client, cfg, ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
        )

        assert report.n_liquidity_deferred == 1
        assert report.n_error == 0
        assert report.n_filled == 0
        r = report.results[0]
        assert r.coin == "HYPE"
        assert r.liquidity_deferred is True
        # A deferred leg never submits → no IOC was sent, ledger left clean.
        assert client.submit_ioc_calls == []
        assert ledger.pending() == []

    def test_executor_fires_when_opposing_depth_sufficient(self, tmp_path: Any) -> None:
        """asks carry 10 HYPE within 1 % of mark → the leg fires & fills."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", target_size=2.41, delta_size=2.41, side="buy")])
        cfg = _cfg(slippage=0.01, liquidity_safety=1.0)
        client = FakeHLClient(
            marks={"HYPE": 64.85},
            l2={"HYPE": {"bids": [], "asks": [(64.9, 10.0)]}},
        )

        report = execute(
            plan, client, cfg, ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
        )

        assert report.n_liquidity_deferred == 0
        assert report.n_filled == 1
        r = report.results[0]
        assert r.liquidity_deferred is False
        assert r.status == "filled"
        assert len(client.submit_ioc_calls) == 1

    def test_executor_defers_leg_when_coin_missing_from_marks(self, tmp_path: Any) -> None:
        """A plan coin ABSENT from the marks snapshot (new-listing / delist race)
        must be DEFERRED — NOT crash the cycle and NOT blind-submit.  Per-coin
        isolation: a missing mark strands only that one leg."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        plan = _plan_with([_order_plan("HYPE", target_size=2.41, delta_size=2.41, side="buy")])
        cfg = _cfg(slippage=0.01, liquidity_safety=1.0)
        # marks is EMPTY → HYPE has no mark; a deep book exists but is irrelevant.
        client = FakeHLClient(
            marks={},
            l2={"HYPE": {"bids": [], "asks": [(64.9, 100.0)]}},
        )

        report = execute(
            plan, client, cfg, ledger,
            strategy_id="haarp", as_of="2026-06-01", now_ts="T",
        )

        assert report.n_liquidity_deferred == 1
        assert report.n_error == 0
        assert report.n_filled == 0
        r = report.results[0]
        assert r.liquidity_deferred is True
        # No mark → no submit (do NOT blind-submit on missing data).
        assert client.submit_ioc_calls == []
        assert ledger.pending() == []
