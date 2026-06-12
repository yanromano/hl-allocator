"""
Tests for automation.runtime.rebalance.

Coverage:
- rebalance fires (bootstrap state) → executor invoked → state committed:
  state.last_rebalanced_target == ta.weights, state file saved.
- hold (baseline == target, no drift) → no execute, NO commit
  (state file unchanged / baseline preserved).
- reconcile-on-boot: a pre-existing PENDING intent is resolved by the
  is_filled resolver (True → DONE) before the cycle executes.
- CycleReport fields match expectations (rebalanced, committed, reason).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

from automation.allocation.base import TargetAllocation
from automation.core.config import CapsConfig, Config, RetryConfig, SignalSourceConfig
from automation.reporting.alerts import Alert, AlertType, Severity
from automation.reporting.state import State
from automation.reporting.state import load as load_state
from automation.runtime.intent_ledger import Intent, IntentLedger
from automation.runtime.rebalance import run_cycle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    client_id: str = "test-client",
    max_notional_per_coin: float = 1_000_000,
    max_total_notional: float = 10_000_000,
    max_turnover_per_cycle: float = 5_000_000,
) -> Config:
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
            max_notional_per_coin=max_notional_per_coin,
            max_total_notional=max_total_notional,
            max_turnover_per_cycle=max_turnover_per_cycle,
            max_order_slippage=0.005,
        ),
        # Liquidity gate OFF: these rebalance tests use a FakeClient with no l2
        # book — arming the gate would defer every leg instead of filling.
        retry=RetryConfig(liquidity_safety=0.0),
    )


def _target(weights: dict[str, float], audience: str = "test-client") -> TargetAllocation:
    cash = 1.0 - sum(weights.values())
    return TargetAllocation(
        as_of=datetime.date(2026, 6, 1),
        weights=weights,
        cash=cash,
        strategy_id="haarp",
        model_rev="test",
        audience=audience,
    )


def _ok_resp(total_sz: float) -> dict[str, Any]:
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


class FakeSource:
    def __init__(self, ta: TargetAllocation) -> None:
        self._ta = ta

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        return self._ta


class FakeClient:
    """Fake client supporting both snapshot reads and order writes."""

    def __init__(
        self,
        equity: float = 100_000.0,
        positions: dict[str, float] | None = None,
        marks: dict[str, float] | None = None,
        sz_decimals: dict[str, int] | None = None,
        fill_sz: float = 0.5,
    ) -> None:
        self._equity = Decimal(str(equity))
        self._positions = {k: Decimal(str(v)) for k, v in (positions or {}).items()}
        self._marks = {k: Decimal(str(v)) for k, v in (marks or {"BTC": 50_000.0, "ETH": 2_000.0}).items()}
        self._sz_decimals = sz_decimals or {"BTC": 3, "ETH": 2}
        self._fill_sz = fill_sz
        self.call_log: list[dict] = []

    def equity(self) -> Decimal:
        return self._equity

    def positions(self) -> dict[str, Decimal]:
        return self._positions

    def marks(self) -> dict[str, Decimal]:
        return self._marks

    def sz_decimals(self) -> dict[str, int]:
        return self._sz_decimals

    def submit_ioc(self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str) -> dict:
        self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid})
        return _ok_resp(sz)

    def market_close(self, coin: str, slippage: float, cloid: str) -> dict:
        self.call_log.append({"method": "market_close", "coin": coin, "cloid": cloid})
        return _ok_resp(self._fill_sz)

    def query_by_cloid(self, cloid: str) -> dict:
        return {}

    @property
    def sub(self) -> str:
        return "0x" + "0" * 40


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRebalanceFires:
    def test_bootstrap_rebalance_commits_state(self, tmp_path: Any) -> None:
        """Bootstrap state → rebalance fires → state.last_rebalanced_target == ta.weights."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()  # bootstrap — no prior baseline
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True
        assert report.committed is True
        # State object is updated in-place
        assert state.last_rebalanced_target == weights
        assert state.last_rebalanced_as_of == "2026-06-01"
        # State file is written on disk
        assert state_path.exists()
        loaded = load_state(state_path)
        assert loaded.last_rebalanced_target == weights

    def test_rebalanced_target_is_frozen_target_not_achieved(self, tmp_path: Any) -> None:
        """State stores ta.weights (frozen NEW target), NOT achieved/live weights."""
        weights = {"BTC": 0.6, "ETH": 0.4}
        ta = _target(weights)
        source = FakeSource(ta)
        # Client has a different live position
        client = FakeClient(
            positions={"BTC": 0.1},  # very different from target
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        # Must be ta.weights exactly — not interpolated or achieved
        assert state.last_rebalanced_target == weights


class _ErrorClient(FakeClient):
    """Every submit_ioc raises OrderRejected → all coins error."""

    def submit_ioc(self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str) -> dict:
        self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid})
        from automation.core.safety import OrderRejected
        raise OrderRejected(f"{coin} rejected")


class _PartialClient(FakeClient):
    """Every submit_ioc fills only a tiny fraction → all coins partial."""

    def submit_ioc(self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str) -> dict:
        self.call_log.append({"method": "submit_ioc", "coin": coin, "cloid": cloid})
        return _ok_resp(sz * 0.1)  # 10% fill, never completes


class TestCommitGate:
    """F-1: state commit is gated on CLEAN execution (no errors, no partials, not aborted).

    The bootstrap target (BTC 0.5 / ETH 0.5, equity 100k, marks 50k/2k) produces
    two BUY orders (no prior positions).  The fill behaviour of the fake client
    determines whether the cycle is clean.
    """

    def test_all_error_does_not_commit(self, tmp_path: Any) -> None:
        """All orders OrderRejected → n_error>0 → NOT committed, baseline frozen."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _ErrorClient()
        state = State()  # bootstrap
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True
        assert report.exec_report is not None
        assert report.exec_report.n_error == 2
        assert report.committed is False
        # Baseline must stay frozen (None — bootstrap was never achieved)
        assert state.last_rebalanced_target is None
        # State file IS written (freshness ts saved after good signal) but
        # last_rebalanced_target must NOT be updated — only committed=True writes that.
        loaded = load_state(state_path)
        assert loaded.last_rebalanced_target is None
        assert loaded.last_successful_signal_ts == "2026-06-01T00:00:00Z"

    def test_partial_fill_does_not_commit(self, tmp_path: Any) -> None:
        """One+ coins partially filled → n_partial>0 → NOT committed."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _PartialClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True
        assert report.exec_report is not None
        assert report.exec_report.n_partial >= 1
        assert report.committed is False
        assert state.last_rebalanced_target is None
        # State file IS written (freshness ts saved after good signal) but
        # last_rebalanced_target must NOT be updated — only committed=True writes that.
        loaded = load_state(state_path)
        assert loaded.last_rebalanced_target is None
        assert loaded.last_successful_signal_ts == "2026-06-01T00:00:00Z"

    def test_clean_fill_commits(self, tmp_path: Any) -> None:
        """All coins fully filled → clean → committed."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()  # default fully fills
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.exec_report is not None
        assert report.exec_report.n_error == 0
        assert report.exec_report.n_partial == 0
        assert report.committed is True
        assert state.last_rebalanced_target == weights

    def test_clean_fill_with_harmless_skip_still_commits(self, tmp_path: Any) -> None:
        """A skipped coin (below_min_notional) does NOT block the commit (F-1).

        We craft a target where one coin's delta is below the $10 min-notional so
        the sizer records it in sp.skipped (NOT in orders).  The remaining coin
        fills cleanly → n_error==0, n_partial==0 → committed.  The skip lives in
        the plan, never reaches exec_report, so it can't block commit.

        With safety_buffer=0.03 the ETH target after scale-down is
        0.5 * 100k * 0.97 / 2000 = 24.25 ETH.  Seeding the live ETH position at
        24.25 makes the ETH delta ≈ 0 → below $10 → skipped.  BTC has no position
        so it gets a full buy that fills cleanly.
        """
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        # ETH already at the post-scale-down target → delta ≈ 0 → skipped.
        client = FakeClient(
            positions={"ETH": 24.25},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State()  # bootstrap → rebalance fires
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True
        assert report.exec_report is not None
        # ETH was skipped in the sizer (below_min_notional) so it never reached the
        # executor — only BTC has a CoinResult.  The skip can't block the commit.
        exec_coins = {r.coin for r in report.exec_report.results}
        assert exec_coins == {"BTC"}, (
            f"Only BTC should reach the executor (ETH skipped); got {exec_coins}"
        )
        assert report.exec_report.n_error == 0
        assert report.exec_report.n_partial == 0
        assert report.committed is True
        assert state.last_rebalanced_target == weights

    def test_cap_breach_aborts_and_does_not_commit(self, tmp_path: Any) -> None:
        """F-5: a pre-flight CapBreach aborts the cycle → NOT committed, zero submits."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        # End-state notional ≈ 97k (after safety_buffer scale-down). Cap total to 10k.
        cfg = _cfg(max_total_notional=10_000)

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.exec_report is not None
        assert report.exec_report.aborted is True
        assert report.exec_report.n_submitted == 0
        assert report.committed is False
        assert state.last_rebalanced_target is None
        order_calls = [c for c in client.call_log if c["method"] in ("submit_ioc", "market_close")]
        assert len(order_calls) == 0


class TestHoldNoop:
    """Hold-cycle tests.

    A genuine hold requires both:
    1. last_rebalanced_target == new target (max_delta < threshold).
    2. Live positions match the target weights (abs_drift < backstop).

    With equity=100k, BTC mark=50k, ETH mark=2k and target BTC=0.5/ETH=0.5:
      BTC target size = 0.5 * 100k / 50k = 1.0 BTC  → weight = 1*50k/100k = 0.5
      ETH target size = 0.5 * 100k /  2k = 25 ETH   → weight = 25*2k/100k = 0.5
    """

    @staticmethod
    def _hold_client() -> FakeClient:
        return FakeClient(
            positions={"BTC": 1.0, "ETH": 25.0},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )

    def test_hold_does_not_commit_state(self, tmp_path: Any) -> None:
        """Hold cycle → no execute, NO commit: last_rebalanced_target unchanged.

        NOTE: the state file IS written (to persist last_successful_signal_ts),
        but ``last_rebalanced_target`` and ``committed`` must remain unchanged.
        The pre-N-3 invariant "state file mtime unchanged on hold" no longer holds
        because step 4 now saves the freshness clock unconditionally after a good
        signal.  The important contract is: baseline NOT overwritten, committed=False.
        """
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = self._hold_client()
        state = State(
            last_rebalanced_target=weights,
            last_rebalanced_as_of="2026-06-01",
        )
        state_path = tmp_path / "state.json"
        from automation.reporting.state import save as save_state
        save_state(state, state_path)

        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is False
        assert report.committed is False
        # The critical invariant: baseline NOT overwritten
        loaded = load_state(state_path)
        assert loaded.last_rebalanced_target == weights

    def test_hold_baseline_unchanged(self, tmp_path: Any) -> None:
        """After a hold cycle, the baseline in state must stay exactly as before."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = self._hold_client()
        original_baseline = dict(weights)
        state = State(
            last_rebalanced_target=original_baseline,
            last_rebalanced_as_of="2026-06-01",
        )
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert state.last_rebalanced_target == original_baseline

    def test_hold_no_order_calls(self, tmp_path: Any) -> None:
        """Hold cycle produces zero order submissions."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = self._hold_client()
        state = State(last_rebalanced_target=weights, last_rebalanced_as_of="2026-06-01")
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        order_calls = [c for c in client.call_log if c["method"] in ("submit_ioc", "market_close")]
        assert len(order_calls) == 0


class TestBootReconciliation:
    def test_pending_intent_resolved_by_true_resolver(self, tmp_path: Any) -> None:
        """A pre-existing PENDING intent is marked DONE when is_filled returns True."""
        # Pre-populate ledger with a PENDING intent
        ledger = IntentLedger(tmp_path / "ledger.json")
        pending_cloid = "0x" + "a" * 32
        ledger.record_pending(
            Intent(
                cloid=pending_cloid,
                as_of="2026-05-31",
                coin="BTC",
                target_w=0.5,
                attempt_seq=0,
                status="PENDING",
                ts="2026-05-31T00:00:00Z",
            )
        )
        assert len(ledger.pending()) == 1

        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"

        # is_filled returns True for all intents → should mark the stale one DONE
        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            is_filled=lambda _: True,
        )

        # After reconciliation, the stale intent should be DONE
        assert len(ledger.pending()) == 0 or all(
            i.cloid != pending_cloid for i in ledger.pending()
        ), "The pre-existing PENDING intent should have been resolved to DONE"

    def test_pending_stays_pending_when_resolver_false(self, tmp_path: Any) -> None:
        """When is_filled returns False, PENDING intent stays PENDING."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        pending_cloid = "0x" + "b" * 32
        ledger.record_pending(
            Intent(
                cloid=pending_cloid,
                as_of="2026-05-31",
                coin="ETH",
                target_w=0.5,
                attempt_seq=0,
                status="PENDING",
                ts="2026-05-31T00:00:00Z",
            )
        )

        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"

        # is_filled returns False → intent stays PENDING
        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            is_filled=lambda _: False,
        )

        # The stale intent must still be PENDING
        pending_cloids = {i.cloid for i in ledger.pending()}
        # Note: the cycle itself adds new PENDING intents (then marks them DONE)
        # but our stale one with the old cloid should remain PENDING
        assert pending_cloid in pending_cloids, (
            f"Stale intent {pending_cloid!r} should still be PENDING "
            f"(is_filled=False).  Pending: {pending_cloids}"
        )

    def test_unresolved_pending_in_report(self, tmp_path: Any) -> None:
        """Unresolved intents are reported in CycleReport.unresolved_pending."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        pending_cloid = "0x" + "c" * 32
        ledger.record_pending(
            Intent(
                cloid=pending_cloid,
                as_of="2026-05-31",
                coin="SOL",
                target_w=0.3,
                attempt_seq=0,
                status="PENDING",
                ts="2026-05-31T00:00:00Z",
            )
        )

        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            is_filled=lambda _: False,
        )

        unresolved_cloids = {i.cloid for i in report.unresolved_pending}
        assert pending_cloid in unresolved_cloids


class TestCycleReportFields:
    def test_rebalance_cycle_report_fields(self, tmp_path: Any) -> None:
        """CycleReport has correct rebalanced, reason, exec_report fields."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True
        assert report.reason == "bootstrap"
        assert report.exec_report is not None

    def test_hold_cycle_report_fields(self, tmp_path: Any) -> None:
        """CycleReport for hold has exec_report=None, committed=False."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        # Positions that match the target to avoid abs-drift backstop
        client = FakeClient(
            positions={"BTC": 1.0, "ETH": 25.0},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State(last_rebalanced_target=weights, last_rebalanced_as_of="2026-06-01")
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is False
        assert report.exec_report is None
        assert report.committed is False


# ---------------------------------------------------------------------------
# Staleness / signal-freshness tests (N-3)
# ---------------------------------------------------------------------------


class _CapturingAlertSink:
    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


class _RaisingSource:
    """Source that always raises a generic exception (signal unavailable)."""

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        raise RuntimeError("signal endpoint unreachable")


class _InvalidSource:
    """Source that returns a TargetAllocation that will fail validate (weight > max_per_asset)."""

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        # max_per_asset is 0.6 in _cfg(); weight=0.9 will trigger AllocationRejected.
        return TargetAllocation(
            as_of=datetime.date(2026, 6, 1),
            weights={"BTC": 0.9},
            cash=0.1,
            strategy_id="haarp",
            model_rev="test",
            audience="test-client",
        )


def _ts(dt: datetime.datetime) -> str:
    """Return an ISO timestamp string (tz-aware UTC)."""
    return dt.isoformat()


_NOW = datetime.datetime(2026, 6, 5, 0, 15, 0, tzinfo=datetime.UTC)
_NOW_TS = _ts(_NOW)
_1D_AGO = _ts(_NOW - datetime.timedelta(days=1))
_5D_AGO = _ts(_NOW - datetime.timedelta(days=5))


class TestSignalFreshnessUpdated:
    """Good signal → freshness clock is advanced and persisted."""

    def test_good_signal_updates_last_successful_ts(self, tmp_path: Any) -> None:
        """A successful fetch+validate cycle sets state.last_successful_signal_ts."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert state.last_successful_signal_ts == _NOW_TS
        loaded = load_state(state_path)
        assert loaded.last_successful_signal_ts == _NOW_TS

    def test_good_signal_updates_ts_even_on_hold(self, tmp_path: Any) -> None:
        """Freshness clock advances even when the sizer decides to hold."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient(
            positions={"BTC": 1.0, "ETH": 25.0},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State(last_rebalanced_target=weights, last_rebalanced_as_of="2026-06-01")
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.rebalanced is False  # sizer held
        assert state.last_successful_signal_ts == _NOW_TS


class TestFirstCycleTs:
    """first_cycle_ts boot anchor: write-once, persisted, anchor preference."""

    def test_first_cycle_ts_set_on_first_run(self, tmp_path: Any) -> None:
        """The boot anchor is set (and persisted) on the very first run_cycle."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()  # first_cycle_ts is None
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert state.first_cycle_ts == _NOW_TS
        loaded = load_state(state_path)
        assert loaded.first_cycle_ts == _NOW_TS

    def test_first_cycle_ts_not_overwritten(self, tmp_path: Any) -> None:
        """A pre-existing first_cycle_ts must survive (write-once)."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        original = "2026-01-01T00:00:00+00:00"
        state = State(first_cycle_ts=original)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert state.first_cycle_ts == original

    def test_anchor_prefers_last_successful_over_first_cycle(self, tmp_path: Any) -> None:
        """When a good signal exists, staleness anchors on it (NOT first_cycle_ts).

        first_cycle_ts is ancient (would trigger de-risk if it were the anchor),
        but last_successful_signal_ts is recent (within window) → must HOLD.
        """
        cfg = _cfg()
        source = _RaisingSource()  # signal fails THIS cycle
        client = FakeClient(positions={"BTC": 1.0})
        ancient_boot = _ts(_NOW - datetime.timedelta(days=100))
        recent_good = _ts(_NOW - datetime.timedelta(days=1))  # within window=2
        state = State(
            first_cycle_ts=ancient_boot,
            last_successful_signal_ts=recent_good,
        )
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        # Anchored on the recent good signal → within window → HOLD, no flatten.
        assert report.derisked is False
        assert report.reason == "hold_stale_within_window"
        close_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(close_calls) == 0


class TestWithinWindowHold:
    """Signal unavailable but within max_signal_staleness_days → HOLD, WARNING."""

    def test_within_window_returns_hold_report(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient()
        # last good signal 1 day ago, window = 2 days → within window
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.rebalanced is False
        assert report.signal_ok is False
        assert report.derisked is False
        assert report.reason == "hold_stale_within_window"

    def test_within_window_emits_stale_signal_warning(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=sink,
        )

        stale = [a for a in sink.received if a.type == AlertType.STALE_SIGNAL]
        assert len(stale) >= 1
        assert stale[0].severity == Severity.WARNING

    def test_within_window_no_flatten_called(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        order_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(order_calls) == 0

    def test_within_window_baseline_unchanged(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient()
        original_target = {"BTC": 0.5, "ETH": 0.5}
        state = State(
            last_successful_signal_ts=_1D_AGO,
            last_rebalanced_target=original_target,
        )
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert state.last_rebalanced_target == original_target


class TestBeyondWindowDeRisk:
    """Signal stale beyond max_signal_staleness_days → CRITICAL alert + flatten to CASH."""

    def test_beyond_window_returns_derisk_report(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        # last good signal 5 days ago, window = 2 days → beyond window
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.signal_ok is False
        assert report.derisked is True
        assert report.committed is True
        assert report.reason == "derisk_to_cash"

    def test_beyond_window_emits_stale_signal_critical(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=sink,
        )

        stale = [a for a in sink.received if a.type == AlertType.STALE_SIGNAL]
        assert len(stale) >= 1
        assert stale[0].severity == Severity.CRITICAL

    def test_beyond_window_flatten_called(self, tmp_path: Any) -> None:
        """flatten_all must issue a market_close for every non-zero position."""
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0, "ETH": 5.0})
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        close_calls = [c for c in client.call_log if c["method"] == "market_close"]
        # One close per non-zero position
        closed_coins = {c["coin"] for c in close_calls}
        assert "BTC" in closed_coins
        assert "ETH" in closed_coins

    def test_beyond_window_baseline_reset_to_empty(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        state = State(
            last_successful_signal_ts=_5D_AGO,
            last_rebalanced_target={"BTC": 0.5, "ETH": 0.5},
        )
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert state.last_rebalanced_target == {}
        # Must be persisted
        loaded = load_state(state_path)
        assert loaded.last_rebalanced_target == {}


class TestInvalidSignal:
    """AllocationRejected (S-11) → INVALID_SIGNAL alert, freshness NOT updated."""

    def test_invalid_signal_emits_invalid_signal_alert(self, tmp_path: Any) -> None:
        source = _InvalidSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=sink,
        )

        invalid = [a for a in sink.received if a.type == AlertType.INVALID_SIGNAL]
        assert len(invalid) >= 1
        # not STALE_SIGNAL
        stale = [a for a in sink.received if a.type == AlertType.STALE_SIGNAL]
        assert len(stale) == 0

    def test_invalid_signal_freshness_not_updated(self, tmp_path: Any) -> None:
        source = _InvalidSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        # Freshness clock must NOT advance — this was NOT a good signal.
        assert state.last_successful_signal_ts == _1D_AGO

    def test_invalid_signal_within_window_holds(self, tmp_path: Any) -> None:
        """AllocationRejected + within window → HOLD, not de-risk."""
        source = _InvalidSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.signal_ok is False
        assert report.derisked is False

    def test_invalid_signal_beyond_window_derisks(self, tmp_path: Any) -> None:
        """AllocationRejected + beyond window → de-risk to CASH."""
        source = _InvalidSource()
        client = FakeClient(positions={"BTC": 1.0})
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=sink,
        )

        assert report.derisked is True
        invalid = [a for a in sink.received if a.type == AlertType.INVALID_SIGNAL]
        assert len(invalid) >= 1
        assert invalid[0].severity == Severity.CRITICAL


class TestNeverHadSignal:
    """Policy: never-had-good-signal is treated as STALE, anchored to first boot.

    A fresh deploy whose first signal fails HOLDs for the staleness window
    (anchor = first_cycle_ts ≈ now → age ≈ 0).  Only after the window elapses
    since first boot WITHOUT a good signal does it de-risk orphan positions.
    """

    def test_fresh_deploy_first_cycle_signal_fails_holds(self, tmp_path: Any) -> None:
        """First-ever cycle (first_cycle_ts just set = now) + signal fails → HOLD.

        Replaces the old "never de-risk" behaviour: anchor = first_cycle_ts ≈ now
        → age ≈ 0 ≤ window → HOLD, NOT an immediate flatten on a transient blip.
        """
        source = _RaisingSource()
        client = FakeClient()
        state = State()  # both freshness AND first_cycle_ts are None
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.signal_ok is False
        assert report.derisked is False
        assert report.reason == "hold_stale_within_window"
        # first_cycle_ts was set on this very first run
        assert state.first_cycle_ts == _NOW_TS

    def test_fresh_deploy_first_cycle_no_flatten(self, tmp_path: Any) -> None:
        """First cycle + signal fails + open orphan position → still HOLD (no flatten)."""
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        close_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(close_calls) == 0

    def test_fresh_deploy_first_cycle_emits_warning(self, tmp_path: Any) -> None:
        """First cycle + signal fails → WARNING (not CRITICAL)."""
        source = _RaisingSource()
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=sink,
        )

        stale = [a for a in sink.received if a.type == AlertType.STALE_SIGNAL]
        assert len(stale) >= 1
        assert stale[0].severity == Severity.WARNING

    def test_never_had_signal_beyond_window_derisks(self, tmp_path: Any) -> None:
        """first_cycle_ts = now − (window+1d), never a good signal, signal fails →
        DE-RISK to cash (flatten orphan positions, baseline {}, CRITICAL)."""
        cfg = _cfg()
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        # First boot was (window + 1) days ago and a good signal NEVER arrived.
        old_boot = _ts(
            _NOW - datetime.timedelta(days=cfg.max_signal_staleness_days + 1)
        )
        state = State(first_cycle_ts=old_boot)  # last_successful_signal_ts still None
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=sink,
        )

        assert report.signal_ok is False
        assert report.derisked is True
        assert report.reason == "derisk_to_cash"
        # flatten issued a close for the orphan BTC position
        close_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(close_calls) >= 1
        # baseline reset to cash
        assert state.last_rebalanced_target == {}
        # CRITICAL stale alert
        stale = [a for a in sink.received if a.type == AlertType.STALE_SIGNAL]
        assert len(stale) >= 1
        assert stale[0].severity == Severity.CRITICAL

    def test_never_had_signal_beyond_window_first_cycle_ts_preserved(
        self, tmp_path: Any
    ) -> None:
        """first_cycle_ts must NOT be overwritten on a subsequent cycle."""
        cfg = _cfg()
        source = _RaisingSource()
        client = FakeClient()
        old_boot = _ts(
            _NOW - datetime.timedelta(days=cfg.max_signal_staleness_days + 1)
        )
        state = State(first_cycle_ts=old_boot)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        # The original boot anchor is preserved (write-once).
        assert state.first_cycle_ts == old_boot


class TestNoAlertSink:
    """All staleness paths must not crash when alert_sink=None."""

    def test_within_window_no_sink_no_crash(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        # Must not raise
        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=None,
        )
        assert report.signal_ok is False

    def test_beyond_window_no_sink_no_crash(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=None,
        )
        assert report.derisked is True

    def test_never_had_signal_no_sink_no_crash(self, tmp_path: Any) -> None:
        source = _RaisingSource()
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=None,
        )
        assert report.signal_ok is False


class TestStalenessBoundary:
    """H2: pin the strict-``>`` boundary at EXACTLY max_signal_staleness_days.

    These tests have a ~1-second margin around the boundary, so a future
    ``>`` → ``>=`` flip in ``_handle_no_good_signal`` would flip the verdict and
    be caught here (the 1d/5d window tests have a 2-day margin and would not).
    """

    def test_exactly_at_window_holds(self, tmp_path: Any) -> None:
        """age_days == max_signal_staleness_days EXACTLY → HOLD (NOT > window)."""
        cfg = _cfg()
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        # last good signal exactly `window` days ago → age == window → NOT > window
        exact = _ts(_NOW - datetime.timedelta(days=cfg.max_signal_staleness_days))
        state = State(last_successful_signal_ts=exact)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is False
        assert report.reason == "hold_stale_within_window"
        # No flatten on the exact boundary
        close_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(close_calls) == 0

    def test_just_over_window_derisks(self, tmp_path: Any) -> None:
        """age_days just over max_signal_staleness_days (by 1s) → DE-RISK (flatten)."""
        cfg = _cfg()
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        # last good signal `window` days + 1 second ago → age > window
        just_over = _ts(
            _NOW
            - datetime.timedelta(days=cfg.max_signal_staleness_days)
            - datetime.timedelta(seconds=1)
        )
        state = State(last_successful_signal_ts=just_over)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is True
        assert report.reason == "derisk_to_cash"
        # flatten must have issued a close for the open BTC position
        close_calls = [c for c in client.call_log if c["method"] == "market_close"]
        assert len(close_calls) >= 1


class TestNaiveTimestampRobustness:
    """H1: a NAIVE last_successful_signal_ts must NOT raise TypeError."""

    def test_naive_last_ts_does_not_raise(self, tmp_path: Any) -> None:
        """Legacy naive last_successful_signal_ts is coerced to UTC, no TypeError."""
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        # NAIVE (no tzinfo) timestamp 5 days before _NOW → must still de-risk
        naive_5d_ago = (_NOW - datetime.timedelta(days=5)).replace(tzinfo=None).isoformat()
        assert "+" not in naive_5d_ago and "Z" not in naive_5d_ago  # confirm naive
        state = State(last_successful_signal_ts=naive_5d_ago)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        # Must not raise TypeError (aware now_ts − naive last_ts)
        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is True

    def test_naive_now_ts_does_not_raise(self, tmp_path: Any) -> None:
        """A naive now_ts against an aware last_ts must not raise either."""
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0})
        # aware last_ts, naive now_ts
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        naive_now = _NOW.replace(tzinfo=None).isoformat()

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=naive_now,
        )

        assert report.derisked is True


class TestDeriskIncomplete:
    """H3: surface a partial kill via CycleReport.derisk_incomplete + audit."""

    def test_clean_kill_sets_incomplete_false(self, tmp_path: Any) -> None:
        """When flatten_all fully closes every position, derisk_incomplete=False."""
        source = _RaisingSource()
        # fill_sz=1.0 fully covers the BTC=1.0 position → status "filled"
        client = FakeClient(positions={"BTC": 1.0}, fill_sz=1.0)
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is True
        assert report.derisk_incomplete is False

    def test_partial_kill_sets_incomplete_true(self, tmp_path: Any) -> None:
        """When a position only partially closes, derisk_incomplete=True."""
        source = _RaisingSource()
        # fill_sz=0.5 < BTC position 1.0 → status "partial" → incomplete kill
        client = FakeClient(positions={"BTC": 1.0}, fill_sz=0.5)
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is True
        assert report.derisk_incomplete is True

    def test_partial_kill_no_sink_still_flagged(self, tmp_path: Any) -> None:
        """H3 core: with alert_sink=None a partial kill is still flagged on the report."""
        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0}, fill_sz=0.5)
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
            alert_sink=None,
        )

        assert report.derisk_incomplete is True

    def test_flat_account_derisk_incomplete_false(self, tmp_path: Any) -> None:
        """De-risk with no open positions → flatten is a no-op → incomplete=False."""
        source = _RaisingSource()
        client = FakeClient(positions={})  # already flat
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is True
        assert report.derisk_incomplete is False

    def test_derisk_incomplete_in_audit_record(self, tmp_path: Any) -> None:
        """The audit record carries derisk_incomplete for a partial kill."""
        from automation.reporting.audit import cycle_record

        source = _RaisingSource()
        client = FakeClient(positions={"BTC": 1.0}, fill_sz=0.5)
        state = State(last_successful_signal_ts=_5D_AGO)
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        rec = cycle_record(ts=_NOW_TS, as_of="2026-06-05", report=report)
        assert "derisk_incomplete" in rec
        assert rec["derisk_incomplete"] is True

    def test_non_derisk_report_incomplete_false(self, tmp_path: Any) -> None:
        """A normal (non-derisk) hold report has derisk_incomplete=False by default."""
        from automation.reporting.audit import cycle_record

        source = _RaisingSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)  # within window → hold
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is False
        assert report.derisk_incomplete is False
        rec = cycle_record(ts=_NOW_TS, as_of="2026-06-05", report=report)
        assert rec["derisk_incomplete"] is False


# ---------------------------------------------------------------------------
# Data-age FORENSIC STAMP tests (observability only, NO alert)
# ---------------------------------------------------------------------------


class TestDataAgeStamp:
    """The good-signal path stamps ``data_age_days = (now.date() - ta.as_of).days``
    onto the CycleReport and the audit record; the no-good-signal / de-risk path
    leaves the documented sentinel ``-1``.  NO alert is ever derived from it."""

    def test_good_signal_stamps_data_age_one_day(self, tmp_path: Any) -> None:
        """now=2026-05-18, ta.as_of=2026-05-17 → data_age_days == 1 (the normal
        closed-daily-bar lag; hand-computed)."""
        ta = TargetAllocation(
            as_of=datetime.date(2026, 5, 17),
            weights={"BTC": 0.5, "ETH": 0.5},
            cash=0.0,
            strategy_id="haarp",
            model_rev="test",
            audience="test-client",
        )
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        now_ts = "2026-05-18T00:15:00+00:00"

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=now_ts,
        )

        assert report.rebalanced is True
        assert report.data_age_days == 1

    def test_data_age_in_audit_record(self, tmp_path: Any) -> None:
        """The good-signal data_age_days appears in the audit dict."""
        from automation.reporting.audit import cycle_record

        ta = TargetAllocation(
            as_of=datetime.date(2026, 5, 17),
            weights={"BTC": 0.5, "ETH": 0.5},
            cash=0.0,
            strategy_id="haarp",
            model_rev="test",
            audience="test-client",
        )
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        now_ts = "2026-05-18T00:15:00+00:00"

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=now_ts,
        )

        rec = cycle_record(ts=now_ts, as_of="2026-05-17", report=report)
        assert "data_age_days" in rec
        assert rec["data_age_days"] == 1

    def test_good_signal_naive_now_ts_stamps_age(self, tmp_path: Any) -> None:
        """A NAIVE now_ts (no tz) is coerced to UTC and still yields the right age."""
        ta = TargetAllocation(
            as_of=datetime.date(2026, 5, 15),
            weights={"BTC": 0.5, "ETH": 0.5},
            cash=0.0,
            strategy_id="haarp",
            model_rev="test",
            audience="test-client",
        )
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        # now=2026-05-18 (naive), as_of=2026-05-15 → age 3.
        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-05-18T00:15:00",
        )

        assert report.data_age_days == 3

    def test_no_alert_emitted_for_data_age(self, tmp_path: Any) -> None:
        """A fresh good-signal cycle (age 1) emits NO data-age alert — only the
        normal cycle alerts (none here on a clean fresh bootstrap)."""
        ta = TargetAllocation(
            as_of=datetime.date(2026, 5, 17),
            weights={"BTC": 0.5, "ETH": 0.5},
            cash=0.0,
            strategy_id="haarp",
            model_rev="test",
            audience="test-client",
        )
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-05-18T00:15:00+00:00",
            alert_sink=sink,
        )

        # No STALE_SIGNAL / INVALID_SIGNAL / any data-age alert on a fresh bar.
        assert all(
            a.type not in (AlertType.STALE_SIGNAL, AlertType.INVALID_SIGNAL)
            for a in sink.received
        )

    def test_derisk_cycle_data_age_sentinel(self, tmp_path: Any) -> None:
        """De-risk (no good signal) path → data_age_days == -1 (documented sentinel)
        on the report AND the audit record."""
        from automation.reporting.audit import cycle_record

        source = _RaisingSource()  # signal unavailable → de-risk
        client = FakeClient(positions={"BTC": 1.0})
        state = State(last_successful_signal_ts=_5D_AGO)  # stale beyond window
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is True
        assert report.data_age_days == -1
        rec = cycle_record(ts=_NOW_TS, as_of="2026-06-05", report=report)
        assert rec["data_age_days"] == -1

    def test_hold_within_window_data_age_sentinel(self, tmp_path: Any) -> None:
        """Hold-stale-within-window (no good signal) path → data_age_days == -1."""
        source = _RaisingSource()
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)  # within window → hold
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts=_NOW_TS,
        )

        assert report.derisked is False
        assert report.signal_ok is False
        assert report.data_age_days == -1

    def test_default_data_age_zero_on_bare_report(self) -> None:
        """A bare CycleReport (e.g. the daemon's exception fallback) defaults
        data_age_days to 0 (additive default)."""
        from automation.runtime.rebalance import CycleReport

        rep = CycleReport(rebalanced=False, reason="exception: boom", exec_report=None)
        assert rep.data_age_days == 0


# ---------------------------------------------------------------------------
# G4a: MIN_NOTIONAL_UNALLOCATED alert tests
# ---------------------------------------------------------------------------


class TestMinNotionalUnallocated:
    """G4a: MIN_NOTIONAL_UNALLOCATED is emitted when the sizer skips a wanted
    coin for below_min_notional or zeroed_by_cap reasons."""

    def test_min_notional_emitted_when_skipped(self, tmp_path: Any) -> None:
        """A sizer skip with below_min_notional → MIN_NOTIONAL_UNALLOCATED WARNING."""
        # ETH already at the post-scale-down target → ETH delta ≈ 0 → below_min_notional.
        # equity=100k, ETH mark=2000, target ETH=0.5 → target notional = 50k,
        # target size = 50k * 0.97 / 2000 = 24.25; seeding position at 24.25 makes delta ≈ 0.
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient(
            positions={"ETH": 24.25},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State()  # bootstrap → rebalance fires
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        assert report.rebalanced is True
        assert report.n_unallocated >= 1
        min_notional_alerts = [
            a for a in sink.received
            if a.type == AlertType.MIN_NOTIONAL_UNALLOCATED
        ]
        assert len(min_notional_alerts) >= 1
        alert = min_notional_alerts[0]
        assert alert.severity == Severity.WARNING
        assert "ETH" in alert.context.get("coins", [])

    def test_min_notional_not_emitted_when_no_skipped(self, tmp_path: Any) -> None:
        """No skip → no MIN_NOTIONAL_UNALLOCATED alert."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()  # no prior positions → both coins get orders
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        min_notional_alerts = [
            a for a in sink.received
            if a.type == AlertType.MIN_NOTIONAL_UNALLOCATED
        ]
        assert len(min_notional_alerts) == 0

    def test_min_notional_not_emitted_for_unpriceable_only(self, tmp_path: Any) -> None:
        """An 'unpriceable' skip does NOT trigger MIN_NOTIONAL_UNALLOCATED."""
        # SOL has no mark → unpriceable (skipped, but not below_min_notional/zeroed_by_cap).
        weights = {"BTC": 0.5, "SOL": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        # SOL has no mark price → sizer records it as "unpriceable"
        client = FakeClient(
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 3},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        min_notional_alerts = [
            a for a in sink.received
            if a.type == AlertType.MIN_NOTIONAL_UNALLOCATED
        ]
        assert len(min_notional_alerts) == 0

    def test_n_unallocated_on_hold_cycle_when_skipped(self, tmp_path: Any) -> None:
        """n_unallocated is threaded into CycleReport on a hold cycle too."""
        # On a hold cycle, sp.skipped is empty (SizePlan.skipped is [] for hold).
        # A hold produces no orders and no skipped entries → n_unallocated==0.
        # Here we just verify the field is present and zero on a clean hold.
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient(
            positions={"BTC": 1.0, "ETH": 25.0},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State(last_rebalanced_target=weights, last_rebalanced_as_of="2026-06-01")
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is False
        assert report.n_unallocated == 0

    def test_min_notional_no_sink_no_crash(self, tmp_path: Any) -> None:
        """alert_sink=None must not crash even when a coin is skipped."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient(
            positions={"ETH": 24.25},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        # Must not raise
        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=None,
        )
        assert report.n_unallocated >= 1


# ---------------------------------------------------------------------------
# G4a: DIVERGENCE alert tests
# ---------------------------------------------------------------------------


class _TwoSnapshotClient(FakeClient):
    """Client whose positions() changes after the first call to simulate post-execute state.

    First snapshot call: no positions (so the sizer generates orders and execute() runs).
    Second snapshot call: returns the far-from-target positions (simulates divergence).
    """

    def __init__(
        self,
        post_exec_positions: dict[str, float],
        raise_on_second_snapshot: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._post_exec_positions = {
            k: __import__("decimal").Decimal(str(v))
            for k, v in post_exec_positions.items()
        }
        self._raise_on_second: bool = raise_on_second_snapshot
        self._snapshot_call_count: int = 0

    def equity(self) -> Any:
        self._snapshot_call_count += 1
        if self._raise_on_second and self._snapshot_call_count > 1:
            raise RuntimeError("simulated post-snapshot failure")
        return self._equity

    def positions(self) -> Any:
        # After the first snapshot (pre-sizer), return far-from-target positions
        if self._snapshot_call_count > 1:
            return self._post_exec_positions
        return self._positions


class TestDivergenceAlert:
    """G4a: DIVERGENCE is emitted when achieved weights after execute diverge from target."""

    def test_divergence_emitted_when_far_from_target(self, tmp_path: Any) -> None:
        """Post-execute positions far from target → DIVERGENCE WARNING or CRITICAL."""
        # Target: BTC 0.5, ETH 0.5 (equity=100k, BTC mark=50k, ETH mark=2k)
        # Post-execute: no positions (flat) → achieved weights = 0 for all
        # max divergence = max(|0.5−0|, |0.5−0|) = 0.5 = 50 pp >> 5 pp threshold
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        # Post-execute positions are empty → achieved weights = 0 → big divergence
        client = _TwoSnapshotClient(
            post_exec_positions={},  # flat after "execution"
            equity=100_000.0,
            positions={},  # initially flat → bootstrap triggers rebalance
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        assert report.rebalanced is True
        # max_dev ≈ 0.5 >> 0.05 threshold
        assert report.max_divergence > 0.05

        div_alerts = [a for a in sink.received if a.type == AlertType.DIVERGENCE]
        assert len(div_alerts) >= 1

    def test_divergence_critical_when_double_threshold(self, tmp_path: Any) -> None:
        """max_div > 2 × threshold → CRITICAL severity."""
        # threshold = 5 pp = 0.05; 2× = 0.10; divergence = 0.5 >> 0.10
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _TwoSnapshotClient(
            post_exec_positions={},
            equity=100_000.0,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        div_alerts = [a for a in sink.received if a.type == AlertType.DIVERGENCE]
        assert len(div_alerts) >= 1
        assert div_alerts[0].severity == Severity.CRITICAL

    def test_divergence_warning_at_one_threshold(self, tmp_path: Any) -> None:
        """max_div just above 1× threshold but ≤ 2× → WARNING severity."""
        # threshold=5pp=0.05; 1×<dev≤2×: set up 7 pp divergence = 0.07
        # Target BTC=0.5; achieved BTC=0.43 (weight diff=0.07); ETH=0 vs 0.5 too, but
        # we want a single coin mismatch near 7pp so let's use a simpler scenario:
        # Use divergence_alert_pct=5 (default); target BTC=0.1, ETH=0;
        # post-execute: BTC weight=0.17 → dev=0.07 > 0.05 but ≤ 0.10 → WARNING.
        #
        # Build: equity=100k, BTC mark=50k, target BTC=0.1, no ETH.
        # post_exec BTC position = 0.34 → weight = 0.34*50k/100k = 0.17; diff = 0.07
        from automation.core.config import (
            CapsConfig,
            Config,
            RetryConfig,
            SignalSourceConfig,
        )

        cfg = Config(
            env="testnet",
            subaccount_address="0x" + "0" * 40,
            signal_source=SignalSourceConfig(type="equal_weight", client_id="test-client"),
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
            divergence_alert_pct=5.0,
            retry=RetryConfig(liquidity_safety=0.0),
        )

        # Target: BTC=0.10 only (no ETH)
        cash = 0.90
        ta_local = TargetAllocation(
            as_of=datetime.date(2026, 6, 1),
            weights={"BTC": 0.10},
            cash=cash,
            strategy_id="haarp",
            model_rev="test",
            audience="test-client",
        )
        source = FakeSource(ta_local)

        client = _TwoSnapshotClient(
            post_exec_positions={"BTC": 0.34},  # weight=0.34*50k/100k=0.17; diff=|0.10-0.17|=0.07
            equity=100_000.0,
            positions={},
            marks={"BTC": 50_000.0},
            sz_decimals={"BTC": 3},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        div_alerts = [a for a in sink.received if a.type == AlertType.DIVERGENCE]
        assert len(div_alerts) >= 1
        assert div_alerts[0].severity == Severity.WARNING

    def test_divergence_not_emitted_when_close_to_target(self, tmp_path: Any) -> None:
        """Post-execute positions ≈ target → no DIVERGENCE alert."""
        # Target BTC=0.5, ETH=0.5; equity=100k; marks BTC=50k, ETH=2k.
        # Achieved: BTC = 1.0 position → weight 0.5; ETH = 25.0 → weight 0.5 → 0 dev.
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _TwoSnapshotClient(
            post_exec_positions={"BTC": 1.0, "ETH": 25.0},  # perfect match
            equity=100_000.0,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        div_alerts = [a for a in sink.received if a.type == AlertType.DIVERGENCE]
        assert len(div_alerts) == 0

    def test_post_snapshot_failure_no_crash_no_divergence(self, tmp_path: Any) -> None:
        """Post-execute snapshot raises → cycle continues, no DIVERGENCE alert, max_divergence=0."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _TwoSnapshotClient(
            post_exec_positions={},
            raise_on_second_snapshot=True,  # raises on 2nd equity() call
            equity=100_000.0,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        # Must not raise
        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=sink,
        )

        assert report.rebalanced is True
        assert report.max_divergence == 0.0
        div_alerts = [a for a in sink.received if a.type == AlertType.DIVERGENCE]
        assert len(div_alerts) == 0

    def test_divergence_no_sink_no_crash(self, tmp_path: Any) -> None:
        """alert_sink=None → no crash even with large divergence."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _TwoSnapshotClient(
            post_exec_positions={},  # flat → big divergence
            equity=100_000.0,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        # Must not raise
        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
            alert_sink=None,
        )
        assert report.max_divergence > 0.0

    def test_max_divergence_in_cycle_report(self, tmp_path: Any) -> None:
        """CycleReport.max_divergence reflects the computed deviation."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _TwoSnapshotClient(
            post_exec_positions={},  # flat → max_divergence ≈ 0.5
            equity=100_000.0,
            positions={},
            marks={"BTC": 50_000.0, "ETH": 2_000.0},
            sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        # Achieved is flat (0), target is 0.5 each → max_dev = 0.5
        assert abs(report.max_divergence - 0.5) < 0.01


# ---------------------------------------------------------------------------
# G4b MF-9: THROTTLE alert from run_cycle (deferred + mid-cycle RateLimited)
# ---------------------------------------------------------------------------


def _make_deferred_exec_report() -> Any:
    """Build an ExecReport that looks like a rate-budget defer."""
    from automation.execution.executor import ExecReport  # noqa: PLC0415
    return ExecReport(
        results=[],
        n_submitted=0,
        n_filled=0,
        n_error=0,
        n_partial=0,
        deferred=True,
        defer_reason="rate budget 3 < est 8 actions (2 orders × 4)",
    )


def _make_throttled_exec_report() -> Any:
    """Build an ExecReport with n_throttled>0 (mid-cycle RateLimited, not deferred)."""
    from automation.execution.executor import CoinResult, ExecReport  # noqa: PLC0415
    cr = CoinResult(
        coin="BTC",
        requested_size=0.5,
        filled_size=0.0,
        attempts=1,
        status="error",
        throttled=True,
    )
    return ExecReport(
        results=[cr],
        n_submitted=1,
        n_filled=0,
        n_error=1,
        n_partial=0,
        n_throttled=1,
    )


class FakeClientWithRateLimit(FakeClient):
    """FakeClient with configurable user_rate_limit() for run_cycle tests."""

    def __init__(
        self,
        rate_limit_response: dict[str, Any] | None = None,
        *,
        rate_limit_raises: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rl_response = rate_limit_response
        self._rl_raises = rate_limit_raises

    def user_rate_limit(self) -> dict[str, Any]:
        if self._rl_raises:
            raise RuntimeError("query failed")
        if self._rl_response is None:
            return {"cumVlm": "0", "nRequestsCap": 1000, "nRequestsUsed": 0}
        return self._rl_response


class TestRebalanceThrottleAlerts:
    """G4b: THROTTLE alert emission from run_cycle."""

    def test_deferred_exec_emits_throttle_warning(self, tmp_path: Any) -> None:
        """A deferred exec_report → THROTTLE WARNING emitted, committed=False,
        CycleReport.deferred=True, baseline UNCHANGED."""
        import unittest.mock as mock  # noqa: PLC0415

        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        # Patch execute() so it returns a deferred report (no real orders)
        deferred_report = _make_deferred_exec_report()
        with mock.patch(
            "automation.runtime.rebalance.execute",
            return_value=deferred_report,
        ):
            report = run_cycle(
                _cfg(), client, source, ledger, state, state_path,
                now_ts="2026-06-01T00:00:00Z",
                alert_sink=sink,
            )

        # CycleReport must be deferred
        assert report.deferred is True
        assert report.committed is False

        # THROTTLE WARNING must be emitted
        throttle_alerts = [a for a in sink.received if a.type == AlertType.THROTTLE]
        assert len(throttle_alerts) >= 1
        assert throttle_alerts[0].severity == Severity.WARNING
        assert "DEFERRED" in throttle_alerts[0].message

        # Baseline must be unchanged (nothing was traded)
        assert state.last_rebalanced_target is None

    def test_deferred_exec_no_divergence_snapshot(self, tmp_path: Any) -> None:
        """A deferred cycle skips the DIVERGENCE post-snapshot (nothing was traded)."""
        import unittest.mock as mock  # noqa: PLC0415

        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        deferred_report = _make_deferred_exec_report()
        with mock.patch(
            "automation.runtime.rebalance.execute",
            return_value=deferred_report,
        ):
            run_cycle(
                _cfg(), client, source, ledger, state, state_path,
                now_ts="2026-06-01T00:00:00Z",
                alert_sink=sink,
            )

        # DIVERGENCE alert must NOT be emitted (no trades → no divergence to measure)
        div_alerts = [a for a in sink.received if a.type == AlertType.DIVERGENCE]
        assert len(div_alerts) == 0

    def test_mid_cycle_throttled_emits_throttle_warning(self, tmp_path: Any) -> None:
        """n_throttled>0 (non-deferred) → THROTTLE WARNING emitted."""
        import unittest.mock as mock  # noqa: PLC0415

        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _CapturingAlertSink()

        throttled_report = _make_throttled_exec_report()
        with mock.patch(
            "automation.runtime.rebalance.execute",
            return_value=throttled_report,
        ):
            report = run_cycle(
                _cfg(), client, source, ledger, state, state_path,
                now_ts="2026-06-01T00:00:00Z",
                alert_sink=sink,
            )

        assert report.deferred is False  # not a defer — mid-cycle throttle
        throttle_alerts = [a for a in sink.received if a.type == AlertType.THROTTLE]
        assert len(throttle_alerts) >= 1
        assert throttle_alerts[0].severity == Severity.WARNING
        # Message must mention rate-limited coins
        assert "rate-limited" in throttle_alerts[0].message.lower()

    def test_deferred_default_field_on_cycle_report(self, tmp_path: Any) -> None:
        """CycleReport.deferred defaults to False (additive; existing reports unaffected)."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        # Normal clean rebalance → deferred must be False
        assert report.deferred is False


# ---------------------------------------------------------------------------
# D5: spot_routing threaded end-to-end through run_cycle
# ---------------------------------------------------------------------------


def _cfg_spot(
    spot_routing: dict[str, str],
    universe_perp_map: dict[str, str],
    client_id: str = "test-client",
) -> Config:
    """Config with spot_routing + an identity universe_perp_map (keyspace guard)."""
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
        retry=RetryConfig(liquidity_safety=0.0),
    )


class _SpotEnabledClient(FakeClient):
    """Hybrid client: perp pool + spot pool.  Records submit_ioc AND submit_spot_ioc.

    The perp pool is supplied via the FakeClient base (equity/positions/marks/
    sz_decimals).  The spot pool is supplied separately and read via the
    spot_balances/spot_marks/spot_sz_decimals methods (D1/D2 contract).
    """

    def __init__(
        self,
        *,
        perp_equity: float = 60_000.0,
        perp_positions: dict[str, float] | None = None,
        perp_marks: dict[str, float] | None = None,
        perp_sz_decimals: dict[str, int] | None = None,
        spot_balances: dict[str, float] | None = None,
        spot_marks: dict[str, float] | None = None,
        spot_sz_decimals: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            equity=perp_equity,
            positions=perp_positions,
            marks=perp_marks or {"BTC": 50_000.0},
            sz_decimals=perp_sz_decimals or {"BTC": 3},
        )
        self._spot_balances = {
            k: Decimal(str(v)) for k, v in (spot_balances or {}).items()
        }
        self._spot_marks = {
            k: Decimal(str(v)) for k, v in (spot_marks or {}).items()
        }
        self._spot_sz_decimals = spot_sz_decimals or {}

    # ----- D1/D2 spot read path ------------------------------------------
    def spot_balances(self) -> dict[str, Decimal]:
        self.call_log.append({"method": "spot_balances"})
        return self._spot_balances

    def spot_marks(self) -> dict[str, Decimal]:
        self.call_log.append({"method": "spot_marks"})
        return self._spot_marks

    def spot_sz_decimals(self) -> dict[str, int]:
        self.call_log.append({"method": "spot_sz_decimals"})
        return self._spot_sz_decimals

    # ----- D3/D4 spot write path -----------------------------------------
    def submit_spot_ioc(
        self, pair: str, is_buy: bool, sz: float, slippage: float, cloid: str
    ) -> dict:
        self.call_log.append(
            {"method": "submit_spot_ioc", "pair": pair, "is_buy": is_buy, "cloid": cloid}
        )
        return _ok_resp(sz)


class TestSpotRoutingEndToEnd:
    """D5: cfg.spot_routing is threaded reconciler→sizer→executor in run_cycle."""

    def test_hybrid_book_routes_spot_and_perp(self, tmp_path: Any) -> None:
        """End-to-end: a spot coin (HYPE) → submit_spot_ioc; a perp coin (BTC) → submit_ioc.

        Universe: BTC (perp) + HYPE (spot).  Equal-weight target = 0.5/0.5.
        Perp pool equity = 60k; spot pool = 40k free USDC → hybrid total = 100k.
        Bootstrap state → rebalance fires.  Both legs are BUYs.
        """
        weights = {"BTC": 0.5, "HYPE": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _SpotEnabledClient(
            perp_equity=60_000.0,
            perp_marks={"BTC": 50_000.0},
            perp_sz_decimals={"BTC": 3},
            spot_balances={"USDC": 40_000.0},  # spot pool is all free USDC
            spot_marks={"@107": 10.0},  # HYPE spot mark
            spot_sz_decimals={"@107": 2},
        )
        state = State()  # bootstrap
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        cfg = _cfg_spot(
            spot_routing={"HYPE": "@107"},
            universe_perp_map={"BTC": "BTC", "HYPE": "HYPE"},
        )

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True

        # The reconciler read the HYBRID snapshot: total equity = 60k perp + 40k spot.
        # If it had read perp-only (60k) the BTC size would differ — assert the spot
        # reads were actually performed (proves the hybrid path was taken).
        methods = [c["method"] for c in client.call_log]
        assert "spot_balances" in methods, "hybrid snapshot must read spot_balances"
        assert "spot_marks" in methods, "hybrid snapshot must read spot_marks"

        # HYPE routed to SPOT; BTC routed to PERP.
        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        perp_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert len(spot_calls) == 1, "HYPE must route to submit_spot_ioc"
        assert spot_calls[0]["pair"] == "@107"
        assert spot_calls[0]["is_buy"] is True
        assert len(perp_calls) == 1, "BTC must route to submit_ioc"
        assert perp_calls[0]["coin"] == "BTC"

        # Sized off the HYBRID equity (100k), not perp-only (60k):
        #   BTC notional = 0.5 * 100k = 50k → size = 50k / 50k = 1.0 BTC
        #   HYPE notional = 0.5 * 100k = 50k → size = 50k / 10 = 5000 HYPE
        assert report.exec_report is not None
        assert report.exec_report.aborted is False

    def test_perp_only_when_spot_routing_empty(self, tmp_path: Any) -> None:
        """spot_routing={} → unchanged perp-only cycle, no spot reads, no spot writes."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = _SpotEnabledClient(
            perp_equity=100_000.0,
            perp_marks={"BTC": 50_000.0, "ETH": 2_000.0},
            perp_sz_decimals={"BTC": 3, "ETH": 2},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        # Empty spot_routing → cfg.spot_routing or None == None inside run_cycle.
        cfg = _cfg(max_notional_per_coin=1_000_000)

        report = run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.rebalanced is True
        methods = [c["method"] for c in client.call_log]
        assert "spot_balances" not in methods, "perp-only cycle must NOT read spot"
        assert "spot_marks" not in methods, "perp-only cycle must NOT read spot"
        spot_calls = [c for c in client.call_log if c["method"] == "submit_spot_ioc"]
        assert len(spot_calls) == 0, "perp-only cycle must NOT submit spot orders"
        # Both BTC and ETH went to perp.
        perp_calls = [c for c in client.call_log if c["method"] == "submit_ioc"]
        assert {c["coin"] for c in perp_calls} == {"BTC", "ETH"}

    def test_divergence_snapshot_uses_hybrid_book(self, tmp_path: Any) -> None:
        """The post-execute DIVERGENCE snapshot must ALSO read the hybrid book.

        If the post-snapshot read perp-only it would understate the denominator and
        the achieved HYPE weight would be inflated → a spurious DIVERGENCE alert.
        Count the spot reads: the hybrid path is taken TWICE (pre + post execute).
        """
        weights = {"BTC": 0.5, "HYPE": 0.5}
        ta = _target(weights)
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
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        cfg = _cfg_spot(
            spot_routing={"HYPE": "@107"},
            universe_perp_map={"BTC": "BTC", "HYPE": "HYPE"},
        )

        run_cycle(
            cfg, client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        # spot_balances is read once per snapshot; pre-execute + post-execute = 2.
        n_spot_balance_reads = sum(
            1 for c in client.call_log if c["method"] == "spot_balances"
        )
        assert n_spot_balance_reads == 2, (
            "spot_balances must be read in BOTH the pre-execute and post-execute "
            f"(divergence) snapshots — got {n_spot_balance_reads}"
        )


# ---------------------------------------------------------------------------
# Task 7: CycleReport.target_weights + n_liquidity_deferred, clean-gate
# extension (liquidity-deferred leg makes the cycle NON-clean), _FrozenSource
# + retry_attempt (one bounded-retry attempt toward a frozen target).
# ---------------------------------------------------------------------------


def _cfg_liq(
    *,
    client_id: str = "test-client",
    liquidity_safety: float = 1.0,
    max_per_asset: float = 1.0,
) -> Config:
    """A cfg with the pre-trade liquidity gate ARMED (liquidity_safety > 0).

    Mirrors ``_cfg`` but turns the gate ON so a thin/empty opposing book defers
    the leg.  ``max_per_asset`` is widened to 1.0 so single-coin targets at
    weight 1.0 pass ``validate_target_allocation``.
    """
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight", client_id=client_id),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=max_per_asset,
        safety_buffer=0.03,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.01,
        ),
        retry=RetryConfig(liquidity_safety=liquidity_safety),
    )


class TestTargetWeightsAndLiquidityDeferred:
    def test_cyclereport_exposes_target_weights_and_liquidity_deferred(
        self, tmp_path: Any
    ) -> None:
        """A normal daily run_cycle carries the served target + a deferred count (0)."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        ta = _target(weights)
        source = FakeSource(ta)
        client = FakeClient()
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert isinstance(report.target_weights, dict)
        assert report.target_weights == weights
        assert report.n_liquidity_deferred == 0

    def test_clean_is_false_when_liquidity_deferred(self, tmp_path: Any) -> None:
        """Arm the gate with an EMPTY asks book for the one target coin → the
        executor DEFERS it → cycle is NOT committed (clean False)."""
        from automation.tests._fakes import FakeHLClient

        ta = _target({"HYPE": 1.0})
        source = FakeSource(ta)
        # Mark present so the gate runs the depth check; asks EMPTY → buy depth 0
        # < required (liquidity_safety=1.0) → leg deferred.
        client = FakeHLClient(
            equity=50_000.0,
            marks={"HYPE": 64.85},
            sz_decimals={"HYPE": 2},
            l2={"HYPE": {"bids": [(64.7, 100.0)], "asks": []}},
        )
        state = State()
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            _cfg_liq(), client, source, ledger, state, state_path,
            now_ts="2026-06-01T00:00:00Z",
        )

        assert report.exec_report is not None
        assert report.n_liquidity_deferred == 1
        assert report.exec_report.n_liquidity_deferred == 1
        assert report.committed is False
        # The frozen baseline must NOT advance — the residual is left for retry.
        assert state.last_rebalanced_target is None
        # The served target is still exposed for the daemon to stash.
        assert report.target_weights == {"HYPE": 1.0}


class TestRetryAttempt:
    def test_retry_attempt_no_double_when_already_at_target(
        self, tmp_path: Any
    ) -> None:
        """Positions already == frozen target (achieved) → sizer holds → 0 NEW orders."""
        from automation.runtime.rebalance import retry_attempt
        from automation.tests._fakes import FakeHLClient

        # equity 50k, TON mark 5.0 → 0.5 weight == 25k notional == 5000 TON.
        client = FakeHLClient(
            equity=50_000.0,
            positions={"TON": 5000.0},
            marks={"TON": 5.0},
            sz_decimals={"TON": 2},
            l2={"TON": {"bids": [(4.99, 1e9)], "asks": [(5.01, 1e9)]}},
        )
        # Baseline frozen at the SAME target → max_delta 0; positions at target →
        # abs_drift below the backstop → sizer HOLDS (no orders).
        state = State(
            last_rebalanced_target={"TON": 0.5},
            last_rebalanced_as_of="2026-05-17",
        )
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = retry_attempt(
            cfg=_cfg_liq(),
            client=client,
            ledger=ledger,
            state=state,
            state_path=state_path,
            retry_target={"TON": 0.5},
            retry_as_of="2026-05-17",
            now_ts="2026-05-18T00:26:00+00:00",
        )

        # No NEW orders submitted.
        assert report.exec_report is None or report.exec_report.n_submitted == 0
        assert client.submit_ioc_calls == []

    def test_retry_attempt_fills_residual_leg(self, tmp_path: Any) -> None:
        """Frozen target wants a leg currently flat; book has liquidity → fills,
        clean, committed."""
        from automation.runtime.rebalance import retry_attempt
        from automation.tests._fakes import FakeHLClient

        client = FakeHLClient(
            equity=50_000.0,
            positions={},  # flat → full buy needed
            marks={"TON": 5.0},
            sz_decimals={"TON": 2},
            l2={"TON": {"bids": [], "asks": [(5.01, 1e9)]}},  # deep asks → fills
        )
        state = State()  # bootstrap → rebalance fires
        state_path = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = retry_attempt(
            cfg=_cfg_liq(),
            client=client,
            ledger=ledger,
            state=state,
            state_path=state_path,
            retry_target={"TON": 1.0},
            retry_as_of="2026-05-17",
            now_ts="2026-05-18T00:26:00+00:00",
        )

        assert report.exec_report is not None
        assert report.exec_report.n_filled == 1
        assert report.committed is True
        assert report.target_weights == {"TON": 1.0}


# ---------------------------------------------------------------------------
# Remote model_rev pin (NOGATE-protection for remote sources)
# ---------------------------------------------------------------------------


class _PinnedSource:
    """Valid-allocation source with a configurable model_rev."""

    def __init__(self, model_rev: str) -> None:
        self._model_rev = model_rev

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        return TargetAllocation(
            as_of=datetime.date(2026, 6, 1),
            weights={"BTC": 0.5},
            cash=0.5,
            strategy_id="haarp",
            model_rev=self._model_rev,
            schema_version=1,
            audience="test-client",
        )


class TestModelRevPin:
    """cfg.pinned_model_rev, when set, must match the served allocation's
    model_rev — the client-side defense that a NOGATE-TEST (ungated) signal
    can never be consumed by a deployment pinned to the production rev."""

    def _run(self, tmp_path: Any, *, pin: str | None, served: str) -> tuple[Any, _CapturingAlertSink]:
        cfg = _cfg().model_copy(update={"pinned_model_rev": pin})
        source = _PinnedSource(served)
        client = FakeClient()
        state = State(last_successful_signal_ts=_1D_AGO)
        sink = _CapturingAlertSink()
        report = run_cycle(
            cfg, client, source, IntentLedger(tmp_path / "ledger.json"),
            state, tmp_path / "state.json",
            now_ts=_NOW_TS, alert_sink=sink,
        )
        return report, sink

    def test_pin_mismatch_is_invalid_signal_hold(self, tmp_path: Any) -> None:
        report, sink = self._run(tmp_path, pin="0c6021efab4711a1", served="0c6021efab4711a1-NOGATE-TEST")
        assert report.signal_ok is False
        assert report.committed is False
        assert any(a.type == AlertType.INVALID_SIGNAL for a in sink.received)

    def test_pin_match_accepts(self, tmp_path: Any) -> None:
        report, _ = self._run(tmp_path, pin="rev-A", served="rev-A")
        assert report.signal_ok is True

    def test_no_pin_accepts_any_rev(self, tmp_path: Any) -> None:
        report, _ = self._run(tmp_path, pin=None, served="whatever-rev")
        assert report.signal_ok is True
