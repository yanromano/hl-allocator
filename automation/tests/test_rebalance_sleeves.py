"""
Tests for the MULTI-SLEEVE ``run_cycle`` rewrite (CRASH Phase 1, Task C5).

The per-sleeve fetch loop consumes N independent signed sources, combines them
into ONE synthetic combined ``TargetAllocation`` (signed, budget-scaled), and
drives the single-target sizer/executor path with the combined target + a
frozen-coin set.  These tests cover:

* two FRESH sleeves (long + short) → ONE combined execution, signed baseline,
  per-sleeve audit attribution;
* a sleeve that fails-to-fetch within its window → ``SLEEVE_HOLD`` WARNING + its
  coins FROZEN (no order, live position carried untouched);
* a sleeve past its window (or 2nd hold short) → ``SLEEVE_DERISKED`` CRITICAL +
  its coins unwind via the combined delta;
* ALL sleeves failing → the existing no-good-signal HOLD path (recheck machinery
  shape unchanged);
* the ``_FrozenSource`` retry cash = ``1 − Σ|w|`` so a signed retry validates;
* a legacy SINGULAR config through the new run_cycle behaves like today;
* the combined gross stays within the sum of the sleeve budgets.

The real legacy pin is ``test_rebalance.py`` (unmodified, all green); this module
adds the new multi-sleeve coverage with fake sources + ``FakeHLClient``.
"""

from __future__ import annotations

import datetime
from typing import Any

import pytest

from automation.allocation.base import AllocationRejected, TargetAllocation
from automation.core.config import CapsConfig, Config, RetryConfig, SignalSourceConfig
from automation.reporting.alerts import Alert, AlertType, Severity
from automation.reporting.state import State
from automation.runtime.intent_ledger import IntentLedger
from automation.runtime.rebalance import _fetch_sleeve, _FrozenSource, run_cycle
from automation.tests._fakes import FakeHLClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_sleeve_cfg(
    *,
    haarp_budget: float = 0.5,
    crash_budget: float = 0.5,
    crash_staleness: int | None = None,
) -> Config:
    """A two-sleeve config: ``haarp`` (long-only) + ``crash`` (allow_short)."""
    crash = {
        "type": "remote",
        "name": "crash",
        "budget": crash_budget,
        "allow_short": True,
        "client_id": "crash-client",
        "url": "https://crash.local",
        "root_key_fingerprint_ref": "CRASH_FP",
    }
    if crash_staleness is not None:
        crash["max_signal_staleness_days"] = crash_staleness
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_sources=[
            SignalSourceConfig(
                type="remote",
                name="haarp",
                budget=haarp_budget,
                allow_short=False,
                client_id="haarp-client",
                url="https://haarp.local",
                root_key_fingerprint_ref="HAARP_FP",
            ),
            SignalSourceConfig(**crash),
        ],
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.9,
        safety_buffer=0.03,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.005,
        ),
        # Liquidity gate OFF — FakeHLClient has no l2 book by default.
        retry=RetryConfig(liquidity_safety=0.0),
    )


def _singular_cfg(client_id: str = "test-client") -> Config:
    """A legacy SINGULAR config (one default sleeve, long-only)."""
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight", client_id=client_id),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.9,
        safety_buffer=0.03,
        universe_perp_map={},
        caps=CapsConfig(
            max_notional_per_coin=1_000_000,
            max_total_notional=10_000_000,
            max_turnover_per_cycle=5_000_000,
            max_order_slippage=0.005,
        ),
        retry=RetryConfig(liquidity_safety=0.0),
    )


class _FakeSource:
    """Serves a fixed ``TargetAllocation`` whose audience == *client_id*."""

    def __init__(
        self,
        weights: dict[str, float],
        *,
        client_id: str,
        strategy_id: str = "strat",
        as_of: datetime.date = datetime.date(2026, 6, 1),
        allow_short: bool = False,
    ) -> None:
        cash = 1.0 - sum(abs(w) for w in weights.values())
        self._ta = TargetAllocation(
            as_of=as_of,
            weights=dict(weights),
            cash=cash,
            strategy_id=strategy_id,
            model_rev="test",
            audience=client_id,
        )

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        return self._ta


class _RaisingSource:
    """A source whose fetch raises — simulates a dark sleeve."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("signal unavailable")

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        raise self._exc


class _RevSource:
    """Serves a fixed ``TargetAllocation`` with a configurable ``model_rev``."""

    def __init__(self, model_rev: str, client_id: str) -> None:
        self._ta = TargetAllocation(
            as_of=datetime.date(2026, 6, 1),
            weights={"BTC": 0.3},
            cash=0.7,
            strategy_id="strat",
            model_rev=model_rev,
            audience=client_id,
        )

    def get_target_allocation(self, as_of: Any = None) -> TargetAllocation:
        return self._ta


class TestPerSleevePin:
    """The model_rev pin is resolved PER SLEEVE: each sleeve consumes a
    different signal (HAARP NOGATE rev vs CRASH gated rev), so a single global
    pin cannot match both.  ``_fetch_sleeve`` must use
    ``cfg.effective_pinned_model_rev(sleeve)``."""

    _AS_OF = datetime.date(2026, 6, 1)

    def test_each_sleeve_accepts_its_own_rev(self) -> None:
        cfg = _two_sleeve_cfg()
        haarp, crash = cfg.sleeves()
        haarp = haarp.model_copy(update={"pinned_model_rev": "haarp-rev"})
        crash = crash.model_copy(update={"pinned_model_rev": "crash-rev"})
        fh = _fetch_sleeve(
            cfg=cfg, sleeve=haarp,
            source=_RevSource("haarp-rev", "haarp-client"), as_of=self._AS_OF,
        )
        fc = _fetch_sleeve(
            cfg=cfg, sleeve=crash,
            source=_RevSource("crash-rev", "crash-client"), as_of=self._AS_OF,
        )
        assert fh.served_ok and not fh.is_rejected
        assert fc.served_ok and not fc.is_rejected

    def test_wrong_rev_is_rejected_on_that_sleeve(self) -> None:
        cfg = _two_sleeve_cfg()
        haarp, _ = cfg.sleeves()
        haarp = haarp.model_copy(update={"pinned_model_rev": "haarp-rev"})
        # The HAARP sleeve is served the CRASH rev → mismatch → AllocationRejected.
        f = _fetch_sleeve(
            cfg=cfg, sleeve=haarp,
            source=_RevSource("crash-rev", "haarp-client"), as_of=self._AS_OF,
        )
        assert not f.served_ok and f.is_rejected

    def test_sleeve_override_beats_global_pin(self) -> None:
        # Global pin would reject the CRASH rev; the per-sleeve override rescues it.
        cfg = _two_sleeve_cfg().model_copy(update={"pinned_model_rev": "haarp-rev"})
        _, crash = cfg.sleeves()
        crash = crash.model_copy(update={"pinned_model_rev": "crash-rev"})
        f = _fetch_sleeve(
            cfg=cfg, sleeve=crash,
            source=_RevSource("crash-rev", "crash-client"), as_of=self._AS_OF,
        )
        assert f.served_ok and not f.is_rejected


def _alerts_of(sink: Any, alert_type: AlertType) -> list[Alert]:
    return [a for a in sink.alerts if a.type == alert_type]


class _RecordingSink:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)


def _marks() -> dict[str, float]:
    return {"BTC": 50_000.0, "TON": 5.0, "ETH": 2_000.0}


def _szd() -> dict[str, int]:
    return {"BTC": 3, "TON": 2, "ETH": 2}


_NOW = "2026-06-02T00:00:00+00:00"
_AS_OF = datetime.date(2026, 6, 1)


# ---------------------------------------------------------------------------
# Two FRESH sleeves → ONE combined execution
# ---------------------------------------------------------------------------


class TestTwoFreshSleeves:
    def test_two_fresh_sleeves_one_combined_execution(self, tmp_path: Any) -> None:
        """long {BTC:+0.3}@.5 + short {TON:-0.4}@.5 → combined {BTC:+0.15, TON:-0.2}.

        ONE plan; audit record carries both sleeves as fresh; committed baseline
        equals the combined SIGNED dict.
        """
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        state = State()  # bootstrap
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF,
        )

        assert report.rebalanced is True
        assert report.committed is True
        # Combined SIGNED baseline.
        assert state.last_rebalanced_target == pytest.approx(
            {"BTC": 0.15, "TON": -0.2}
        )
        assert report.target_weights == pytest.approx({"BTC": 0.15, "TON": -0.2})
        # Two orders: a BTC buy (long open) and a TON sell (short open).
        assert {c["coin"] for c in client.submit_ioc_calls} == {"BTC", "TON"}
        ton = next(c for c in client.submit_ioc_calls if c["coin"] == "TON")
        assert ton["is_buy"] is False  # short open = sell
        btc = next(c for c in client.submit_ioc_calls if c["coin"] == "BTC")
        assert btc["is_buy"] is True

    def test_audit_record_has_per_sleeve_outcomes(self, tmp_path: Any) -> None:
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        state = State()
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        assert set(report.sleeves.keys()) == {"haarp", "crash"}
        assert report.sleeves["haarp"]["outcome"] == "fresh"
        assert report.sleeves["crash"]["outcome"] == "fresh"
        assert report.sleeves["haarp"]["weights"] == {"BTC": 0.3}
        assert report.sleeves["crash"]["weights"] == {"TON": -0.4}
        assert report.sleeves["haarp"]["as_of"] == "2026-06-01"
        # The audit dict is built by cycle_record from report.sleeves.
        from automation.reporting.audit import cycle_record

        rec = cycle_record(ts=_NOW, as_of="2026-06-01", report=report)
        assert rec["sleeves"]["crash"]["outcome"] == "fresh"

    def test_per_sleeve_ledger_written_on_fresh(self, tmp_path: Any) -> None:
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        state = State()
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF)

        assert state.sleeves["haarp"]["last_weights"] == {"BTC": 0.3}
        assert state.sleeves["crash"]["last_weights"] == {"TON": -0.4}
        assert state.sleeves["haarp"]["consecutive_holds"] == 0
        assert state.sleeves["crash"]["last_success_ts"] == _NOW
        assert state.last_successful_signal_ts == _NOW


# ---------------------------------------------------------------------------
# A HELD sleeve freezes its coins
# ---------------------------------------------------------------------------


class TestSleeveHoldFreezes:
    def test_sleeve_hold_freezes_its_coins(self, tmp_path: Any) -> None:
        """Sleeve B (crash) fetch raises; it has prior last_weights {TON:-0.3} →
        SLEEVE_HOLD WARNING; TON gets NO order (frozen); sleeve A executes; the
        live TON position is untouched."""
        cfg = _two_sleeve_cfg()
        # Live TON short already on the book (the crash sleeve's prior position).
        client = FakeHLClient(
            equity=100_000.0,
            positions={"TON": -6000.0},  # -0.3 * 100k / 5 = -6000
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _RaisingSource(),
        }
        # Prior ledger: crash served {TON:-0.3} yesterday + a baseline reflecting it.
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": -0.3},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            sleeves={
                "haarp": {
                    "last_weights": {"BTC": 0.0},
                    "last_as_of": "2026-06-01",
                    "last_success_ts": "2026-06-01T00:00:00+00:00",
                    "consecutive_holds": 0,
                },
                "crash": {
                    "last_weights": {"TON": -0.3},
                    "last_as_of": "2026-06-01",
                    "last_success_ts": "2026-06-01T00:00:00+00:00",
                    "consecutive_holds": 0,
                },
            },
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _RecordingSink()

        report = run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF, alert_sink=sink,
        )

        # SLEEVE_HOLD WARNING for crash.
        holds = _alerts_of(sink, AlertType.SLEEVE_HOLD)
        assert len(holds) == 1
        assert holds[0].severity == Severity.WARNING
        assert holds[0].context["sleeve"] == "crash"
        assert "TON" in holds[0].context["frozen_coins"]
        # TON got NO order — its live short is carried untouched.
        assert all(c["coin"] != "TON" for c in client.submit_ioc_calls)
        assert all(c["coin"] != "TON" for c in client.market_close_calls)
        assert client.positions()["TON"] == pytest.approx(-6000.0)
        # haarp executed (BTC buy).
        assert any(c["coin"] == "BTC" for c in client.submit_ioc_calls)
        # Crash audit outcome is hold; its consecutive_holds incremented.
        assert report.sleeves["crash"]["outcome"] == "hold"
        assert state.sleeves["crash"]["consecutive_holds"] == 1
        # The held sleeve's last_weights are preserved.
        assert state.sleeves["crash"]["last_weights"] == {"TON": -0.3}


# ---------------------------------------------------------------------------
# A STALE sleeve de-risks its coins
# ---------------------------------------------------------------------------


class TestSleeveStaleDerisks:
    def test_sleeve_stale_derisks(self, tmp_path: Any) -> None:
        """Crash sleeve fails on its SECOND consecutive hold (short fast de-risk) →
        SLEEVE_DERISKED CRITICAL; its TON short unwinds (close order to zero)."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={"TON": -6000.0},  # crash's prior short
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _RaisingSource(),
        }
        # crash already held ONCE (consecutive_holds=1) → this fetch failure is the
        # 2nd consecutive HOLD → short fast de-risk → STALE.
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": -0.3},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            sleeves={
                "haarp": {
                    "last_weights": {"BTC": 0.0},
                    "last_as_of": "2026-06-01",
                    "last_success_ts": "2026-06-01T00:00:00+00:00",
                    "consecutive_holds": 0,
                },
                "crash": {
                    "last_weights": {"TON": -0.3},
                    "last_as_of": "2026-06-01",
                    "last_success_ts": "2026-06-01T00:00:00+00:00",
                    "consecutive_holds": 1,
                },
            },
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _RecordingSink()

        report = run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF, alert_sink=sink,
        )

        # SLEEVE_DERISKED CRITICAL for crash.
        derisks = _alerts_of(sink, AlertType.SLEEVE_DERISKED)
        assert len(derisks) == 1
        assert derisks[0].severity == Severity.CRITICAL
        assert derisks[0].context["sleeve"] == "crash"
        # TON unwinds — its target is absent → close order brings it to flat.
        ton_closed = (
            any(c["coin"] == "TON" for c in client.market_close_calls)
            or any(
                c["coin"] == "TON" and c["is_buy"] is True
                for c in client.submit_ioc_calls
            )
        )
        assert ton_closed
        assert report.sleeves["crash"]["outcome"] == "stale"


# ---------------------------------------------------------------------------
# ALL sleeves fail → no-good-signal hold path
# ---------------------------------------------------------------------------


class TestAllSleevesFail:
    def test_all_sleeves_fail_hold_path(self, tmp_path: Any) -> None:
        """Every sleeve fetch fails within the window → no-good-signal HOLD report
        (recheck machinery shape intact: reason hold, committed False, exec None)."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        sources = {
            "haarp": _RaisingSource(),
            "crash": _RaisingSource(),
        }
        # first_cycle_ts == now → age 0 → within window → HOLD (not de-risk).
        state = State(first_cycle_ts=_NOW)
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        assert report.rebalanced is False
        assert report.committed is False
        assert report.signal_ok is False
        assert report.exec_report is None
        assert report.reason == "hold_stale_within_window"
        # No orders submitted.
        assert client.submit_ioc_calls == []

    def test_all_sleeves_fail_invalid_routes_invalid_signal(self, tmp_path: Any) -> None:
        """When the first failing sleeve raised AllocationRejected → INVALID_SIGNAL."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        sources = {
            "haarp": _RaisingSource(AllocationRejected("bad signal")),
            "crash": _RaisingSource(),
        }
        state = State(first_cycle_ts=_NOW)
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _RecordingSink()

        run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF, alert_sink=sink,
        )

        assert _alerts_of(sink, AlertType.INVALID_SIGNAL)
        assert not _alerts_of(sink, AlertType.STALE_SIGNAL)


# ---------------------------------------------------------------------------
# _FrozenSource signed cash (red-team H1)
# ---------------------------------------------------------------------------


class TestFrozenSourceSignedCash:
    def test_frozen_source_signed_cash(self) -> None:
        """A frozen SIGNED target → cash = 1 − Σ|w| (so the validator accepts it)."""
        src = _FrozenSource(
            {"TON": -0.4, "BTC": 0.3}, "2026-06-01", client_id="combined"
        )
        ta = src.get_target_allocation()
        # 1 - (0.4 + 0.3) = 0.3
        assert ta.cash == pytest.approx(0.3)
        # The allow_short unity check (Σ|w| + cash == 1) holds.
        gross = sum(abs(w) for w in ta.weights.values())
        assert gross + ta.cash == pytest.approx(1.0)

    def test_frozen_source_retry_signed_validates(self, tmp_path: Any) -> None:
        """A retry whose frozen combined target is net-short validates + sizes."""
        from automation.runtime.rebalance import retry_attempt

        cfg = _two_sleeve_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        state = State()  # bootstrap → rebalance fires toward the frozen target
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = retry_attempt(
            cfg=cfg,
            client=client,
            ledger=ledger,
            state=state,
            state_path=sp,
            retry_target={"TON": -0.4},  # net-short frozen combined target
            retry_as_of="2026-06-01",
            now_ts=_NOW,
        )

        # The signed retry was validated + sized (a TON sell-to-open).
        assert report.signal_ok is True
        ton = [c for c in client.submit_ioc_calls if c["coin"] == "TON"]
        assert ton and ton[0]["is_buy"] is False

    def test_multi_sleeve_retry_does_not_write_default_sleeve(
        self, tmp_path: Any
    ) -> None:
        """C5-review cleanup: a retry on a MULTI-sleeve config drives the cycle with
        a synthetic ``"default"`` sleeve — but ``"default"`` is NOT a configured
        sleeve here (the real sleeves are ``haarp``/``crash``), so no phantom
        ``state.sleeves["default"]`` ledger record is written."""
        from automation.runtime.rebalance import retry_attempt

        cfg = _two_sleeve_cfg()  # sleeves: haarp + crash (no "default")
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        state = State()
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        retry_attempt(
            cfg=cfg,
            client=client,
            ledger=ledger,
            state=state,
            state_path=sp,
            retry_target={"TON": -0.4},
            retry_as_of="2026-06-01",
            now_ts=_NOW,
        )

        # The synthetic "default" sleeve did NOT leak into the per-sleeve ledger.
        assert "default" not in state.sleeves


# ---------------------------------------------------------------------------
# Legacy singular config unchanged
# ---------------------------------------------------------------------------


class TestLegacySingularUnchanged:
    def test_legacy_singular_config_unchanged(self, tmp_path: Any) -> None:
        """A singular cfg through the new run_cycle behaves like today: a single
        source, served weights == combined, baseline committed."""
        cfg = _singular_cfg()
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        # equal_weight over the universe is empty here, so pass an explicit source.
        source = _FakeSource({"BTC": 0.5, "ETH": 0.5}, client_id="test-client")
        state = State()
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        # NOTE: passed POSITIONALLY as a bare source — the legacy call shape.
        report = run_cycle(
            cfg, client, source, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        assert report.rebalanced is True
        assert report.committed is True
        # Single sleeve at budget 1.0 → combined == served.
        assert state.last_rebalanced_target == pytest.approx({"BTC": 0.5, "ETH": 0.5})
        assert report.target_weights == pytest.approx({"BTC": 0.5, "ETH": 0.5})
        # One "default" sleeve in the audit, fresh.
        assert set(report.sleeves.keys()) == {"default"}
        assert report.sleeves["default"]["outcome"] == "fresh"


# ---------------------------------------------------------------------------
# Combined gross within budgets
# ---------------------------------------------------------------------------


class TestFrozenCoinGrossCap:
    def test_frozen_live_notional_counts_toward_gross_cap(self, tmp_path: Any) -> None:
        """A frozen sleeve's LIVE |notional| consumes the gross cap: the traded
        sleeve is scaled to fit cap − frozen, and the combined book never exceeds
        the cap (no SizerError).  This pins the C5 sizer-note accounting."""
        from decimal import Decimal

        from automation.allocation.base import TargetAllocation
        from automation.core.config import (
            CapsConfig,
            Config,
            RetryConfig,
            SignalSourceConfig,
        )
        from automation.execution.reconciler import Snapshot
        from automation.execution.sizer import plan

        cfg = Config(
            env="testnet",
            subaccount_address="0x" + "0" * 40,
            signal_source=SignalSourceConfig(type="equal_weight", client_id="c"),
            rebalance_utc_time="00:15",
            threshold_pct=3.0,
            max_per_asset=1.0,
            safety_buffer=0.0,
            universe_perp_map={},
            caps=CapsConfig(
                max_notional_per_coin=1e9,
                max_total_notional=1e9,
                max_turnover_per_cycle=1e9,
                max_order_slippage=0.005,
            ),
            retry=RetryConfig(liquidity_safety=0.0),
        )
        # Equity 100k, cap 100k.  Frozen TON short = 50k live notional.  Traded BTC
        # 0.8 = 80k → combined 130k > cap → traded scales to cap − frozen = 50k.
        snap = Snapshot(
            equity=Decimal("100000"),
            positions={"TON": Decimal("-10000")},
            marks={"BTC": Decimal("50000"), "TON": Decimal("5")},
            sz_decimals={"BTC": 3, "TON": 2},
        )
        ta = TargetAllocation(
            as_of=datetime.date(2026, 6, 1),
            weights={"BTC": 0.8},
            cash=0.2,
            strategy_id="s",
            model_rev="m",
            audience="c",
        )
        sp = plan(
            ta,
            snap,
            State(),
            cfg,
            allow_short=True,
            frozen_coins=frozenset({"TON"}),
        )

        assert sp.scaled_down is True
        # Combined book (traded + frozen) at the cap, not over it.
        assert sp.gross_notional == pytest.approx(100_000.0)
        # BTC traded notional == cap − frozen.
        assert sp.target_sizes["BTC"] * 50_000 == pytest.approx(50_000.0)
        # Frozen TON gets NO order.
        assert all(o.coin != "TON" for o in sp.orders)

    def test_frozen_alone_over_cap_raises_sizer_error(self, tmp_path: Any) -> None:
        """C5-review cleanup: when the FROZEN book alone exceeds the cap the traded
        book scales to 0 but ``post_gross`` (= frozen) still violates the invariant
        → SizerError (fail-loud-safe), NOT a silent clamp.  The daemon catches this
        as an EXEC_ERROR rather than trading into an over-leveraged frozen book."""
        from decimal import Decimal

        from automation.allocation.base import TargetAllocation
        from automation.core.config import (
            CapsConfig,
            Config,
            RetryConfig,
            SignalSourceConfig,
        )
        from automation.execution.reconciler import Snapshot
        from automation.execution.sizer import SizerError, plan

        cfg = Config(
            env="testnet",
            subaccount_address="0x" + "0" * 40,
            signal_source=SignalSourceConfig(type="equal_weight", client_id="c"),
            rebalance_utc_time="00:15",
            threshold_pct=3.0,
            max_per_asset=1.0,
            safety_buffer=0.0,
            universe_perp_map={},
            caps=CapsConfig(
                max_notional_per_coin=1e9,
                max_total_notional=1e9,
                max_turnover_per_cycle=1e9,
                max_order_slippage=0.005,
            ),
            retry=RetryConfig(liquidity_safety=0.0),
        )
        # Equity 100k, cap 100k.  Frozen TON short alone = 150k live notional > cap.
        snap = Snapshot(
            equity=Decimal("100000"),
            positions={"TON": Decimal("-30000")},  # 30000 * 5 = 150k
            marks={"BTC": Decimal("50000"), "TON": Decimal("5")},
            sz_decimals={"BTC": 3, "TON": 2},
        )
        ta = TargetAllocation(
            as_of=datetime.date(2026, 6, 1),
            weights={"BTC": 0.3},
            cash=0.7,
            strategy_id="s",
            model_rev="m",
            audience="c",
        )
        with pytest.raises(SizerError, match="money invariant"):
            plan(
                ta,
                snap,
                State(),
                cfg,
                allow_short=True,
                frozen_coins=frozenset({"TON"}),
            )


class TestCombinedGrossWithinBudgets:
    def test_combined_gross_within_budgets(self, tmp_path: Any) -> None:
        """fresh+fresh → Σ|combined| ≤ Σ budgets (each sleeve at its full budget)."""
        cfg = _two_sleeve_cfg(haarp_budget=0.5, crash_budget=0.5)
        client = FakeHLClient(equity=100_000.0, marks=_marks(), sz_decimals=_szd())
        sources = {
            # haarp uses its full budget (Σ|w|=1.0 sleeve-relative → 0.5 scaled).
            "haarp": _FakeSource({"BTC": 0.6, "ETH": 0.4}, client_id="haarp-client"),
            # crash uses its full budget short.
            "crash": _FakeSource(
                {"TON": -0.9}, client_id="crash-client", allow_short=True
            ),
        }
        state = State()
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        report = run_cycle(
            cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        combined = report.target_weights or {}
        gross = sum(abs(w) for w in combined.values())
        total_budget = sum(s.budget for s in cfg.sleeves())
        assert gross <= total_budget + 1e-9


# ---------------------------------------------------------------------------
# C6 — post-liquidation re-entry cooldown (spec §5, red-team M3)
# ---------------------------------------------------------------------------


class TestLiquidationDetection:
    """A baseline-nonzero coin that vanished from the live snapshot with no exit
    order of ours was LIQUIDATED → POSITION_LIQUIDATED CRITICAL + arm a cooldown."""

    def test_liquidation_detected_sets_cooldown(self, tmp_path: Any) -> None:
        """Baseline {TON:-0.3}, snapshot has NO TON (vanished) → liquidation:
        POSITION_LIQUIDATED CRITICAL + ``state.liq_cooldowns["TON"] == today``."""
        cfg = _two_sleeve_cfg()
        # Snapshot has NO TON position — the short vanished (liquidated by a squeeze).
        client = FakeHLClient(
            equity=100_000.0,
            positions={},  # TON ABSENT
            marks=_marks(),
            sz_decimals=_szd(),
        )
        # Both sleeves serve fresh this cycle (so the cycle proceeds normally), but
        # the PRIOR baseline carried a TON short that is now gone from the book.
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": -0.3},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            # TON was ACTUALLY HELD last cycle (live book) — a genuine liquidation.
            last_known_positions={"TON": -3.0},
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _RecordingSink()

        run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF, alert_sink=sink,
        )

        liq = _alerts_of(sink, AlertType.POSITION_LIQUIDATED)
        assert len(liq) == 1
        assert liq[0].severity == Severity.CRITICAL
        assert liq[0].context["coin"] == "TON"
        # Cooldown armed for TON at TODAY's date (now_ts's date).
        assert state.liq_cooldowns["TON"] == "2026-06-02"

    def test_unfilled_target_not_flagged_liquidation(self, tmp_path: Any) -> None:
        """Baseline {TON:-0.3} but TON was NEVER actually held (absent from
        last_known_positions) — a committed-but-dropped target, NOT a liquidation.
        The held_last_cycle guard must NOT arm a phantom cooldown (the stranded-
        flat-account bug)."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={},  # TON absent — but it was NEVER held
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        # Baseline records TON:-0.3 (a prior cycle COMMITTED it) but the order was
        # DROPPED before filling → TON never entered the live book → NOT in
        # last_known_positions.
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": -0.3},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            last_known_positions={},  # TON NEVER held
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _RecordingSink()

        run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF, alert_sink=sink,
        )

        # NO phantom liquidation: the never-filled target must not arm a cooldown.
        assert _alerts_of(sink, AlertType.POSITION_LIQUIDATED) == []
        assert "TON" not in state.liq_cooldowns

    def test_last_known_positions_recorded_from_snapshot(self, tmp_path: Any) -> None:
        """run_cycle records the live book into state.last_known_positions for the
        NEXT cycle's liquidation detection (perp → szi, zeros dropped)."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={"BTC": 0.5, "ETH": 0.0},  # BTC held, ETH flat
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource({}, client_id="crash-client", allow_short=True),
        }
        state = State(first_cycle_ts="2026-06-01T00:00:00+00:00")
        run_cycle(
            cfg, client, sources, IntentLedger(tmp_path / "ledger.json"),
            state, tmp_path / "state.json", now_ts=_NOW, as_of=_AS_OF,
            alert_sink=_RecordingSink(),
        )
        assert state.last_known_positions == {"BTC": 0.5}  # ETH (0.0) dropped

    def test_normal_exit_not_flagged_liquidation(self, tmp_path: Any) -> None:
        """We targeted TON→0 last cycle and it filled (so baseline records TON:0.0);
        TON is absent from the snapshot because WE closed it → NOT a liquidation."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={},  # TON absent — but because WE closed it (baseline 0.0)
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        # Baseline records TON:0.0 — last cycle we INTENTIONALLY targeted it to flat.
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": 0.0},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")
        sink = _RecordingSink()

        run_cycle(
            cfg, client, sources, ledger, state, sp,
            now_ts=_NOW, as_of=_AS_OF, alert_sink=sink,
        )

        assert _alerts_of(sink, AlertType.POSITION_LIQUIDATED) == []
        assert "TON" not in state.liq_cooldowns


class TestCooldownSkip:
    """A coin in cooldown skips SAME-direction re-entries (sizer.plan)."""

    def _crash_target(self, ton_weight: float) -> TargetAllocation:
        return TargetAllocation(
            as_of=_AS_OF,
            weights={"TON": ton_weight},
            cash=1.0 - abs(ton_weight),
            strategy_id="crash",
            model_rev="test",
            audience="combined",
        )

    def test_cooldown_blocks_same_direction_reentry(self, tmp_path: Any) -> None:
        """A cooldown-blocked coin (run_cycle decided SAME-direction) gets NO order
        (skipped with reason ``cooldown``).  run_cycle owns the direction decision;
        the sizer just skips every coin in the blocked frozenset."""
        from decimal import Decimal

        from automation.execution.reconciler import Snapshot
        from automation.execution.sizer import plan

        cfg = _two_sleeve_cfg()
        snap = Snapshot(
            equity=Decimal("100000"),
            positions={},  # flat (liquidated yesterday)
            marks={"TON": Decimal("5"), "BTC": Decimal("50000")},
            sz_decimals={"TON": 2, "BTC": 3},
        )
        state = State(last_rebalanced_target={"TON": -0.3})
        # New target re-enters SHORT TON; run_cycle flagged TON as cooldown-blocked.
        sp = plan(
            self._crash_target(-0.4),
            snap,
            state,
            cfg,
            allow_short=True,
            cooldown_blocked=frozenset({"TON"}),
        )
        assert all(o.coin != "TON" for o in sp.orders)
        cooldown_skips = [s for s in sp.skipped if s["reason"] == "cooldown"]
        assert any(s["coin"] == "TON" for s in cooldown_skips)

    def test_cooldown_allows_opposite_direction(self, tmp_path: Any) -> None:
        """TON in cooldown from a SHORT; new combined target is LONG TON → run_cycle
        does NOT add TON to the blocked set → order allowed (squeeze flipped thesis).
        This exercises the run_cycle direction logic end-to-end."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={},  # flat after the short was liquidated
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            # crash now wants LONG TON (the opposite of the liquidated short).
            "crash": _FakeSource(
                {"TON": 0.4}, client_id="crash-client", allow_short=True
            ),
        }
        # Baseline records the liquidated SHORT; cooldown armed today.
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": -0.3},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            liq_cooldowns={"TON": "2026-06-02"},
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        # Opposite-direction LONG TON re-entry is allowed → a TON BUY landed.
        ton = [c for c in client.submit_ioc_calls if c["coin"] == "TON"]
        assert ton and ton[0]["is_buy"] is True

    def test_cooldown_blocks_same_direction_reentry_end_to_end(
        self, tmp_path: Any
    ) -> None:
        """run_cycle: TON liquidated short (baseline −0.3, cooldown today), new
        combined target re-enters SHORT TON → run_cycle blocks it → NO TON order."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={},  # flat after liquidation
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            # crash re-enters SHORT TON (SAME direction as the liquidated short).
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": -0.3},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            liq_cooldowns={"TON": "2026-06-02"},
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        # Same-direction short re-entry is blocked → no TON order; BTC still trades.
        assert all(c["coin"] != "TON" for c in client.submit_ioc_calls)
        assert any(c["coin"] == "BTC" for c in client.submit_ioc_calls)


class TestCooldownExpiry:
    """An expired cooldown trades normally and is cleared from the map."""

    def test_cooldown_expires(self, tmp_path: Any) -> None:
        """TON cooldown date = today − liquidation_cooldown_days → trades normally;
        the entry is cleared from ``state.liq_cooldowns``."""
        cfg = _two_sleeve_cfg()
        client = FakeHLClient(
            equity=100_000.0,
            positions={},
            marks=_marks(),
            sz_decimals=_szd(),
        )
        sources = {
            "haarp": _FakeSource({"BTC": 0.3}, client_id="haarp-client"),
            "crash": _FakeSource(
                {"TON": -0.4}, client_id="crash-client", allow_short=True
            ),
        }
        # Cooldown armed 1 day ago (= liquidation_cooldown_days) → EXPIRED today.
        # Baseline TON:0.0 so no NEW liquidation is detected this cycle.
        state = State(
            last_rebalanced_target={"BTC": 0.0, "TON": 0.0},
            last_rebalanced_as_of="2026-06-01",
            first_cycle_ts="2026-06-01T00:00:00+00:00",
            last_successful_signal_ts="2026-06-01T00:00:00+00:00",
            liq_cooldowns={"TON": "2026-06-01"},  # today (06-02) − 1 day
        )
        sp = tmp_path / "state.json"
        ledger = IntentLedger(tmp_path / "ledger.json")

        run_cycle(
            cfg, client, sources, ledger, state, sp, now_ts=_NOW, as_of=_AS_OF
        )

        # The expired cooldown is cleared, and TON trades (a short re-entry).
        assert "TON" not in state.liq_cooldowns
        assert any(c["coin"] == "TON" for c in client.submit_ioc_calls)
