"""
Tests for automation.runtime.daemon.

Coverage:
- run_once fires exactly once per as_of (second call same day → None,
  state.last_scheduled_as_of set)
- run_once with run_cycle that raises → EXEC_ERROR alert + as_of still marked
  fired + audit line written
- run_once with a clean rebalance → audit line + committed=True, no error alert
- run_once before fire time → returns None, no side-effects
- backfill_fills dedup (overlapping fills across two reconnects → no dupes)
- boot: emits DAEMON_RESTART alert
- boot: reconciles PENDING intents
"""

from __future__ import annotations

import datetime
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from automation.reporting.alerts import Alert, AlertType, Severity
from automation.reporting.audit import AuditLog
from automation.reporting.state import State, load
from automation.runtime import daemon as _daemon_mod
from automation.runtime.daemon import Daemon, backfill_fills
from automation.runtime.intent_ledger import IntentLedger
from automation.runtime.rebalance import CycleReport

UTC = datetime.UTC


# ---------------------------------------------------------------------------
# Fake / stub helpers
# ---------------------------------------------------------------------------


class _CapturingAlertSink:
    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


class _FakeCfg:
    rebalance_utc_time: str = "00:15"
    signal_source: Any = None
    universe_perp_map: dict = {}
    max_per_asset: float = 1.0
    # Boot-leverage knobs: default OFF for the generic fake-cfg boot tests so the
    # existing _FakeClient (no update_leverage) is never called.  The dedicated
    # leverage tests below construct a cfg with set_leverage_on_boot=True.
    target_leverage: int = 1
    leverage_is_cross: bool = False
    set_leverage_on_boot: bool = False
    enforce_leverage_on_drift: bool = True
    tradeable_coins: Any = None
    strategy_id: str = "test-strategy"

    # Position-health monitor (PH-4): the post-cycle hook is gated on
    # ``health.enabled``.  Default OFF for the generic fake-cfg so the existing
    # _FakeClient (no account_snapshot) is never touched; the dedicated
    # TestDaemonHealthHook class flips it on with a breaching fake client.
    class health:
        enabled: bool = False
        funding_drag_pct: float = 5.0
        margin_maint_frac_warn: float = 0.5
        liq_distance_pct_warn: float = 10.0
        spot_collateral_warn_usd: float = 1.0

    class caps:
        max_order_slippage: float = 0.01

    class retry:
        # Bounded intra-cycle retry knobs (mirrors automation.core.config.RetryConfig
        # defaults).  ``enabled=True`` so the retry-window tests exercise the branch;
        # interval 600 s (10 min), window 10 800 s (3 h).
        enabled: bool = True
        interval_seconds: int = 600
        window_seconds: int = 10800
        liquidity_safety: float = 1.0

    class signal_recheck:
        # No-signal re-check knobs (mirrors automation.core.config.SignalRecheckConfig
        # defaults, SP3 §5).  ``enabled=True`` so the recheck tests exercise the
        # branch; interval 180 s (3 min), window 7 200 s (2 h).
        enabled: bool = True
        interval_seconds: int = 180
        window_seconds: int = 7200


class _FakeSource:
    def get_target_allocation(self, as_of: Any = None) -> Any:
        return None


class _FakeClient:
    def positions(self) -> dict:
        return {}

    def equity(self) -> Decimal:
        return Decimal("10000")


def _make_exec_report(
    n_submitted: int = 1,
    n_filled: int = 1,
    n_error: int = 0,
    n_partial: int = 0,
    aborted: bool = False,
) -> Any:
    from automation.execution.executor import ExecReport  # noqa: PLC0415

    return ExecReport(
        results=[],
        n_submitted=n_submitted,
        n_filled=n_filled,
        n_error=n_error,
        n_partial=n_partial,
        aborted=aborted,
        abort_reason="cap" if aborted else None,
    )


def _clean_report() -> CycleReport:
    return CycleReport(
        rebalanced=True,
        reason="bootstrap",
        exec_report=_make_exec_report(n_submitted=1, n_filled=1),
        unresolved_pending=[],
        committed=True,
    )


def _hold_report() -> CycleReport:
    return CycleReport(
        rebalanced=False,
        reason="hold",
        exec_report=None,
        unresolved_pending=[],
        committed=False,
    )


def _make_daemon(
    tmp_path: Path,
    state: State | None = None,
    sink: _CapturingAlertSink | None = None,
) -> tuple[Daemon, AuditLog, _CapturingAlertSink, Path, Path]:
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    ledger_path = tmp_path / "ledger.json"

    if state is None:
        state = State()

    audit = AuditLog(audit_path)
    if sink is None:
        sink = _CapturingAlertSink()
    ledger = IntentLedger(ledger_path)

    d = Daemon(
        cfg=_FakeCfg,  # type: ignore[arg-type]
        client=_FakeClient(),
        source=_FakeSource(),
        ledger=ledger,
        state=state,
        state_path=state_path,
        audit=audit,
        alert_sink=sink,
        strategy_id="test-strategy",
    )
    return d, audit, sink, state_path, audit_path


def _dt(year: int, month: int, day: int, hour: int = 0, minute: int = 15) -> datetime.datetime:
    return datetime.datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# backfill_fills — pure function
# ---------------------------------------------------------------------------


class TestBackfillFills:
    def test_no_prior_seen_returns_all(self) -> None:
        fills = [
            {"oid": 1, "tid": 100, "coin": "BTC"},
            {"oid": 2, "tid": 200, "coin": "ETH"},
        ]
        new, seen = backfill_fills(fills, set())
        assert len(new) == 2
        assert (1, 100) in seen
        assert (2, 200) in seen

    def test_already_seen_deduped(self) -> None:
        fills = [
            {"oid": 1, "tid": 100},
            {"oid": 2, "tid": 200},
        ]
        new, seen = backfill_fills(fills, {(1, 100)})
        assert len(new) == 1
        assert new[0]["oid"] == 2

    def test_overlapping_reconnect_no_dupes(self) -> None:
        """Two reconnects with overlapping fills produce no duplicates."""
        batch1 = [{"oid": 1, "tid": 10}, {"oid": 2, "tid": 20}]
        new1, seen1 = backfill_fills(batch1, set())
        assert len(new1) == 2

        # Second reconnect includes old fills + new ones
        batch2 = [{"oid": 1, "tid": 10}, {"oid": 3, "tid": 30}]
        new2, seen2 = backfill_fills(batch2, seen1)
        assert len(new2) == 1
        assert new2[0]["oid"] == 3

    def test_does_not_mutate_input_seen(self) -> None:
        original = {(1, 10)}
        _, updated = backfill_fills([{"oid": 2, "tid": 20}], original)
        # original should be unchanged
        assert original == {(1, 10)}
        assert (2, 20) in updated

    def test_empty_fills_returns_empty(self) -> None:
        new, seen = backfill_fills([], {(1, 10)})
        assert new == []
        assert (1, 10) in seen

    def test_missing_oid_tid_keys_handled(self) -> None:
        """Fills missing oid/tid keys get (None, None) key — still deduplicated."""
        fills = [{"coin": "BTC"}, {"coin": "BTC"}]
        new, seen = backfill_fills(fills, set())
        # Both map to (None, None) — second is a dupe
        assert len(new) == 1


# ---------------------------------------------------------------------------
# Daemon.run_once
# ---------------------------------------------------------------------------


class TestDaemonRunOnce:
    def test_before_fire_time_returns_none(self, tmp_path: Path, monkeypatch: Any) -> None:
        d, _, sink, _, _ = _make_daemon(tmp_path)
        # 00:10 UTC — before 00:15 fire time
        now = _dt(2026, 6, 2, 0, 10)
        result = d.run_once(now, now.isoformat())
        assert result is None
        assert d.state.last_scheduled_as_of is None
        # No alerts
        assert len(sink.received) == 0

    def test_fires_once_and_marks_as_of(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())
        d, _, sink, state_path, audit_path = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        result = d.run_once(now, now.isoformat())

        assert result is not None
        assert result.rebalanced is True
        assert d.state.last_scheduled_as_of == "2026-06-01"

        # State should be persisted
        loaded = load(state_path)
        assert loaded.last_scheduled_as_of == "2026-06-01"

    def test_second_call_same_day_returns_none(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        first = d.run_once(now, now.isoformat())
        assert first is not None

        # Tick again same time — must return None (already fired)
        second = d.run_once(now, now.isoformat())
        assert second is None

        # On-disk idempotency key must remain set after the second (no-op) tick.
        loaded = load(state_path)
        assert loaded.last_scheduled_as_of == "2026-06-01"

    def test_mark_scheduled_persisted_before_cycle_crash_no_refire(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """F1: simulate commit-then-crash.

        run_cycle commits the trade internally, THEN the process is hard-killed
        before run_once's post-cycle save.  Because mark-scheduled is persisted
        FIRST (before run_cycle), the on-disk State already carries the
        idempotency key.  On reboot, should_fire must NOT re-fire the same as_of.
        """
        from automation.runtime.scheduler import should_fire  # noqa: PLC0415

        crashed: dict[str, bool] = {"did": False}

        def _commit_then_crash(*a: Any, **kw: Any) -> CycleReport:
            # run_cycle has committed the trade by now (its own internal save).
            # Simulate the hard kill: raise a non-Exception-catchable death by
            # raising BaseException AFTER the mark-scheduled save already hit disk.
            crashed["did"] = True
            raise KeyboardInterrupt("simulated SIGKILL mid-cycle")

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _commit_then_crash)
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        # BaseException (KeyboardInterrupt) is NOT caught by run_once's
        # `except Exception` — it propagates like a real process death would,
        # AFTER mark-scheduled has been saved to disk.
        with pytest.raises(KeyboardInterrupt):
            d.run_once(now, now.isoformat())
        assert crashed["did"] is True

        # "Reboot": load State fresh from disk into a brand-new daemon.
        reloaded = load(state_path)
        assert reloaded.last_scheduled_as_of == "2026-06-01"

        # should_fire must NOT re-fire this as_of → no double-rebalance.
        fire, as_of = should_fire(now, reloaded.last_scheduled_as_of, "00:15")
        assert fire is False
        assert as_of == "2026-06-01"

    def test_clean_rebalance_writes_audit_line(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())
        d, audit, _, _, audit_path = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["rebalanced"] is True
        assert rec["committed"] is True
        assert rec["as_of"] == "2026-06-01"

    def test_clean_rebalance_no_error_alert(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())
        d, _, sink, _, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        error_alerts = [a for a in sink.received if a.type == AlertType.EXEC_ERROR]
        assert len(error_alerts) == 0

    def test_run_cycle_raises_exec_error_alert_emitted(self, tmp_path: Path, monkeypatch: Any) -> None:
        def _bad_cycle(*a: Any, **kw: Any) -> CycleReport:
            raise RuntimeError("exchange down")

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _bad_cycle)
        d, _, sink, _, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        result = d.run_once(now, now.isoformat())

        # Should still return a (failed) CycleReport, not re-raise
        assert result is not None
        assert result.rebalanced is False

        error_alerts = [a for a in sink.received if a.type == AlertType.EXEC_ERROR]
        assert len(error_alerts) >= 1

    def test_run_cycle_raises_as_of_still_marked(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Even on exception, last_scheduled_as_of must be set (idempotent once/day)."""
        def _bad_cycle(*a: Any, **kw: Any) -> CycleReport:
            raise RuntimeError("exchange down")

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _bad_cycle)
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        assert d.state.last_scheduled_as_of == "2026-06-01"
        loaded = load(state_path)
        assert loaded.last_scheduled_as_of == "2026-06-01"

    def test_run_cycle_raises_audit_line_written(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Even on exception, an audit line must be written."""
        def _bad_cycle(*a: Any, **kw: Any) -> CycleReport:
            raise RuntimeError("exchange down")

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _bad_cycle)
        d, _, _, _, audit_path = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["as_of"] == "2026-06-01"
        assert rec["rebalanced"] is False

    def test_exec_error_alert_on_n_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        def _error_cycle(*a: Any, **kw: Any) -> CycleReport:
            return CycleReport(
                rebalanced=True,
                reason="bootstrap",
                exec_report=_make_exec_report(n_error=1, n_filled=1, n_submitted=2),
                unresolved_pending=[],
                committed=False,
            )

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _error_cycle)
        d, _, sink, _, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        assert any(a.type == AlertType.EXEC_ERROR for a in sink.received)

    def test_cycle_incomplete_alert_on_rebalanced_not_committed(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        def _uncommitted_cycle(*a: Any, **kw: Any) -> CycleReport:
            return CycleReport(
                rebalanced=True,
                reason="bootstrap",
                exec_report=_make_exec_report(),
                unresolved_pending=[],
                committed=False,
            )

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _uncommitted_cycle)
        d, _, sink, _, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        incomplete_alerts = [a for a in sink.received if a.type == AlertType.CYCLE_INCOMPLETE]
        assert len(incomplete_alerts) >= 1

    def test_hold_cycle_no_error_alert(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _hold_report())
        d, _, sink, _, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        error_alerts = [a for a in sink.received if a.type == AlertType.EXEC_ERROR]
        assert len(error_alerts) == 0

    def test_next_day_fires_new_cycle(self, tmp_path: Path, monkeypatch: Any) -> None:
        """After day-1 fires, day-2 should fire with new as_of."""
        call_count = 0

        def _counting_cycle(*a: Any, **kw: Any) -> CycleReport:
            nonlocal call_count
            call_count += 1
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _counting_cycle)
        d, _, _, _, _ = _make_daemon(tmp_path)

        # Day 1
        day1 = _dt(2026, 6, 2, 0, 20)
        d.run_once(day1, day1.isoformat())
        assert call_count == 1
        assert d.state.last_scheduled_as_of == "2026-06-01"

        # Day 2
        day2 = _dt(2026, 6, 3, 0, 20)
        d.run_once(day2, day2.isoformat())
        assert call_count == 2
        assert d.state.last_scheduled_as_of == "2026-06-02"

    def test_run_once_threads_alert_sink_into_run_cycle(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once must pass its alert_sink to run_cycle.

        We inject a real raising source so run_cycle's STALE_SIGNAL path fires,
        which only emits if alert_sink is wired correctly.
        """
        import datetime as _dt_mod
        from decimal import Decimal

        from automation.core.config import CapsConfig, Config, SignalSourceConfig
        from automation.reporting.audit import AuditLog
        from automation.reporting.state import save as _save
        from automation.runtime.intent_ledger import IntentLedger

        # State: last good signal 5 days ago → beyond window → STALE_SIGNAL CRITICAL
        now = _dt(2026, 6, 5, 0, 15)
        last_ts = (now - _dt_mod.timedelta(days=5)).isoformat()
        state = State(last_successful_signal_ts=last_ts)
        sink = _CapturingAlertSink()

        class _RaisingSource:
            def get_target_allocation(self, as_of: Any = None) -> Any:
                raise RuntimeError("signal down")

        class _FakeClientFlat:
            def positions(self) -> dict:
                return {}  # already flat → flatten_all is idempotent no-op

            def equity(self) -> Decimal:
                return Decimal("10000")

            def marks(self) -> dict:
                return {}

            def sz_decimals(self) -> dict:
                return {}

        cfg = Config(
            env="testnet",
            subaccount_address="0x" + "0" * 40,
            signal_source=SignalSourceConfig(type="equal_weight", client_id="test"),
            rebalance_utc_time="00:15",
            threshold_pct=3.0,
            max_per_asset=0.6,
            safety_buffer=0.03,
            universe_perp_map={},
            max_signal_staleness_days=2,
            caps=CapsConfig(
                max_notional_per_coin=1_000_000,
                max_total_notional=10_000_000,
                max_turnover_per_cycle=5_000_000,
                max_order_slippage=0.005,
            ),
        )

        state_path = tmp_path / "state.json"
        _save(state, state_path)

        d = Daemon(
            cfg=cfg,
            client=_FakeClientFlat(),
            source=_RaisingSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=state,
            state_path=state_path,
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )

        d.run_once(now, now.isoformat())

        stale = [a for a in sink.received if a.type == AlertType.STALE_SIGNAL]
        assert len(stale) >= 1, (
            f"Expected at least one STALE_SIGNAL alert — got: {[a.type for a in sink.received]}"
        )

    def test_audit_line_contains_signal_ok_and_derisked(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Audit record must include signal_ok and derisked fields."""

        def _stale_cycle(*a: Any, **kw: Any) -> CycleReport:
            return CycleReport(
                rebalanced=False,
                reason="hold_stale_within_window",
                exec_report=None,
                unresolved_pending=[],
                committed=False,
                signal_ok=False,
                derisked=False,
            )

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _stale_cycle)
        d, audit, _, _, audit_path = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert "signal_ok" in rec
        assert rec["signal_ok"] is False
        assert "derisked" in rec
        assert rec["derisked"] is False

    def test_audit_line_derisked_true_when_derisked(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Audit record has derisked=True when the cycle de-risked to cash."""

        def _derisk_cycle(*a: Any, **kw: Any) -> CycleReport:
            return CycleReport(
                rebalanced=False,
                reason="derisk_to_cash",
                exec_report=None,
                unresolved_pending=[],
                committed=True,
                signal_ok=False,
                derisked=True,
            )

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _derisk_cycle)
        d, audit, _, _, audit_path = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        lines = audit_path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        assert rec["derisked"] is True
        assert rec["signal_ok"] is False


# ---------------------------------------------------------------------------
# Daemon.boot
# ---------------------------------------------------------------------------


class TestDaemonBoot:
    def test_daemon_restart_alert_emitted(self, tmp_path: Path) -> None:
        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.boot("2026-06-01T00:15:00")

        restart_alerts = [a for a in sink.received if a.type == AlertType.DAEMON_RESTART]
        assert len(restart_alerts) == 1
        assert restart_alerts[0].severity == Severity.INFO

    def test_boot_reconciles_pending(self, tmp_path: Path) -> None:
        """boot() calls reconcile_pending on the ledger (conservative: no fills)."""
        from automation.runtime.intent_ledger import Intent  # noqa: PLC0415

        ledger_path = tmp_path / "ledger.json"
        ledger = IntentLedger(ledger_path)
        ledger.record_pending(
            Intent(
                cloid="0x" + "a" * 32,
                as_of="2026-05-31",
                coin="BTC",
                target_w=0.5,
                attempt_seq=0,
                status="PENDING",
                ts="2026-05-31T00:15:00",
            )
        )
        assert len(ledger.pending()) == 1

        state_path = tmp_path / "state.json"
        audit = AuditLog(tmp_path / "audit.jsonl")
        sink = _CapturingAlertSink()
        d = Daemon(
            cfg=_FakeCfg,  # type: ignore[arg-type]
            client=_FakeClient(),
            source=_FakeSource(),
            ledger=ledger,
            state=State(),
            state_path=state_path,
            audit=audit,
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot("2026-06-01T00:15:00")

        # Conservative reconcile → intents stay PENDING (is_filled always False)
        # boot should not raise; the restart alert should be emitted
        assert any(a.type == AlertType.DAEMON_RESTART for a in sink.received)

    def test_agent_expiry_alert_when_field_present(self, tmp_path: Path) -> None:
        """If cfg.agent_valid_until_ms is set and < 14 days, AGENT_EXPIRY alert."""

        # now = 2026-06-01T00:15:00 UTC ≈ 1748733300s
        # expires in 5 days from now
        now_ts = "2026-06-01T00:15:00"
        now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()
        expires_in = 5 * 24 * 3600  # 5 days
        valid_until_ms = int((now_unix + expires_in) * 1000)

        class _CfgWithExpiry(_FakeCfg):
            agent_valid_until_ms = valid_until_ms

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgWithExpiry  # type: ignore[assignment]
        d.boot(now_ts)

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 1
        assert expiry_alerts[0].severity == Severity.WARNING  # 5 days → WARNING

    def test_no_agent_expiry_alert_when_field_absent(self, tmp_path: Path) -> None:
        """If cfg.agent_valid_until_ms is absent, AGENT_EXPIRY is NOT emitted."""
        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.boot("2026-06-01T00:15:00")

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 0

    # -----------------------------------------------------------------------
    # G5 / S-12 boot tests
    # -----------------------------------------------------------------------

    def test_boot_no_alert_when_agent_healthy(self, tmp_path: Path) -> None:
        """boot with cfg.agent_valid_until_ms = now+20d → NO AGENT_EXPIRY alert."""
        now_ts = "2026-06-01T00:15:00"
        now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()
        valid_until_ms = int((now_unix + 20 * 24 * 3600) * 1000)  # +20 days

        class _CfgHealthy(_FakeCfg):
            agent_valid_until_ms = valid_until_ms

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgHealthy  # type: ignore[assignment]
        d.boot(now_ts)

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 0

    def test_boot_warning_when_agent_expires_in_10d(self, tmp_path: Path) -> None:
        """boot with cfg.agent_valid_until_ms = now+10d → AGENT_EXPIRY WARNING."""
        now_ts = "2026-06-01T00:15:00"
        now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()
        valid_until_ms = int((now_unix + 10 * 24 * 3600) * 1000)  # +10 days

        class _CfgSoon(_FakeCfg):
            agent_valid_until_ms = valid_until_ms

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgSoon  # type: ignore[assignment]
        d.boot(now_ts)

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 1
        assert expiry_alerts[0].severity == Severity.WARNING

    def test_boot_critical_when_agent_expired(self, tmp_path: Path) -> None:
        """boot with cfg.agent_valid_until_ms = now-1d (expired) → AGENT_EXPIRY CRITICAL."""
        now_ts = "2026-06-01T00:15:00"
        now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()
        valid_until_ms = int((now_unix - 1 * 24 * 3600) * 1000)  # -1 day (past)

        class _CfgExpired(_FakeCfg):
            agent_valid_until_ms = valid_until_ms

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgExpired  # type: ignore[assignment]
        d.boot(now_ts)

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 1
        assert expiry_alerts[0].severity == Severity.CRITICAL

    def test_boot_none_agent_valid_no_alert_no_crash(self, tmp_path: Path) -> None:
        """boot with cfg.agent_valid_until_ms = None → no AGENT_EXPIRY alert, no crash."""
        d, _, sink, _, _ = _make_daemon(tmp_path)
        # _FakeCfg has no agent_valid_until_ms attribute → getattr returns None
        d.boot("2026-06-01T00:15:00")

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 0


# ---------------------------------------------------------------------------
# Daemon.boot — per-coin leverage set (1x isolated)
# ---------------------------------------------------------------------------


class _CfgLeverage(_FakeCfg):
    """Fake cfg that turns boot-leverage ON with a 2-coin tradeable set."""

    target_leverage: int = 1
    leverage_is_cross: bool = False
    set_leverage_on_boot: bool = True
    tradeable_coins: Any = ["BTC", "ETH"]


class TestDaemonBootLeverage:
    def _make_lev_daemon(
        self, tmp_path: Path, cfg: Any, client: Any
    ) -> Daemon:
        return Daemon(
            cfg=cfg,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=_CapturingAlertSink(),
            strategy_id="test",
        )

    def test_boot_sets_leverage_for_each_tradeable_coin(self, tmp_path: Path) -> None:
        """set_leverage_on_boot=True → update_leverage(coin, 1, is_cross=False) per coin."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        client = FakeHLClient()
        d = self._make_lev_daemon(tmp_path, _CfgLeverage, client)
        d.boot("2026-06-01T00:15:00")

        assert client.leverage_calls == [("BTC", 1, False), ("ETH", 1, False)]

    def test_boot_skips_leverage_when_flag_off(self, tmp_path: Path) -> None:
        """set_leverage_on_boot=False → no update_leverage calls at all."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        class _CfgOff(_CfgLeverage):
            set_leverage_on_boot: bool = False

        client = FakeHLClient()
        d = self._make_lev_daemon(tmp_path, _CfgOff, client)
        d.boot("2026-06-01T00:15:00")

        assert client.leverage_calls == []

    def test_boot_one_coin_failure_does_not_abort(self, tmp_path: Path) -> None:
        """One coin's update_leverage raising must not abort boot; others still set."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        class _RaisingClient(FakeHLClient):
            def update_leverage(
                self, coin: str, leverage: int, *, is_cross: bool
            ) -> dict:
                if coin == "BTC":
                    raise RuntimeError("venue rejected BTC leverage")
                return super().update_leverage(coin, leverage, is_cross=is_cross)

        client = _RaisingClient()
        d = self._make_lev_daemon(tmp_path, _CfgLeverage, client)
        # Must not raise.
        d.boot("2026-06-01T00:15:00")

        # ETH still got set even though BTC failed.
        assert client.leverage_calls == [("ETH", 1, False)]

    def test_boot_falls_back_to_universe_perp_map_when_tradeable_none(
        self, tmp_path: Path
    ) -> None:
        """tradeable_coins=None → iterate universe_perp_map keys instead."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        class _CfgUniverse(_FakeCfg):
            target_leverage: int = 1
            leverage_is_cross: bool = False
            set_leverage_on_boot: bool = True
            tradeable_coins: Any = None
            universe_perp_map: dict = {"SOL": "SOL", "BNB": "BNB"}

        client = FakeHLClient()
        d = self._make_lev_daemon(tmp_path, _CfgUniverse, client)
        d.boot("2026-06-01T00:15:00")

        assert client.leverage_calls == [("SOL", 1, False), ("BNB", 1, False)]

    def test_boot_skips_leverage_when_agent_expired(self, tmp_path: Path) -> None:
        """EXPIRED agent → NO signed update_leverage actions, boot still completes.

        update_leverage is a signed action; the "never sign on an expired key"
        invariant (run_once returns None when expired) must hold at boot too.
        """
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        now_ts = "2026-06-01T00:15:00"
        now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()
        valid_until_ms = int((now_unix - 1 * 24 * 3600) * 1000)  # expired 1 day ago

        class _CfgExpiredLev(_CfgLeverage):
            agent_valid_until_ms = valid_until_ms

        client = FakeHLClient()
        d = self._make_lev_daemon(tmp_path, _CfgExpiredLev, client)
        d.boot(now_ts)  # must not raise

        assert client.leverage_calls == [], (
            "no signed leverage actions may be attempted on an expired agent"
        )

    # -- drift reconciliation (leverage TYPE mismatch on a HELD position) -----

    def test_boot_type_mismatch_enforced_flattens_then_sets(self, tmp_path: Path) -> None:
        """Held BTC in CROSS but target ISOLATED + enforce=True →
        _flatten_for_leverage (market_close) for BTC + LEVERAGE_DRIFT alert +
        update_leverage still called after (position now flat → sets default)."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        sink = _CapturingAlertSink()
        client = FakeHLClient(
            positions={"BTC": 0.5},
            positions_leverage={"BTC": {"type": "cross", "value": 10}},
        )
        d = Daemon(
            cfg=_CfgLeverage,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot("2026-06-01T00:15:00")

        # BTC flattened (reduce-only market_close recorded).
        assert [c["coin"] for c in client.market_close_calls] == ["BTC"]
        # LEVERAGE_DRIFT alert emitted for BTC.
        drift = [a for a in sink.received if a.type == AlertType.LEVERAGE_DRIFT]
        assert len(drift) == 1
        assert drift[0].context["coin"] == "BTC"
        # update_leverage STILL called for both coins (BTC fell through after flatten).
        assert client.leverage_calls == [("BTC", 1, False), ("ETH", 1, False)]

    def test_boot_type_mismatch_not_enforced_alert_only(self, tmp_path: Path) -> None:
        """Held BTC in CROSS, target ISOLATED + enforce=False → NO flatten,
        LEVERAGE_DRIFT alert emitted, update_leverage NOT called for BTC (continue)."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        class _CfgNoEnforce(_CfgLeverage):
            enforce_leverage_on_drift: bool = False

        sink = _CapturingAlertSink()
        client = FakeHLClient(
            positions={"BTC": 0.5},
            positions_leverage={"BTC": {"type": "cross", "value": 10}},
        )
        d = Daemon(
            cfg=_CfgNoEnforce,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot("2026-06-01T00:15:00")

        assert client.market_close_calls == []  # NO flatten
        drift = [a for a in sink.received if a.type == AlertType.LEVERAGE_DRIFT]
        assert len(drift) == 1 and drift[0].context["coin"] == "BTC"
        # BTC skipped (continue); ETH (no held position) still set.
        assert client.leverage_calls == [("ETH", 1, False)]

    def test_boot_value_mismatch_same_type_updates_in_place(self, tmp_path: Path) -> None:
        """Held BTC ISOLATED but wrong VALUE (10 vs target 1) → NO flatten,
        update_leverage called in place (HL allows value change, same type)."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        sink = _CapturingAlertSink()
        client = FakeHLClient(
            positions={"BTC": 0.5},
            positions_leverage={"BTC": {"type": "isolated", "value": 10}},
        )
        d = Daemon(
            cfg=_CfgLeverage,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot("2026-06-01T00:15:00")

        assert client.market_close_calls == []  # NO flatten (type matches)
        assert [a for a in sink.received if a.type == AlertType.LEVERAGE_DRIFT] == []
        assert client.leverage_calls == [("BTC", 1, False), ("ETH", 1, False)]

    def test_boot_no_held_position_sets_default(self, tmp_path: Path) -> None:
        """No held position (coin flat) → NO flatten, update_leverage sets default."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        sink = _CapturingAlertSink()
        client = FakeHLClient()  # no positions, no positions_leverage
        d = Daemon(
            cfg=_CfgLeverage,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot("2026-06-01T00:15:00")

        assert client.market_close_calls == []
        assert [a for a in sink.received if a.type == AlertType.LEVERAGE_DRIFT] == []
        assert client.leverage_calls == [("BTC", 1, False), ("ETH", 1, False)]

    def test_boot_drift_skips_when_agent_expired(self, tmp_path: Path) -> None:
        """EXPIRED agent → NO flatten and NO update_leverage even with a TYPE
        mismatch held (no signed actions on an expired agent)."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        now_ts = "2026-06-01T00:15:00"
        now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()
        valid_until_ms = int((now_unix - 1 * 24 * 3600) * 1000)

        class _CfgExpiredDrift(_CfgLeverage):
            agent_valid_until_ms = valid_until_ms

        sink = _CapturingAlertSink()
        client = FakeHLClient(
            positions={"BTC": 0.5},
            positions_leverage={"BTC": {"type": "cross", "value": 10}},
        )
        d = Daemon(
            cfg=_CfgExpiredDrift,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot(now_ts)  # must not raise

        assert client.market_close_calls == []
        assert client.leverage_calls == []

    def test_boot_drift_flatten_raises_continues(self, tmp_path: Path) -> None:
        """enforce=True but _flatten_for_leverage RAISES for one coin → caught,
        boot continues, other coins still processed."""
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        class _RaisingCloseClient(FakeHLClient):
            def market_close(self, coin: str, slippage: float, cloid: str) -> dict:
                if coin == "BTC":
                    raise RuntimeError("venue rejected BTC close")
                return super().market_close(coin, slippage, cloid)

        sink = _CapturingAlertSink()
        client = _RaisingCloseClient(
            positions={"BTC": 0.5},
            positions_leverage={"BTC": {"type": "cross", "value": 10}},
        )
        d = Daemon(
            cfg=_CfgLeverage,
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(tmp_path / "ledger.json"),
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=sink,
            strategy_id="test",
        )
        d.boot("2026-06-01T00:15:00")  # must not raise

        # Drift alert still emitted; update_leverage still attempted for both.
        drift = [a for a in sink.received if a.type == AlertType.LEVERAGE_DRIFT]
        assert len(drift) == 1 and drift[0].context["coin"] == "BTC"
        assert client.leverage_calls == [("BTC", 1, False), ("ETH", 1, False)]

    def test_boot_leverage_set_applies_to_held_short(self, tmp_path: Path) -> None:
        """Boot leverage reconciliation is direction-blind: a coin held SHORT (szi < 0)
        with correct leverage type and value is classified 'ok' and update_leverage is
        still called for it — no sign-check or direction branch crash.

        Spec §5 (plan task B5): ``update_leverage`` is per-coin, direction-blind,
        and operates identically whether the held position is long or short.
        This test pins that invariant so any accidental sign-guard in the boot
        path would be caught immediately.
        """
        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        # ETH held SHORT at the target leverage/mode — should land in 'ok' and still
        # receive update_leverage (the boot path calls it for every tradeable coin).
        client = FakeHLClient(
            positions={"ETH": -2.0},  # negative szi = short
            positions_leverage={"ETH": {"type": "isolated", "value": 1}},
        )
        d = self._make_lev_daemon(tmp_path, _CfgLeverage, client)
        d.boot("2026-06-10T00:00:00")

        # update_leverage must be called for ETH regardless of its position direction.
        # BTC has no held position but is tradeable → also gets the default set.
        coins_levered = {coin for coin, _lev, _cross in client.leverage_calls}
        assert "ETH" in coins_levered, (
            "boot must call update_leverage for ETH even though its position is short"
        )
        # Confirm no unhandled exception was raised (implicit: we reached here).
        # Check the ETH call carries the target params: leverage=1, is_cross=False.
        eth_calls = [
            (lev, cross)
            for coin, lev, cross in client.leverage_calls
            if coin == "ETH"
        ]
        assert eth_calls == [(1, False)], (
            "update_leverage for a short position must use the same target params as "
            "for a long — no direction branching"
        )


# ---------------------------------------------------------------------------
# Daemon.run_once — G5 / S-12 fail-safe HOLD tests
# ---------------------------------------------------------------------------


class TestDaemonRunOnceExpiry:
    _now_ts = "2026-06-01T00:15:00"
    _now = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC)
    _now_unix = datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp()

    def _expired_ms(self) -> int:
        return int((self._now_unix - 1 * 24 * 3600) * 1000)  # 1 day in the past

    def _future_ms(self) -> int:
        return int((self._now_unix + 30 * 24 * 3600) * 1000)  # 30 days in the future

    def test_run_once_expired_returns_none(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once with expired agent → returns None (fail-safe HOLD)."""

        class _CfgExpired(_FakeCfg):
            agent_valid_until_ms = int(
                (datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp() - 86400) * 1000
            )

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgExpired  # type: ignore[assignment]

        result = d.run_once(self._now, self._now_ts)
        assert result is None

    def test_run_once_expired_emits_critical_alert(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once with expired agent → emits AGENT_EXPIRY CRITICAL."""

        class _CfgExpired(_FakeCfg):
            agent_valid_until_ms = int(
                (datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp() - 86400) * 1000
            )

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgExpired  # type: ignore[assignment]

        d.run_once(self._now, self._now_ts)

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) >= 1
        assert any(a.severity == Severity.CRITICAL for a in expiry_alerts)

    def test_run_once_expired_does_not_call_run_cycle(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once with expired agent → does NOT call run_cycle."""
        called: dict[str, bool] = {"did": False}

        def _spy_cycle(*a: Any, **kw: Any) -> Any:
            called["did"] = True
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _spy_cycle)

        class _CfgExpired(_FakeCfg):
            agent_valid_until_ms = int(
                (datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp() - 86400) * 1000
            )

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgExpired  # type: ignore[assignment]

        d.run_once(self._now, self._now_ts)
        assert called["did"] is False

    def test_run_once_expired_does_not_consume_as_of(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once with expired agent → does NOT mark state.last_scheduled_as_of.

        After the operator rotates the agent, the next run_once must still be
        able to fire the scheduled cycle (the as_of must not have been consumed).
        """
        before = None  # initial state has last_scheduled_as_of = None

        class _CfgExpired(_FakeCfg):
            agent_valid_until_ms = int(
                (datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp() - 86400) * 1000
            )

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgExpired  # type: ignore[assignment]

        d.run_once(self._now, self._now_ts)

        # last_scheduled_as_of must remain unchanged (None → cycle not consumed)
        assert d.state.last_scheduled_as_of == before

    def test_run_once_healthy_agent_behaves_normally(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once with a healthy (future) agent → expiry guard is a no-op,
        normal cycle fires."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        class _CfgHealthy(_FakeCfg):
            agent_valid_until_ms = int(
                (datetime.datetime(2026, 6, 1, 0, 15, tzinfo=UTC).timestamp() + 30 * 86400) * 1000
            )

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = _CfgHealthy  # type: ignore[assignment]

        result = d.run_once(self._now, self._now_ts)

        # Should fire normally (cycle ran)
        assert result is not None
        # No expiry alert on a healthy agent
        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) == 0

    def test_run_once_unreadable_expiry_fail_safe_hold(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """ID-12: a parse error in _agent_expiry → fail-safe HOLD.

        run_once must NOT raise; instead it returns None, emits AGENT_EXPIRY
        CRITICAL, does NOT call run_cycle, and does NOT consume the as_of.
        (Unreachable via pydantic Config, but symmetric with boot's try/except.)
        """
        called: dict[str, bool] = {"did": False}

        def _spy_cycle(*a: Any, **kw: Any) -> Any:
            called["did"] = True
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _spy_cycle)

        d, _, sink, _, _ = _make_daemon(tmp_path)

        def _boom(_now_ts: str) -> tuple[bool, float | None]:
            raise ValueError("malformed agent_valid_until_ms")

        monkeypatch.setattr(d, "_agent_expiry", _boom)

        before = d.state.last_scheduled_as_of  # None initially
        result = d.run_once(self._now, self._now_ts)

        assert result is None
        assert called["did"] is False  # run_cycle NOT called
        assert d.state.last_scheduled_as_of == before  # as_of NOT consumed

        expiry_alerts = [a for a in sink.received if a.type == AlertType.AGENT_EXPIRY]
        assert len(expiry_alerts) >= 1
        assert any(a.severity == Severity.CRITICAL for a in expiry_alerts)


# ---------------------------------------------------------------------------
# Daemon.run_once — G3 Tier-1 kill-switch tests
# ---------------------------------------------------------------------------


class _FakeClientWithPositions:
    """Fake HLClient that holds open positions and records market_close calls.

    ``persist_positions=True`` keeps the position OPEN after ``market_close``
    (instead of removing it) — used to prove the flatten path runs on EVERY
    armed tick (a position that keeps reappearing must keep being closed).
    """

    def __init__(
        self,
        positions: dict[str, Any] | None = None,
        *,
        persist_positions: bool = False,
    ) -> None:
        from decimal import Decimal  # noqa: PLC0415

        self._positions: dict[str, Any] = (
            {"BTC": Decimal("0.5")} if positions is None else positions
        )
        self._persist = persist_positions
        self.market_close_calls: list[tuple[str, float, str]] = []
        self.positions_calls: int = 0

    def positions(self) -> dict[str, Any]:
        self.positions_calls += 1
        return dict(self._positions)

    def equity(self) -> Any:
        from decimal import Decimal  # noqa: PLC0415

        return Decimal("10000")

    def market_close(self, coin: str, slippage: float, cloid: str) -> dict[str, Any]:
        self.market_close_calls.append((coin, slippage, cloid))
        # Simulate a successful full close — remove from positions so post-query
        # is flat (unless persist_positions: the position stays OPEN, modelling a
        # resting order that keeps refilling while armed).
        from decimal import Decimal  # noqa: PLC0415

        size = abs(float(self._positions.get(coin, Decimal("0"))))
        if not self._persist:
            self._positions.pop(coin, None)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"totalSz": str(size), "avgPx": "50000", "oid": 1}}
                    ]
                }
            },
        }


def _make_kill_daemon(
    tmp_path: Path,
    kill_flag_path: Path | None = None,
    client: Any = None,
) -> tuple[Any, Any, Path]:
    """Helper: build a Daemon wired with kill_flag_path and a recording client."""
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    ledger_path = tmp_path / "ledger.json"

    if client is None:
        from decimal import Decimal  # noqa: PLC0415

        client = _FakeClientWithPositions({"BTC": Decimal("0.5")})

    sink = _CapturingAlertSink()

    d = Daemon(
        cfg=_FakeCfg,  # type: ignore[arg-type]
        client=client,
        source=_FakeSource(),
        ledger=IntentLedger(ledger_path),
        state=State(),
        state_path=state_path,
        audit=AuditLog(audit_path),
        alert_sink=sink,
        strategy_id="test-strategy",
        kill_flag_path=kill_flag_path,
    )
    return d, sink, state_path


class TestDaemonKillSwitch:
    """G3 — Tier-1 in-process kill-switch (file-flag) tests."""

    _now = datetime.datetime(2026, 6, 2, 0, 15, tzinfo=UTC)
    _now_ts = "2026-06-02T00:15:00+00:00"

    # ------------------------------------------------------------------
    # Armed → flatten + halt
    # ------------------------------------------------------------------

    def test_armed_halts_returns_none(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Flag present → run_once returns None (HALT)."""
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag)

        # run_cycle must NOT be called
        called: dict[str, bool] = {"did": False}

        def _spy(*a: Any, **kw: Any) -> Any:
            called["did"] = True
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _spy)

        result = d.run_once(self._now, self._now_ts)

        assert result is None
        assert called["did"] is False

    def test_armed_emits_kill_switch_critical(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Flag present → KILL_SWITCH CRITICAL alert emitted."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag)

        d.run_once(self._now, self._now_ts)

        kill_alerts = [a for a in sink.received if a.type == AlertType.KILL_SWITCH]
        assert any(a.severity == Severity.CRITICAL for a in kill_alerts)

    def test_armed_issues_market_close_for_open_positions(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Flag present → market_close called for the open BTC position."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        from decimal import Decimal  # noqa: PLC0415

        client = _FakeClientWithPositions({"BTC": Decimal("0.5")})
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=client)

        d.run_once(self._now, self._now_ts)

        assert len(client.market_close_calls) >= 1
        coins_closed = {call[0] for call in client.market_close_calls}
        assert "BTC" in coins_closed

    def test_armed_does_not_consume_as_of(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Flag present → state.last_scheduled_as_of is NOT updated."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, state_path = _make_kill_daemon(tmp_path, kill_flag_path=flag)

        before = d.state.last_scheduled_as_of
        d.run_once(self._now, self._now_ts)

        assert d.state.last_scheduled_as_of == before

    # ------------------------------------------------------------------
    # Armed + already flat — no market_close, still HALT
    # ------------------------------------------------------------------

    def test_armed_already_flat_no_market_close(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Flag present, no open positions → no market_close calls, still HALT (None)."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        flat_client = _FakeClientWithPositions({})
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=flat_client)

        result = d.run_once(self._now, self._now_ts)

        assert result is None
        assert len(flat_client.market_close_calls) == 0
        kill_alerts = [a for a in sink.received if a.type == AlertType.KILL_SWITCH]
        assert any(a.severity == Severity.CRITICAL for a in kill_alerts)

    # ------------------------------------------------------------------
    # One-shot: armed twice → flatten + alert ONLY on first tick
    # ------------------------------------------------------------------

    def test_alert_edge_gated_but_flatten_every_tick(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Two consecutive armed ticks: the daemon's 'armed' CRITICAL alert is
        edge-gated (fires ONCE), but the flatten path is attempted on BOTH ticks.

        Uses a persist_positions fake so the position stays OPEN after each close
        — proving market_close is attempted every armed tick (not one-shot)."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        from decimal import Decimal  # noqa: PLC0415

        # Position persists after close → flatten must keep closing it each tick.
        client = _FakeClientWithPositions({"BTC": Decimal("0.5")}, persist_positions=True)
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=client)

        # The daemon emits its OWN "armed — flattening" CRITICAL alert; identify it
        # by message so it can be distinguished from flatten_all's own KILL_SWITCH
        # alerts (which fire every tick the flatten runs).
        def _armed_alerts() -> list[Any]:
            return [
                a
                for a in sink.received
                if a.type == AlertType.KILL_SWITCH and "kill flag armed" in a.message
            ]

        # Tick 1
        d.run_once(self._now, self._now_ts)
        close_after_t1 = len(client.market_close_calls)
        assert close_after_t1 >= 1  # flatten ran
        assert len(_armed_alerts()) == 1  # arming-edge alert fired once

        # Tick 2: flag still present — flatten must run AGAIN (every armed tick)…
        d.run_once(self._now, self._now_ts)
        assert len(client.market_close_calls) > close_after_t1  # new close attempted
        # …but the daemon's own 'armed' CRITICAL alert is edge-gated → still 1.
        assert len(_armed_alerts()) == 1

    def test_k2_new_position_while_armed_is_flattened(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """K2 regression: a NEW position appears while STILL armed (no disarmed
        tick) → the next armed run_once flattens it.  Proves no silent dead kill."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        from decimal import Decimal  # noqa: PLC0415

        # Start flat — first armed tick is a no-op flatten (nothing to close).
        client = _FakeClientWithPositions({})
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=client)

        # Tick 1: armed but flat → no market_close.
        d.run_once(self._now, self._now_ts)
        assert len(client.market_close_calls) == 0

        # A NEW position appears while STILL armed (no disarmed tick in between).
        client._positions["BTC"] = Decimal("0.5")

        # Tick 2: still armed → must flatten the newly-appeared position.
        d.run_once(self._now, self._now_ts)
        assert len(client.market_close_calls) >= 1
        assert any(call[0] == "BTC" for call in client.market_close_calls)

    # ------------------------------------------------------------------
    # Disarm resumes normal trading; re-arm flattens again
    # ------------------------------------------------------------------

    def test_disarm_resumes_normal_trading(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Arm → run_once (halt); remove flag → run_once behaves normally (scheduler path)."""
        cycle_count: dict[str, int] = {"n": 0}

        def _spy_cycle(*a: Any, **kw: Any) -> Any:
            cycle_count["n"] += 1
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _spy_cycle)

        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag)

        # Arm: first tick must HALT
        r1 = d.run_once(self._now, self._now_ts)
        assert r1 is None
        assert cycle_count["n"] == 0

        # Disarm
        flag.unlink()

        # Next tick: scheduler fires normally (fire time is 00:15, as_of not consumed)
        r2 = d.run_once(self._now, self._now_ts)
        assert r2 is not None
        assert cycle_count["n"] == 1

    def test_rearm_flattens_again(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Arm → run_once; disarm → run_once; re-arm → flattens again (edge reset)."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        from decimal import Decimal  # noqa: PLC0415

        # Use a client that re-opens a position after each close so re-arm matters
        client = _FakeClientWithPositions({"BTC": Decimal("0.5")})
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=client)

        # Arm tick 1 — flatten fires
        d.run_once(self._now, self._now_ts)
        closes_after_arm1 = len(client.market_close_calls)
        assert closes_after_arm1 >= 1

        # Disarm — _kill_handled must reset
        flag.unlink()
        d.run_once(self._now, self._now_ts)  # normal tick (scheduler may or may not fire)
        assert not d._kill_handled  # reset on disarm

        # Re-open a position so re-arm has something to flatten
        client._positions["ETH"] = Decimal("1.0")

        # Re-arm — fresh flatten must trigger
        flag.touch()
        d.run_once(self._now, self._now_ts)
        assert len(client.market_close_calls) > closes_after_arm1

    # ------------------------------------------------------------------
    # kill_flag_path=None disables the feature
    # ------------------------------------------------------------------

    def test_kill_flag_none_disabled(self, tmp_path: Path, monkeypatch: Any) -> None:
        """kill_flag_path=None → no flatten, no KILL_SWITCH alert, normal execution."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        # Daemon built WITHOUT kill_flag_path (default None via _make_daemon)
        d, _, sink, _, _ = _make_daemon(tmp_path)
        assert d.kill_flag_path is None

        d.run_once(self._now, self._now_ts)

        kill_alerts = [a for a in sink.received if a.type == AlertType.KILL_SWITCH]
        assert len(kill_alerts) == 0

    # ------------------------------------------------------------------
    # Kill wins over expiry (ordering check)
    # ------------------------------------------------------------------

    def test_kill_wins_over_expired_agent(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Both kill flag AND expired agent: kill guard fires first (highest priority)."""
        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report())

        now_ts = "2026-06-02T00:15:00+00:00"
        now_unix = datetime.datetime(2026, 6, 2, 0, 15, tzinfo=UTC).timestamp()

        class _CfgExpired(_FakeCfg):
            agent_valid_until_ms = int((now_unix - 86400) * 1000)  # 1 day in the past

        from decimal import Decimal  # noqa: PLC0415

        client = _FakeClientWithPositions({"BTC": Decimal("0.5")})
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=client)
        d.cfg = _CfgExpired  # type: ignore[assignment]

        result = d.run_once(self._now, now_ts)

        # Kill guard fires: returns None
        assert result is None
        # flatten_all attempted (reduce-only market_close)
        assert len(client.market_close_calls) >= 1
        # KILL_SWITCH alert emitted (not only AGENT_EXPIRY)
        kill_alerts = [a for a in sink.received if a.type == AlertType.KILL_SWITCH]
        assert len(kill_alerts) >= 1

    # ------------------------------------------------------------------
    # K1 — unreadable flag → fail-safe HALT (loudly)
    # ------------------------------------------------------------------

    def test_k1_unreadable_flag_fail_safe_halt(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """K1 regression: reading the kill flag raises (e.g. permissions/IO error)
        → run_once must NOT propagate; instead fail-safe HALT: returns None, emits
        KILL_SWITCH CRITICAL, attempts a flatten, does NOT call run_cycle, and
        does NOT consume the as_of.  An unreadable flag must HALT loudly."""
        called: dict[str, bool] = {"did": False}

        def _spy_cycle(*a: Any, **kw: Any) -> Any:
            called["did"] = True
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _spy_cycle)

        from decimal import Decimal  # noqa: PLC0415

        client = _FakeClientWithPositions({"BTC": Decimal("0.5")})
        flag = tmp_path / "kill.flag"
        flag.touch()
        d, sink, _ = _make_kill_daemon(tmp_path, kill_flag_path=flag, client=client)

        # Make the flag-existence check raise — models a permissions / IO error.
        # Patch the daemon's own _kill_armed so only the kill guard's read fails
        # (the ledger / state atomic writes still use a healthy Path).
        def _boom() -> bool:
            raise PermissionError("kill flag unreadable")

        monkeypatch.setattr(d, "_kill_armed", _boom)

        before = d.state.last_scheduled_as_of
        result = d.run_once(self._now, self._now_ts)

        assert result is None  # fail-safe HALT
        assert called["did"] is False  # run_cycle NOT called
        assert d.state.last_scheduled_as_of == before  # as_of NOT consumed
        # Attempted a flatten on the open position (fail-safe close).
        assert len(client.market_close_calls) >= 1
        # KILL_SWITCH CRITICAL emitted.
        kill_alerts = [a for a in sink.received if a.type == AlertType.KILL_SWITCH]
        assert len(kill_alerts) >= 1
        assert any(a.severity == Severity.CRITICAL for a in kill_alerts)
        assert any("unreadable" in a.message for a in kill_alerts)


# ---------------------------------------------------------------------------
# Daemon.boot — sweep_open_orders wiring
# ---------------------------------------------------------------------------


class _FakeClientWithOpenOrders:
    """Fake client that records open_orders / cancel calls; used for sweep tests."""

    def __init__(
        self,
        open_orders: list[dict] | None = None,
        *,
        sweep_raises: bool = False,
    ) -> None:
        self._open_orders = open_orders or []
        self._sweep_raises = sweep_raises
        self.open_orders_calls: int = 0
        self.cancel_oid_calls: list[tuple[str, int]] = []
        self.cancel_by_cloid_calls: list[tuple[str, str]] = []

    # Required by daemon boot / snapshot / etc.
    def positions(self) -> dict:
        return {}

    def equity(self) -> Decimal:
        return Decimal("10000")

    def marks(self) -> dict:
        return {}

    def sz_decimals(self) -> dict:
        return {}

    def open_orders(self) -> list[dict]:
        self.open_orders_calls += 1
        if self._sweep_raises:
            raise RuntimeError("exchange down")
        return list(self._open_orders)

    def cancel_oid(self, coin: str, oid: int) -> dict:
        self.cancel_oid_calls.append((coin, oid))
        return {}

    def cancel_by_cloid(self, coin: str, cloid: str) -> dict:
        self.cancel_by_cloid_calls.append((coin, cloid))
        return {}


class TestDaemonBootSweep:
    """boot() calls sweep_open_orders after reconcile_pending."""

    def _make_sweep_daemon(
        self,
        tmp_path: Path,
        open_orders: list[dict] | None = None,
        *,
        sweep_raises: bool = False,
    ) -> tuple[Daemon, _CapturingAlertSink, _FakeClientWithOpenOrders]:
        state_path = tmp_path / "state.json"
        audit_path = tmp_path / "audit.jsonl"
        ledger_path = tmp_path / "ledger.json"
        sink = _CapturingAlertSink()
        client = _FakeClientWithOpenOrders(open_orders, sweep_raises=sweep_raises)
        d = Daemon(
            cfg=_FakeCfg,  # type: ignore[arg-type]
            client=client,
            source=_FakeSource(),
            ledger=IntentLedger(ledger_path),
            state=State(),
            state_path=state_path,
            audit=AuditLog(audit_path),
            alert_sink=sink,
            strategy_id="test-strategy",
        )
        return d, sink, client

    def test_boot_calls_sweep_when_open_orders_present(self, tmp_path: Path) -> None:
        """boot() calls open_orders and cancels any orphan orders."""
        orphan = {"coin": "BTC", "oid": 42, "sz": "0.5", "side": "B"}
        d, sink, client = self._make_sweep_daemon(tmp_path, open_orders=[orphan])

        d.boot("2026-06-01T00:00:00Z")

        # open_orders was queried
        assert client.open_orders_calls >= 1
        # orphan was cancelled via cancel_oid (no cloid on the order)
        assert len(client.cancel_oid_calls) == 1
        coin, oid = client.cancel_oid_calls[0]
        assert coin == "BTC"
        assert oid == 42

    def test_boot_emits_divergence_alert_for_orphan(self, tmp_path: Path) -> None:
        """boot() emits CHAIN_LEDGER_DIVERGENCE WARNING when orphan orders found."""
        orphan = {"coin": "ETH", "oid": 7, "sz": "1.0", "side": "B"}
        d, sink, client = self._make_sweep_daemon(tmp_path, open_orders=[orphan])

        d.boot("2026-06-01T00:00:00Z")

        div_alerts = [a for a in sink.received if a.type == AlertType.CHAIN_LEDGER_DIVERGENCE]
        assert len(div_alerts) >= 1
        assert div_alerts[0].severity == Severity.WARNING

    def test_boot_no_alert_when_no_open_orders(self, tmp_path: Path) -> None:
        """boot() with no open orders emits NO CHAIN_LEDGER_DIVERGENCE alert."""
        d, sink, client = self._make_sweep_daemon(tmp_path, open_orders=[])

        d.boot("2026-06-01T00:00:00Z")

        div_alerts = [a for a in sink.received if a.type == AlertType.CHAIN_LEDGER_DIVERGENCE]
        assert len(div_alerts) == 0

    def test_boot_sweep_exception_does_not_block_boot(self, tmp_path: Path) -> None:
        """A sweep failure must log and continue — boot must not raise."""
        d, sink, client = self._make_sweep_daemon(tmp_path, sweep_raises=True)

        # Must not raise even though open_orders() raises
        d.boot("2026-06-01T00:00:00Z")

        # The DAEMON_RESTART alert should still have been emitted (boot continued)
        restart_alerts = [a for a in sink.received if a.type == AlertType.DAEMON_RESTART]
        assert len(restart_alerts) == 1


# ---------------------------------------------------------------------------
# G4b MF-9: defer-retry — daemon restores last_scheduled_as_of on deferred cycle
# ---------------------------------------------------------------------------


def _make_deferred_cycle_report() -> CycleReport:
    """Build a CycleReport that looks like a rate-budget deferred cycle."""
    from automation.execution.executor import ExecReport  # noqa: PLC0415

    deferred_exec = ExecReport(
        results=[],
        n_submitted=0,
        n_filled=0,
        n_error=0,
        n_partial=0,
        deferred=True,
        defer_reason="rate budget 3 < est 8 actions",
    )
    return CycleReport(
        rebalanced=True,
        reason="bootstrap",
        exec_report=deferred_exec,
        unresolved_pending=[],
        committed=False,
        deferred=True,
    )


class TestDaemonDeferRetry:
    """G4b: defer-retry — last_scheduled_as_of is RESTORED on a deferred cycle so
    should_fire re-fires the same as_of on the next tick."""

    def test_deferred_cycle_restores_last_scheduled_as_of(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_once where run_cycle returns deferred=True →
        state.last_scheduled_as_of is RESTORED to its prior value (retry next tick)."""
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle",
            lambda *a, **kw: _make_deferred_cycle_report(),
        )
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        prev_scheduled = d.state.last_scheduled_as_of  # None initially
        now = _dt(2026, 6, 2, 0, 15)
        result = d.run_once(now, now.isoformat())

        # Cycle was deferred → last_scheduled_as_of must be RESTORED to prev_scheduled
        assert d.state.last_scheduled_as_of == prev_scheduled, (
            f"Deferred cycle must restore last_scheduled_as_of to {prev_scheduled!r}; "
            f"got {d.state.last_scheduled_as_of!r}"
        )
        # Deferred report is still returned (not None)
        assert result is not None

    def test_deferred_cycle_restore_written_to_disk(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """After a deferred cycle the RESTORED as_of is persisted to disk."""
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle",
            lambda *a, **kw: _make_deferred_cycle_report(),
        )
        d, _, _, state_path, _ = _make_daemon(tmp_path)
        prev = d.state.last_scheduled_as_of  # None

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        loaded = load(state_path)
        assert loaded.last_scheduled_as_of == prev, (
            "Defer-restore must be written to disk so next boot also retries"
        )

    def test_deferred_cycle_should_fire_retries_same_as_of(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """After a deferred cycle, should_fire sees the restored as_of and
        re-fires the same as_of on the next tick (retry)."""
        from automation.runtime.scheduler import should_fire  # noqa: PLC0415

        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle",
            lambda *a, **kw: _make_deferred_cycle_report(),
        )
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        # Load from disk and check that should_fire would still fire
        reloaded = load(state_path)
        fire, as_of_val = should_fire(now, reloaded.last_scheduled_as_of, "00:15")
        assert fire is True, (
            "After a deferred cycle, should_fire must re-fire the same as_of on next tick"
        )

    def test_non_deferred_cycle_consumes_as_of(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Regression: a normal/clean cycle → as_of stays consumed (B10 F1 intact)."""
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle",
            lambda *a, **kw: _clean_report(),
        )
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        # Normal cycle → last_scheduled_as_of must NOT be restored (consumed)
        assert d.state.last_scheduled_as_of == "2026-06-01", (
            "Normal cycle must keep as_of consumed (B10 F1 intact)"
        )
        loaded = load(state_path)
        assert loaded.last_scheduled_as_of == "2026-06-01"

    def test_error_cycle_exception_consumes_as_of(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Regression: run_cycle raises → as_of stays consumed (exception path, B10 F1)."""
        def _bad_cycle(*a: Any, **kw: Any) -> CycleReport:
            raise RuntimeError("exchange down")

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _bad_cycle)
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        # Exception path → as_of stays consumed (not restored)
        assert d.state.last_scheduled_as_of == "2026-06-01"


# ---------------------------------------------------------------------------
# Task 8 — bounded intra-cycle retry: window lifecycle in run_once
# ---------------------------------------------------------------------------


def _make_residual_cycle_report(
    target: dict[str, float] | None = None,
) -> CycleReport:
    """A non-clean daily cycle that left a RETRYABLE residual (a leg was
    liquidity-deferred) and did NOT commit — the condition that OPENS a window."""
    if target is None:
        target = {"TON": 0.5, "HYPE": 0.075}
    from automation.execution.executor import ExecReport  # noqa: PLC0415

    er = ExecReport(
        results=[],
        n_submitted=1,
        n_filled=0,
        n_error=0,
        n_partial=0,
        n_liquidity_deferred=1,
    )
    return CycleReport(
        rebalanced=True,
        reason="bootstrap",
        exec_report=er,
        unresolved_pending=[],
        committed=False,
        target_weights=dict(target),
        n_liquidity_deferred=1,
    )


class TestDaemonRetryWindow:
    """run_once opens / advances / closes the bounded-retry window."""

    def test_run_once_opens_retry_window_on_nonclean_daily(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Daily cycle defers a leg (n_liquidity_deferred>0, not committed) → window opens.
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle",
            lambda *a, **kw: _make_residual_cycle_report({"TON": 0.5, "HYPE": 0.075}),
        )
        d, _, _, state_path, _ = _make_daemon(tmp_path)
        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())

        assert d.state.retry_as_of == "2026-06-01"
        assert d.state.retry_target == {"TON": 0.5, "HYPE": 0.075}
        assert d.state.retry_window_start == now.isoformat()
        assert d.state.retry_last_attempt == now.isoformat()
        assert d.state.retry_attempt_count == 0
        # persisted to disk
        loaded = load(state_path)
        assert loaded.retry_as_of == "2026-06-01"

    def test_run_once_no_window_when_clean_daily(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Clean daily cycle (committed=True) → no window opened.
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report()
        )
        d, _, _, _, _ = _make_daemon(tmp_path)
        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())
        assert d.state.retry_as_of is None

    def test_run_once_no_window_when_disabled(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # retry disabled → never opens a window even on a non-clean daily cycle.
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle",
            lambda *a, **kw: _make_residual_cycle_report(),
        )
        monkeypatch.setattr(_FakeCfg.retry, "enabled", False)
        d, _, _, _, _ = _make_daemon(tmp_path)
        now = _dt(2026, 6, 2, 0, 15)
        d.run_once(now, now.isoformat())
        assert d.state.retry_as_of is None

    def test_run_once_fires_retry_after_interval_and_clears_on_clean(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Window pre-set; >interval since last attempt; retry fills cleanly →
        # window cleared, baseline committed.
        state = State()
        state.last_scheduled_as_of = "2026-05-17"  # already fired today → no new daily cycle
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 1

        calls: dict[str, Any] = {}

        def _fake_retry(*a: Any, **kw: Any) -> CycleReport:
            calls["fired"] = True
            calls["retry_as_of"] = kw.get("retry_as_of")
            # Commit the baseline as a clean retry would.
            st: State = a[3]
            st.last_rebalanced_as_of = "2026-05-17"
            st.last_rebalanced_target = {"TON": 0.5}
            return CycleReport(
                rebalanced=True, reason="retry", exec_report=_make_exec_report(),
                unresolved_pending=[], committed=True, target_weights={"TON": 0.5},
            )

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "retry_attempt", _fake_retry)
        d, _, _, state_path, _ = _make_daemon(tmp_path, state=state)

        now = _dt(2026, 5, 18, 0, 27)  # +12 min after last attempt, within 3 h window
        d.run_once(now, now.isoformat())

        assert calls.get("fired") is True
        assert calls["retry_as_of"] == "2026-05-17"
        assert d.state.retry_as_of is None  # cleared after clean retry
        assert d.state.last_rebalanced_as_of == "2026-05-17"
        loaded = load(state_path)
        assert loaded.retry_as_of is None

    def test_run_once_skips_retry_before_interval(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Window open; now only +5 min since last attempt → no retry fired,
        # window unchanged, attempt_count unchanged.
        state = State()
        state.last_scheduled_as_of = "2026-05-17"
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 1

        fired: dict[str, bool] = {"did": False}

        def _fake_retry(*a: Any, **kw: Any) -> CycleReport:
            fired["did"] = True
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "retry_attempt", _fake_retry)
        d, _, _, _, _ = _make_daemon(tmp_path, state=state)

        now = _dt(2026, 5, 18, 0, 20)  # +5 min < 10 min interval
        d.run_once(now, now.isoformat())

        assert fired["did"] is False
        assert d.state.retry_as_of == "2026-05-17"  # window unchanged
        assert d.state.retry_last_attempt == "2026-05-18T00:15:00+00:00"
        assert d.state.retry_attempt_count == 1

    def test_run_once_best_effort_commit_on_window_expiry(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Window open; now = +3 h 05 (past the 3 h window) → best-effort commit +
        # UNFILLED alert + cleared.
        state = State()
        state.last_scheduled_as_of = "2026-05-17"
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5, "HYPE": 0.075}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 3

        d, _, sink, state_path, _ = _make_daemon(tmp_path, state=state)
        now = _dt(2026, 5, 18, 3, 20)  # +3 h 05 > 3 h window
        d.run_once(now, now.isoformat())

        assert d.state.retry_as_of is None
        assert d.state.last_rebalanced_target == {"TON": 0.5, "HYPE": 0.075}
        assert d.state.last_rebalanced_as_of == "2026-05-17"
        assert any(a.type == AlertType.UNFILLED for a in sink.received)
        loaded = load(state_path)
        assert loaded.retry_as_of is None
        assert loaded.last_rebalanced_as_of == "2026-05-17"

    def test_run_once_best_effort_commit_on_day_rollover(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Window open for 2026-05-17; now is the NEXT day's bar → the old window's
        # expected_as_of no longer matches → best-effort commit + cleared.
        # last_scheduled_as_of left at 2026-05-18 so the new day's should_fire does
        # NOT fire a fresh daily cycle this tick (we only assert the old window cleared).
        state = State()
        state.last_scheduled_as_of = "2026-05-18"
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 2

        d, _, sink, _, _ = _make_daemon(tmp_path, state=state)
        # 2026-05-19 00:20 → expected_as_of = 2026-05-18 != retry_as_of 2026-05-17.
        now = _dt(2026, 5, 19, 0, 20)
        d.run_once(now, now.isoformat())

        assert d.state.retry_as_of is None
        assert d.state.last_rebalanced_target == {"TON": 0.5}
        assert d.state.last_rebalanced_as_of == "2026-05-17"
        assert any(a.type == AlertType.UNFILLED for a in sink.received)


class TestDaemonBootOrphanReconcile:
    """boot() reconciles orphan PENDING intents via the CHAIN resolver (Layer 4)."""

    def test_boot_reconciles_orphan_pending_via_chain_resolver(
        self, tmp_path: Path
    ) -> None:
        from automation.runtime.intent_ledger import Intent  # noqa: PLC0415

        ledger_path = tmp_path / "ledger.json"
        ledger = IntentLedger(ledger_path)
        ledger.record_pending(
            Intent(
                cloid="0x" + "b" * 32,
                as_of="2026-05-17",
                coin="TON",
                target_w=0.5,
                attempt_seq=0,
                status="PENDING",
                ts="2026-05-18T00:15:00",
            )
        )
        assert len(ledger.pending()) == 1

        # Chain resolver reports the orphan PENDING as actually FILLED on-chain.
        d = Daemon(
            cfg=_FakeCfg,  # type: ignore[arg-type]
            client=_FakeClient(),
            source=_FakeSource(),
            ledger=ledger,
            state=State(),
            state_path=tmp_path / "state.json",
            audit=AuditLog(tmp_path / "audit.jsonl"),
            alert_sink=_CapturingAlertSink(),
            strategy_id="test",
            is_filled=lambda _intent: True,
        )
        d.boot("2026-05-18T00:30:00")

        # Orphan PENDING must be reconciled to DONE (zero remaining PENDING).
        assert len(ledger.pending()) == 0

    def test_boot_default_resolver_leaves_pending(self, tmp_path: Path) -> None:
        # No is_filled supplied → conservative resolver leaves the intent PENDING
        # (regression: default behaviour unchanged).
        from automation.runtime.intent_ledger import Intent  # noqa: PLC0415

        ledger = IntentLedger(tmp_path / "ledger.json")
        ledger.record_pending(
            Intent(
                cloid="0x" + "c" * 32, as_of="2026-05-17", coin="TON",
                target_w=0.5, attempt_seq=0, status="PENDING",
                ts="2026-05-18T00:15:00",
            )
        )
        d, _, _, _, _ = _make_daemon(tmp_path)
        d.ledger = ledger
        d.boot("2026-05-18T00:30:00")
        assert len(ledger.pending()) == 1


# ---------------------------------------------------------------------------
# Holistic-review fixes: C1 (supersede stale window on new daily fire) and
# I1 (drain window when retry disabled)
# ---------------------------------------------------------------------------


def _clean_report_for(target: dict[str, float]) -> CycleReport:
    """A CLEAN committed daily cycle that also carries a fresh baseline target.

    Mimics run_cycle on the success path: it commits ``last_rebalanced_*`` to
    the served target before returning ``committed=True``.  We bake the
    state-mutation into the monkeypatched callable below (the report object
    alone does not mutate state).
    """
    return CycleReport(
        rebalanced=True,
        reason="bootstrap",
        exec_report=_make_exec_report(n_submitted=1, n_filled=1),
        unresolved_pending=[],
        committed=True,
        target_weights=dict(target),
    )


class TestDaemonSupersedeStaleWindow:
    """C1: a new daily cycle for as_of_new must SUPERSEDE a stale window for a
    DIFFERENT as_of BEFORE run_cycle commits — so the later no-fire tick can't
    best-effort-commit the stale target and clobber the fresh baseline."""

    def test_new_daily_fire_supersedes_stale_window_and_keeps_fresh_baseline(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Open a stale window for D = 2026-05-17.
        state = State()
        # last_scheduled_as_of = D so should_fire fires the NEXT day's D+1.
        state.last_scheduled_as_of = "2026-05-17"
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 2

        d_plus_1_target = {"BTC": 0.8, "ETH": 0.1}

        def _clean_commit(*a: Any, **kw: Any) -> CycleReport:
            # run_cycle commits the fresh D+1 baseline as the real one would.
            st: State = a[4]
            st.last_rebalanced_as_of = "2026-05-18"
            st.last_rebalanced_target = dict(d_plus_1_target)
            return _clean_report_for(d_plus_1_target)

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _clean_commit)
        d, _, sink, state_path, _ = _make_daemon(tmp_path, state=state)

        # D+1 = 2026-05-19 fire time → expected_as_of = 2026-05-18 (D+1).
        now = _dt(2026, 5, 19, 0, 15)
        d.run_once(now, now.isoformat())

        # Stale window superseded (cleared) in the SAME run_once, fresh baseline intact.
        assert d.state.retry_as_of is None
        assert d.state.last_rebalanced_as_of == "2026-05-18"
        assert d.state.last_rebalanced_target == d_plus_1_target
        # No spurious UNFILLED from a best-effort-commit of the stale D target.
        assert not any(a.type == AlertType.UNFILLED for a in sink.received)
        loaded = load(state_path)
        assert loaded.retry_as_of is None
        assert loaded.last_rebalanced_as_of == "2026-05-18"

    def test_later_no_fire_tick_does_not_clobber_fresh_baseline(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Same setup; after the D+1 fire, a later no-fire tick must NOT clobber
        # the D+1 baseline (the stale window is already gone).
        state = State()
        state.last_scheduled_as_of = "2026-05-17"
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 2

        d_plus_1_target = {"BTC": 0.8, "ETH": 0.1}

        def _clean_commit(*a: Any, **kw: Any) -> CycleReport:
            st: State = a[4]
            st.last_rebalanced_as_of = "2026-05-18"
            st.last_rebalanced_target = dict(d_plus_1_target)
            return _clean_report_for(d_plus_1_target)

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _clean_commit)
        d, _, sink, _, _ = _make_daemon(tmp_path, state=state)

        d.run_once(_dt(2026, 5, 19, 0, 15), _dt(2026, 5, 19, 0, 15).isoformat())
        # A later no-fire tick the same day (post-fire, idempotent hold).
        later = _dt(2026, 5, 19, 0, 30)
        d.run_once(later, later.isoformat())

        assert d.state.retry_as_of is None
        assert d.state.last_rebalanced_as_of == "2026-05-18"
        assert d.state.last_rebalanced_target == d_plus_1_target
        assert not any(a.type == AlertType.UNFILLED for a in sink.received)


class TestDaemonDrainWindowWhenDisabled:
    """I1: if retry.enabled flips False while a window is open, the next no-fire
    tick must DRAIN it (best-effort commit + clear), not leak it forever."""

    def test_disabled_while_open_drains_window(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        state = State()
        state.last_scheduled_as_of = "2026-05-17"
        state.retry_as_of = "2026-05-17"
        state.retry_target = {"TON": 0.5, "HYPE": 0.075}
        state.retry_window_start = "2026-05-18T00:15:00+00:00"
        state.retry_last_attempt = "2026-05-18T00:15:00+00:00"
        state.retry_attempt_count = 1

        # Operator disabled the feature while the window was open.
        monkeypatch.setattr(_FakeCfg.retry, "enabled", False)

        fired: dict[str, bool] = {"did": False}

        def _fake_retry(*a: Any, **kw: Any) -> CycleReport:
            fired["did"] = True
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "retry_attempt", _fake_retry)
        d, _, sink, state_path, _ = _make_daemon(tmp_path, state=state)

        # No-fire tick (already fired today; +5 min — would be too-early if enabled).
        now = _dt(2026, 5, 18, 0, 20)
        d.run_once(now, now.isoformat())

        # Window drained, NOT a retry fired.
        assert fired["did"] is False
        assert d.state.retry_as_of is None
        assert d.state.last_rebalanced_target == {"TON": 0.5, "HYPE": 0.075}
        assert d.state.last_rebalanced_as_of == "2026-05-17"
        unfilled = [a for a in sink.received if a.type == AlertType.UNFILLED]
        assert len(unfilled) == 1
        loaded = load(state_path)
        assert loaded.retry_as_of is None


# ---------------------------------------------------------------------------
# PH-4 — post-cycle position-health hook
# ---------------------------------------------------------------------------
class TestDaemonHealthHook:
    """The guarded post-cycle health monitor (PH-4 UNIT B).

    Runs AFTER the cycle completes (state saved, alerts emitted), gated on
    ``cfg.health.enabled``, and fully wrapped so a monitoring failure NEVER
    aborts/affects the cycle.  Driven on a real FIRED cycle (run_cycle
    monkeypatched to a clean report) with a breaching FakeHLClient as the
    daemon's ``self.client``.
    """

    _now = _dt(2026, 6, 2, 0, 15)
    _now_ts = _dt(2026, 6, 2, 0, 15).isoformat()

    @staticmethod
    def _breaching_client() -> Any:
        from decimal import Decimal as _D  # noqa: PLC0415

        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        return FakeHLClient(
            account_snapshot={
                "equity": 1000.0,
                "maint_margin": 600.0,  # 0.6 > 0.5 → MARGIN_RATIO_LOW
                "positions": {
                    "BTC": {
                        "szi": 0.5,
                        "position_value": 30000.0,
                        "unrealized_pnl": 0.0,
                        "liquidation_px": 97.0,  # 3% from mark=100 → CRITICAL
                        "funding_since_open": 60.0,  # 6% of equity → FUNDING_DRAG
                        "leverage_type": "cross",
                        "leverage_value": 3,
                    },
                },
            },
            marks={"BTC": _D("100")},
            venue_meta={},  # BTC absent → COIN_VENUE_DEGRADED
            spot_balances={"USDC": _D("5.0")},  # 5 > 1 → COLLATERAL_IN_SPOT
        )

    class _CfgHealthOn(_FakeCfg):
        class health:
            enabled: bool = True
            funding_drag_pct: float = 5.0
            margin_maint_frac_warn: float = 0.5
            liq_distance_pct_warn: float = 10.0
            spot_collateral_warn_usd: float = 1.0

    def test_fired_cycle_emits_health_findings(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report()
        )
        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = self._CfgHealthOn  # type: ignore[assignment]
        d.client = self._breaching_client()

        result = d.run_once(self._now, self._now_ts)

        # Cycle is unaffected.
        assert result is not None
        assert result.rebalanced is True

        types = {a.type for a in sink.received}
        assert AlertType.FUNDING_DRAG in types
        assert AlertType.MARGIN_RATIO_LOW in types
        assert AlertType.COIN_VENUE_DEGRADED in types
        assert AlertType.COLLATERAL_IN_SPOT in types

    def test_disabled_emits_no_health_findings(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report()
        )
        # Default _FakeCfg.health.enabled is False.
        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.client = self._breaching_client()

        result = d.run_once(self._now, self._now_ts)
        assert result is not None

        health_types = {
            AlertType.FUNDING_DRAG,
            AlertType.MARGIN_RATIO_LOW,
            AlertType.COIN_VENUE_DEGRADED,
            AlertType.COLLATERAL_IN_SPOT,
        }
        assert not any(a.type in health_types for a in sink.received)

    def test_health_failure_never_aborts_cycle(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """gather/assess raising is caught — run_once still returns the report."""
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report()
        )

        from automation.tests._fakes import FakeHLClient  # noqa: PLC0415

        class _BoomClient(FakeHLClient):
            def account_snapshot(self) -> dict[str, Any]:
                raise RuntimeError("health read boom")

        d, _, sink, _, _ = _make_daemon(tmp_path)
        d.cfg = self._CfgHealthOn  # type: ignore[assignment]
        d.client = _BoomClient()

        # No exception propagates; the cycle report is returned intact.
        result = d.run_once(self._now, self._now_ts)
        assert result is not None
        assert result.rebalanced is True

        health_types = {
            AlertType.FUNDING_DRAG,
            AlertType.MARGIN_RATIO_LOW,
            AlertType.COIN_VENUE_DEGRADED,
            AlertType.COLLATERAL_IN_SPOT,
        }
        assert not any(a.type in health_types for a in sink.received)


# ---------------------------------------------------------------------------
# Heartbeat liveness — run_once beats every tick (incl. idle ticks)
# ---------------------------------------------------------------------------


class TestDaemonHeartbeat:
    def test_run_once_beats_heartbeat(self, tmp_path: Path) -> None:
        """run_once must beat the attached heartbeat each tick, even an idle one.

        Fire BEFORE the rebalance time (00:15) so should_fire declines and no
        cycle runs — the beat must still land (the beat is at the VERY TOP of
        run_once, before every guard, including scheduling).
        """
        from automation.runtime.health import Heartbeat

        d, _, _, _, _ = _make_daemon(tmp_path)
        hb = Heartbeat(max_silence_seconds=180)
        d.heartbeat = hb
        now = _dt(2026, 5, 18, hour=0, minute=0)  # before fire time => idle tick
        now_ts = now.isoformat()
        d.run_once(now, now_ts)
        # A near-future time within the silence window is still healthy.
        assert hb.is_healthy(now_ts="2026-05-18T00:02:00+00:00") is True

    def test_run_once_beats_heartbeat_when_kill_armed(self, tmp_path: Path) -> None:
        """A KILL-armed daemon is ALIVE and looping → it must STILL beat (HEALTHY).

        Liveness answers "is the process alive and looping?", NOT "is it
        trading?".  If a KILL-armed daemon stopped beating, /healthz would go
        503 and the supervisor would restart-churn it forever against the
        operator's deliberate stop.  So: arm the kill flag, run a tick, and
        assert run_once HALTed (returned None) yet the heartbeat beat.
        """
        from automation.runtime.health import Heartbeat

        kill_flag = tmp_path / "KILL"
        kill_flag.write_text("armed", encoding="utf-8")  # arm the Tier-1 kill flag

        d, _, _, _, _ = _make_daemon(tmp_path)
        d.kill_flag_path = kill_flag  # flat _FakeClient → flatten_all is a no-op
        hb = Heartbeat(max_silence_seconds=180)
        d.heartbeat = hb

        now = _dt(2026, 5, 18, hour=0, minute=15)  # at fire time, but KILL preempts
        now_ts = now.isoformat()
        result = d.run_once(now, now_ts)

        assert result is None  # HALT: kill guard returned None, no cycle fired
        # Killed-but-alive daemon stays HEALTHY (beat landed before the guard).
        assert hb.is_healthy(now_ts="2026-05-18T00:15:01+00:00") is True


# ---------------------------------------------------------------------------
# SP3 §5 — no-signal re-fire: bounded re-check window
# ---------------------------------------------------------------------------


def _no_signal_hold_report() -> CycleReport:
    """The EXACT no-good-signal HOLD shape returned by
    ``rebalance._handle_no_good_signal`` within the staleness window:
    uncommitted, structurally zero orders (``exec_report=None`` — the executor
    was never reached), ``signal_ok=False``, NOT deferred."""
    return CycleReport(
        rebalanced=False,
        reason="hold_stale_within_window",
        exec_report=None,
        unresolved_pending=[],
        committed=False,
        signal_ok=False,
        derisked=False,
        data_age_days=-1,
    )


def _derisk_report() -> CycleReport:
    """A de-risk-to-cash report: the staleness clock expired and the cycle
    flattened — ``committed=True`` (that day ACTED), ``signal_ok=False``."""
    return CycleReport(
        rebalanced=False,
        reason="derisk_to_cash",
        exec_report=None,
        unresolved_pending=[],
        committed=True,
        signal_ok=False,
        derisked=True,
        data_age_days=-1,
    )


class TestNoSignalRecheck:
    """SP3 §5 — a NO-GOOD-SIGNAL HOLD must not consume the day: the daemon
    restores the idempotency key (MF-9 mechanism — zero orders submitted →
    re-fire is double-trade-safe) and re-checks on a bounded cadence until a
    non-hold outcome lands or the window expires."""

    _T0 = _dt(2026, 6, 2, 0, 15)  # fire tick → as_of "2026-06-01"
    _AS_OF = "2026-06-01"

    @staticmethod
    def _install_counting_hold(monkeypatch: Any) -> dict[str, int]:
        """Monkeypatch run_cycle with a COUNTING fake that always holds."""
        calls = {"n": 0}

        def _hold_cycle(*a: Any, **kw: Any) -> CycleReport:
            calls["n"] += 1
            return _no_signal_hold_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _hold_cycle)
        return calls

    def test_hold_restores_last_scheduled_and_opens_window(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        calls = self._install_counting_hold(monkeypatch)
        d, _, _, _, _ = _make_daemon(tmp_path)

        result = d.run_once(self._T0, self._T0.isoformat())

        assert result is not None
        assert calls["n"] == 1
        # Idempotency key RESTORED to prev value (None) — day NOT consumed.
        assert d.state.last_scheduled_as_of is None
        # Window opened for this as_of, first attempt recorded.
        assert d.state.recheck_as_of == self._AS_OF
        assert d.state.recheck_window_start == self._T0.isoformat()
        assert d.state.recheck_last_attempt == self._T0.isoformat()
        assert d.state.recheck_attempt_count == 1

    def test_restore_and_window_in_single_persisted_save(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        self._install_counting_hold(monkeypatch)
        d, _, _, state_path, _ = _make_daemon(tmp_path)

        d.run_once(self._T0, self._T0.isoformat())

        # Reload from DISK: restore AND window fields both present (atomicity
        # of the combined save — no state where one landed without the other).
        loaded = load(state_path)
        assert loaded.last_scheduled_as_of is None
        assert loaded.recheck_as_of == self._AS_OF
        assert loaded.recheck_window_start == self._T0.isoformat()
        assert loaded.recheck_last_attempt == self._T0.isoformat()
        assert loaded.recheck_attempt_count == 1

    def test_refire_blocked_within_interval(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        calls = self._install_counting_hold(monkeypatch)
        d, _, _, _, _ = _make_daemon(tmp_path)
        d.run_once(self._T0, self._T0.isoformat())

        t1 = self._T0 + datetime.timedelta(seconds=30)  # +30 s < 180 s interval
        result = d.run_once(t1, t1.isoformat())

        assert result is None
        assert calls["n"] == 1  # cycle NOT re-run
        assert d.state.recheck_attempt_count == 1
        assert d.state.recheck_last_attempt == self._T0.isoformat()  # untouched

    def test_refire_fires_after_interval(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        calls = self._install_counting_hold(monkeypatch)
        d, _, _, _, _ = _make_daemon(tmp_path)
        d.run_once(self._T0, self._T0.isoformat())

        t1 = self._T0 + datetime.timedelta(seconds=181)  # interval+1 s
        result = d.run_once(t1, t1.isoformat())

        assert result is not None
        assert calls["n"] == 2  # cycle re-ran
        assert d.state.recheck_attempt_count == 2
        assert d.state.recheck_last_attempt == t1.isoformat()
        # Window start NOT reset on a subsequent hold for the SAME as_of.
        assert d.state.recheck_window_start == self._T0.isoformat()

    def test_window_expiry_consumes_day_and_alerts(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        calls = self._install_counting_hold(monkeypatch)
        d, _, sink, state_path, _ = _make_daemon(tmp_path)
        d.run_once(self._T0, self._T0.isoformat())

        # Exactly window_seconds later: >= semantics (retry_window_expired is
        # the authoritative expiry check) → consume the day.
        t_exp = self._T0 + datetime.timedelta(seconds=7200)
        result = d.run_once(t_exp, t_exp.isoformat())

        assert result is None
        assert calls["n"] == 1  # NO cycle ran on the expiry tick
        assert d.state.last_scheduled_as_of == self._AS_OF  # day consumed
        assert d.state.recheck_as_of is None
        assert d.state.recheck_window_start is None
        assert d.state.recheck_last_attempt is None
        assert d.state.recheck_attempt_count == 0
        assert d.state.recheck_exhausted_as_of == self._AS_OF
        ex = [a for a in sink.received if a.type == AlertType.SIGNAL_RECHECK_EXHAUSTED]
        assert len(ex) == 1
        assert ex[0].severity == Severity.CRITICAL
        assert ex[0].context["as_of"] == self._AS_OF
        assert ex[0].context["attempts"] == 1
        assert ex[0].context["window_seconds"] == 7200
        loaded = load(state_path)
        assert loaded.last_scheduled_as_of == self._AS_OF
        assert loaded.recheck_exhausted_as_of == self._AS_OF

    def test_stale_window_superseded_on_new_day(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # A re-check window is still OPEN for D="2026-05-17" (idempotency key
        # restored to D-1 by the prior holds), and the NEW day D+1 fires.
        state = State()
        state.last_scheduled_as_of = "2026-05-16"
        state.recheck_as_of = "2026-05-17"
        state.recheck_window_start = "2026-05-18T23:00:00+00:00"
        state.recheck_last_attempt = "2026-05-18T23:55:00+00:00"
        state.recheck_attempt_count = 5
        d, _, _, state_path, _ = _make_daemon(tmp_path, state=state)

        # Capture-at-call-time fake: record the recheck state AS SEEN INSIDE
        # run_cycle (in memory AND on disk).  End-state-only assertions cannot
        # pin the supersede block — the hold path's fresh-open produces the
        # same final state — so we prove D's window was cleared AND PERSISTED
        # BEFORE the cycle ran (the on-disk clear is what keeps the boot
        # crash-detection marker valid during the new day's cycle).
        seen: dict[str, Any] = {}

        def _capturing_hold(*a: Any, **kw: Any) -> CycleReport:
            seen["mem_recheck_as_of"] = d.state.recheck_as_of
            seen["disk_recheck_as_of"] = load(state_path).recheck_as_of
            return _no_signal_hold_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _capturing_hold)

        # D+1 fire tick: 2026-05-19 00:15 → expected as_of "2026-05-18".
        now = _dt(2026, 5, 19, 0, 15)
        d.run_once(now, now.isoformat())

        # D's window was cleared AND persisted BEFORE run_cycle ran (an empty
        # ``seen`` would also fail here — the cycle must have fired exactly so).
        assert seen == {"mem_recheck_as_of": None, "disk_recheck_as_of": None}
        # The D+1 hold then opened a FRESH window — never inheriting
        # yesterday's start/attempt clock.
        assert d.state.recheck_as_of == "2026-05-18"
        assert d.state.recheck_window_start == now.isoformat()
        assert d.state.recheck_last_attempt == now.isoformat()
        assert d.state.recheck_attempt_count == 1
        # Supersede never touches the exhausted marker.
        assert d.state.recheck_exhausted_as_of is None

    def test_exception_path_closes_window(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # A window is OPEN from a prior hold and the re-check attempt's
        # run_cycle RAISES.  The exception path's minimal CycleReport keeps
        # ``signal_ok=True`` (the default _is_no_signal_hold relies on), so it
        # is NOT a hold: the day stays consumed (F1 error-loop protection) and
        # the close branch clears the now-orphaned window.
        def _boom(*a: Any, **kw: Any) -> CycleReport:
            raise RuntimeError("exchange down")

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _boom)
        state = State()
        state.last_scheduled_as_of = None  # restored by the prior hold
        state.recheck_as_of = self._AS_OF
        state.recheck_window_start = self._T0.isoformat()
        state.recheck_last_attempt = self._T0.isoformat()
        state.recheck_attempt_count = 1
        d, _, _, state_path, _ = _make_daemon(tmp_path, state=state)

        t1 = self._T0 + datetime.timedelta(seconds=181)  # cadence elapsed → re-fire
        result = d.run_once(t1, t1.isoformat())

        assert result is not None
        assert result.committed is False
        # Day stays consumed — the exception path never restores.
        assert d.state.last_scheduled_as_of == self._AS_OF
        # The orphaned window is CLOSED (in memory and on disk).
        assert d.state.recheck_as_of is None
        assert d.state.recheck_window_start is None
        assert d.state.recheck_last_attempt is None
        assert d.state.recheck_attempt_count == 0
        loaded = load(state_path)
        assert loaded.recheck_as_of is None
        assert loaded.last_scheduled_as_of == self._AS_OF

    def test_success_clears_recheck_state(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # First cycle holds (no signal), second re-check gets a GOOD signal.
        reports = [_no_signal_hold_report(), _clean_report()]
        calls = {"n": 0}

        def _seq_cycle(*a: Any, **kw: Any) -> CycleReport:
            r = reports[calls["n"]]
            calls["n"] += 1
            return r

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _seq_cycle)
        d, _, _, _, _ = _make_daemon(tmp_path)

        d.run_once(self._T0, self._T0.isoformat())
        t1 = self._T0 + datetime.timedelta(seconds=181)
        result = d.run_once(t1, t1.isoformat())

        assert result is not None
        assert result.committed is True
        assert calls["n"] == 2
        # Good cycle CLOSED the window and the day stays consumed.
        assert d.state.recheck_as_of is None
        assert d.state.recheck_window_start is None
        assert d.state.recheck_last_attempt is None
        assert d.state.recheck_attempt_count == 0
        assert d.state.last_scheduled_as_of == self._AS_OF

    def test_derisk_does_not_restore(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # De-risk (committed=True, exec_report None, signal_ok False): that day
        # ACTED — the day stays consumed and NO re-check window opens.
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _derisk_report()
        )
        d, _, _, _, _ = _make_daemon(tmp_path)

        d.run_once(self._T0, self._T0.isoformat())

        assert d.state.last_scheduled_as_of == self._AS_OF  # consumed
        assert d.state.recheck_as_of is None
        assert d.state.recheck_window_start is None
        assert d.state.recheck_attempt_count == 0

    def test_disabled_legacy_behavior(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # signal_recheck.enabled=False → a hold consumes the day (legacy).
        monkeypatch.setattr(_FakeCfg.signal_recheck, "enabled", False)
        calls = self._install_counting_hold(monkeypatch)
        d, _, _, _, _ = _make_daemon(tmp_path)

        d.run_once(self._T0, self._T0.isoformat())

        assert calls["n"] == 1
        assert d.state.last_scheduled_as_of == self._AS_OF  # consumed (no restore)
        assert d.state.recheck_as_of is None  # no window opened

    def test_mf9_deferred_not_treated_as_hold(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod,
            "run_cycle",
            lambda *a, **kw: _make_deferred_cycle_report(),
        )
        # Part 1 — fresh daemon: a deferred report takes ONLY the MF-9 restore;
        # NO re-check window opens (a defer says nothing about the signal).
        d, _, _, _, _ = _make_daemon(tmp_path)
        d.run_once(self._T0, self._T0.isoformat())
        assert d.state.last_scheduled_as_of is None  # MF-9 restore intact
        assert d.state.recheck_as_of is None  # no recheck window opened

        # Part 2 — an OPEN window from a prior hold stays UNTOUCHED when a
        # re-check attempt comes back deferred.
        state = State()
        state.last_scheduled_as_of = None  # restored by the prior hold
        state.recheck_as_of = self._AS_OF
        state.recheck_window_start = self._T0.isoformat()
        state.recheck_last_attempt = self._T0.isoformat()
        state.recheck_attempt_count = 2
        d2, _, _, _, _ = _make_daemon(tmp_path, state=state)

        t1 = self._T0 + datetime.timedelta(seconds=181)
        result = d2.run_once(t1, t1.isoformat())

        assert result is not None
        assert result.deferred is True
        # Window untouched — neither advanced nor closed.
        assert d2.state.recheck_as_of == self._AS_OF
        assert d2.state.recheck_window_start == self._T0.isoformat()
        assert d2.state.recheck_last_attempt == self._T0.isoformat()
        assert d2.state.recheck_attempt_count == 2
        # MF-9 restore happened as before.
        assert d2.state.last_scheduled_as_of is None

    def test_boot_warns_on_suspect_dropped_day(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        calls = {"n": 0}

        def _cycle(*a: Any, **kw: Any) -> CycleReport:
            calls["n"] += 1
            return _clean_report()

        monkeypatch.setattr(_daemon_mod._rebalance_mod, "run_cycle", _cycle)
        # The crash marker: day consumed, no recorded rebalance for it, no open
        # retry/recheck windows, and the window was NOT deliberately exhausted.
        state = State()
        state.last_scheduled_as_of = self._AS_OF  # == expected_as_of at _T0
        state.last_rebalanced_as_of = "2026-05-31"  # lags the scheduled day
        d, _, sink, _, _ = _make_daemon(tmp_path, state=state)

        r1 = d.run_once(self._T0, self._T0.isoformat())
        t1 = self._T0 + datetime.timedelta(seconds=60)
        r2 = d.run_once(t1, t1.isoformat())

        # NO re-fire — detection only (fail-safe wins over availability).
        assert r1 is None
        assert r2 is None
        assert calls["n"] == 0
        assert d.state.last_scheduled_as_of == self._AS_OF  # state unchanged
        warns = [
            a
            for a in sink.received
            if a.type == AlertType.DAEMON_RESTART and a.severity == Severity.WARNING
        ]
        assert len(warns) == 1  # exactly ONCE across two ticks
        assert warns[0].context["suspect_dropped_as_of"] == self._AS_OF

    def test_boot_no_warn_when_window_was_exhausted(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            _daemon_mod._rebalance_mod, "run_cycle", lambda *a, **kw: _clean_report()
        )
        # Same marker as above, but the day was DELIBERATELY consumed by a
        # re-check window expiry → silent.
        state = State()
        state.last_scheduled_as_of = self._AS_OF
        state.last_rebalanced_as_of = "2026-05-31"
        state.recheck_exhausted_as_of = self._AS_OF
        d, _, sink, _, _ = _make_daemon(tmp_path, state=state)

        result = d.run_once(self._T0, self._T0.isoformat())

        assert result is None
        warns = [
            a
            for a in sink.received
            if a.type == AlertType.DAEMON_RESTART and a.severity == Severity.WARNING
        ]
        assert len(warns) == 0
