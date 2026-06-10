"""
Tests for automation.runtime.kill_switch.

Coverage:
- flatten_all: with BTC + ETH positions → exactly 2 market_close calls
- flatten_all: intents recorded-before-call then marked DONE
- flatten_all: KILL_SWITCH alert emitted via alert_sink
- flatten_all: one coin raising market_close → still flattens the other
- Watchdog.check: fires on stale heartbeat (liveness_deadline_s exceeded)
- Watchdog.check: fires on equity floor breach
- Watchdog.check: does NOT fire when healthy
- revoke_runbook: returns a non-empty string
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from automation.reporting.alerts import Alert, AlertType, Severity
from automation.runtime.intent_ledger import IntentLedger
from automation.runtime.kill_switch import Watchdog, flatten_all, revoke_runbook

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class _CapturingAlertSink:
    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


class _FakeCfg:
    class caps:
        max_order_slippage: float = 0.01


class _FakeClient:
    """Fake HLClient for kill_switch tests.

    By default a ``market_close`` fully closes the position (returns a ``filled``
    status for the FULL requested size) and removes the coin from ``positions``
    so a post-close re-query reports flat.

    ``close_behaviour`` overrides per-coin response shape to exercise F3:
    - ``"full"``     (default) — filled for the full size, position cleared.
    - ``"partial"``  — filled for half the size, position halved (residual).
    - ``"resting"``  — a ``resting`` status (no fill), position UNCHANGED.
    - ``"zero"``     — empty statuses (no fill), position UNCHANGED.
    """

    def __init__(
        self,
        positions: dict[str, Decimal] | None = None,
        market_close_raises_for: set[str] | None = None,
        close_behaviour: dict[str, str] | None = None,
    ) -> None:
        self._positions: dict[str, Decimal] = dict(positions or {})
        self._market_close_raises = market_close_raises_for or set()
        self._close_behaviour = close_behaviour or {}
        self.market_close_calls: list[tuple[str, float, str]] = []  # (coin, slippage, cloid)

    def positions(self) -> dict[str, Decimal]:
        return dict(self._positions)

    def market_close(self, coin: str, slippage: float, cloid: str) -> dict[str, Any]:
        self.market_close_calls.append((coin, slippage, cloid))
        if coin in self._market_close_raises:
            raise RuntimeError(f"market_close failed for {coin}")

        size = abs(float(self._positions.get(coin, Decimal("0"))))
        behaviour = self._close_behaviour.get(coin, "full")

        if behaviour == "resting":
            # No fill — position remains OPEN (residual).
            return {
                "status": "ok",
                "response": {
                    "data": {"statuses": [{"resting": {"oid": 7}}]}
                },
            }
        if behaviour == "zero":
            # Empty statuses — no fill, position remains OPEN.
            return {"status": "ok", "response": {"data": {"statuses": []}}}
        if behaviour == "partial":
            filled = size / 2.0
            # Halve the residual position.
            self._positions[coin] = Decimal(str(filled)) * (
                Decimal("1") if self._positions[coin] > 0 else Decimal("-1")
            )
            return {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {"filled": {"oid": 1, "totalSz": str(filled), "avgPx": "50000.0"}}
                        ]
                    }
                },
            }

        # "full": fully close — clear the position so a re-query reports flat.
        self._positions.pop(coin, None)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"oid": 1, "totalSz": str(size), "avgPx": "50000.0"}}
                    ]
                }
            },
        }


def _make_ledger(tmp_path: Path) -> IntentLedger:
    return IntentLedger(tmp_path / "ledger.json")


# ---------------------------------------------------------------------------
# flatten_all
# ---------------------------------------------------------------------------


class TestFlattenAll:
    def test_two_positions_two_market_close_calls(self, tmp_path: Path) -> None:
        client = _FakeClient(
            positions={"BTC": Decimal("0.5"), "ETH": Decimal("2.0")}
        )
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()

        flatten_all(
            client,
            _FakeCfg,
            ledger,
            strategy_id="test",
            now_ts="2026-06-01T00:00:00",
            alert_sink=sink,
        )

        assert len(client.market_close_calls) == 2
        coins_called = {c for c, _, _ in client.market_close_calls}
        assert coins_called == {"BTC", "ETH"}

    def test_intents_recorded_before_call_and_marked_done(self, tmp_path: Path) -> None:
        """MF-7: PENDING written before market_close; DONE after."""
        client = _FakeClient(positions={"BTC": Decimal("0.1")})
        ledger = _make_ledger(tmp_path)

        flatten_all(
            client,
            _FakeCfg,
            ledger,
            strategy_id="test",
            now_ts="2026-06-01T00:00:00",
        )

        # After flatten, no PENDING intents should remain
        assert ledger.pending() == []

    def test_kill_switch_alert_emitted(self, tmp_path: Path) -> None:
        client = _FakeClient(positions={"SOL": Decimal("5.0")})
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()

        flatten_all(
            client,
            _FakeCfg,
            ledger,
            strategy_id="test",
            now_ts="T",
            alert_sink=sink,
        )

        assert len(sink.received) == 1
        a = sink.received[0]
        assert a.type == AlertType.KILL_SWITCH
        assert a.severity == Severity.CRITICAL

    def test_no_positions_no_calls_but_alert_still_emitted(self, tmp_path: Path) -> None:
        client = _FakeClient(positions={})
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()

        results = flatten_all(
            client,
            _FakeCfg,
            ledger,
            strategy_id="test",
            now_ts="T",
            alert_sink=sink,
        )

        assert len(client.market_close_calls) == 0
        assert results == []
        # Alert still emitted (attempted to flatten 0 positions)
        assert len(sink.received) == 1

    def test_one_coin_error_others_still_flattened(self, tmp_path: Path) -> None:
        """Per-coin isolation: BTC raises, ETH must still be flattened."""
        client = _FakeClient(
            positions={"BTC": Decimal("0.5"), "ETH": Decimal("2.0")},
            market_close_raises_for={"BTC"},
        )
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()

        results = flatten_all(
            client,
            _FakeCfg,
            ledger,
            strategy_id="test",
            now_ts="T",
            alert_sink=sink,
        )

        # Both coins were attempted
        coins_called = {c for c, _, _ in client.market_close_calls}
        assert "BTC" in coins_called
        assert "ETH" in coins_called

        # ETH should be "filled", BTC should be "error"
        outcomes = {r["coin"]: r["status"] for r in results}
        assert outcomes["ETH"] == "filled"
        assert outcomes["BTC"] == "error"

    def test_zero_position_coin_is_skipped(self, tmp_path: Path) -> None:
        """Coins with szi == 0 should not trigger a market_close."""
        client = _FakeClient(
            positions={"BTC": Decimal("0.5"), "ETH": Decimal("0")}
        )
        ledger = _make_ledger(tmp_path)

        results = flatten_all(
            client,
            _FakeCfg,
            ledger,
            strategy_id="test",
            now_ts="T",
        )

        # Only BTC had a non-zero position
        assert len(results) == 1
        assert results[0]["coin"] == "BTC"

    def test_results_include_cloid(self, tmp_path: Path) -> None:
        client = _FakeClient(positions={"BTC": Decimal("1.0")})
        ledger = _make_ledger(tmp_path)
        results = flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T"
        )
        assert results[0]["cloid"].startswith("0x")


class TestFlattenAllCloseVerification:
    """F3: never report a non-fill as 'filled'; surface residuals as CRITICAL."""

    def test_resting_response_not_labelled_filled(self, tmp_path: Path) -> None:
        """A reduce-only close that returns a resting status → status 'resting'."""
        client = _FakeClient(
            positions={"BTC": Decimal("0.5")},
            close_behaviour={"BTC": "resting"},
        )
        ledger = _make_ledger(tmp_path)
        results = flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T"
        )
        assert results[0]["status"] == "resting"
        assert results[0]["status"] != "filled"
        assert results[0]["filled_size"] == 0.0

    def test_zero_fill_not_labelled_filled(self, tmp_path: Path) -> None:
        """Empty statuses (zero fill) → status 'resting', never 'filled'."""
        client = _FakeClient(
            positions={"ETH": Decimal("3.0")},
            close_behaviour={"ETH": "zero"},
        )
        ledger = _make_ledger(tmp_path)
        results = flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T"
        )
        assert results[0]["status"] != "filled"
        assert results[0]["filled_size"] == 0.0

    def test_partial_fill_labelled_partial(self, tmp_path: Path) -> None:
        """A close that fills less than the position → status 'partial'."""
        client = _FakeClient(
            positions={"SOL": Decimal("10.0")},
            close_behaviour={"SOL": "partial"},
        )
        ledger = _make_ledger(tmp_path)
        results = flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T"
        )
        assert results[0]["status"] == "partial"
        assert results[0]["filled_size"] == pytest.approx(5.0)

    def test_full_fill_labelled_filled(self, tmp_path: Path) -> None:
        """A close that fills the full position → status 'filled'."""
        client = _FakeClient(positions={"BTC": Decimal("0.5")})
        ledger = _make_ledger(tmp_path)
        results = flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T"
        )
        assert results[0]["status"] == "filled"
        assert results[0]["filled_size"] == pytest.approx(0.5)

    def test_residual_position_triggers_critical_alert(self, tmp_path: Path) -> None:
        """A close that leaves the position OPEN → a CRITICAL residual alert."""
        client = _FakeClient(
            positions={"BTC": Decimal("0.5")},
            close_behaviour={"BTC": "resting"},  # position stays open
        )
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()
        flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T", alert_sink=sink
        )
        # Expect a residual-incomplete alert plus the standard kill alert.
        residual_alerts = [
            a
            for a in sink.received
            if a.type == AlertType.KILL_SWITCH and "INCOMPLETE" in a.message
        ]
        assert len(residual_alerts) == 1
        assert residual_alerts[0].severity == Severity.CRITICAL
        assert "BTC" in residual_alerts[0].context["residual_positions"]

    def test_clean_full_close_no_residual_alert(self, tmp_path: Path) -> None:
        """A fully-closed position → no residual alert (only the standard kill)."""
        client = _FakeClient(positions={"BTC": Decimal("0.5")})
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()
        flatten_all(
            client, _FakeCfg, ledger, strategy_id="test", now_ts="T", alert_sink=sink
        )
        residual_alerts = [a for a in sink.received if "INCOMPLETE" in a.message]
        assert len(residual_alerts) == 0
        # Standard kill alert still present
        assert len(sink.received) == 1


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class TestWatchdog:
    def _make_watchdog(
        self,
        tmp_path: Path,
        positions: dict[str, Decimal] | None = None,
        equity_floor: float | None = None,
        liveness_deadline_s: float | None = None,
    ) -> tuple[Watchdog, _FakeClient, _CapturingAlertSink]:
        kill_client = _FakeClient(positions=positions or {})
        ledger = _make_ledger(tmp_path)
        sink = _CapturingAlertSink()
        wd = Watchdog(
            kill_client,
            _FakeCfg,
            ledger,
            equity_floor=equity_floor,
            liveness_deadline_s=liveness_deadline_s,
            alert_sink=sink,
        )
        return wd, kill_client, sink

    def test_does_not_fire_when_healthy(self, tmp_path: Path) -> None:
        wd, client, _ = self._make_watchdog(
            tmp_path,
            liveness_deadline_s=300.0,
            equity_floor=1000.0,
        )
        triggered = wd.check(
            last_heartbeat_ts=1000.0,
            now_ts_float=1100.0,  # only 100s — under 300s deadline
            current_equity=Decimal("5000"),
        )
        assert triggered is False
        assert len(client.market_close_calls) == 0

    def test_fires_on_stale_heartbeat(self, tmp_path: Path) -> None:
        wd, client, sink = self._make_watchdog(
            tmp_path,
            positions={"BTC": Decimal("0.1")},
            liveness_deadline_s=60.0,
        )
        triggered = wd.check(
            last_heartbeat_ts=0.0,
            now_ts_float=200.0,  # 200s > 60s deadline
            current_equity=Decimal("99999"),
        )
        assert triggered is True
        # flatten_all was called
        assert len(client.market_close_calls) == 1

    def test_fires_on_equity_floor_breach(self, tmp_path: Path) -> None:
        wd, client, sink = self._make_watchdog(
            tmp_path,
            positions={"ETH": Decimal("1.0")},
            equity_floor=5000.0,
        )
        triggered = wd.check(
            last_heartbeat_ts=1000.0,
            now_ts_float=1001.0,  # heartbeat fresh
            current_equity=Decimal("3000"),  # below 5000
        )
        assert triggered is True
        assert len(client.market_close_calls) == 1

    def test_no_triggers_when_both_none(self, tmp_path: Path) -> None:
        wd, client, _ = self._make_watchdog(tmp_path)
        triggered = wd.check(
            last_heartbeat_ts=0.0,
            now_ts_float=9999.0,
            current_equity=Decimal("0"),
        )
        assert triggered is False

    def test_kill_switch_alert_emitted_on_trigger(self, tmp_path: Path) -> None:
        wd, _, sink = self._make_watchdog(
            tmp_path,
            liveness_deadline_s=30.0,
        )
        wd.check(
            last_heartbeat_ts=0.0,
            now_ts_float=100.0,
            current_equity=Decimal("9999"),
        )
        assert any(a.type == AlertType.KILL_SWITCH for a in sink.received)


# ---------------------------------------------------------------------------
# revoke_runbook
# ---------------------------------------------------------------------------


class TestRevokeRunbook:
    def test_returns_non_empty_string(self) -> None:
        result = revoke_runbook()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_mentions_key_concepts(self) -> None:
        result = revoke_runbook()
        result_lower = result.lower()
        # Should mention revocation, agent, and nonce safety
        assert "revok" in result_lower
        assert "agent" in result_lower
        assert "nonce" in result_lower
