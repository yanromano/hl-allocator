"""
Tests for automation.reporting.alerts.

Coverage:
- LogAlertSink logs at right level (CRITICAL→error, else→warning)
- WebhookAlertSink posts correct JSON to a fake http client
- WebhookAlertSink swallows POST failure (never raises)
- MultiSink fan-out to all sinks
- MultiSink isolates one raising sink (others still receive)
- Alert is frozen (immutable)
"""

from __future__ import annotations

from typing import Any

import pytest

from automation.reporting.alerts import (
    Alert,
    AlertType,
    LogAlertSink,
    MultiSink,
    Severity,
    WebhookAlertSink,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alert(
    alert_type: AlertType = AlertType.EXEC_ERROR,
    severity: Severity = Severity.WARNING,
    message: str = "test alert",
    context: dict[str, Any] | None = None,
) -> Alert:
    return Alert(
        type=alert_type,
        severity=severity,
        message=message,
        context=context or {},
    )


class _CapturingSink:
    """Fake sink that records all emitted alerts."""

    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


class _RaisingSink:
    """Fake sink that always raises on emit."""

    def emit(self, alert: Alert) -> None:
        raise RuntimeError("sink exploded")


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


class TestAlert:
    def test_alert_is_frozen(self) -> None:
        a = _alert()
        with pytest.raises((AttributeError, TypeError)):
            a.message = "mutated"  # type: ignore[misc]

    def test_alert_default_context_is_empty(self) -> None:
        a = Alert(type=AlertType.KILL_SWITCH, severity=Severity.INFO, message="m")
        assert a.context == {}

    def test_alert_types_are_strings(self) -> None:
        # AlertType(str, Enum) — values should be usable as strings
        assert AlertType.KILL_SWITCH == "KILL_SWITCH"
        assert Severity.CRITICAL == "CRITICAL"


# ---------------------------------------------------------------------------
# LogAlertSink
# ---------------------------------------------------------------------------


class TestLogAlertSink:
    """LogAlertSink tests use monkeypatching to verify which log method is called."""

    def test_warning_severity_calls_logger_warning(self, monkeypatch: Any) -> None:
        """WARNING severity → logger.warning is called."""
        calls: list[tuple[str, str]] = []

        class _FakeLog:
            def warning(self, event: str, **kw: Any) -> None:
                calls.append(("warning", kw.get("message", event)))

            def error(self, event: str, **kw: Any) -> None:
                calls.append(("error", kw.get("message", event)))

        sink = LogAlertSink()
        monkeypatch.setattr(sink, "_log", _FakeLog())
        sink.emit(_alert(severity=Severity.WARNING, message="warn me"))
        assert ("warning", "warn me") in calls

    def test_info_severity_calls_logger_warning(self, monkeypatch: Any) -> None:
        """INFO severity → logger.warning is called (same as WARNING, not error)."""
        calls: list[tuple[str, str]] = []

        class _FakeLog:
            def warning(self, event: str, **kw: Any) -> None:
                calls.append(("warning", kw.get("message", event)))

            def error(self, event: str, **kw: Any) -> None:
                calls.append(("error", kw.get("message", event)))

        sink = LogAlertSink()
        monkeypatch.setattr(sink, "_log", _FakeLog())
        sink.emit(_alert(severity=Severity.INFO, message="info me"))
        assert ("warning", "info me") in calls
        assert not any(level == "error" for level, _ in calls)

    def test_critical_severity_calls_logger_error(self, monkeypatch: Any) -> None:
        """CRITICAL severity → logger.error is called, NOT logger.warning."""
        calls: list[tuple[str, str]] = []

        class _FakeLog:
            def warning(self, event: str, **kw: Any) -> None:
                calls.append(("warning", kw.get("message", event)))

            def error(self, event: str, **kw: Any) -> None:
                calls.append(("error", kw.get("message", event)))

        sink = LogAlertSink()
        monkeypatch.setattr(sink, "_log", _FakeLog())
        sink.emit(_alert(severity=Severity.CRITICAL, message="critical situation"))
        assert ("error", "critical situation") in calls
        assert not any(level == "warning" for level, _ in calls)

    def test_alert_type_in_log_kwargs(self, monkeypatch: Any) -> None:
        """Alert type value should appear in the keyword arguments."""
        captured_kwargs: list[dict[str, Any]] = []

        class _FakeLog:
            def warning(self, event: str, **kw: Any) -> None:
                captured_kwargs.append(kw)

            def error(self, event: str, **kw: Any) -> None:
                captured_kwargs.append(kw)

        sink = LogAlertSink()
        monkeypatch.setattr(sink, "_log", _FakeLog())
        sink.emit(_alert(alert_type=AlertType.KILL_SWITCH, message="ks"))
        assert any("KILL_SWITCH" in str(kw) for kw in captured_kwargs)


# ---------------------------------------------------------------------------
# WebhookAlertSink
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _FakeHttpClient:
    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[dict[str, Any]] = []
        self._status_code = status_code

    def post(self, url: str, json: Any = None) -> _FakeResponse:
        self.calls.append({"url": url, "json": json})
        return _FakeResponse(self._status_code)


class _RaisingHttpClient:
    def post(self, url: str, json: Any = None) -> _FakeResponse:
        raise ConnectionError("network down")


class TestWebhookAlertSink:
    def test_posts_correct_json_fields(self) -> None:
        http = _FakeHttpClient()
        sink = WebhookAlertSink("https://example.com/webhook", http_client=http)
        a = _alert(
            alert_type=AlertType.EXEC_ERROR,
            severity=Severity.CRITICAL,
            message="execution failed",
            context={"coin": "BTC"},
        )
        sink.emit(a)

        assert len(http.calls) == 1
        payload = http.calls[0]["json"]
        assert payload["type"] == "EXEC_ERROR"
        assert payload["severity"] == "CRITICAL"
        assert payload["message"] == "execution failed"
        assert payload["context"] == {"coin": "BTC"}
        assert http.calls[0]["url"] == "https://example.com/webhook"

    def test_non_2xx_does_not_raise(self) -> None:
        http = _FakeHttpClient(status_code=500)
        sink = WebhookAlertSink("https://example.com/webhook", http_client=http)
        # Must not raise
        sink.emit(_alert())

    def test_network_failure_is_swallowed(self) -> None:
        http = _RaisingHttpClient()
        sink = WebhookAlertSink("https://example.com/webhook", http_client=http)
        # Must not raise even when the network is down
        sink.emit(_alert())

    def test_payload_does_not_include_secrets(self) -> None:
        """Context keys should be mirrored as-is; the caller must not pass secrets."""
        http = _FakeHttpClient()
        sink = WebhookAlertSink("https://example.com/webhook", http_client=http)
        # Legitimate non-secret context
        sink.emit(_alert(context={"n_error": 2, "as_of": "2026-06-01"}))
        payload = http.calls[0]["json"]
        assert "agent_key" not in payload
        assert "private_key" not in payload


# ---------------------------------------------------------------------------
# MultiSink
# ---------------------------------------------------------------------------


class TestMultiSink:
    def test_fan_out_to_all_sinks(self) -> None:
        a = _CapturingSink()
        b = _CapturingSink()
        multi = MultiSink([a, b])
        alert = _alert(message="broadcast")
        multi.emit(alert)

        assert len(a.received) == 1
        assert len(b.received) == 1
        assert a.received[0] is alert
        assert b.received[0] is alert

    def test_raising_sink_does_not_stop_others(self) -> None:
        good = _CapturingSink()
        bad = _RaisingSink()
        multi = MultiSink([bad, good])
        alert = _alert(message="isolated")
        # Must not raise
        multi.emit(alert)
        # The good sink should still have received the alert
        assert len(good.received) == 1
        assert good.received[0] is alert

    def test_all_sinks_raising_does_not_propagate(self) -> None:
        multi = MultiSink([_RaisingSink(), _RaisingSink()])
        # Must not raise
        multi.emit(_alert())

    def test_empty_sinks_is_noop(self) -> None:
        multi = MultiSink([])
        multi.emit(_alert())  # Must not raise

    def test_order_preserved(self) -> None:
        """Sinks receive alerts in the order they were registered."""
        order: list[str] = []

        class _OrderedSink:
            def __init__(self, name: str) -> None:
                self._name = name

            def emit(self, alert: Alert) -> None:
                order.append(self._name)

        multi = MultiSink([_OrderedSink("first"), _OrderedSink("second"), _OrderedSink("third")])
        multi.emit(_alert())
        assert order == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# AlertType.UNFILLED
# ---------------------------------------------------------------------------


def test_unfilled_alert_type_exists() -> None:
    from automation.reporting.alerts import AlertType

    assert AlertType.UNFILLED
    assert AlertType.UNFILLED.value == "UNFILLED"


def test_leverage_drift_alert_type_exists() -> None:
    from automation.reporting.alerts import AlertType

    assert AlertType.LEVERAGE_DRIFT
    assert AlertType.LEVERAGE_DRIFT.value == "LEVERAGE_DRIFT"


# ---------------------------------------------------------------------------
# PH-1 alert types
# ---------------------------------------------------------------------------


def test_funding_drag_alert_type_exists() -> None:
    from automation.reporting.alerts import AlertType

    assert AlertType.FUNDING_DRAG
    assert AlertType.FUNDING_DRAG.value == "FUNDING_DRAG"


def test_margin_ratio_low_alert_type_exists() -> None:
    from automation.reporting.alerts import AlertType

    assert AlertType.MARGIN_RATIO_LOW
    assert AlertType.MARGIN_RATIO_LOW.value == "MARGIN_RATIO_LOW"


def test_coin_venue_degraded_alert_type_exists() -> None:
    from automation.reporting.alerts import AlertType

    assert AlertType.COIN_VENUE_DEGRADED
    assert AlertType.COIN_VENUE_DEGRADED.value == "COIN_VENUE_DEGRADED"


def test_collateral_in_spot_alert_type_exists() -> None:
    from automation.reporting.alerts import AlertType

    assert AlertType.COLLATERAL_IN_SPOT
    assert AlertType.COLLATERAL_IN_SPOT.value == "COLLATERAL_IN_SPOT"
