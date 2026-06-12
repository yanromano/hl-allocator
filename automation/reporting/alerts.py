"""
automation.reporting.alerts — Structured alert emission with sink abstraction.

Sinks MUST NEVER crash the daemon — every emit() implementation catches and
swallows its own internal failures (network, serialisation) to guarantee the
daemon loop remains alive even when all downstream channels are degraded.

Secrets must not appear in payloads — caller's responsibility to exclude them;
the redacted logger is a backstop for anything that leaks via message/context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from automation.core.redaction import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Alert taxonomy
# ---------------------------------------------------------------------------


class AlertType(StrEnum):
    """Canonical alert types for the hl-allocator daemon."""

    STALE_SIGNAL = "STALE_SIGNAL"
    INVALID_SIGNAL = "INVALID_SIGNAL"
    SIGNAL_RECHECK_EXHAUSTED = "SIGNAL_RECHECK_EXHAUSTED"
    EXEC_ERROR = "EXEC_ERROR"
    DIVERGENCE = "DIVERGENCE"
    THROTTLE = "THROTTLE"
    CYCLE_INCOMPLETE = "CYCLE_INCOMPLETE"
    DAEMON_RESTART = "DAEMON_RESTART"
    AGENT_EXPIRY = "AGENT_EXPIRY"
    MIN_NOTIONAL_UNALLOCATED = "MIN_NOTIONAL_UNALLOCATED"
    CHAIN_LEDGER_DIVERGENCE = "CHAIN_LEDGER_DIVERGENCE"
    KILL_SWITCH = "KILL_SWITCH"
    UNFILLED = "UNFILLED"
    LEVERAGE_DRIFT = "LEVERAGE_DRIFT"
    FUNDING_DRAG = "FUNDING_DRAG"
    MARGIN_RATIO_LOW = "MARGIN_RATIO_LOW"
    COIN_VENUE_DEGRADED = "COIN_VENUE_DEGRADED"
    COLLATERAL_IN_SPOT = "COLLATERAL_IN_SPOT"
    # CRASH Phase 1 multi-sleeve (spec §4.3, red-team L1).
    SLEEVE_HOLD = "SLEEVE_HOLD"
    """A sleeve failed to fetch/validate within its staleness window — its coins
    are FROZEN (live positions carried untouched, no orders this cycle).  WARNING."""
    SLEEVE_DERISKED = "SLEEVE_DERISKED"
    """A sleeve exceeded its staleness window (or a short sleeve hit its second
    consecutive HOLD) — its coins unwind to cash via the combined delta.  CRITICAL."""
    POSITION_LIQUIDATED = "POSITION_LIQUIDATED"
    """A position present in the baseline vanished from the live snapshot with no
    committed exit order of ours — an apparent liquidation.  Triggers a
    same-direction re-entry cooldown (spec §5/M3, consumed by Task C6).  CRITICAL."""


class Severity(StrEnum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    """An immutable, non-secret alert record.

    The caller is responsible for ensuring ``context`` contains no secrets.
    The redacted logger provides a backstop for anything that leaks.
    """

    type: AlertType
    severity: Severity
    message: str
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AlertSink(Protocol):
    """Structural protocol for alert sinks.

    Any object with an ``emit(alert: Alert) -> None`` method satisfies this
    protocol — no inheritance required.  A sink's ``emit`` MUST swallow its
    own internal errors so the daemon loop stays alive.
    """

    def emit(self, alert: Alert) -> None:
        """Emit an alert to the underlying channel."""
        ...


# ---------------------------------------------------------------------------
# Concrete sinks
# ---------------------------------------------------------------------------


class LogAlertSink:
    """Emit alerts via the redacted structlog logger.

    CRITICAL → ``logger.error``; WARNING and INFO → ``logger.warning``.
    The redaction processor scrubs anything resembling a secret from the output.
    """

    def __init__(self) -> None:
        self._log = get_logger(__name__)

    def emit(self, alert: Alert) -> None:
        """Log the alert at the appropriate level."""
        payload = {
            "alert_type": alert.type.value,
            "severity": alert.severity.value,
            "message": alert.message,
            **alert.context,
        }
        if alert.severity == Severity.CRITICAL:
            self._log.error("alert", **payload)
        else:
            self._log.warning("alert", **payload)


class WebhookAlertSink:
    """Emit alerts by POSTing JSON to a webhook URL.

    Failures (network errors, non-2xx responses, serialisation errors) are
    logged and swallowed — a broken webhook MUST NOT crash the daemon.

    Parameters
    ----------
    webhook_url:
        HTTPS endpoint to POST the alert payload to.
    http_client:
        Optional injectable HTTP client (must have a ``post(url, json=...)``
        method returning a response with a ``status_code`` attribute).
        Defaults to ``httpx.Client()`` when ``None``.

    Security note
    -------------
    The payload contains only ``type``, ``severity``, ``message``, and
    ``context``.  NEVER include secrets (keys, tokens, passphrases).
    """

    def __init__(self, webhook_url: str, http_client: Any = None) -> None:
        self._url = webhook_url
        self._http_client = http_client

    def _get_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        import httpx  # noqa: PLC0415 — lazy import to avoid hard dep in tests

        return httpx.Client()

    def emit(self, alert: Alert) -> None:
        """POST the alert as JSON.  All failures are swallowed."""
        payload: dict[str, Any] = {
            "type": alert.type.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "context": alert.context,
        }
        try:
            client = self._get_client()
            resp = client.post(self._url, json=payload)
            if resp.status_code >= 400:
                _logger.warning(
                    "WebhookAlertSink: non-2xx response",
                    status_code=resp.status_code,
                    alert_type=alert.type.value,
                )
        except Exception as exc:  # noqa: BLE001
            # Never propagate — a broken webhook MUST NOT crash the daemon.
            _logger.warning(
                "WebhookAlertSink: emit failed (swallowed)",
                error=str(exc),
                alert_type=alert.type.value,
            )


class MultiSink:
    """Fan-out to multiple sinks.

    One sink raising MUST NOT prevent the remaining sinks from receiving the
    alert.  Errors from individual sinks are logged and swallowed.

    Parameters
    ----------
    sinks:
        Ordered list of ``AlertSink``-like objects.
    """

    def __init__(self, sinks: list[AlertSink]) -> None:
        self._sinks = sinks

    def emit(self, alert: Alert) -> None:
        """Emit to all sinks; isolate per-sink failures."""
        for sink in self._sinks:
            try:
                sink.emit(alert)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "MultiSink: sink.emit raised (swallowed, continuing)",
                    sink=type(sink).__name__,
                    error=str(exc),
                    alert_type=alert.type.value,
                )
