"""
Tests for automation.core.safety.

Covers:
- parse_order_response: resting/filled OK path; request-level status:"err" raises
  OrderRejected; per-order statuses[i].error raises OrderRejected.
- with_backoff: retries then succeeds (patched sleep); re-raises after max_retries;
  does not retry OrderRejected.
- enforce_caps: raises CapBreach on per-coin or total breach; passes within caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from automation.core.safety import (
    CapBreach,
    OrderRejected,
    RateLimited,
    enforce_caps,
    parse_order_response,
    with_backoff,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_resp(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "ok", "response": {"data": {"statuses": statuses}}}


def _err_resp(msg: str) -> dict[str, Any]:
    return {"status": "err", "response": msg}


@dataclass
class MockCaps:
    max_notional_per_coin: float
    max_total_notional: float
    max_turnover_per_cycle: float


# ---------------------------------------------------------------------------
# parse_order_response
# ---------------------------------------------------------------------------


def test_parse_resting_ok() -> None:
    """A resting status returns a list with the resting dict."""
    resp = _ok_resp([{"resting": {"oid": 42}}])
    result = parse_order_response(resp)
    assert result == [{"resting": {"oid": 42}}]


def test_parse_filled_ok() -> None:
    """A filled status returns a list with the filled dict."""
    resp = _ok_resp([{"filled": {"oid": 99, "price": "1000"}}])
    result = parse_order_response(resp)
    assert result == [{"filled": {"oid": 99, "price": "1000"}}]


def test_parse_multiple_statuses() -> None:
    """Multiple resting/filled statuses all collected."""
    resp = _ok_resp([
        {"resting": {"oid": 1}},
        {"filled": {"oid": 2}},
    ])
    result = parse_order_response(resp)
    assert len(result) == 2
    assert result[0] == {"resting": {"oid": 1}}
    assert result[1] == {"filled": {"oid": 2}}


def test_parse_request_level_error_raises() -> None:
    """status != 'ok' at the request level raises OrderRejected with the response payload."""
    resp = _err_resp("Order would immediately cross")
    with pytest.raises(OrderRejected, match="Order would immediately cross"):
        parse_order_response(resp)


def test_parse_request_level_unknown_status_raises() -> None:
    """Any non-'ok' status raises OrderRejected."""
    resp = {"status": "rejected", "response": "margin too low"}
    with pytest.raises(OrderRejected):
        parse_order_response(resp)


def test_parse_per_order_error_raises() -> None:
    """A per-order 'error' key inside statuses raises OrderRejected."""
    resp = _ok_resp([{"error": "Insufficient margin"}])
    with pytest.raises(OrderRejected, match="Insufficient margin"):
        parse_order_response(resp)


def test_parse_per_order_error_raises_on_second_status() -> None:
    """Raises even if the first status is fine but a later one has an error."""
    resp = _ok_resp([
        {"resting": {"oid": 1}},
        {"error": "Bad price tick"},
    ])
    with pytest.raises(OrderRejected, match="Bad price tick"):
        parse_order_response(resp)


def test_parse_empty_statuses() -> None:
    """Empty statuses list returns an empty list (no orders placed, no error)."""
    resp = _ok_resp([])
    assert parse_order_response(resp) == []


def test_parse_missing_response_key() -> None:
    """Missing 'response' key with status != 'ok' raises OrderRejected gracefully."""
    resp = {"status": "err"}
    with pytest.raises(OrderRejected):
        parse_order_response(resp)


# ---------------------------------------------------------------------------
# with_backoff
# ---------------------------------------------------------------------------


def test_with_backoff_succeeds_on_first_try() -> None:
    """If fn() succeeds immediately, returns the value with no sleep."""
    sleep_mock = MagicMock()
    result = with_backoff(lambda: 42, _sleep=sleep_mock)
    assert result == 42
    sleep_mock.assert_not_called()


def test_with_backoff_retries_then_succeeds() -> None:
    """Retries on transient errors and eventually returns the successful result."""
    sleep_mock = MagicMock()
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient failure")
        return "success"

    result = with_backoff(flaky, max_retries=5, base=0.001, cap=0.01, _sleep=sleep_mock)
    assert result == "success"
    assert calls["count"] == 3
    assert sleep_mock.call_count == 2  # slept twice (after attempt 0 and 1)


def test_with_backoff_reraises_after_max_retries() -> None:
    """After max_retries attempts are exhausted, the last exception is re-raised."""
    sleep_mock = MagicMock()
    attempt_count = {"n": 0}

    def always_fails() -> None:
        attempt_count["n"] += 1
        raise RuntimeError("persistent failure")

    with pytest.raises(RuntimeError, match="persistent failure"):
        with_backoff(always_fails, max_retries=3, base=0.001, cap=0.01, _sleep=sleep_mock)

    # Total calls = 1 initial + 3 retries = 4
    assert attempt_count["n"] == 4
    assert sleep_mock.call_count == 3  # slept after each failed attempt except the last


def test_with_backoff_does_not_retry_order_rejected() -> None:
    """OrderRejected is a deterministic failure and must NOT be retried."""
    sleep_mock = MagicMock()
    call_count = {"n": 0}

    def raise_rejected() -> None:
        call_count["n"] += 1
        raise OrderRejected("price too far from mark")

    with pytest.raises(OrderRejected, match="price too far from mark"):
        with_backoff(raise_rejected, max_retries=5, _sleep=sleep_mock)

    assert call_count["n"] == 1  # called exactly once
    sleep_mock.assert_not_called()


def test_with_backoff_rate_limited_honors_retry_after() -> None:
    """RateLimited with retry_after sleeps for exactly retry_after seconds."""
    sleep_mock = MagicMock()
    calls = {"n": 0}

    def rate_limited_then_ok() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimited("too many requests", retry_after=7.5)
        return "ok"

    result = with_backoff(rate_limited_then_ok, max_retries=3, _sleep=sleep_mock)
    assert result == "ok"
    # First sleep must be retry_after=7.5
    assert sleep_mock.call_args_list[0] == call(7.5)


def test_with_backoff_rate_limited_reraises_after_max() -> None:
    """RateLimited is retriable; re-raises after max_retries exhausted."""
    sleep_mock = MagicMock()

    def always_rate_limited() -> None:
        raise RateLimited("burst limit", retry_after=0.001)

    with pytest.raises(RateLimited):
        with_backoff(always_rate_limited, max_retries=2, _sleep=sleep_mock)

    assert sleep_mock.call_count == 2


def test_with_backoff_returns_none_on_success() -> None:
    """fn() returning None (common for void functions) works correctly."""
    result = with_backoff(lambda: None, _sleep=MagicMock())
    assert result is None


# ---------------------------------------------------------------------------
# enforce_caps
# ---------------------------------------------------------------------------


def test_enforce_caps_passes_within_limits() -> None:
    """No exception when all orders are within caps."""
    caps = MockCaps(
        max_notional_per_coin=10_000.0,
        max_total_notional=50_000.0,
        max_turnover_per_cycle=60_000.0,
    )
    orders = [
        {"coin": "BTC", "notional": 5_000.0},
        {"coin": "ETH", "notional": 4_000.0},
    ]
    enforce_caps(orders, caps)  # must not raise


def test_enforce_caps_passes_empty_orders() -> None:
    """Empty order list never breaches any cap."""
    caps = MockCaps(1_000.0, 5_000.0, 6_000.0)
    enforce_caps([], caps)  # must not raise


def test_enforce_caps_raises_on_per_coin_breach() -> None:
    """Per-coin notional breach raises CapBreach."""
    caps = MockCaps(
        max_notional_per_coin=3_000.0,
        max_total_notional=50_000.0,
        max_turnover_per_cycle=60_000.0,
    )
    orders = [{"coin": "BTC", "notional": 5_000.0}]
    with pytest.raises((CapBreach, ValueError)):
        enforce_caps(orders, caps)


def test_enforce_caps_raises_on_total_notional_breach() -> None:
    """Total notional breach raises CapBreach even if each individual order is fine."""
    caps = MockCaps(
        max_notional_per_coin=10_000.0,
        max_total_notional=15_000.0,
        max_turnover_per_cycle=60_000.0,
    )
    orders = [
        {"coin": "BTC", "notional": 9_000.0},
        {"coin": "ETH", "notional": 9_000.0},
    ]
    with pytest.raises((CapBreach, ValueError)):
        enforce_caps(orders, caps)


def test_enforce_caps_raises_on_turnover_breach() -> None:
    """Total turnover breach raises CapBreach."""
    caps = MockCaps(
        max_notional_per_coin=10_000.0,
        max_total_notional=50_000.0,
        max_turnover_per_cycle=5_000.0,
    )
    orders = [
        {"coin": "BTC", "notional": 3_000.0, "turnover": 3_000.0},
        {"coin": "ETH", "notional": 3_000.0, "turnover": 3_000.0},
    ]
    with pytest.raises((CapBreach, ValueError)):
        enforce_caps(orders, caps)


def test_enforce_caps_error_message_names_coin() -> None:
    """CapBreach message must mention the offending coin name."""
    caps = MockCaps(1_000.0, 50_000.0, 60_000.0)
    orders = [{"coin": "DOGE", "notional": 5_000.0}]
    with pytest.raises((CapBreach, ValueError)) as exc_info:
        enforce_caps(orders, caps)
    assert "DOGE" in str(exc_info.value)


def test_enforce_caps_exact_limit_does_not_raise() -> None:
    """Orders at exactly the limit must NOT raise (boundary condition)."""
    caps = MockCaps(
        max_notional_per_coin=5_000.0,
        max_total_notional=10_000.0,
        max_turnover_per_cycle=10_000.0,
    )
    orders = [
        {"coin": "BTC", "notional": 5_000.0},
        {"coin": "ETH", "notional": 5_000.0},
    ]
    enforce_caps(orders, caps)  # must not raise
