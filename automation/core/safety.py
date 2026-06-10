"""
automation.core.safety — HTTP-response parsing, backoff, and order-cap enforcement.

Key insight from Hyperliquid API (§10): **HTTP 200 ≠ success.**  A 200 response
with ``status != "ok"`` is a top-level rejection; a 200 response with
``status == "ok"`` may still contain per-order ``error`` entries inside
``response.data.statuses[i]``.  Both must be checked.

Modules
-------
- ``OrderRejected`` — raised for any order-level or request-level rejection.
- ``RateLimited``   — raised when the exchange returns a rate-limit signal.
- ``parse_order_response`` — the canonical HTTP-200-is-not-success parser.
- ``with_backoff``  — bounded exponential backoff + jitter retry loop.
- ``enforce_caps``  — pre-flight notional / turnover cap check.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from automation.core.redaction import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OrderRejected(Exception):
    """Raised when Hyperliquid rejects an order at the request or per-order level."""


class RateLimited(Exception):
    """Raised when the exchange signals a rate-limit condition.

    Parameters
    ----------
    retry_after:
        Optional number of seconds to wait before retrying (if provided by
        the exchange in the response headers or body).
    """

    def __init__(self, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_order_response(resp: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a Hyperliquid order-placement response, raising on any rejection.

    HTTP 200 ≠ success.  This function enforces both levels of checking:

    1. **Request-level**: ``resp["status"] != "ok"`` → raise ``OrderRejected``
       with the top-level error payload (a plain string).
    2. **Per-order level**: any status in ``resp["response"]["data"]["statuses"]``
       that contains an ``"error"`` key → raise ``OrderRejected`` with that message.

    Parameters
    ----------
    resp:
        The decoded JSON body of the HTTP response.

    Returns
    -------
    list[dict]
        The list of ``resting`` / ``filled`` status dicts for all successfully
        placed orders.

    Raises
    ------
    OrderRejected
        On any request-level or per-order rejection.
    """
    # Level 1: request-level status check
    if resp.get("status") != "ok":
        raise OrderRejected(str(resp.get("response", "unknown error")))

    # Level 2: per-order status check
    statuses: list[dict[str, Any]] = resp["response"]["data"]["statuses"]
    parsed: list[dict[str, Any]] = []
    for st in statuses:
        if "error" in st:
            raise OrderRejected(st["error"])
        # Collect resting / filled dicts
        if "resting" in st:
            parsed.append({"resting": st["resting"]})
        elif "filled" in st:
            parsed.append({"filled": st["filled"]})
        else:
            # Passthrough for any other valid status shape
            parsed.append(st)
    return parsed


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------


def with_backoff(
    fn: Callable[[], Any],
    *,
    max_retries: int = 6,
    base: float = 1.0,
    cap: float = 60.0,
    _sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn()`` with bounded exponential backoff + full jitter.

    Retries on:
    - ``RateLimited`` — honors ``retry_after`` if set; otherwise uses backoff.
    - Any ``Exception`` that is not ``OrderRejected`` (treated as transient).

    Does NOT retry ``OrderRejected`` — these are deterministic rejections.

    Parameters
    ----------
    fn:
        Zero-argument callable to invoke (may raise).
    max_retries:
        Maximum number of retry attempts after the first failure (total
        attempts = ``max_retries + 1``).
    base:
        Base delay in seconds for the exponential schedule.
    cap:
        Maximum delay in seconds (before jitter).
    _sleep:
        Injectable sleep function (default ``time.sleep``; override in tests).

    Returns
    -------
    Any
        The return value of ``fn()`` on first success.

    Raises
    ------
    Exception
        Re-raises the last exception after ``max_retries`` attempts are
        exhausted, or re-raises ``OrderRejected`` immediately (no retry).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except OrderRejected:
            # Deterministic rejection — do not retry
            raise
        except RateLimited as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = exc.retry_after if exc.retry_after is not None else _backoff_delay(attempt, base, cap)
            logger.warning(
                "Rate limited; retrying",
                attempt=attempt,
                delay=delay,
            )
            _sleep(delay)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _backoff_delay(attempt, base, cap)
            logger.warning(
                "Transient error; retrying",
                attempt=attempt,
                delay=delay,
                error=str(exc),
            )
            _sleep(delay)

    assert last_exc is not None  # always set when we reach here
    raise last_exc


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Full-jitter exponential backoff: ``uniform(0, min(cap, base * 2^attempt))``."""
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0, ceiling)


# ---------------------------------------------------------------------------
# Order-cap enforcement
# ---------------------------------------------------------------------------


@dataclass
class CapBreach(Exception):
    """Raised by ``enforce_caps`` when a notional or turnover cap is exceeded."""

    message: str

    def __str__(self) -> str:
        return self.message


def enforce_caps(
    orders: list[dict[str, Any]],
    caps: Any,
) -> None:
    """Raise ``CapBreach`` if any per-coin or aggregate cap is breached.

    Parameters
    ----------
    orders:
        List of ``{"coin": str, "notional": float}`` dicts describing the
        intended orders for this cycle.
    caps:
        An object with attributes ``max_notional_per_coin``,
        ``max_total_notional``, and ``max_turnover_per_cycle``.

    Raises
    ------
    CapBreach
        On the first cap breach detected.
    """
    total_notional: float = 0.0
    total_turnover: float = 0.0

    for order in orders:
        coin: str = order["coin"]
        notional: float = float(order["notional"])
        turnover: float = float(order.get("turnover", notional))  # fallback to notional

        if notional > caps.max_notional_per_coin:
            raise CapBreach(
                f"Per-coin cap breached for {coin}: "
                f"notional={notional:.2f} > max={caps.max_notional_per_coin:.2f}"
            )

        total_notional += notional
        total_turnover += turnover

    if total_notional > caps.max_total_notional:
        raise CapBreach(
            f"Total notional cap breached: "
            f"{total_notional:.2f} > max={caps.max_total_notional:.2f}"
        )

    if total_turnover > caps.max_turnover_per_cycle:
        raise CapBreach(
            f"Total turnover cap breached: "
            f"{total_turnover:.2f} > max={caps.max_turnover_per_cycle:.2f}"
        )
