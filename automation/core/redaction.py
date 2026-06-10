"""
automation.core.redaction — Global secret redaction for all logs.

Provides:
- ``redact(text)``            — replace 64-hex / long base64 blobs with «redacted»
- ``redact_event(...)``       — structlog processor: scrubs sensitive keys and hex values
- ``install_global_redaction()`` — idempotent one-call setup
- ``get_logger(name)``        — structlog logger with redaction baked in

Security model
--------------
No private key, agent key, passphrase, or bearer token should ever appear in a
log line. This module enforces that at the structlog-processor layer and via the
global sys.excepthook wrapper.
"""

from __future__ import annotations

import re
import sys
import traceback
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lowercased key names whose values are always redacted verbatim.
SECRET_KEY_NAMES: frozenset[str] = frozenset(
    {
        "signature",
        "secret_key",
        "agent_key",
        "private_key",
        "passphrase",
        "password",
        "token",
        "authorization",
    }
)

#: Matches EXACTLY a 64-character hex string (with or without a leading ``0x``).
#: This covers raw private keys (32 bytes) and Ethereum signature halves.
#:
#: Critical: it must NOT match a 40-hex Ethereum address (public, loggable) and
#: must NOT partially match a longer hex run.  The surrounding ``(?![0-9a-fA-F])``
#: / ``(?<![0-9a-fA-F])`` boundaries anchor the run to *exactly* 64 hex digits.
HEX64: re.Pattern[str] = re.compile(
    r"(?<![0-9a-fA-F])0x[0-9a-fA-F]{64}(?![0-9a-fA-F])"  # 0x-prefixed 64-hex
    r"|(?<![0-9a-fA-Fx])[0-9a-fA-F]{64}(?![0-9a-fA-F])"  # bare 64-hex
)

#: Ethereum address shape: ``0x`` + EXACTLY 40 hex digits.  PUBLIC — these are
#: matched first in :func:`redact` and deliberately left UNCHANGED so that
#: routine ``sub=<address>`` logging stays readable.
_ETH_ADDRESS: re.Pattern[str] = re.compile(
    r"(?<![0-9a-fA-F])0x[0-9a-fA-F]{40}(?![0-9a-fA-F])"
)

#: Minimum length of a base64 run to be treated as a secret blob.
_BASE64_MIN_LEN = 40

#: Matches a contiguous run of base64 characters (incl. padding) of sufficient
#: length AND containing at least one character that is NOT a plain hex digit
#: (one of ``[g-zG-Z+/]`` or padding ``=``).  Requiring a non-hex char stops the
#: heuristic from devouring plain hex strings (handled by HEX64 / spared by
#: the address rule).  Ethereum addresses are consumed by the address branch in
#: the combined :data:`_REDACT_RE` before this branch can match them.
_BASE64_RE: re.Pattern[str] = re.compile(
    r"(?=[A-Za-z0-9+/]{" + str(_BASE64_MIN_LEN) + r",}={0,2})"  # length >= min
    r"[A-Za-z0-9+/]*[g-zG-Z+/=][A-Za-z0-9+/]*={0,2}"  # >=1 non-hex base64 char
)

#: Combined single-pass pattern.  Order matters: the address branch is FIRST so
#: a public ``0x``+40-hex address is consumed (and preserved) before HEX64 or the
#: greedy base64 branch can mangle it.  The replacement callback returns the
#: address untouched and ``«redacted»`` for the secret branches.
_REDACT_RE: re.Pattern[str] = re.compile(
    r"(?P<addr>" + _ETH_ADDRESS.pattern + r")"
    r"|(?P<hex64>" + HEX64.pattern + r")"
    r"|(?P<b64>" + _BASE64_RE.pattern + r")"
)

_REDACTED = "«redacted»"

_REDACTION_INSTALLED = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _redact_replace(m: re.Match[str]) -> str:
    """Replacement callback for the combined :data:`_REDACT_RE` pass.

    Public Ethereum addresses (the ``addr`` group) are returned UNCHANGED;
    every other branch (64-hex keys, base64 blobs) becomes ``«redacted»``.
    """
    if m.lastgroup == "addr":
        return m.group(0)
    return _REDACTED


def redact(text: str) -> str:
    """Replace 64-hex private keys and long base64 secret blobs with ``«redacted»``.

    Public Ethereum addresses (``0x`` + 40 hex) are explicitly left UNCHANGED —
    they are not secret and are logged routinely (e.g. ``sub=<address>``).

    Parameters
    ----------
    text:
        Arbitrary string (log message, traceback, etc.).

    Returns
    -------
    str
        The sanitised string with genuine secrets replaced by ``«redacted»``
        and public addresses preserved.
    """
    return _REDACT_RE.sub(_redact_replace, text)


def redact_event(
    logger: Any,  # noqa: ARG001 — structlog passes logger but we don't need it
    method: str,  # noqa: ARG001 — structlog passes method but we don't need it
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor: redact secrets from the event dictionary.

    - If a key (lowercased) is in ``SECRET_KEY_NAMES``, the value is replaced
      with ``«redacted»`` regardless of its content.
    - Otherwise, if the value is a ``str``, it is passed through :func:`redact`
      to strip embedded hex-64 / base64 blobs.

    Parameters
    ----------
    logger:
        The structlog bound logger (unused; required by the processor protocol).
    method:
        The log method name (unused; required by the processor protocol).
    event_dict:
        The current event dictionary being built.

    Returns
    -------
    dict
        The (possibly modified) event dictionary.
    """
    cleaned: dict[str, Any] = {}
    for key, value in event_dict.items():
        if key.lower() in SECRET_KEY_NAMES:
            cleaned[key] = _REDACTED
        elif isinstance(value, str):
            cleaned[key] = redact(value)
        else:
            cleaned[key] = value
    return cleaned


def install_global_redaction() -> None:
    """Configure structlog with redaction and install a sys.excepthook wrapper.

    Idempotent — calling this multiple times is safe.

    Effects
    -------
    1. Configures structlog with ``redact_event`` as an early processor so that
       all log lines (from any structlog logger) pass through redaction.
    2. Wraps ``sys.excepthook`` so that formatted tracebacks are redacted before
       being printed to stderr.
    3. Sets ``httpx``, ``eth_account``, and ``hyperliquid`` stdlib loggers to
       WARNING level to reduce noise (and surface area for accidental key leaks
       through verbose SDK logging).
    """
    global _REDACTION_INSTALLED  # noqa: PLW0603
    if _REDACTION_INSTALLED:
        return

    # ---- 1. Configure structlog -----------------------------------------
    # Cast redact_event to Any to satisfy structlog's narrow Processor type
    # signature without losing the readable inline type on the function itself.
    from typing import cast as _cast

    # NOTE: we use PrintLoggerFactory (not stdlib logging), so the processor
    # chain must contain ONLY processors that work with a non-stdlib logger.
    # `structlog.stdlib.add_logger_name` reads ``logger.name`` which a
    # PrintLogger does NOT have -> it would raise AttributeError on every call.
    # We therefore use the structlog-native `processors.add_log_level` and skip
    # `add_logger_name`; the logger name is bound explicitly in `get_logger`.
    _processors: list[Any] = [
        _cast(Any, redact_event),
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
    structlog.configure(
        processors=_processors,
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ---- 2. Wrap sys.excepthook -----------------------------------------
    _original_excepthook = sys.excepthook

    def _redacting_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> None:
        lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        safe = redact("".join(lines))
        sys.stderr.write(safe)
        sys.stderr.flush()

    # Only replace if not already our wrapper (guards double-install in tests)
    if sys.excepthook is _original_excepthook:
        sys.excepthook = _redacting_excepthook

    # ---- 3. Quiet noisy SDK loggers ------------------------------------
    import logging

    for name in ("httpx", "eth_account", "hyperliquid"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _REDACTION_INSTALLED = True


def get_logger(name: str) -> Any:
    """Return a structlog logger bound with redaction.

    Calls :func:`install_global_redaction` if it hasn't been called yet, so
    callers never need to bootstrap manually.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.BoundLogger
        A bound logger whose output passes through the redaction processor.
    """
    install_global_redaction()
    # Bind the name as a regular event field (``logger=<name>``) rather than
    # relying on the stdlib-only `add_logger_name` processor, which would crash
    # on the PrintLogger we use as the factory.
    return structlog.get_logger().bind(logger=name)
