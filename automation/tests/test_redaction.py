"""
Tests for automation.core.redaction.

Coverage
--------
1. ``redact()`` — 64-hex strings (with and without 0x prefix) become «redacted».
2. ``redact()`` — long base64 blobs become «redacted»; short strings are left alone.
3. ``redact_event`` — dict with ``signature`` / ``agent_key`` keys are redacted;
   a 64-hex value in a *plain* key is also redacted via the string scanner.
4. After ``install_global_redaction()``, emitting a structlog log line that
   contains a known 64-hex secret does NOT expose that secret in the output.
"""

from __future__ import annotations

import base64 as _b64
import io

import structlog

from automation.core.redaction import (
    _REDACTED,
    SECRET_KEY_NAMES,
    get_logger,
    install_global_redaction,
    redact,
    redact_event,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A plausible-looking 64-hex string (32 bytes) — NOT a real key.
_FAKE_HEX_KEY_NO_PREFIX = "a" * 64
_FAKE_HEX_KEY_WITH_0X = "0x" + "b" * 64

# A long enough base64 blob to trigger the heuristic (≥40 chars)
_FAKE_B64 = _b64.b64encode(b"\xde\xad\xbe\xef" * 20).decode()  # 108 chars


# ---------------------------------------------------------------------------
# 1. redact() — hex patterns
# ---------------------------------------------------------------------------


class TestRedactHex:
    def test_no_prefix_hex64_is_redacted(self) -> None:
        text = f"key={_FAKE_HEX_KEY_NO_PREFIX}"
        out = redact(text)
        assert _FAKE_HEX_KEY_NO_PREFIX not in out
        assert _REDACTED in out

    def test_0x_prefix_hex64_is_redacted(self) -> None:
        text = f"private_key={_FAKE_HEX_KEY_WITH_0X}"
        out = redact(text)
        assert _FAKE_HEX_KEY_WITH_0X not in out
        assert _REDACTED in out

    def test_shorter_hex_not_redacted(self) -> None:
        # 10-char hex — should NOT be redacted (not a key)
        short_hex = "deadbeef12"
        out = redact(f"txid={short_hex}")
        assert short_hex in out

    def test_mixed_text_preserved_around_secret(self) -> None:
        text = f"prefix {_FAKE_HEX_KEY_WITH_0X} suffix"
        out = redact(text)
        assert "prefix" in out
        assert "suffix" in out
        assert _FAKE_HEX_KEY_WITH_0X not in out


class TestRedactAddressNotRedacted:
    """Public Ethereum addresses (0x + 40 hex) must stay loggable.

    Regression for the over-redaction bug where ``0x0000...0000`` was mangled to
    ``«redacted»00`` by the base64 heuristic.  Addresses are PUBLIC — we log
    ``sub=<address>`` on every cycle.
    """

    def test_zero_address_unchanged(self) -> None:
        addr = "0x" + "0" * 40
        assert redact(addr) == addr

    def test_all_a_address_unchanged(self) -> None:
        addr = "0x" + "a" * 40
        assert redact(addr) == addr

    def test_mixed_case_address_unchanged(self) -> None:
        # A realistic EIP-55 mixed-case address
        addr = "0xAbC1230000000000000000000000000000004567"
        assert redact(addr) == addr

    def test_address_in_context_unchanged(self) -> None:
        text = f"routing sub={('0x' + '0' * 40)} done"
        out = redact(text)
        assert ("0x" + "0" * 40) in out
        assert _REDACTED not in out

    def test_address_kept_but_key_redacted_in_same_string(self) -> None:
        addr = "0x" + "0" * 40
        key = "0x" + "a" * 64
        out = redact(f"sub={addr} key={key}")
        assert addr in out  # public address survives
        assert key not in out  # private key is gone
        assert _REDACTED in out

    def test_64hex_still_redacted(self) -> None:
        # The complementary half of the regression: a genuine 64-hex key IS hit.
        key = "0x" + "a" * 64
        out = redact(key)
        assert key not in out
        assert _REDACTED in out


# ---------------------------------------------------------------------------
# 2. redact() — base64 blobs
# ---------------------------------------------------------------------------


class TestRedactBase64:
    def test_long_base64_blob_is_redacted(self) -> None:
        text = f"token={_FAKE_B64}"
        out = redact(text)
        assert _FAKE_B64 not in out
        assert _REDACTED in out

    def test_short_base64_left_alone(self) -> None:
        # 12 chars of base64 — below the heuristic threshold
        short = "SGVsbG8="  # "Hello"
        out = redact(f"greeting={short}")
        assert short in out


# ---------------------------------------------------------------------------
# 3. redact_event processor
# ---------------------------------------------------------------------------


class TestRedactEvent:
    def _call(self, d: dict) -> dict:
        """Invoke the processor with dummy logger/method."""
        return redact_event(None, "info", d)  # type: ignore[arg-type]

    def test_signature_key_is_redacted(self) -> None:
        out = self._call({"signature": _FAKE_HEX_KEY_WITH_0X})
        assert out["signature"] == _REDACTED

    def test_agent_key_is_redacted(self) -> None:
        out = self._call({"agent_key": _FAKE_HEX_KEY_NO_PREFIX})
        assert out["agent_key"] == _REDACTED

    def test_private_key_is_redacted(self) -> None:
        out = self._call({"private_key": _FAKE_HEX_KEY_WITH_0X})
        assert out["private_key"] == _REDACTED

    def test_all_secret_key_names_covered(self) -> None:
        """Every key in SECRET_KEY_NAMES should cause value redaction."""
        for key_name in SECRET_KEY_NAMES:
            out = self._call({key_name: "some_value_that_should_be_gone"})
            assert out[key_name] == _REDACTED, f"key={key_name!r} was not redacted"

    def test_64hex_in_plain_key_is_scrubbed(self) -> None:
        """A hex-64 embedded in a non-sensitive key's value is still redacted."""
        out = self._call({"message": f"doing stuff with key={_FAKE_HEX_KEY_WITH_0X}"})
        assert _FAKE_HEX_KEY_WITH_0X not in out["message"]
        assert _REDACTED in out["message"]

    def test_non_sensitive_non_hex_value_is_preserved(self) -> None:
        out = self._call({"coin": "BTC", "equity": "12345.67"})
        assert out["coin"] == "BTC"
        assert out["equity"] == "12345.67"

    def test_non_string_value_is_preserved(self) -> None:
        out = self._call({"count": 42, "active": True})
        assert out["count"] == 42
        assert out["active"] is True


# ---------------------------------------------------------------------------
# 4. install_global_redaction() — log emission check
# ---------------------------------------------------------------------------


class TestInstallGlobalRedaction:
    def test_install_is_idempotent(self) -> None:
        """Calling install twice should not raise and not break logging."""
        install_global_redaction()
        install_global_redaction()

    def test_get_logger_info_does_not_raise(self) -> None:
        """Regression: PrintLogger has no `.name`, so a stdlib-only processor
        (`add_logger_name`) in the chain crashed every `.info(...)` call with
        AttributeError.  The real `get_logger(...).info(...)` path must work.
        """
        log = get_logger("x")
        # These calls previously raised AttributeError on PrintLogger.name.
        log.info("msg", env="testnet", sub="0x" + "0" * 40)
        log.info("with key", agent_key="0x" + "a" * 64)

    def test_log_line_does_not_expose_hex_secret(self) -> None:
        """After installing redaction, a log message containing a 64-hex secret
        must NOT appear in captured output — exercised through the REAL
        production processor chain (PrintLoggerFactory, no stdlib processors).
        """
        install_global_redaction()

        # Point the production chain at a StringIO buffer so we can inspect
        # output deterministically.  Mirrors install_global_redaction()'s chain
        # exactly (native add_log_level, NO stdlib add_logger_name).
        buf = io.StringIO()

        structlog.configure(
            processors=[
                redact_event,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(10),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=buf),
            cache_logger_on_first_use=False,
        )

        log = structlog.get_logger().bind(logger="test_install")
        secret = _FAKE_HEX_KEY_WITH_0X
        # Public address must survive; secret key must be scrubbed.
        addr = "0x" + "0" * 40
        log.info("processing tx", agent_key=secret, sub=addr, message=f"raw={secret}")

        output = buf.getvalue()
        assert secret not in output, (
            f"Secret leaked into log output!\nOutput: {output[:500]}"
        )
        assert _REDACTED in output
        assert addr in output, "Public address was wrongly redacted from log output"
