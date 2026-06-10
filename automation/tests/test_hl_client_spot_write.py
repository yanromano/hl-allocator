"""
Tests for D3: HLClient.submit_spot_ioc — spot IOC buy/sell via SDK market_open.

All tests are offline — no network calls, no real orders submitted.

The fake-exchange pattern mirrors test_hl_client_nonce.py: a ``FakeExchange``
records the nonce that ``hyperliquid.exchange.get_timestamp_ms()`` returns at
call time (so we can prove the nonce override is active during the call), plus
the positional/keyword arguments that reached ``market_open`` (so we can prove
the spot pair name, is_buy and sz are forwarded verbatim).

Spot venue note
---------------
``submit_spot_ioc`` is a thin wrapper around the SAME ``Exchange.market_open``
the perp path uses — the SDK does spot asset resolution (asset id >= 10000) and
8-decimal price rounding internally when ``name`` is a spot pair (e.g. ``@107``).
A spot SELL (``is_buy=False``) is how a spot position is reduced/flattened —
there is no reduce-only ``market_close`` for spot.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import eth_account
import hyperliquid.exchange as _hl_exchange
import pytest
from hyperliquid.utils.signing import Cloid

from automation.core.hl_client import HLClient
from automation.core.nonce import ClockError, NonceManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SUB = "0x" + "a" * 40
_FAKE_CLOID = "0x" + "ab" * 16  # 32 hex chars = 16 bytes
_SPOT_PAIR = "@107"  # HL spot pair name (e.g. HYPE/USDC)


def _agent_key() -> str:
    """Generate a throwaway agent private key (no network needed)."""
    return eth_account.Account.create().key.hex()


class FakeExchange:
    """Exchange stub that records the nonce + arguments seen during market_open.

    ``market_open`` reads ``hyperliquid.exchange.get_timestamp_ms()`` at call
    time (mirroring the real SDK's ``bulk_orders()``) so tests can assert the
    nonce override is active during the call.  It also captures the positional
    args and ``cloid`` keyword so tests can assert the spot pair, is_buy and sz
    are forwarded verbatim.
    """

    def __init__(self, vault_address: str) -> None:
        self.vault_address: str = vault_address
        self.recorded_nonces: list[int] = []
        self.call_count: int = 0
        self.calls: list[dict[str, Any]] = []
        self.response: dict[str, Any] = {"status": "ok", "fake": True}

    def market_open(
        self,
        name: str,
        is_buy: bool,
        sz: float,
        px: object,
        slippage: float,
        *,
        cloid: object = None,
    ) -> dict[str, Any]:
        nonce = _hl_exchange.get_timestamp_ms()
        self.recorded_nonces.append(nonce)
        self.call_count += 1
        self.calls.append(
            {
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "px": px,
                "slippage": slippage,
                "cloid": cloid,
            }
        )
        return self.response


def _client_with_fake_exchange(
    tmp_path: Path,
) -> tuple[HLClient, FakeExchange, NonceManager]:
    """Build an HLClient backed by a FakeExchange and a real NonceManager."""
    nonce_path = tmp_path / "nonce.txt"
    nm = NonceManager(nonce_path)
    k = _agent_key()
    client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
    fake = FakeExchange(vault_address=_FAKE_SUB)
    client.exchange = fake  # type: ignore[assignment]
    return client, fake, nm


# ---------------------------------------------------------------------------
# 1. submit_spot_ioc forwards the spot pair / is_buy / sz / cloid to market_open
# ---------------------------------------------------------------------------


class TestSpotIocForwarding:
    def test_spot_buy_forwards_pair_and_args(self, tmp_path: Path) -> None:
        """A spot BUY forwards the @107 pair, is_buy=True, sz, slippage, px=None."""
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        resp = client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)

        assert fake.call_count == 1
        call = fake.calls[0]
        assert call["name"] == _SPOT_PAIR, "spot pair name must be forwarded verbatim"
        assert call["is_buy"] is True
        assert call["sz"] == 30.0, "sz is in base-token units, forwarded as-is"
        assert call["slippage"] == 0.01
        # px must be None so the SDK computes the aggressive spot price itself.
        assert call["px"] is None
        # Response is the raw exchange dict, untouched.
        assert resp == fake.response

    def test_spot_buy_passes_cloid_object(self, tmp_path: Path) -> None:
        """market_open receives a Cloid built from the cloid string."""
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)

        cloid_seen = fake.calls[0]["cloid"]
        assert isinstance(cloid_seen, Cloid), "cloid must be a Cloid instance"
        assert cloid_seen.to_raw() == Cloid.from_str(_FAKE_CLOID).to_raw()

    def test_spot_sell_is_flatten(self, tmp_path: Path) -> None:
        """A spot SELL (is_buy=False) is forwarded — this is how spot flatten works."""
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        client.submit_spot_ioc(_SPOT_PAIR, False, 12.5, 0.01, _FAKE_CLOID)

        call = fake.calls[0]
        assert call["is_buy"] is False, (
            "spot SELL reduces/flattens a spot holding (no reduce-only on spot)"
        )
        assert call["name"] == _SPOT_PAIR
        assert call["sz"] == 12.5

    def test_returns_raw_response_dict(self, tmp_path: Path) -> None:
        """The raw response dict is returned unchanged for safety.parse_order_response."""
        client, fake, _ = _client_with_fake_exchange(tmp_path)
        fake.response = {"status": "ok", "response": {"type": "order"}}

        resp = client.submit_spot_ioc(_SPOT_PAIR, True, 1.0, 0.01, _FAKE_CLOID)

        assert resp is fake.response


# ---------------------------------------------------------------------------
# 2. Nonce override is active during the call (consumed) and is the manager's
# ---------------------------------------------------------------------------


class TestSpotIocNonce:
    def test_nonce_from_manager(self, tmp_path: Path) -> None:
        """The nonce the fake saw IS the manager's issued nonce (override active)."""
        client, fake, nm = _client_with_fake_exchange(tmp_path)

        before_ms = int(time.time() * 1000)
        client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)
        after_ms = int(time.time() * 1000) + 1

        assert fake.call_count == 1
        assert len(fake.recorded_nonces) == 1
        nonce = fake.recorded_nonces[0]
        # EXACT: the fake saw precisely the nonce the manager issued + persisted.
        assert nonce == nm._last, (
            f"nonce seen by fake {nonce} != manager's last issued nonce {nm._last} "
            "— the override is a no-op (regression)"
        )
        assert before_ms <= nonce <= after_ms + 100

    def test_nonce_file_advanced(self, tmp_path: Path) -> None:
        """After submit_spot_ioc, the nonce state file records the issued nonce."""
        nonce_path = tmp_path / "nonce.txt"
        nm = NonceManager(nonce_path)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        client.submit_spot_ioc(_SPOT_PAIR, False, 5.0, 0.01, _FAKE_CLOID)

        assert nonce_path.exists()
        persisted = int(nonce_path.read_text().strip())
        assert persisted == fake.recorded_nonces[0]


# ---------------------------------------------------------------------------
# 3. Override restored after the call (success + exception paths)
# ---------------------------------------------------------------------------


class TestSpotIocOverrideRestored:
    def test_restored_after_successful_call(self, tmp_path: Path) -> None:
        """After submit_spot_ioc returns normally, get_timestamp_ms is the original."""
        original_fn = _hl_exchange.get_timestamp_ms
        client, _fake, _ = _client_with_fake_exchange(tmp_path)

        client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was NOT restored after submit_spot_ioc returned"
        )

    def test_restored_after_exchange_raises(self, tmp_path: Path) -> None:
        """Even if the fake exchange raises, get_timestamp_ms is restored."""
        original_fn = _hl_exchange.get_timestamp_ms

        class RaisingExchange(FakeExchange):
            def market_open(self, *args: Any, **kwargs: Any) -> Any:
                _hl_exchange.get_timestamp_ms()  # consume the override
                raise RuntimeError("network error")

        nonce_path = tmp_path / "nonce.txt"
        nm = NonceManager(nonce_path)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        client.exchange = RaisingExchange(vault_address=_FAKE_SUB)  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="network error"):
            client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was NOT restored after exchange raised"
        )


# ---------------------------------------------------------------------------
# 4. ClockError path — refuse-before-send (mirrors submit_ioc)
# ---------------------------------------------------------------------------


class TestSpotIocClockError:
    def test_clock_error_propagates_and_refuses_before_send(
        self, tmp_path: Path
    ) -> None:
        """A backward clock jump raises ClockError and the exchange is never called."""
        nonce_path = tmp_path / "nonce.txt"
        far_future_ms = int(time.time() * 1000) + 70_000
        nonce_path.write_text(f"{far_future_ms}\n")

        nm = NonceManager(nonce_path, backward_tolerance_ms=5000)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        with pytest.raises(ClockError):
            client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)

        assert fake.call_count == 0, (
            f"market_open was called {fake.call_count} time(s) despite ClockError "
            "— spot order was NOT refused before send"
        )

    def test_clock_error_restores_override(self, tmp_path: Path) -> None:
        """Even on ClockError, get_timestamp_ms is restored to the original."""
        original_fn = _hl_exchange.get_timestamp_ms

        nonce_path = tmp_path / "nonce.txt"
        far_future_ms = int(time.time() * 1000) + 70_000
        nonce_path.write_text(f"{far_future_ms}\n")

        nm = NonceManager(nonce_path, backward_tolerance_ms=5000)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        with pytest.raises(ClockError):
            client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was NOT restored after ClockError"
        )


# ---------------------------------------------------------------------------
# 5. exchange is None — assertion guard (read-only client)
# ---------------------------------------------------------------------------


class TestSpotIocRequiresExchange:
    def test_raises_when_no_exchange(self) -> None:
        """submit_spot_ioc asserts when the client has no agent key (exchange None)."""
        client = HLClient("testnet", _FAKE_SUB)  # read-only — no agent_key
        assert client.exchange is None

        with pytest.raises(AssertionError):
            client.submit_spot_ioc(_SPOT_PAIR, True, 30.0, 0.01, _FAKE_CLOID)
