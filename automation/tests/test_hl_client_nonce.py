"""
Tests for G1 (MF-10): NonceManager wired into live order-submit path.

All tests are offline — no network calls, no real orders submitted.

The fake exchange pattern
--------------------------
Each test replaces ``client.exchange`` with a ``FakeExchange`` whose
``market_open``/``market_close`` capture the value that
``hyperliquid.exchange.get_timestamp_ms()`` returns at call time.  This
lets us assert that:

- The nonce seen by the fake equals the value that ``NonceManager.next()``
  would produce (i.e. the override IS active during the call).
- After the call (or exception) the original function is restored.

The fake deliberately calls ``import hyperliquid.exchange as _hl_exchange``
and reads ``_hl_exchange.get_timestamp_ms()`` at call time — exactly as
the real SDK's ``bulk_orders()`` does — so it faithfully models the
module-namespace lookup that the override targets.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import eth_account
import hyperliquid.exchange as _hl_exchange
import pytest

from automation.core.hl_client import HLClient
from automation.core.nonce import ClockError, NonceManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SUB = "0x" + "a" * 40
_FAKE_CLOID = "0x" + "ab" * 16  # 32 hex chars = 16 bytes


def _agent_key() -> str:
    """Generate a throwaway agent private key (no network needed)."""
    return eth_account.Account.create().key.hex()


class FakeExchange:
    """Minimal exchange stub that records the nonce seen during each call.

    ``market_open`` and ``market_close`` both call
    ``hyperliquid.exchange.get_timestamp_ms()`` at invocation time — mirroring
    the real SDK's ``bulk_orders()`` — and store the result in
    ``self.recorded_nonces``.  They also record whether they were called at all
    so tests can assert refuse-before-send.

    ``vault_address`` is stored to satisfy the routing assertion already in
    the test_hl_client.py suite (not used by nonce tests directly).
    """

    def __init__(self, vault_address: str) -> None:
        self.vault_address: str = vault_address
        self.recorded_nonces: list[int] = []
        self.call_count: int = 0
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
        return self.response

    def market_close(
        self,
        coin: str,
        sz: object = None,
        px: object = None,
        slippage: float = 0.01,
        *,
        cloid: object = None,
    ) -> dict[str, Any]:
        nonce = _hl_exchange.get_timestamp_ms()
        self.recorded_nonces.append(nonce)
        self.call_count += 1
        return self.response


def _client_with_fake_exchange(tmp_path: Path) -> tuple[HLClient, FakeExchange, NonceManager]:
    """Build an HLClient backed by a FakeExchange and a real NonceManager."""
    nonce_path = tmp_path / "nonce.txt"
    nm = NonceManager(nonce_path)
    k = _agent_key()
    client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
    fake = FakeExchange(vault_address=_FAKE_SUB)
    client.exchange = fake  # type: ignore[assignment]
    return client, fake, nm


# ---------------------------------------------------------------------------
# 1. Nonce comes from the manager — override is active during call
# ---------------------------------------------------------------------------


class TestNonceComesFromManager:
    def test_submit_ioc_nonce_from_manager(self, tmp_path: Path) -> None:
        """submit_ioc: the nonce the fake saw IS the manager's issued nonce.

        Tightened (F7): assert exact equality against ``nm._last`` (the value the
        manager persisted on this ``next()``), so this test FAILS if the override
        regresses to a no-op (which would let the fake read the real SDK
        ``get_timestamp_ms`` instead — a different value the manager never
        recorded).
        """
        client, fake, nm = _client_with_fake_exchange(tmp_path)

        before_ms = int(time.time() * 1000)
        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)
        after_ms = int(time.time() * 1000) + 1

        assert fake.call_count == 1
        assert len(fake.recorded_nonces) == 1
        nonce = fake.recorded_nonces[0]
        # EXACT: the fake saw precisely the nonce the manager issued + persisted.
        assert nonce == nm._last, (
            f"nonce seen by fake {nonce} != manager's last issued nonce {nm._last} "
            "— the override is a no-op (regression)"
        )
        # Sanity: it is also a plausible current-time ms value.
        assert before_ms <= nonce <= after_ms + 100, (
            f"nonce {nonce} is not near call time [{before_ms}, {after_ms}]"
        )

    def test_submit_ioc_nonce_file_advanced(self, tmp_path: Path) -> None:
        """After submit_ioc, the nonce state file records the issued nonce."""
        nonce_path = tmp_path / "nonce.txt"
        nm = NonceManager(nonce_path)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        client.submit_ioc("ETH", False, 0.01, 0.005, _FAKE_CLOID)

        assert nonce_path.exists()
        persisted = int(nonce_path.read_text().strip())
        assert persisted == fake.recorded_nonces[0], (
            f"persisted={persisted} != nonce seen by fake={fake.recorded_nonces[0]}"
        )

    def test_market_close_nonce_from_manager(self, tmp_path: Path) -> None:
        """market_close: the nonce the fake saw IS the manager's issued nonce.

        Tightened (F7): exact equality against ``nm._last`` so a regressed no-op
        override (fake reading the real SDK timestamp) would fail.
        """
        client, fake, nm = _client_with_fake_exchange(tmp_path)

        before_ms = int(time.time() * 1000)
        client.market_close("SOL", 0.005, _FAKE_CLOID)
        after_ms = int(time.time() * 1000) + 1

        assert fake.call_count == 1
        nonce = fake.recorded_nonces[0]
        assert nonce == nm._last, (
            f"nonce seen by fake {nonce} != manager's last issued nonce {nm._last} "
            "— the override is a no-op (regression)"
        )
        assert before_ms <= nonce <= after_ms + 100


# ---------------------------------------------------------------------------
# 2. Override restored after call (and after exception)
# ---------------------------------------------------------------------------


class TestOverrideRestored:
    def _original_fn(self) -> object:
        """Capture original get_timestamp_ms before any client is created."""
        return _hl_exchange.get_timestamp_ms

    def test_restored_after_successful_submit_ioc(self, tmp_path: Path) -> None:
        """After submit_ioc returns normally, get_timestamp_ms is the original."""
        original_fn = _hl_exchange.get_timestamp_ms
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was NOT restored after submit_ioc returned normally"
        )

    def test_restored_after_successful_market_close(self, tmp_path: Path) -> None:
        """After market_close returns normally, get_timestamp_ms is the original."""
        original_fn = _hl_exchange.get_timestamp_ms
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        client.market_close("ETH", 0.005, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn

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
            client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was NOT restored after exchange raised"
        )


# ---------------------------------------------------------------------------
# 3. Monotonic across two submits
# ---------------------------------------------------------------------------


class TestMonotonicAcrossSubmits:
    def test_two_submits_monotonic(self, tmp_path: Path) -> None:
        """Second submit's nonce is strictly > first's; both are persisted."""
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)
        client.submit_ioc("ETH", True, 0.01, 0.005, _FAKE_CLOID)

        n1, n2 = fake.recorded_nonces
        assert n2 > n1, f"Second nonce {n2} must be strictly > first {n1}"

    def test_mixed_submit_ioc_and_market_close_monotonic(self, tmp_path: Path) -> None:
        """Alternating submit_ioc / market_close produce strictly increasing nonces."""
        client, fake, _ = _client_with_fake_exchange(tmp_path)

        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)
        client.market_close("BTC", 0.005, _FAKE_CLOID)
        client.submit_ioc("ETH", False, 0.01, 0.005, _FAKE_CLOID)

        nonces = fake.recorded_nonces
        assert len(nonces) == 3
        for i in range(1, len(nonces)):
            assert nonces[i] > nonces[i - 1], (
                f"nonce[{i}]={nonces[i]} not > nonce[{i - 1}]={nonces[i - 1]}"
            )


# ---------------------------------------------------------------------------
# 4. Cross-restart continuity
# ---------------------------------------------------------------------------


class TestCrossRestartContinuity:
    def test_second_client_same_path_continues_above_first(self, tmp_path: Path) -> None:
        """A second HLClient with a new NonceManager on the same path continues above."""
        nonce_path = tmp_path / "nonce.txt"

        # First client — issue several nonces
        nm1 = NonceManager(nonce_path)
        k = _agent_key()
        c1 = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm1)
        fake1 = FakeExchange(vault_address=_FAKE_SUB)
        c1.exchange = fake1  # type: ignore[assignment]

        for _ in range(3):
            c1.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        last_nonce_c1 = fake1.recorded_nonces[-1]

        # Simulate daemon restart: new NonceManager + new HLClient on same path
        nm2 = NonceManager(nonce_path)
        c2 = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm2)
        fake2 = FakeExchange(vault_address=_FAKE_SUB)
        c2.exchange = fake2  # type: ignore[assignment]

        c2.submit_ioc("ETH", True, 0.01, 0.005, _FAKE_CLOID)
        first_nonce_c2 = fake2.recorded_nonces[0]

        assert first_nonce_c2 > last_nonce_c1, (
            f"After restart, first nonce {first_nonce_c2} must be > "
            f"last nonce from previous session {last_nonce_c1}"
        )


# ---------------------------------------------------------------------------
# 5. Backward-clock refusal (refuse-before-send)
# ---------------------------------------------------------------------------


class TestBackwardClockRefusal:
    def test_clock_error_propagates_from_submit_ioc(self, tmp_path: Path) -> None:
        """submit_ioc raises ClockError when clock jumps backward beyond tolerance."""
        nonce_path = tmp_path / "nonce.txt"

        # Pre-seed the nonce file with a far-future value that is well beyond
        # tolerance from any real clock reading (current time + 60s + tolerance).
        far_future_ms = int(time.time() * 1000) + 70_000
        nonce_path.write_text(f"{far_future_ms}\n")

        # Build a fresh NonceManager that loads that far-future value as _last
        nm2 = NonceManager(nonce_path, backward_tolerance_ms=5000)

        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm2)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        with pytest.raises(ClockError):
            client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        # The fake exchange must NOT have been called (refuse before send)
        assert fake.call_count == 0, (
            f"exchange.market_open was called {fake.call_count} time(s) "
            "despite ClockError — order was NOT refused before send"
        )

    def test_clock_error_propagates_from_market_close(self, tmp_path: Path) -> None:
        """market_close raises ClockError and the fake exchange is never called."""
        nonce_path = tmp_path / "nonce.txt"
        far_future_ms = int(time.time() * 1000) + 70_000
        nonce_path.write_text(f"{far_future_ms}\n")

        nm = NonceManager(nonce_path, backward_tolerance_ms=5000)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        with pytest.raises(ClockError):
            client.market_close("ETH", 0.005, _FAKE_CLOID)

        assert fake.call_count == 0

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
            client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was NOT restored after ClockError"
        )


# ---------------------------------------------------------------------------
# 6. nonce_manager=None path — unchanged behaviour
# ---------------------------------------------------------------------------


class TestNoneManagerPathUnchanged:
    def test_submit_ioc_no_manager_does_not_touch_module_global(
        self, tmp_path: Path
    ) -> None:
        """With nonce_manager=None, the module-level get_timestamp_ms is never swapped."""
        original_fn = _hl_exchange.get_timestamp_ms

        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k)  # no nonce_manager
        assert client._nonce is None

        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        # get_timestamp_ms must be the original (unpatched) function
        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "get_timestamp_ms was unexpectedly modified when nonce_manager=None"
        )
        # The fake WAS called (order not blocked)
        assert fake.call_count == 1

    def test_market_close_no_manager_does_not_touch_module_global(
        self, tmp_path: Path
    ) -> None:
        original_fn = _hl_exchange.get_timestamp_ms

        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        client.market_close("ETH", 0.005, _FAKE_CLOID)

        assert _hl_exchange.get_timestamp_ms is original_fn
        assert fake.call_count == 1

    def test_none_manager_nonce_seen_by_fake_is_real_timestamp(
        self, tmp_path: Path
    ) -> None:
        """With no manager, the fake sees the real SDK get_timestamp_ms value."""
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        before_ms = int(time.time() * 1000)
        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)
        after_ms = int(time.time() * 1000) + 1

        # The nonce is the real get_timestamp_ms() value — should be near now
        nonce = fake.recorded_nonces[0]
        assert before_ms <= nonce <= after_ms + 100, (
            f"nonce {nonce} should be near current time [{before_ms}, {after_ms}] "
            "when no NonceManager is wired"
        )


# ---------------------------------------------------------------------------
# 7. for_trading factory
# ---------------------------------------------------------------------------


class TestForTradingFactory:
    def test_for_trading_wires_nonce_manager(self, tmp_path: Path) -> None:
        """for_trading() builds an HLClient with a NonceManager on the given path."""
        nonce_path = tmp_path / "nonce.txt"
        k = _agent_key()

        client = HLClient.for_trading(
            "testnet", _FAKE_SUB, k, nonce_path, backward_tolerance_ms=5000
        )

        assert client._nonce is not None, "_nonce should be a NonceManager"
        assert isinstance(client._nonce, NonceManager)

    def test_for_trading_nonce_path_matches(self, tmp_path: Path) -> None:
        """for_trading() NonceManager is backed by the supplied state path."""
        nonce_path = tmp_path / "state" / "nonce.txt"
        k = _agent_key()

        client = HLClient.for_trading("testnet", _FAKE_SUB, k, nonce_path)

        assert client._nonce is not None
        # Trigger a next() — file should be created at the given path
        client._nonce.next()
        assert nonce_path.exists(), f"State file not created at {nonce_path}"

    def test_for_trading_submits_use_nonce_manager(self, tmp_path: Path) -> None:
        """for_trading() client's submits use the wired NonceManager nonces."""
        nonce_path = tmp_path / "nonce.txt"
        k = _agent_key()

        client = HLClient.for_trading("testnet", _FAKE_SUB, k, nonce_path)
        fake = FakeExchange(vault_address=_FAKE_SUB)
        client.exchange = fake  # type: ignore[assignment]

        before_ms = int(time.time() * 1000)
        client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)
        after_ms = int(time.time() * 1000) + 1

        assert fake.call_count == 1
        nonce = fake.recorded_nonces[0]
        assert before_ms <= nonce <= after_ms + 100

        # State file persisted
        assert nonce_path.exists()
        persisted = int(nonce_path.read_text().strip())
        assert persisted == nonce

    def test_for_trading_exchange_attribute_set(self, tmp_path: Path) -> None:
        """for_trading() returns a client with exchange set (not None)."""
        nonce_path = tmp_path / "nonce.txt"
        k = _agent_key()

        client = HLClient.for_trading("testnet", _FAKE_SUB, k, nonce_path)

        assert client.exchange is not None
        assert client.sub == _FAKE_SUB

    def test_for_trading_default_tolerance(self, tmp_path: Path) -> None:
        """for_trading() with no backward_tolerance_ms uses default 5000."""
        nonce_path = tmp_path / "nonce.txt"
        k = _agent_key()

        client = HLClient.for_trading("testnet", _FAKE_SUB, k, nonce_path)

        assert client._nonce is not None
        assert client._nonce._tolerance == 5000


# ---------------------------------------------------------------------------
# 8. Re-entrancy guard on the override (F4)
# ---------------------------------------------------------------------------


class TestReentrancyGuard:
    def test_nested_override_raises_and_restores(self, tmp_path: Path) -> None:
        """A re-entrant (nested) _nonce_override raises RuntimeError.

        Simulates a concurrent same-process submit: an outer override is active
        (global already tagged) when an inner submit tries to swap again.  The
        inner attempt must raise BEFORE mutating the global, and the outer
        ``finally`` must still restore the TRUE original.
        """
        original_fn = _hl_exchange.get_timestamp_ms

        # The fake's market_open, while the OUTER override is active, performs a
        # nested submit — which must raise the concurrent-override RuntimeError.
        class NestingExchange(FakeExchange):
            def __init__(self, vault_address: str, client: HLClient) -> None:
                super().__init__(vault_address)
                self._client = client

            def market_open(self, *args: Any, **kwargs: Any) -> Any:
                # While the outer override is installed, attempt a nested submit.
                return self._client.submit_ioc("ETH", True, 0.01, 0.005, _FAKE_CLOID)

        nonce_path = tmp_path / "nonce.txt"
        nm = NonceManager(nonce_path)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)
        client.exchange = NestingExchange(_FAKE_SUB, client)  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="concurrent nonce override"):
            client.submit_ioc("BTC", True, 0.001, 0.005, _FAKE_CLOID)

        # Outer finally must have restored the TRUE original (not the lambda).
        assert _hl_exchange.get_timestamp_ms is original_fn, (
            "global get_timestamp_ms was NOT restored to the true original after "
            "a re-entrant override was rejected"
        )

    def test_direct_nested_context_manager_raises(self, tmp_path: Path) -> None:
        """Entering _nonce_override twice (nested) on the same client raises."""
        original_fn = _hl_exchange.get_timestamp_ms

        nonce_path = tmp_path / "nonce.txt"
        nm = NonceManager(nonce_path)
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k, nonce_manager=nm)

        with client._nonce_override():  # noqa: SIM117 — nested-override test, must stay nested
            with pytest.raises(RuntimeError, match="concurrent nonce override"):
                # Nested entry on the same client must raise (re-entrancy guard).
                with client._nonce_override():
                    pass  # pragma: no cover — should never reach here

        # After the outer block exits, the global is restored to the original.
        assert _hl_exchange.get_timestamp_ms is original_fn
