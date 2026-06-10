"""
Tests for peer-issue-1 hardening: exclusive nonce-file lock in HLClient.for_trading.

Single-authoritative-signer invariant (spec §7/§23): if a second process is
misconfigured onto the same agent key + same nonce state file it MUST fail
loudly at boot (RuntimeError) instead of silently double-signing.

All tests are offline — no network calls, no real orders.  ``for_trading``
builds an Info + Exchange via the SDK constructor but does NOT make any HTTP
call at construction time (only read/write methods hit the network).  Tested
patterns:

1. Second process simulation → RuntimeError (same path).
2. Release allows re-acquire (same path, serial).
3. Distinct paths → both succeed (daemon vs watchdog path separation).
4. close() idempotent; no-op on read-only (no-lock) client.
5. Context-manager: lock held inside ``with`` block, released on exit.
"""

from __future__ import annotations

from pathlib import Path

import eth_account
import pytest
from filelock import FileLock

from automation.core.hl_client import HLClient

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_SUB = "0x" + "a" * 40


def _agent_key() -> str:
    """Generate a throwaway private key — no network needed."""
    return eth_account.Account.create().key.hex()


def _for_trading(path: Path, key: str | None = None) -> HLClient:
    """Build a for_trading client using a throwaway key (offline)."""
    k = key if key is not None else _agent_key()
    return HLClient.for_trading("testnet", _FAKE_SUB, k, path)


# ---------------------------------------------------------------------------
# 1. Second process simulation: same nonce_state_path → RuntimeError
# ---------------------------------------------------------------------------


class TestSecondProcessBlocked:
    def test_second_for_trading_same_path_raises(self, tmp_path: Path) -> None:
        """A second for_trading on the SAME nonce_state_path raises RuntimeError.

        Simulates two misconfigured processes sharing the same agent/state file.
        The first acquires the OS advisory lock; the second must be rejected.
        """
        nonce_path = tmp_path / "nonce.txt"
        client_a = _for_trading(nonce_path)
        try:
            with pytest.raises(RuntimeError, match="single authoritative signer"):
                _for_trading(nonce_path)
        finally:
            client_a.close()

    def test_error_message_mentions_lock(self, tmp_path: Path) -> None:
        """RuntimeError message mentions the nonce lock (not a generic error)."""
        nonce_path = tmp_path / "nonce.txt"
        client_a = _for_trading(nonce_path)
        try:
            with pytest.raises(RuntimeError) as exc_info:
                _for_trading(nonce_path)
            msg = str(exc_info.value)
            assert "nonce lock" in msg or "authoritative signer" in msg, (
                f"Error message should explain the lock purpose; got: {msg!r}"
            )
        finally:
            client_a.close()

    def test_raw_filelock_holder_blocks_for_trading(self, tmp_path: Path) -> None:
        """A raw FileLock (simulating an external process) blocks for_trading.

        This is the canonical "process A is alive with the lock, process B
        tries to start" simulation without requiring an actual subprocess.
        """
        nonce_path = tmp_path / "nonce.txt"
        # Simulate process A holding the lock
        process_a_lock = FileLock(str(nonce_path) + ".lock")
        process_a_lock.acquire(timeout=0)
        try:
            with pytest.raises(RuntimeError, match="single authoritative signer"):
                _for_trading(nonce_path)
        finally:
            process_a_lock.release()


# ---------------------------------------------------------------------------
# 2. Release allows re-acquire (restart pattern)
# ---------------------------------------------------------------------------


class TestReleaseAllowsReacquire:
    def test_close_then_new_for_trading_succeeds(self, tmp_path: Path) -> None:
        """After close(), a subsequent for_trading on the same path SUCCEEDS.

        This is the clean-restart pattern: the daemon calls shutdown() (which
        calls client.close()) and a new process starts immediately after.
        """
        nonce_path = tmp_path / "nonce.txt"

        # First client acquires the lock.
        client_a = _for_trading(nonce_path)
        client_a.close()

        # Second client (simulating the restarted daemon) must succeed.
        client_b = _for_trading(nonce_path)
        assert client_b._nonce_lock is not None
        client_b.close()

    def test_context_manager_releases_lock_on_exit(self, tmp_path: Path) -> None:
        """Context manager exit releases the lock; a subsequent for_trading succeeds."""
        nonce_path = tmp_path / "nonce.txt"

        with _for_trading(nonce_path):
            pass  # lock is held here

        # After the context manager exits the lock is released.
        client_b = _for_trading(nonce_path)
        client_b.close()


# ---------------------------------------------------------------------------
# 3. Distinct paths: daemon + watchdog paths both succeed
# ---------------------------------------------------------------------------


class TestDistinctPaths:
    def test_two_different_paths_both_succeed(self, tmp_path: Path) -> None:
        """for_trading on DIFFERENT nonce_state_paths both acquire their locks.

        This proves the daemon + watchdog separation: the watchdog uses a
        different state file, so they never contend (spec §7/§23).
        """
        daemon_path = tmp_path / "daemon_nonce.txt"
        watchdog_path = tmp_path / "watchdog_nonce.txt"

        client_daemon = _for_trading(daemon_path)
        client_watchdog = _for_trading(watchdog_path)
        try:
            # Both clients are alive with their respective locks.
            assert client_daemon._nonce_lock is not None
            assert client_watchdog._nonce_lock is not None
        finally:
            client_daemon.close()
            client_watchdog.close()

    def test_nested_dir_path_created_automatically(self, tmp_path: Path) -> None:
        """for_trading creates parent directories for the nonce_state_path."""
        deep_path = tmp_path / "a" / "b" / "c" / "nonce.txt"
        assert not deep_path.parent.exists()

        client = _for_trading(deep_path)
        try:
            assert deep_path.parent.exists(), "Parent directory must be created"
            assert client._nonce_lock is not None
        finally:
            client.close()


# ---------------------------------------------------------------------------
# 4. close() idempotent; no-op on read-only client
# ---------------------------------------------------------------------------


class TestCloseIdempotentAndNoop:
    def test_close_twice_is_idempotent(self, tmp_path: Path) -> None:
        """Calling close() twice does not raise."""
        nonce_path = tmp_path / "nonce.txt"
        client = _for_trading(nonce_path)
        client.close()
        client.close()  # second call: must not raise

    def test_close_on_readonly_client_is_noop(self) -> None:
        """HLClient(env, sub) without agent_key has no lock; close() is a no-op."""
        client = HLClient("testnet", _FAKE_SUB)
        assert client._nonce_lock is None
        client.close()  # must not raise
        assert client._nonce_lock is None

    def test_lock_attribute_always_present_on_plain_client(self) -> None:
        """_nonce_lock attribute exists even on a plain (non-for_trading) client."""
        k = _agent_key()
        client = HLClient("testnet", _FAKE_SUB, agent_key=k)
        assert hasattr(client, "_nonce_lock"), "_nonce_lock must always exist"
        assert client._nonce_lock is None


# ---------------------------------------------------------------------------
# 5. Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_yields_self(self, tmp_path: Path) -> None:
        """``with HLClient.for_trading(...) as c:`` yields the client itself."""
        nonce_path = tmp_path / "nonce.txt"
        k = _agent_key()
        with HLClient.for_trading("testnet", _FAKE_SUB, k, nonce_path) as c:
            assert c is not None
            assert c._nonce_lock is not None, "Lock must be held inside context"

    def test_lock_released_after_context_exit(self, tmp_path: Path) -> None:
        """After the context manager exits, the lock is released."""
        nonce_path = tmp_path / "nonce.txt"

        with _for_trading(nonce_path) as _:
            pass  # lock held here

        # Verify the lock is released by successfully acquiring a new one.
        client_b = _for_trading(nonce_path)
        client_b.close()

    def test_lock_released_even_on_exception(self, tmp_path: Path) -> None:
        """Context manager releases the lock even if the body raises."""
        nonce_path = tmp_path / "nonce.txt"

        try:
            with _for_trading(nonce_path):
                raise ValueError("body exception")
        except ValueError:
            pass

        # After the exception the lock must be released.
        client_b = _for_trading(nonce_path)
        client_b.close()
