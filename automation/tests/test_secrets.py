"""
Tests for automation.core.secrets — TDD, RED phase written before implementation.

Security contract under test:
- Round-trip: encrypt → write JSON → decrypt == original private key.
- Wrong passphrase raises (eth_account raises ValueError/Exception).
- Mainnet + EnvBackend → PermissionError (hardcoded refusal).
- testnet + EnvBackend → allowed (dev convenience).
- assert_loadable returns an Ethereum address (0x…, len 42), NOT the private key.
- Key is never returned by assert_loadable; no log/print leakage tested structurally.
"""

import json
from pathlib import Path

import eth_account
import pytest

from automation.core.secrets import (
    EnvBackend,
    KeystoreKeyProvider,
    RawEnvKeyProvider,
    create_keystore,
)


# ---------------------------------------------------------------------------
# Tiny test-only backend — not part of the public API
# ---------------------------------------------------------------------------
class _StaticBackend:
    """Returns a fixed passphrase. Test use only — never ship to prod."""

    def __init__(self, passphrase: str) -> None:
        self._passphrase = passphrase

    def get(self) -> str:  # noqa: D102
        return self._passphrase


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
# Use very low scrypt n so each keystore op takes ~10ms instead of ~1s.
_SCRYPT_TEST_N = 1024
_PASSPHRASE = "hunter2_test_pw"


@pytest.fixture(scope="module")
def agent_account():
    """Fresh eth_account once per module — reused across tests to save key-gen time."""
    return eth_account.Account.create()


@pytest.fixture()
def keystore_file(tmp_path: Path, agent_account) -> Path:
    """Write a low-n scrypt keystore to a temp file and return the path."""
    ks = create_keystore(
        agent_account.key.hex(),
        _PASSPHRASE,
        kdf="scrypt",
        iterations=_SCRYPT_TEST_N,
    )
    p = tmp_path / "agent_keystore.json"
    p.write_text(json.dumps(ks))
    return p


# ---------------------------------------------------------------------------
# Test 1: Round-trip (create → write → load → compare)
# ---------------------------------------------------------------------------
def test_round_trip(keystore_file: Path, agent_account) -> None:
    """Decrypted key must equal the original private key (0x-hex, case-insensitive)."""
    provider = KeystoreKeyProvider(
        keystore_path=keystore_file,
        passphrase_backend=_StaticBackend(_PASSPHRASE),
        env="testnet",
    )
    loaded_key = provider.load_agent_key()

    original_hex = "0x" + agent_account.key.hex()
    assert loaded_key.lower() == original_hex.lower(), (
        f"Key mismatch: got {loaded_key[:10]}…, expected {original_hex[:10]}…"
    )
    # Must be a 32-byte hex (0x + 64 hex chars = 66 chars total)
    assert len(loaded_key) == 66
    assert loaded_key.startswith("0x") or loaded_key.startswith("0X")


# ---------------------------------------------------------------------------
# Test 2: Wrong passphrase → decrypt raises
# ---------------------------------------------------------------------------
def test_wrong_passphrase_raises(keystore_file: Path) -> None:
    """Loading with an incorrect passphrase must raise an exception."""
    provider = KeystoreKeyProvider(
        keystore_path=keystore_file,
        passphrase_backend=_StaticBackend("wrong_password_xyz"),
        env="testnet",
    )
    with pytest.raises((ValueError, Exception)):  # eth_account raises ValueError on bad passphrase
        provider.load_agent_key()


# ---------------------------------------------------------------------------
# Test 3a: Mainnet + EnvBackend → PermissionError (forbidden)
# ---------------------------------------------------------------------------
def test_mainnet_env_backend_refused(keystore_file: Path, monkeypatch) -> None:
    """EnvBackend on mainnet must be refused before any keystore I/O."""
    monkeypatch.setenv("AGENT_PK_PASSPHRASE", _PASSPHRASE)
    provider = KeystoreKeyProvider(
        keystore_path=keystore_file,
        passphrase_backend=EnvBackend("AGENT_PK_PASSPHRASE"),
        env="mainnet",
    )
    with pytest.raises(PermissionError, match="mainnet"):
        provider.load_agent_key()


# ---------------------------------------------------------------------------
# Test 3b: testnet + EnvBackend → allowed
# ---------------------------------------------------------------------------
def test_testnet_env_backend_allowed(keystore_file: Path, agent_account, monkeypatch) -> None:
    """EnvBackend is permitted on testnet (dev convenience)."""
    monkeypatch.setenv("AGENT_PK_PASSPHRASE", _PASSPHRASE)
    provider = KeystoreKeyProvider(
        keystore_path=keystore_file,
        passphrase_backend=EnvBackend("AGENT_PK_PASSPHRASE"),
        env="testnet",
    )
    loaded_key = provider.load_agent_key()
    original_hex = "0x" + agent_account.key.hex()
    assert loaded_key.lower() == original_hex.lower()


# ---------------------------------------------------------------------------
# Test 4: assert_loadable returns an eth address, not the private key
# ---------------------------------------------------------------------------
def test_assert_loadable_returns_address_not_key(keystore_file: Path, agent_account) -> None:
    """assert_loadable returns a checksummed Ethereum address (42 chars, 0x prefix)."""
    provider = KeystoreKeyProvider(
        keystore_path=keystore_file,
        passphrase_backend=_StaticBackend(_PASSPHRASE),
        env="testnet",
    )
    address = provider.assert_loadable()

    # Must look like an Ethereum address
    assert isinstance(address, str)
    assert address.startswith("0x") or address.startswith("0X")
    assert len(address) == 42, f"Expected 42-char address, got {len(address)}: {address}"

    # Must NOT equal the private key
    original_pk = "0x" + agent_account.key.hex()
    assert address.lower() != original_pk.lower(), (
        "assert_loadable returned the private key instead of the address!"
    )

    # The expected address for this account
    expected_address = agent_account.address
    assert address.lower() == expected_address.lower()


# ---------------------------------------------------------------------------
# Test 5: create_keystore produces a valid V3 keystore dict
# ---------------------------------------------------------------------------
def test_create_keystore_structure() -> None:
    """create_keystore must produce a dict with the V3 required top-level keys."""
    acct = eth_account.Account.create()
    ks = create_keystore(acct.key.hex(), "pw", kdf="scrypt", iterations=_SCRYPT_TEST_N)
    assert isinstance(ks, dict)
    assert "crypto" in ks or "Crypto" in ks  # V3 spec uses lowercase; eth_account uses lowercase
    assert "version" in ks
    assert ks.get("version") == 3
    assert "address" in ks


# ---------------------------------------------------------------------------
# Test 6: RawEnvKeyProvider reads the agent key from an env var
# ---------------------------------------------------------------------------
def test_raw_env_key_provider_reads_env(monkeypatch) -> None:
    """RawEnvKeyProvider must load the key from the env var and round-trip the address."""
    acct = eth_account.Account.create()
    monkeypatch.setenv("HL_AGENT_KEY", acct.key.hex())
    p = RawEnvKeyProvider("HL_AGENT_KEY", "testnet")
    k = p.load_agent_key()
    assert eth_account.Account.from_key(k).address == acct.address


def test_raw_env_key_provider_returns_0x_hex(monkeypatch) -> None:
    """Return type must match KeystoreKeyProvider: a 0x-prefixed 64-hex string."""
    acct = eth_account.Account.create()
    # Set a BARE-hex value (no 0x prefix) and require normalisation to 0x form.
    monkeypatch.setenv("HL_AGENT_KEY", acct.key.hex())
    k = RawEnvKeyProvider("HL_AGENT_KEY", "testnet").load_agent_key()
    assert isinstance(k, str)
    assert k.startswith("0x")
    assert len(k) == 66  # "0x" + 64 hex chars


def test_raw_env_key_provider_accepts_0x_prefixed(monkeypatch) -> None:
    """A 0x-prefixed env value must also be accepted (normalised, not double-prefixed)."""
    acct = eth_account.Account.create()
    monkeypatch.setenv("HL_AGENT_KEY", "0x" + acct.key.hex())
    k = RawEnvKeyProvider("HL_AGENT_KEY", "testnet").load_agent_key()
    assert k.startswith("0x")
    assert len(k) == 66
    assert eth_account.Account.from_key(k).address == acct.address


def test_raw_env_key_provider_missing_raises(monkeypatch) -> None:
    """An unset env var must raise a clear error."""
    monkeypatch.delenv("HL_AGENT_KEY", raising=False)
    with pytest.raises(KeyError):
        RawEnvKeyProvider("HL_AGENT_KEY", "testnet").load_agent_key()


def test_raw_env_key_provider_empty_raises(monkeypatch) -> None:
    """An empty env var must raise a clear error (not silently load garbage)."""
    monkeypatch.setenv("HL_AGENT_KEY", "")
    with pytest.raises(KeyError):
        RawEnvKeyProvider("HL_AGENT_KEY", "testnet").load_agent_key()
