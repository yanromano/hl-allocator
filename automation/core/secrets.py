"""
automation.core.secrets — Cross-platform agent key loading (V3 scrypt keystore).

Security model
--------------
The bot holds ONE key: the **agent/API wallet** private key.  This is never
the master key.  The key is stored as an encrypted Ethereum V3 keystore
(scrypt KDF); the passphrase is supplied by a pluggable backend.

Passphrase backends
-------------------
- OsKeyringBackend   — OS secret store (system keyring / Keychain / GNOME keyring).
                       NOTE: headless / service-account environments may not be
                       able to unlock the OS keyring.  In those cases choose a
                       different backend (e.g. SystemdCredsBackend on Linux
                       systemd services, or KmsBackend in cloud environments).
- PromptBackend      — interactive getpass prompt (CI / developer workstation).
- EnvBackend         — reads an env variable.  **DEV / testnet ONLY** — refused on
                       mainnet by KeystoreKeyProvider.
- KmsBackend         — stub; raises NotImplementedError.  Real impl: use a cloud
                       KMS (AWS KMS, GCP KMS, Azure Key Vault) to decrypt a
                       ciphertext stored alongside the keystore.
- SystemdCredsBackend — stub; raises NotImplementedError.  Real impl: use
                       `systemd-creds decrypt` (systemd ≥ 250) to read a
                       credential injected at service start.

NEVER log, print, or expose the private key or passphrase.
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import keyring
from eth_account import Account
from eth_account.hdaccount.mnemonic import Mnemonic  # noqa: F401 — keep eth_account import chain

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class PassphraseBackend(Protocol):
    """Return the keystore passphrase as a plain string.

    Implementors MUST NOT log, cache, or otherwise persist the returned value.
    """

    def get(self) -> str:
        """Return the passphrase."""
        ...


@runtime_checkable
class KeyProvider(Protocol):
    """Load and return the agent private key as a 0x-hex string."""

    def load_agent_key(self) -> str:
        """Decrypt and return the 32-byte agent private key (0x-prefixed hex).

        The caller is responsible for zeroing the string from memory when done.
        """
        ...


# ---------------------------------------------------------------------------
# Passphrase backends
# ---------------------------------------------------------------------------


class OsKeyringBackend:
    """Retrieve the passphrase from the OS keyring (Keychain / GNOME / Windows
    Credential Manager).

    NOTE for service accounts / headless environments
    -------------------------------------------------
    The OS keyring is typically unlocked by the logged-in user session.  In
    daemon / systemd service contexts the keyring may not be available.
    Use SystemdCredsBackend or KmsBackend instead.
    """

    def __init__(self, service: str, username: str) -> None:
        self._service = service
        self._username = username

    def get(self) -> str:  # noqa: D102
        passphrase = keyring.get_password(self._service, self._username)
        if passphrase is None:
            raise RuntimeError(
                f"No passphrase found in OS keyring for service={self._service!r}, "
                f"username={self._username!r}.  "
                "Store it first: keyring.set_password(service, username, passphrase)"
            )
        return passphrase


class PromptBackend:
    """Ask the user for the passphrase interactively via getpass (no echo)."""

    def __init__(self, prompt: str = "Keystore passphrase: ") -> None:
        self._prompt = prompt

    def get(self) -> str:  # noqa: D102
        return getpass.getpass(self._prompt)


class EnvBackend:
    """Read the passphrase from an environment variable.

    WARNING — DEV / TESTNET ONLY
    ----------------------------
    EnvBackend is **explicitly forbidden on mainnet** by KeystoreKeyProvider.
    Storing passphrases as plaintext env vars is acceptable in local dev and
    testnet CI, but must never reach a mainnet deployment.
    """

    def __init__(self, var: str) -> None:
        self._var = var

    def get(self) -> str:  # noqa: D102
        value = os.environ.get(self._var)
        if value is None:
            raise KeyError(
                f"Environment variable {self._var!r} is not set.  "
                "Set it before starting the process."
            )
        return value


class KmsBackend:
    """Cloud KMS passphrase backend — stub.

    Real implementation: use AWS KMS / GCP Cloud KMS / Azure Key Vault to
    decrypt a ciphertext stored alongside the keystore.  The ciphertext must be
    committed to the repository or config store; the plaintext never should be.
    """

    def get(self) -> str:  # noqa: D102
        raise NotImplementedError(
            "configure KmsBackend: provide a cloud KMS client that decrypts the "
            "passphrase ciphertext.  See automation/core/secrets.py for details."
        )


class SystemdCredsBackend:
    """Systemd credentials passphrase backend — stub.

    Real implementation: use `systemd-creds decrypt <name> -` (systemd ≥ 250)
    to read a credential injected at service start via
    `LoadCredentialEncrypted=` in the unit file.
    """

    def get(self) -> str:  # noqa: D102
        raise NotImplementedError(
            "configure SystemdCredsBackend: call `systemd-creds decrypt <name> -` "
            "and read stdout.  Requires systemd ≥ 250 and a credential injected at "
            "service start.  See automation/core/secrets.py for details."
        )


# ---------------------------------------------------------------------------
# Keystore helper
# ---------------------------------------------------------------------------

#: Environments where plaintext-ish backends (EnvBackend) are forbidden.
_MAINNET_ENVS: frozenset[str] = frozenset({"mainnet", "production", "prod"})


def create_keystore(
    private_key: str,
    passphrase: str,
    kdf: Literal["scrypt", "pbkdf2"] = "scrypt",
    **encrypt_kwargs: Any,
) -> dict[str, Any]:
    """Encrypt *private_key* with *passphrase* and return a V3 keystore dict.

    Parameters
    ----------
    private_key:
        Raw 32-byte private key as a hex string (with or without 0x prefix).
    passphrase:
        Encryption passphrase.
    kdf:
        Key derivation function — ``"scrypt"`` (default, recommended) or
        ``"pbkdf2"``.
    **encrypt_kwargs:
        Forwarded verbatim to ``eth_account.Account.encrypt``.  Useful in
        tests to pass ``iterations=1024`` for faster scrypt.

    Returns
    -------
    dict
        V3 keystore as a plain Python dict (ready for ``json.dumps``).

    Notes
    -----
    NEVER log or store *private_key* or *passphrase* beyond this call.
    """
    # Normalise key: remove an exact 0x/0X prefix if present; eth_account accepts
    # raw hex.  NOTE: use removeprefix (NOT lstrip("0x")) — lstrip strips *every*
    # leading 0/x character, which corrupts keys that legitimately start with 0
    # (producing an odd-length hex string and a fromhex() crash).
    pk = private_key
    if pk[:2] in ("0x", "0X"):
        pk = pk[2:]
    # eth_account.Account.encrypt expects bytes
    pk_bytes = bytes.fromhex(pk)
    if len(pk_bytes) != 32:
        raise ValueError(
            f"private_key must encode exactly 32 bytes, got {len(pk_bytes)}"
        )
    keystore: dict[str, Any] = Account.encrypt(pk_bytes, passphrase, kdf=kdf, **encrypt_kwargs)
    return keystore


# ---------------------------------------------------------------------------
# KeystoreKeyProvider
# ---------------------------------------------------------------------------


class KeystoreKeyProvider:
    """Load the agent private key from a V3 keystore file.

    Parameters
    ----------
    keystore_path:
        Path to the JSON keystore file on disk.
    passphrase_backend:
        Any object implementing ``PassphraseBackend.get() -> str``.
    env:
        Deployment environment name (e.g. ``"mainnet"``, ``"testnet"``,
        ``"devnet"``, ``"local"``).  Used only for the plaintext-backend
        safety check.

    Security notes
    --------------
    * The private key is **never** logged, printed, or returned by any method
      other than ``load_agent_key``.
    * ``EnvBackend`` (and any env-var-based backend) is explicitly refused
      when ``env`` is one of ``{"mainnet", "production", "prod"}``.
    * The caller should zero the key from memory as soon as it has been
      consumed by the signing library.
    """

    def __init__(
        self,
        keystore_path: str | Path,
        passphrase_backend: PassphraseBackend,
        env: str,
    ) -> None:
        self._keystore_path = Path(keystore_path)
        self._backend = passphrase_backend
        self._env = env.lower()

    def _check_backend_policy(self) -> None:
        """Raise PermissionError if an insecure backend is used on mainnet."""
        if self._env in _MAINNET_ENVS and isinstance(self._backend, EnvBackend):
            raise PermissionError(
                "env/plaintext passphrase backend (EnvBackend) is forbidden on mainnet.  "
                "Use OsKeyringBackend, KmsBackend, or SystemdCredsBackend instead."
            )

    def load_agent_key(self) -> str:
        """Decrypt the keystore and return the agent private key as a 0x-hex str.

        Returns
        -------
        str
            ``"0x"`` + 64 lowercase hex characters (32-byte key).

        Raises
        ------
        PermissionError
            If the passphrase backend is insecure (e.g. EnvBackend on mainnet).
        ValueError
            If the keystore JSON is malformed or the decrypted key is not 32
            bytes.
        Exception
            Any exception raised by eth_account on bad passphrase / corrupted
            keystore.

        Security
        --------
        The returned string is the only place the key leaves this method.
        It is NEVER logged or stored in any instance attribute.
        """
        self._check_backend_policy()

        keystore_json = json.loads(self._keystore_path.read_text())
        passphrase = self._backend.get()

        # eth_account.Account.decrypt returns a HexBytes / bytes-like object
        key_bytes = Account.decrypt(keystore_json, passphrase)

        if len(key_bytes) != 32:
            raise ValueError(
                f"Decrypted key has unexpected length {len(key_bytes)} bytes "
                "(expected 32).  Keystore may be corrupted."
            )

        # Build the canonical 0x-prefixed hex string
        return "0x" + key_bytes.hex()

    def assert_loadable(self) -> str:
        """Health-check: decrypt the keystore and return the Ethereum *address* only.

        The private key is derived and immediately used to compute the address,
        then discarded.  This method is suitable for startup self-tests and
        monitoring checks — it leaks NO secret material.

        Returns
        -------
        str
            EIP-55 checksummed Ethereum address (``"0x"`` + 40 hex chars, 42
            total).

        Raises
        ------
        PermissionError
            If the passphrase backend is insecure on mainnet.
        Exception
            Any error from load_agent_key.
        """
        key_hex = self.load_agent_key()
        # Derive the address from the key — this is the non-secret output
        account = Account.from_key(key_hex)
        address: str = account.address
        # Explicitly delete the key reference before returning
        del key_hex, account
        return address


# ---------------------------------------------------------------------------
# RawEnvKeyProvider
# ---------------------------------------------------------------------------


class RawEnvKeyProvider:
    """Load the agent private key directly from an environment variable.

    Unattended-runs key source
    ---------------------------
    Unlike :class:`KeystoreKeyProvider` (encrypted V3 keystore + passphrase
    backend), this provider reads the *raw* agent private key from an env var.
    It is the key source for **24/7 unattended deployments** where no keystore
    passphrase can be supplied interactively and no OS keyring is available
    (e.g. a systemd service, a container, a cloud VM).

    The trade-off is explicit: the secret lives as a plaintext env var for the
    lifetime of the process.  This is acceptable only for an **agent/API
    wallet** key (never the master key) whose blast radius is bounded by the
    on-chain agent approval (trade-only, time-limited).

    Return-type parity with KeystoreKeyProvider
    -------------------------------------------
    ``load_agent_key`` returns the **same type** as
    :meth:`KeystoreKeyProvider.load_agent_key` — a ``"0x"``-prefixed 64-hex
    string — so it is a drop-in :class:`KeyProvider` for
    ``HLClient.for_trading`` (which expects ``agent_key: str``).

    Parameters
    ----------
    env_var:
        Name of the environment variable holding the raw private key.  The
        value may be 0x-prefixed or bare hex; it is normalised to the
        canonical 0x-prefixed lowercase form.
    env:
        Deployment environment name (``"mainnet"``, ``"testnet"``, …).  Kept
        for interface symmetry with :class:`KeystoreKeyProvider` (e.g. future
        policy checks); not used to gate loading today.

    Security
    --------
    The key is NEVER logged, printed, or stored in any instance attribute —
    only the *name* of the env var is held.  Global redaction is installed at
    the process level, but this provider additionally avoids ever emitting the
    secret.
    """

    def __init__(self, env_var: str, env: str) -> None:
        self._env_var = env_var
        self._env = env.lower()

    def load_agent_key(self) -> str:
        """Read, validate, and return the agent private key as a 0x-hex str.

        Returns
        -------
        str
            ``"0x"`` + 64 lowercase hex characters (32-byte key) — identical in
            shape to :meth:`KeystoreKeyProvider.load_agent_key`.

        Raises
        ------
        KeyError
            If the environment variable is unset or empty.
        ValueError
            If the value does not decode to exactly 32 bytes.

        Security
        --------
        The returned string is the only place the key leaves this method.  It
        is NEVER logged or stored in any instance attribute.
        """
        value = os.environ.get(self._env_var)
        if not value:
            raise KeyError(
                f"Environment variable {self._env_var!r} is unset or empty.  "
                "Set it to the raw agent private key (0x-prefixed or bare hex) "
                "before starting the process."
            )

        # Normalise: strip an exact 0x/0X prefix (NOT lstrip("0x") — that strips
        # every leading 0/x and corrupts keys that legitimately start with 0).
        pk = value.strip()
        if pk[:2] in ("0x", "0X"):
            pk = pk[2:]

        try:
            key_bytes = bytes.fromhex(pk)
        except ValueError as exc:
            # Do NOT echo the value — only report the failure shape.
            raise ValueError(
                f"Environment variable {self._env_var!r} does not contain valid "
                "hex for a private key."
            ) from exc

        if len(key_bytes) != 32:
            raise ValueError(
                f"Agent key from {self._env_var!r} must encode exactly 32 bytes, "
                f"got {len(key_bytes)}."
            )

        # Canonical 0x-prefixed lowercase hex — same shape KeystoreKeyProvider returns.
        return "0x" + key_bytes.hex()
