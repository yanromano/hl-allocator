"""
automation.allocation.remote_signal — RemoteSignalSource: HAARP signed-signal consumer.

Security model
--------------
The HAARP signal service sends a *signed* daily allocation payload.  This module
trusts the service for ALPHA but applies a full defence-in-depth gauntlet before
returning a ``TargetAllocation`` to the executor.  Any failure at any check
raises ``SignalRejected`` and the daemon holds its current position (fail-safe).

Nine mandatory checks (spec §5.2, MF-14, MF-17, S-2):

  1. Fetch the signed key manifest; verify its root signature against the
     pinned ROOT key fingerprint (defeats key-substitution attacks, N-1 rule).
  2. Fetch the signal {payload, signature, key_id} over HTTPS with Bearer auth.
  3. ``key_id`` must be ACTIVE (not revoked/absent) in the manifest.
  4. Ed25519 signature over canonical(payload) must be valid.
  5. ``payload["audience"]`` must equal our ``client_id`` (MF-17).
  6. Freshness: ``now < expires_at`` AND ``|now - issued_at| <= max_skew_seconds`` (S-2).
  7. Anti-replay: ``seq`` must be STRICTLY GREATER than the last persisted seq
     AND ``as_of`` strictly greater than the last persisted as_of.
  8. Build TargetAllocation and run ``validate_target_allocation`` with the
     effective max_per_asset = min(local_max_per_asset, server bounds.max_per_asset).
  9. Persist (seq, as_of) ATOMICALLY on full success; return TargetAllocation.

Default-deny: any exception inside the gauntlet is caught, logged (redacted),
and re-raised as ``SignalRejected``.  There is NO fallback to a cached value.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nacl.encoding
import nacl.exceptions
import nacl.signing

from automation.allocation.base import (
    AllocationRejected,
    TargetAllocation,
    validate_target_allocation,
)
from automation.core.redaction import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class SignalRejected(Exception):
    """Raised by ``RemoteSignalSource.get_target_allocation`` on ANY check failure.

    The daemon MUST catch this, hold the current position, and alert operators.
    It must NEVER be silently swallowed.
    """


# ---------------------------------------------------------------------------
# Canonical serialisation (used both for sign and verify)
# ---------------------------------------------------------------------------


def canonical(payload: dict[str, Any]) -> bytes:
    """Deterministic, compact JSON serialisation used for Ed25519 signing.

    ``sort_keys=True`` and ``separators=(",", ":")`` guarantee a byte-for-byte
    identical encoding regardless of insertion order or Python version.

    Parameters
    ----------
    payload:
        The raw payload dict (as received from the wire — do NOT modify it first).

    Returns
    -------
    bytes
        UTF-8 encoded canonical JSON.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


# ---------------------------------------------------------------------------
# Key manifest parser + verifier
# ---------------------------------------------------------------------------


class KeyManifest:
    """Parsed and root-verified signed key manifest.

    Wire format::

        {
          "keys": [
            {"key_id": "...", "public_key": "<base64>", "status": "active"|"revoked"},
            ...
          ],
          "issued_at": "<ISO8601>",
          "signature": "<base64 Ed25519 over canonical({keys, issued_at})>",
          "key_id": "<root-key-id>"   # identifies which root key signed this
        }

    The root signature covers ``canonical({"issued_at": ..., "keys": [...]})``.
    Verification happens in :meth:`verify_root`.

    Parameters
    ----------
    raw:
        Parsed JSON dict from the manifest endpoint (``/keys/manifest``).
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        # Build a fast lookup: key_id → {public_key_bytes, status}
        #
        # F4b — sticky revocation on duplicate key_id.  A naive ``dict`` would let
        # the LAST duplicate win, so a malicious manifest with
        # ``[{id, revoked}, {id, active}]`` would RESURRECT a revoked key.  We
        # reject the entire manifest if any ``key_id`` appears more than once —
        # a well-formed manifest never has duplicates, so this is purely a guard
        # against the resurrection attack.
        self._keys: dict[str, dict[str, Any]] = {}
        for entry in raw.get("keys", []):
            kid = entry["key_id"]
            if kid in self._keys:
                raise SignalRejected(
                    f"KeyManifest contains duplicate key_id {kid!r} — "
                    "rejected (sticky-revocation guard, F4b)"
                )
            self._keys[kid] = {
                "public_key": base64.b64decode(entry["public_key"]),
                "status": entry["status"],
            }
        self._issued_at: str = raw.get("issued_at", "")
        self._manifest_signature: str = raw.get("signature", "")

    def verify_root(self, root_verify_key: nacl.signing.VerifyKey) -> None:
        """Verify the manifest's own Ed25519 signature using the pinned root key.

        The signed message is ``canonical({"issued_at": ..., "keys": [...]})``.

        Parameters
        ----------
        root_verify_key:
            The pinned ROOT VerifyKey (derived from the out-of-band root fingerprint).

        Raises
        ------
        SignalRejected
            If the signature is absent, malformed, or cryptographically invalid.
        """
        try:
            sig_bytes = base64.b64decode(self._manifest_signature)
        except Exception as exc:
            raise SignalRejected(
                f"KeyManifest root signature is not valid base64: {exc}"
            ) from exc

        # Signed message covers only the content fields (not the signature itself).
        signed_body = canonical(
            {"issued_at": self._issued_at, "keys": self._raw.get("keys", [])}
        )
        try:
            root_verify_key.verify(signed_body, sig_bytes)
        except nacl.exceptions.BadSignatureError as exc:
            raise SignalRejected("KeyManifest root signature verification FAILED") from exc

    def pubkey_for(self, key_id: str) -> nacl.signing.VerifyKey | None:
        """Return the VerifyKey for ``key_id`` if it exists and is ACTIVE.

        Parameters
        ----------
        key_id:
            The key identifier from the signal's ``key_id`` field.

        Returns
        -------
        nacl.signing.VerifyKey | None
            The verify key if the entry exists and has ``status == "active"``,
            else ``None`` (missing OR revoked both return None).
        """
        entry = self._keys.get(key_id)
        if entry is None:
            return None
        if entry["status"] != "active":
            return None  # revoked
        return nacl.signing.VerifyKey(entry["public_key"])


# ---------------------------------------------------------------------------
# Anti-replay persistent state
# ---------------------------------------------------------------------------

_STATE_FILENAME = "remote_signal_state.json"
_STATE_KEY_SEQ = "seq"
_STATE_KEY_AS_OF = "as_of"

# Path suffix appended to the signal-URL base to locate the key manifest.
_MANIFEST_PATH_SUFFIX = "/keys/manifest"


def _load_state(state_path: Path) -> dict[str, Any]:
    """Load (seq, as_of) from the state file; return empty dict if absent."""
    try:
        result: dict[str, Any] = json.loads(state_path.read_text())
        return result
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise SignalRejected(
            f"Cannot read anti-replay state file {state_path}: {exc}"
        ) from exc


def _save_state_atomic(state_path: Path, data: dict[str, Any]) -> None:
    """Persist state atomically using a same-dir temp file + os.replace."""
    parent = state_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".remote_signal_state_")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp_path, state_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# RemoteSignalSource
# ---------------------------------------------------------------------------


class RemoteSignalSource:
    """Fetch, verify, and return the HAARP daily allocation from the signed signal API.

    Implements the ``AllocationSource`` protocol.  This is the most security-critical
    component in the executor — every check is mandatory and the default is DENY.

    Parameters
    ----------
    url:
        Base URL of the signal service (e.g. ``"https://haarp.internal/signal"``).
        The manifest is fetched from ``<url-base>/keys/manifest``.  The signal is
        fetched from ``url`` directly.
    client_id:
        This executor's scoped identifier.  Must match ``payload["audience"]``.
    root_key_fingerprint:
        First 16 hex chars of ``sha256(root_pubkey_bytes).hexdigest()``.  Pinned
        out-of-band; used to validate the key manifest's root signature.
    state_path:
        Path to the persistent anti-replay state file (JSON).  Created if absent.
    bearer_token:
        Optional Bearer token for authenticated signal fetch.
    max_skew_seconds:
        Maximum absolute clock skew from ``issued_at`` (default 900 = 15 min).
    max_validity_seconds:
        Maximum allowed signal validity window ``expires_at - issued_at``
        (default 10800 = 3h, per spec §5.2/S-2).  Caps how long any single
        signal can live so a malicious server cannot mint one that outlives
        key rotation (F6).
    http_client:
        Injectable HTTP client (must have a ``.get(url, headers=...)`` method that
        returns an object with ``.json()`` and ``.raise_for_status()``).
        Defaults to a real ``httpx.Client``.
    now_fn:
        Injectable clock returning a tz-aware UTC datetime.  Defaults to
        ``lambda: datetime.datetime.now(datetime.UTC)``.
    local_max_per_asset:
        Hard local cap on any single-asset weight.  Server bounds can only be MORE
        restrictive; the effective cap is ``min(local_max_per_asset, server_bound)``.
    local_whitelist:
        Set of coin symbols this executor is allowed to hold.
    local_caps:
        Reserved for future use (e.g. turnover caps).

    Raises
    ------
    ValueError
        If ``root_key_fingerprint`` is not a lowercase hex string of length >= 16
        (F1b — a weak/short pin would match many candidate roots).
    """

    #: Number of leading hex chars of ``sha256(root_pubkey)`` used as the pin.
    _FINGERPRINT_LEN = 16

    def __init__(
        self,
        url: str,
        client_id: str,
        root_key_fingerprint: str,
        state_path: str | Path,
        *,
        bearer_token: str | None = None,
        max_skew_seconds: int = 900,
        max_validity_seconds: int = 3 * 3600,
        http_client: Any = None,
        now_fn: Callable[[], datetime.datetime] | None = None,
        local_max_per_asset: float,
        local_whitelist: set[str],
        local_caps: Any = None,  # noqa: ARG002 — reserved
    ) -> None:
        # ---- F1b: pin strength validation -------------------------------
        # A short or non-hex pin is a security hole: an empty/short pin would
        # match a huge space of candidate root keys.  Require lowercase hex,
        # length >= 16 (64-bit minimum collision resistance).
        if (
            not isinstance(root_key_fingerprint, str)
            or len(root_key_fingerprint) < self._FINGERPRINT_LEN
            or any(c not in "0123456789abcdef" for c in root_key_fingerprint)
        ):
            raise ValueError(
                "root_key_fingerprint must be a lowercase-hex string of length >= "
                f"{self._FINGERPRINT_LEN} (got len={len(root_key_fingerprint) if isinstance(root_key_fingerprint, str) else 'non-str'})"
            )

        self._url = url
        self._client_id = client_id
        self._root_key_fingerprint = root_key_fingerprint
        self._state_path = Path(state_path)
        self._bearer_token = bearer_token
        self._max_skew_seconds = max_skew_seconds
        self._max_validity_seconds = max_validity_seconds
        self._local_max_per_asset = local_max_per_asset
        self._local_whitelist = local_whitelist

        if http_client is None:
            import httpx

            self._http_client: Any = httpx.Client()
        else:
            self._http_client = http_client

        self._now_fn: Callable[[], datetime.datetime]
        if now_fn is None:
            self._now_fn = lambda: datetime.datetime.now(datetime.UTC)
        else:
            self._now_fn = now_fn

        # Cached manifest (refreshed each call for simplicity/safety)
        self._manifest: KeyManifest | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_target_allocation(
        self, as_of: datetime.date | None = None
    ) -> TargetAllocation:
        """Fetch, verify, and return the current signed allocation.

        Every check is run in strict order.  ANY failure raises ``SignalRejected``
        with a redacted log message BEFORE the exception propagates.  State is
        persisted ONLY on full success.

        Parameters
        ----------
        as_of:
            Optional reference date.  When provided (F6b), the server payload's
            ``as_of`` MUST equal this date (string compare) or the signal is
            rejected — this prevents acting on a stale/out-of-band date.  When
            ``None``, only the strict-increase anti-replay rule applies.

        Returns
        -------
        TargetAllocation
            A fully validated, audience-checked, freshness-checked, replay-checked
            allocation that has also passed the local safety validator.

        Raises
        ------
        SignalRejected
            On any verification failure.
        """
        try:
            return self._get_verified_allocation(expected_as_of=as_of)
        except SignalRejected:
            raise
        except Exception as exc:
            _logger.warning(
                "RemoteSignalSource: unexpected error during allocation fetch — rejecting",
                error=str(exc),
                url=self._url,
            )
            raise SignalRejected(
                f"Unexpected error fetching allocation: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal gauntlet (9 checks)
    # ------------------------------------------------------------------

    def _get_verified_allocation(
        self, expected_as_of: datetime.date | None = None
    ) -> TargetAllocation:
        """Execute the 9-step verification gauntlet."""

        # ----------------------------------------------------------------
        # CHECK 1 — Fetch and root-verify the key manifest
        # ----------------------------------------------------------------
        manifest = self._fetch_and_verify_manifest()

        # ----------------------------------------------------------------
        # CHECK 2 — Fetch the signal payload
        # ----------------------------------------------------------------
        wire = self._fetch_signal()
        payload: dict[str, Any] = wire["payload"]
        signature_b64: str = wire["signature"]
        key_id: str = wire["key_id"]

        # ----------------------------------------------------------------
        # CHECK 3 — key_id must be active in the manifest
        # ----------------------------------------------------------------
        verify_key = manifest.pubkey_for(key_id)
        if verify_key is None:
            _logger.warning(
                "RemoteSignalSource: key_id is revoked or absent in manifest — rejecting",
                key_id=key_id,
            )
            raise SignalRejected(
                f"key_id {key_id!r} is revoked or not present in the key manifest"
            )

        # ----------------------------------------------------------------
        # CHECK 4 — Ed25519 signature verification
        # ----------------------------------------------------------------
        self._verify_payload_signature(payload, signature_b64, verify_key)

        # ----------------------------------------------------------------
        # CHECK 5 — Audience must match our client_id (MF-17)
        # ----------------------------------------------------------------
        audience = payload.get("audience", "")
        if audience != self._client_id:
            _logger.warning(
                "RemoteSignalSource: audience mismatch — rejecting",
                audience=audience,
                expected=self._client_id,
            )
            raise SignalRejected(
                f"Signal audience {audience!r} != client_id {self._client_id!r} (MF-17)"
            )

        # ----------------------------------------------------------------
        # CHECK 6 — Freshness (S-2)
        # ----------------------------------------------------------------
        self._check_freshness(payload)

        # ----------------------------------------------------------------
        # CHECK 6b — Caller-requested as_of must match payload as_of (F6b)
        # ----------------------------------------------------------------
        if expected_as_of is not None:
            payload_as_of = str(payload.get(_STATE_KEY_AS_OF, ""))
            if payload_as_of != expected_as_of.isoformat():
                _logger.warning(
                    "RemoteSignalSource: payload as_of != requested as_of — rejecting (F6b)",
                    payload_as_of=payload_as_of,
                    requested_as_of=expected_as_of.isoformat(),
                )
                raise SignalRejected(
                    f"Signal as_of {payload_as_of!r} != requested as_of "
                    f"{expected_as_of.isoformat()!r}"
                )

        # ----------------------------------------------------------------
        # CHECK 7 — Anti-replay: seq and as_of must be strictly increasing
        # ----------------------------------------------------------------
        self._check_anti_replay(payload)

        # ----------------------------------------------------------------
        # CHECK 8 — Build TargetAllocation + run local safety validator
        # ----------------------------------------------------------------
        ta = self._build_and_validate_ta(payload)

        # ----------------------------------------------------------------
        # CHECK 9 — Persist state ONLY on full success (atomic)
        # ----------------------------------------------------------------
        new_state = {
            _STATE_KEY_SEQ: payload[_STATE_KEY_SEQ],
            _STATE_KEY_AS_OF: payload[_STATE_KEY_AS_OF],
        }
        try:
            _save_state_atomic(self._state_path, new_state)
        except Exception as exc:
            # Persistence failure is fatal — we cannot allow a replay attack
            # from state corruption.  Fail loudly.
            _logger.warning(
                "RemoteSignalSource: failed to persist anti-replay state — rejecting",
                error=str(exc),
            )
            raise SignalRejected(
                f"Cannot persist anti-replay state: {exc}"
            ) from exc

        _logger.info(
            "RemoteSignalSource: allocation accepted",
            strategy_id=ta.strategy_id,
            model_rev=ta.model_rev,
            as_of=str(ta.as_of),
            seq=payload[_STATE_KEY_SEQ],
            audience=ta.audience,
        )
        return ta

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _fetch_and_verify_manifest(self) -> KeyManifest:
        """Fetch /keys/manifest and verify its root signature (Check 1)."""
        # Derive the manifest URL from the signal URL base
        # Convention: signal URL is the full signal endpoint;
        # manifest lives at <base>/keys/manifest where base is the URL up to the last path segment.
        # If the URL has no query, we can safely append /keys/manifest to the base.
        base_url = self._url.rstrip("/")
        # Navigate up one level to the "base" for the keys endpoint.
        # e.g. https://host/v1/signal → https://host/v1/keys/manifest
        parts = base_url.rsplit("/", 1)
        url_base = parts[0] if len(parts) > 1 else base_url
        manifest_url = url_base + _MANIFEST_PATH_SUFFIX

        try:
            headers = self._make_headers()
            resp = self._http_client.get(manifest_url, headers=headers)
            resp.raise_for_status()
            raw: dict[str, Any] = resp.json()
        except SignalRejected:
            raise
        except Exception as exc:
            _logger.warning(
                "RemoteSignalSource: failed to fetch key manifest — rejecting",
                manifest_url=manifest_url,
                error=str(exc),
            )
            raise SignalRejected(
                f"Failed to fetch key manifest from {manifest_url}: {exc}"
            ) from exc

        manifest = KeyManifest(raw)

        # Resolve the root VerifyKey from the pinned fingerprint.
        root_vk = self._resolve_root_key_from_manifest(raw)
        try:
            manifest.verify_root(root_vk)
        except SignalRejected:
            raise
        except Exception as exc:
            raise SignalRejected(
                f"Key manifest root verification raised unexpected error: {exc}"
            ) from exc

        self._manifest = manifest
        return manifest

    def _resolve_root_key_from_manifest(self, raw: dict[str, Any]) -> nacl.signing.VerifyKey:
        """Find the key whose fingerprint matches ``self._root_key_fingerprint``.

        The manifest MAY include a ``"root_public_key"`` field for the signing key
        (the key that signed the manifest itself).  We verify its fingerprint
        against our pinned value before using it.

        Raises
        ------
        SignalRejected
            If no key in the manifest matches the pinned fingerprint, or if the
            ``root_public_key`` field is missing.
        """
        # The manifest carries its own root public key for bootstrap verification.
        root_pubkey_b64 = raw.get("root_public_key")
        if not root_pubkey_b64:
            _logger.warning(
                "RemoteSignalSource: manifest missing root_public_key — rejecting"
            )
            raise SignalRejected("Key manifest does not contain 'root_public_key' field")

        try:
            root_pubkey_bytes = base64.b64decode(root_pubkey_b64)
        except Exception as exc:
            raise SignalRejected(
                f"root_public_key in manifest is not valid base64: {exc}"
            ) from exc

        # F1b — Compare a FIXED-LENGTH prefix of the COMPUTED digest against a
        # FIXED-LENGTH prefix of the pinned value.  Slicing the computed digest
        # to ``len(pin)`` (the old behaviour) would let a short pin match a huge
        # space of roots.  ``__init__`` already enforces ``len(pin) >= 16``, so
        # we anchor both sides to exactly ``_FINGERPRINT_LEN`` hex chars and use
        # a constant-time comparison.
        actual_fp = hashlib.sha256(root_pubkey_bytes).hexdigest()[: self._FINGERPRINT_LEN]
        expected_fp = self._root_key_fingerprint[: self._FINGERPRINT_LEN]
        if not hmac.compare_digest(actual_fp, expected_fp):
            _logger.warning(
                "RemoteSignalSource: root key fingerprint mismatch — rejecting (N-1 rule)",
                expected_fingerprint=expected_fp,
                # Do NOT log actual fingerprint if it might leak key material
                # (fingerprints are public but log it redacted for consistency)
            )
            raise SignalRejected(
                f"Root key fingerprint mismatch: expected {expected_fp!r}"
            )

        try:
            return nacl.signing.VerifyKey(root_pubkey_bytes)
        except Exception as exc:
            raise SignalRejected(
                f"root_public_key bytes are not a valid Ed25519 key: {exc}"
            ) from exc

    def _fetch_signal(self) -> dict[str, Any]:
        """Fetch the raw signal wire payload (Check 2)."""
        try:
            headers = self._make_headers()
            resp = self._http_client.get(self._url, headers=headers)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except SignalRejected:
            raise
        except Exception as exc:
            _logger.warning(
                "RemoteSignalSource: failed to fetch signal — rejecting",
                url=self._url,
                error=str(exc),
            )
            raise SignalRejected(
                f"Failed to fetch signal from {self._url}: {exc}"
            ) from exc

    def _verify_payload_signature(
        self,
        payload: dict[str, Any],
        signature_b64: str,
        verify_key: nacl.signing.VerifyKey,
    ) -> None:
        """Verify the Ed25519 signature over canonical(payload) (Check 4)."""
        try:
            sig_bytes = base64.b64decode(signature_b64)
        except Exception as exc:
            _logger.warning(
                "RemoteSignalSource: signal signature is not valid base64 — rejecting"
            )
            raise SignalRejected(
                f"Signal signature is not valid base64: {exc}"
            ) from exc

        message = canonical(payload)
        try:
            verify_key.verify(message, sig_bytes)
        except nacl.exceptions.BadSignatureError as exc:
            _logger.warning(
                "RemoteSignalSource: Ed25519 signature verification FAILED — rejecting"
            )
            raise SignalRejected("Signal Ed25519 signature verification failed") from exc

    def _check_freshness(self, payload: dict[str, Any]) -> None:
        """Enforce freshness: expiry, clock-skew, and validity window (Check 6, S-2)."""
        now = self._now_fn()

        issued_at_str: str = payload.get("issued_at", "")
        expires_at_str: str = payload.get("expires_at", "")

        try:
            issued_at = datetime.datetime.fromisoformat(
                issued_at_str.replace("Z", "+00:00")
            )
            expires_at = datetime.datetime.fromisoformat(
                expires_at_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError) as exc:
            _logger.warning(
                "RemoteSignalSource: cannot parse issued_at/expires_at — rejecting",
                issued_at=issued_at_str,
                expires_at=expires_at_str,
            )
            raise SignalRejected(
                f"Cannot parse freshness timestamps: {exc}"
            ) from exc

        # Must not be expired
        if now >= expires_at:
            _logger.warning(
                "RemoteSignalSource: signal is expired — rejecting (S-2)",
                expires_at=expires_at_str,
            )
            raise SignalRejected(
                f"Signal expired at {expires_at_str} (now={now.isoformat()})"
            )

        # F6 — Upper-bound the validity window.  A malicious server could mint a
        # signal valid for years (outliving key rotation).  Reject if
        # ``expires_at - issued_at`` exceeds ``max_validity_seconds``.
        validity = (expires_at - issued_at).total_seconds()
        if validity > self._max_validity_seconds:
            _logger.warning(
                "RemoteSignalSource: validity window too long — rejecting (F6)",
                validity_seconds=round(validity, 1),
                max_validity_seconds=self._max_validity_seconds,
                issued_at=issued_at_str,
                expires_at=expires_at_str,
            )
            raise SignalRejected(
                f"Validity window {validity:.0f}s exceeds "
                f"max_validity_seconds={self._max_validity_seconds}"
            )

        # Clock skew must not exceed max_skew_seconds
        skew = abs((now - issued_at).total_seconds())
        if skew > self._max_skew_seconds:
            _logger.warning(
                "RemoteSignalSource: clock skew exceeds limit — rejecting (S-2)",
                skew_seconds=round(skew, 1),
                max_skew_seconds=self._max_skew_seconds,
                issued_at=issued_at_str,
            )
            raise SignalRejected(
                f"Clock skew {skew:.1f}s exceeds max_skew_seconds={self._max_skew_seconds}"
            )

    def _check_anti_replay(self, payload: dict[str, Any]) -> None:
        """Enforce strictly-increasing seq and as_of (Check 7).

        F7c — ``seq`` MUST be an ``int`` (reject floats like ``100.0001``; note
        ``bool`` is an ``int`` subclass so it is explicitly excluded).
        F7b — when the loaded state is non-empty it MUST carry BOTH ``seq`` and
        ``as_of``; a partial state file (missing either axis) is rejected rather
        than failing open on the missing dimension.
        """
        seq = payload[_STATE_KEY_SEQ]
        if not isinstance(seq, int) or isinstance(seq, bool):
            _logger.warning(
                "RemoteSignalSource: seq is not an int — rejecting (F7c)",
                received_seq=str(seq),
                seq_type=type(seq).__name__,
            )
            raise SignalRejected(f"seq must be an int, got {type(seq).__name__}: {seq!r}")

        as_of_str: str = payload[_STATE_KEY_AS_OF]

        state = _load_state(self._state_path)

        # F7b — a non-empty state file must contain BOTH replay axes.  If either
        # is missing we cannot safely compare on that axis, so reject outright
        # (do NOT fail open on the missing dimension).
        if state and (_STATE_KEY_SEQ not in state or _STATE_KEY_AS_OF not in state):
            _logger.warning(
                "RemoteSignalSource: anti-replay state is partial — rejecting (F7b)",
                has_seq=_STATE_KEY_SEQ in state,
                has_as_of=_STATE_KEY_AS_OF in state,
            )
            raise SignalRejected(
                "Anti-replay state file is partial (missing seq or as_of) — refusing to trade"
            )

        last_seq: int | None = state.get(_STATE_KEY_SEQ)
        last_as_of_str: str | None = state.get(_STATE_KEY_AS_OF)

        if last_seq is not None and seq <= last_seq:
            _logger.warning(
                "RemoteSignalSource: seq not strictly increasing — replay attack? Rejecting",
                received_seq=seq,
                last_accepted_seq=last_seq,
            )
            raise SignalRejected(
                f"Anti-replay: seq {seq} <= last accepted seq {last_seq}"
            )

        if last_as_of_str is not None and as_of_str <= last_as_of_str:
            _logger.warning(
                "RemoteSignalSource: as_of not strictly increasing — rejecting",
                received_as_of=as_of_str,
                last_accepted_as_of=last_as_of_str,
            )
            raise SignalRejected(
                f"Anti-replay: as_of {as_of_str!r} <= last accepted as_of {last_as_of_str!r}"
            )

    def _build_and_validate_ta(self, payload: dict[str, Any]) -> TargetAllocation:
        """Build TargetAllocation and run the local safety validator (Check 8)."""
        bounds: dict[str, Any] = payload.get("bounds", {})
        try:
            server_max_per_asset: float = float(bounds.get("max_per_asset", 1.0))
        except (TypeError, ValueError) as exc:
            _logger.warning(
                "RemoteSignalSource: server bounds.max_per_asset is not numeric — rejecting (F8b)",
                raw_value=str(bounds.get("max_per_asset")),
            )
            raise SignalRejected(
                f"Server bounds.max_per_asset is not numeric: {exc}"
            ) from exc

        # F8b — reject a non-finite (NaN/inf) server bound BEFORE the ``min``
        # combine.  ``min(local, nan)`` is order-dependent and can silently
        # return ``nan`` (which would then disable the per-asset cap).
        if not math.isfinite(server_max_per_asset):
            _logger.warning(
                "RemoteSignalSource: server bounds.max_per_asset is non-finite — rejecting (F8b)",
                server_max_per_asset=str(server_max_per_asset),
            )
            raise SignalRejected(
                f"Server bounds.max_per_asset is non-finite: {server_max_per_asset}"
            )

        # Server bounds can only TIGHTEN local caps, never loosen.
        effective_max_per_asset = min(self._local_max_per_asset, server_max_per_asset)

        try:
            ta = TargetAllocation(
                as_of=datetime.date.fromisoformat(payload[_STATE_KEY_AS_OF]),
                weights=dict(payload["weights"]),
                cash=float(payload["cash"]),
                strategy_id=str(payload["strategy_id"]),
                model_rev=str(payload["model_rev"]),
                schema_version=int(payload.get("schema_version", 1)),
                audience=str(payload["audience"]),
            )
        except Exception as exc:
            _logger.warning(
                "RemoteSignalSource: cannot construct TargetAllocation — rejecting",
                error=str(exc),
            )
            raise SignalRejected(
                f"Cannot build TargetAllocation from payload: {exc}"
            ) from exc

        try:
            validate_target_allocation(
                ta,
                max_per_asset=effective_max_per_asset,
                whitelist=self._local_whitelist,
                client_id=self._client_id,
            )
        except AllocationRejected as exc:
            _logger.warning(
                "RemoteSignalSource: local validation failed — rejecting",
                reason=str(exc),
            )
            raise SignalRejected(
                f"Local validation rejected the allocation: {exc}"
            ) from exc

        return ta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_headers(self) -> dict[str, str]:
        """Build HTTP headers (including optional Bearer token)."""
        headers: dict[str, str] = {}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        return headers
