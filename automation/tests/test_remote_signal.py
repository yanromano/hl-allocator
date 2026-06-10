"""
Tests for automation.allocation.remote_signal.RemoteSignalSource.

Security gauntlet: 11 rejection cases + 1 happy path, using:
- PyNaCl test keys (ROOT signing key + per-signal signing key)
- A fake HTTP client that serves canned manifest + signal responses
- An injectable ``now_fn`` for deterministic freshness control

All 9 mandatory checks are exercised:
  1. Root fingerprint mismatch            → (a) test_rejects_root_fingerprint_mismatch
  2. Key not in manifest                  → (b) test_rejects_key_not_in_manifest
  3. Key revoked                          → (c) test_rejects_revoked_key
  4. Tampered payload (bad signature)     → (d) test_rejects_tampered_payload
  5. Wrong audience                       → (e) test_rejects_wrong_audience
  6. Expired signal                       → (f) test_rejects_expired_signal
  7. Clock skew                           → (g) test_rejects_clock_skew
  8. Replay (seq <= last)                 → (h) test_rejects_replay_seq
  9. Older as_of                          → (i) test_rejects_older_as_of
  10. Server bounds breach                → (j) test_rejects_server_bounds_breach
  11. Local max_per_asset breach          → (k) test_rejects_local_max_per_asset_breach

  Happy path                              → test_happy_path_returns_target_allocation
"""

from __future__ import annotations

import base64
import datetime
import hashlib as _hashlib
import json
from pathlib import Path
from typing import Any

import nacl.encoding
import nacl.signing
import pytest

from automation.allocation.remote_signal import (
    KeyManifest,
    RemoteSignalSource,
    SignalRejected,
    canonical,
)

# ---------------------------------------------------------------------------
# Test key generation (done ONCE per module load — deterministic)
# ---------------------------------------------------------------------------

# Root key: signs the manifest itself (offline, never rotated in tests)
_ROOT_SIGNING_KEY = nacl.signing.SigningKey.generate()
_ROOT_VERIFY_KEY = _ROOT_SIGNING_KEY.verify_key
_ROOT_PUBKEY_BYTES = bytes(_ROOT_VERIFY_KEY)
_ROOT_PUBKEY_B64 = base64.b64encode(_ROOT_PUBKEY_BYTES).decode()

# Signal key: signs individual signal payloads (rotatable)
_SIGNAL_SIGNING_KEY = nacl.signing.SigningKey.generate()
_SIGNAL_VERIFY_KEY = _SIGNAL_SIGNING_KEY.verify_key
_SIGNAL_PUBKEY_BYTES = bytes(_SIGNAL_VERIFY_KEY)
_SIGNAL_PUBKEY_B64 = base64.b64encode(_SIGNAL_PUBKEY_BYTES).decode()

_SIGNAL_KEY_ID = "signal-key-v1"

_ROOT_FP = _hashlib.sha256(_ROOT_PUBKEY_BYTES).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Fixed timestamps for deterministic freshness
# ---------------------------------------------------------------------------

_NOW = datetime.datetime(2026, 6, 1, 12, 0, 0, tzinfo=datetime.UTC)
_ISSUED_AT = _NOW - datetime.timedelta(minutes=5)
# Default validity window must stay within the 3h cap (F6): issued -5min,
# expires +2h → window = 2h5min < 3h.
_EXPIRES_AT = _NOW + datetime.timedelta(hours=2)

_AS_OF = "2026-06-01"
_SEQ = 100

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _make_manifest_body(keys: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the signable manifest body (without the root signature)."""
    if keys is None:
        keys = [
            {
                "key_id": _SIGNAL_KEY_ID,
                "public_key": _SIGNAL_PUBKEY_B64,
                "status": "active",
            }
        ]
    return {
        "issued_at": "2026-06-01T00:00:00Z",
        "keys": keys,
    }


def _sign_manifest(body: dict[str, Any], root_key: nacl.signing.SigningKey | None = None) -> dict[str, Any]:
    """Add root signature to a manifest body."""
    if root_key is None:
        root_key = _ROOT_SIGNING_KEY
    sig = root_key.sign(canonical(body)).signature
    return {
        **body,
        "signature": base64.b64encode(sig).decode(),
        "root_public_key": base64.b64encode(bytes(root_key.verify_key)).decode(),
    }


def _default_manifest() -> dict[str, Any]:
    """A valid signed manifest with one active signal key."""
    return _sign_manifest(_make_manifest_body())


def _make_payload(
    audience: str = "test-client",
    as_of: str = _AS_OF,
    seq: int = _SEQ,
    weights: dict[str, float] | None = None,
    cash: float | None = None,
    strategy_id: str = "haarp-v3",
    model_rev: str = "2026-06-01",
    schema_version: int = 1,
    issued_at: datetime.datetime | None = None,
    expires_at: datetime.datetime | None = None,
    bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid payload dict."""
    if weights is None:
        weights = {"BTC": 0.4, "ETH": 0.3, "SOL": 0.2}
    if cash is None:
        cash = 1.0 - sum(weights.values())
    if issued_at is None:
        issued_at = _ISSUED_AT
    if expires_at is None:
        expires_at = _EXPIRES_AT
    if bounds is None:
        bounds = {"max_per_asset": 0.6, "max_turnover": 0.5}
    return {
        "strategy_id": strategy_id,
        "model_rev": model_rev,
        "schema_version": schema_version,
        "audience": audience,
        "as_of": as_of,
        "weights": weights,
        "cash": cash,
        "bounds": bounds,
        "seq": seq,
        "issued_at": issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _sign_payload(
    payload: dict[str, Any],
    signing_key: nacl.signing.SigningKey | None = None,
) -> str:
    """Return base64-encoded Ed25519 signature over canonical(payload)."""
    if signing_key is None:
        signing_key = _SIGNAL_SIGNING_KEY
    sig = signing_key.sign(canonical(payload)).signature
    return base64.b64encode(sig).decode()


def _make_wire(
    payload: dict[str, Any],
    key_id: str = _SIGNAL_KEY_ID,
    signing_key: nacl.signing.SigningKey | None = None,
) -> dict[str, Any]:
    """Build a complete wire response {payload, signature, key_id}."""
    return {
        "payload": payload,
        "signature": _sign_payload(payload, signing_key),
        "key_id": key_id,
    }


# ---------------------------------------------------------------------------
# Fake HTTP client
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal fake httpx.Response."""

    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self._data = data
        self._status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        if self._status_code >= 400:
            raise RuntimeError(f"HTTP {self._status_code}")


class _FakeHttpClient:
    """Injectable HTTP client whose responses are set per URL."""

    def __init__(
        self,
        manifest_data: dict[str, Any],
        signal_data: dict[str, Any],
    ) -> None:
        self._manifest_data = manifest_data
        self._signal_data = signal_data
        self.calls: list[str] = []

    def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        self.calls.append(url)
        if "keys/manifest" in url:
            return _FakeResponse(self._manifest_data)
        return _FakeResponse(self._signal_data)


# ---------------------------------------------------------------------------
# Fixture: temporary state directory
# ---------------------------------------------------------------------------


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    """Return a path to a (non-existent) state file inside a temp dir."""
    return tmp_path / "state.json"


# ---------------------------------------------------------------------------
# Builder: default RemoteSignalSource with injectable overrides
# ---------------------------------------------------------------------------


def _make_source(
    state_file: Path,
    manifest_data: dict[str, Any] | None = None,
    signal_data: dict[str, Any] | None = None,
    root_key_fingerprint: str = _ROOT_FP,
    client_id: str = "test-client",
    local_max_per_asset: float = 0.5,
    local_whitelist: set[str] | None = None,
    now_fn: Any = None,
    payload: dict[str, Any] | None = None,
    key_id: str = _SIGNAL_KEY_ID,
    signing_key: nacl.signing.SigningKey | None = None,
) -> RemoteSignalSource:
    """Build a RemoteSignalSource with all defaults injectable for testing."""
    if local_whitelist is None:
        local_whitelist = {"BTC", "ETH", "SOL"}
    if payload is None:
        payload = _make_payload(audience=client_id)
    if manifest_data is None:
        manifest_data = _default_manifest()
    if signal_data is None:
        signal_data = _make_wire(payload, key_id=key_id, signing_key=signing_key)
    if now_fn is None:
        now_fn = lambda: _NOW  # noqa: E731

    http_client = _FakeHttpClient(manifest_data, signal_data)
    return RemoteSignalSource(
        url="https://haarp.internal/v1/signal",
        client_id=client_id,
        root_key_fingerprint=root_key_fingerprint,
        state_path=state_file,
        bearer_token="test-bearer-token",
        max_skew_seconds=900,
        http_client=http_client,
        now_fn=now_fn,
        local_max_per_asset=local_max_per_asset,
        local_whitelist=local_whitelist,
    )


# ===========================================================================
# Happy path
# ===========================================================================


def test_happy_path_returns_target_allocation(state_file: Path) -> None:
    """Valid manifest + signal → returns TargetAllocation; persists seq+as_of."""
    source = _make_source(state_file)
    ta = source.get_target_allocation()

    assert ta.audience == "test-client"
    assert ta.strategy_id == "haarp-v3"
    assert ta.as_of == datetime.date(2026, 6, 1)
    assert abs(sum(ta.weights.values()) + ta.cash - 1.0) < 1e-9

    # State must be persisted
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["seq"] == _SEQ
    assert state["as_of"] == _AS_OF


def test_happy_path_bearer_token_sent(state_file: Path) -> None:
    """Bearer token is passed as Authorization header on every request."""
    payload = _make_payload(audience="test-client")
    http_client = _FakeHttpClient(_default_manifest(), _make_wire(payload))
    source = RemoteSignalSource(
        url="https://haarp.internal/v1/signal",
        client_id="test-client",
        root_key_fingerprint=_ROOT_FP,
        state_path=state_file,
        bearer_token="super-secret-bearer",
        http_client=http_client,
        now_fn=lambda: _NOW,
        local_max_per_asset=0.5,
        local_whitelist={"BTC", "ETH", "SOL"},
    )
    source.get_target_allocation()
    # Two GET calls: manifest + signal
    assert len(http_client.calls) == 2


# ===========================================================================
# (a) Root fingerprint mismatch — Check 1
# ===========================================================================


def test_rejects_root_fingerprint_mismatch(state_file: Path) -> None:
    """Wrong root fingerprint → SignalRejected; no state persisted."""
    wrong_fp = "deadbeefdeadbeef"  # different from _ROOT_FP
    source = _make_source(state_file, root_key_fingerprint=wrong_fp)
    with pytest.raises(SignalRejected, match="fingerprint"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_manifest_signed_by_different_root(state_file: Path) -> None:
    """Manifest signed by a key whose fingerprint doesn't match pinned fp → rejected."""
    # Generate a *different* root key and sign the manifest with it
    evil_root = nacl.signing.SigningKey.generate()
    manifest_body = _make_manifest_body()
    evil_manifest = _sign_manifest(manifest_body, root_key=evil_root)

    source = _make_source(state_file, manifest_data=evil_manifest)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (b) Signal signed by a key NOT in manifest — Check 3
# ===========================================================================


def test_rejects_key_not_in_manifest(state_file: Path) -> None:
    """key_id not present in manifest → SignalRejected; no state persisted."""
    payload = _make_payload(audience="test-client")
    wire = _make_wire(payload, key_id="unknown-key-xyz")
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="revoked or not present"):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (c) Key revoked — Check 3
# ===========================================================================


def test_rejects_revoked_key(state_file: Path) -> None:
    """A revoked key_id → SignalRejected; no state persisted."""
    revoked_manifest_body = _make_manifest_body(
        keys=[
            {
                "key_id": _SIGNAL_KEY_ID,
                "public_key": _SIGNAL_PUBKEY_B64,
                "status": "revoked",
            }
        ]
    )
    revoked_manifest = _sign_manifest(revoked_manifest_body)
    source = _make_source(state_file, manifest_data=revoked_manifest)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (d) Tampered payload (bad signature) — Check 4
# ===========================================================================


def test_rejects_tampered_payload(state_file: Path) -> None:
    """Payload modified after signing → signature verification fails → rejected."""
    payload = _make_payload(audience="test-client")
    wire = _make_wire(payload)
    # Tamper with the payload AFTER signing (change a weight)
    wire["payload"]["weights"]["BTC"] = 0.99
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="signature"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_wrong_signing_key(state_file: Path) -> None:
    """Payload signed by a different key (not the one in manifest) → rejected."""
    other_key = nacl.signing.SigningKey.generate()
    payload = _make_payload(audience="test-client")
    wire = _make_wire(payload, signing_key=other_key)  # signed by non-manifest key
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (e) Wrong audience — Check 5
# ===========================================================================


def test_rejects_wrong_audience(state_file: Path) -> None:
    """Payload audience != client_id → rejected; no state persisted."""
    payload = _make_payload(audience="DIFFERENT-CLIENT")
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire, client_id="test-client")
    with pytest.raises(SignalRejected, match="audience"):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (f) Expired signal — Check 6
# ===========================================================================


def test_rejects_expired_signal(state_file: Path) -> None:
    """Signal where now >= expires_at → rejected."""
    expired_payload = _make_payload(
        audience="test-client",
        expires_at=_NOW - datetime.timedelta(hours=1),  # already expired
    )
    wire = _make_wire(expired_payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="[Ee]xpir"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_signal_exactly_at_expiry(state_file: Path) -> None:
    """Signal where now == expires_at → rejected (now >= expires_at condition)."""
    expired_payload = _make_payload(
        audience="test-client",
        expires_at=_NOW,  # expires_at == now → expired
    )
    wire = _make_wire(expired_payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (g) Clock skew — Check 6
# ===========================================================================


def test_rejects_clock_skew_too_old(state_file: Path) -> None:
    """issued_at is 2h in the past; max_skew_seconds=900 (15min) → rejected.

    Window kept within the 3h validity cap (issued -2h, expires +30min →
    2h30min) so the SKEW check is what fires, not the F6 validity gate.
    """
    stale_payload = _make_payload(
        audience="test-client",
        issued_at=_NOW - datetime.timedelta(hours=2),
        expires_at=_NOW + datetime.timedelta(minutes=30),  # not expired, window < 3h
    )
    wire = _make_wire(stale_payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="[Ss]kew"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_clock_skew_from_future(state_file: Path) -> None:
    """issued_at is 20 minutes in the FUTURE; max_skew=900s → rejected.

    Window kept within the 3h validity cap so the SKEW check fires (not F6).
    """
    future_payload = _make_payload(
        audience="test-client",
        issued_at=_NOW + datetime.timedelta(minutes=20),
        expires_at=_NOW + datetime.timedelta(hours=2),  # window 1h40min < 3h
    )
    wire = _make_wire(future_payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="[Ss]kew"):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (h) Anti-replay: seq <= last persisted — Check 7
# ===========================================================================


def test_rejects_replay_seq(state_file: Path) -> None:
    """seq == last_seq → rejected as replay; state unchanged."""
    # Pre-populate state with the SAME seq we're about to receive
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"seq": _SEQ, "as_of": "2026-05-31"}))

    source = _make_source(state_file)
    with pytest.raises(SignalRejected, match="[Aa]nti.replay|seq"):
        source.get_target_allocation()

    # State must NOT have been updated
    state = json.loads(state_file.read_text())
    assert state["seq"] == _SEQ
    assert state["as_of"] == "2026-05-31"  # unchanged


def test_rejects_replay_seq_strictly_less(state_file: Path) -> None:
    """seq < last_seq → rejected."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"seq": _SEQ + 10, "as_of": "2026-05-31"}))

    source = _make_source(state_file)
    with pytest.raises(SignalRejected, match="[Aa]nti.replay|seq"):
        source.get_target_allocation()
    assert not state_file.exists() or json.loads(state_file.read_text())["seq"] == _SEQ + 10


# ===========================================================================
# (i) Anti-replay: as_of not strictly increasing — Check 7
# ===========================================================================


def test_rejects_older_as_of(state_file: Path) -> None:
    """as_of <= last_as_of (even with higher seq) → rejected; state unchanged."""
    # Last as_of is TODAY; signal also has TODAY → not strictly greater
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"seq": _SEQ - 1, "as_of": _AS_OF}))

    source = _make_source(state_file)
    with pytest.raises(SignalRejected, match="[Aa]nti.replay|as_of"):
        source.get_target_allocation()

    # State must not have changed
    state = json.loads(state_file.read_text())
    assert state["as_of"] == _AS_OF


def test_rejects_as_of_strictly_earlier(state_file: Path) -> None:
    """as_of earlier than last → rejected."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # Last accepted was 2026-06-02; incoming is 2026-06-01 (earlier)
    state_file.write_text(json.dumps({"seq": _SEQ - 1, "as_of": "2026-06-02"}))

    source = _make_source(state_file)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()


# ===========================================================================
# (j) Server bounds breach — Check 8
# ===========================================================================


def test_rejects_server_bounds_breach(state_file: Path) -> None:
    """Weight exceeds server bounds.max_per_asset → rejected."""
    # server says max_per_asset=0.3 but BTC weight is 0.4
    payload = _make_payload(
        audience="test-client",
        weights={"BTC": 0.4, "ETH": 0.3, "SOL": 0.2},
        cash=0.1,
        bounds={"max_per_asset": 0.3, "max_turnover": 0.5},
    )
    wire = _make_wire(payload)
    source = _make_source(
        state_file,
        signal_data=wire,
        local_max_per_asset=0.9,  # local is generous; server bound is what triggers
    )
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


def test_server_bounds_tighten_local_cap(state_file: Path) -> None:
    """Server bound (0.3) < local cap (0.5): effective cap is 0.3 → breach rejected."""
    payload = _make_payload(
        audience="test-client",
        weights={"BTC": 0.35, "ETH": 0.3, "SOL": 0.2},  # BTC 0.35 > server 0.3
        cash=0.15,
        bounds={"max_per_asset": 0.3, "max_turnover": 0.5},
    )
    wire = _make_wire(payload)
    source = _make_source(
        state_file,
        signal_data=wire,
        local_max_per_asset=0.5,  # local allows 0.5, server only 0.3
    )
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# (k) Local max_per_asset breach — Check 8
# ===========================================================================


def test_rejects_local_max_per_asset_breach(state_file: Path) -> None:
    """Weight within server bounds but exceeds local cap → rejected."""
    # server says max_per_asset=0.9 (generous), local says 0.3
    # BTC weight = 0.4 → fine for server, breaches local
    payload = _make_payload(
        audience="test-client",
        weights={"BTC": 0.4, "ETH": 0.3, "SOL": 0.2},
        cash=0.1,
        bounds={"max_per_asset": 0.9, "max_turnover": 0.5},
    )
    wire = _make_wire(payload)
    source = _make_source(
        state_file,
        signal_data=wire,
        local_max_per_asset=0.3,  # local cap: 0.3, BTC=0.4 → breach
    )
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# State persistence — happy path and rejection invariant
# ===========================================================================


def test_state_not_persisted_on_rejection(state_file: Path) -> None:
    """A rejected signal (bad signature) must NOT update the state file."""
    payload = _make_payload(audience="test-client")
    wire = _make_wire(payload)
    wire["payload"]["weights"]["BTC"] = 0.999  # tamper AFTER signing

    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


def test_state_persisted_only_after_full_success(state_file: Path) -> None:
    """Two sequential calls with increasing seq/as_of both succeed; state grows."""
    # First call
    payload1 = _make_payload(audience="test-client", seq=1, as_of="2026-06-01")
    source = _make_source(state_file, payload=payload1)
    source.get_target_allocation()
    state = json.loads(state_file.read_text())
    assert state["seq"] == 1
    assert state["as_of"] == "2026-06-01"

    # Second call — new source with higher seq + as_of
    payload2 = _make_payload(audience="test-client", seq=2, as_of="2026-06-02")
    source2 = _make_source(state_file, payload=payload2)
    source2.get_target_allocation()
    state2 = json.loads(state_file.read_text())
    assert state2["seq"] == 2
    assert state2["as_of"] == "2026-06-02"


# ===========================================================================
# canonical() unit tests
# ===========================================================================


def test_canonical_is_stable() -> None:
    """canonical() is deterministic and key-order-independent."""
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert canonical(d1) == canonical(d2)


def test_canonical_compact_separators() -> None:
    """canonical() uses compact separators (no spaces)."""
    result = canonical({"k": "v"}).decode()
    assert result == '{"k":"v"}'


# ===========================================================================
# KeyManifest unit tests
# ===========================================================================


def test_key_manifest_pubkey_for_active() -> None:
    """pubkey_for returns a VerifyKey for an active key."""
    manifest_body = _make_manifest_body()
    manifest_raw = _sign_manifest(manifest_body)
    km = KeyManifest(manifest_raw)
    vk = km.pubkey_for(_SIGNAL_KEY_ID)
    assert vk is not None
    assert isinstance(vk, nacl.signing.VerifyKey)


def test_key_manifest_pubkey_for_revoked_returns_none() -> None:
    """pubkey_for returns None for a revoked key."""
    manifest_body = _make_manifest_body(
        keys=[{"key_id": "k1", "public_key": _SIGNAL_PUBKEY_B64, "status": "revoked"}]
    )
    manifest_raw = _sign_manifest(manifest_body)
    km = KeyManifest(manifest_raw)
    assert km.pubkey_for("k1") is None


def test_key_manifest_pubkey_for_missing_returns_none() -> None:
    """pubkey_for returns None for an unknown key_id."""
    manifest_raw = _sign_manifest(_make_manifest_body())
    km = KeyManifest(manifest_raw)
    assert km.pubkey_for("does-not-exist") is None


def test_key_manifest_verify_root_success() -> None:
    """verify_root does not raise for a correctly signed manifest."""
    manifest_raw = _sign_manifest(_make_manifest_body())
    km = KeyManifest(manifest_raw)
    km.verify_root(_ROOT_VERIFY_KEY)  # must not raise


def test_key_manifest_verify_root_bad_signature_raises() -> None:
    """verify_root raises SignalRejected for a bad root signature."""
    manifest_body = _make_manifest_body()
    manifest_raw = _sign_manifest(manifest_body)
    # Corrupt the signature
    manifest_raw["signature"] = base64.b64encode(b"\x00" * 64).decode()
    km = KeyManifest(manifest_raw)
    with pytest.raises(SignalRejected, match="[Ss]ignature"):
        km.verify_root(_ROOT_VERIFY_KEY)


# ===========================================================================
# No-state first-run (no anti-replay file) — happy path
# ===========================================================================


def test_first_run_no_state_file(state_file: Path) -> None:
    """When no state file exists, first-run signal with any seq is accepted."""
    assert not state_file.exists()
    source = _make_source(state_file)
    ta = source.get_target_allocation()
    assert ta is not None
    assert state_file.exists()


# ===========================================================================
# F12b (a) — cash=NaN must be rejected (F11 in base.py, surfaced via remote)
# ===========================================================================


def test_rejects_nan_cash(state_file: Path) -> None:
    """A fully-signed payload with cash=NaN → SignalRejected; no state persisted.

    The unity-sum gate ``abs(weight_sum + cash - 1.0) > tol`` would evaluate to
    ``nan > tol == False`` and silently PASS — F11 closes this by adding a
    finiteness guard on ``cash`` in validate_target_allocation.
    """
    # Empty weights + NaN cash: would slip past unity-sum without the F11 guard.
    payload = _make_payload(
        audience="test-client",
        weights={},
        cash=float("nan"),
    )
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_nan_cash_with_weights(state_file: Path) -> None:
    """cash=NaN with non-empty weights also rejected; no state persisted."""
    payload = _make_payload(
        audience="test-client",
        weights={"BTC": 0.4, "ETH": 0.3},
        cash=float("nan"),
    )
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# F12b (b) — validity window too long (F6)
# ===========================================================================


def test_rejects_validity_window_too_long(state_file: Path) -> None:
    """expires_at ~10 years out with recent issued_at → SignalRejected (F6)."""
    payload = _make_payload(
        audience="test-client",
        issued_at=_NOW - datetime.timedelta(minutes=1),  # fresh issue
        expires_at=_NOW + datetime.timedelta(days=3650),  # ~10 years
    )
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="[Vv]alidity"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_validity_window_exactly_at_cap_passes(state_file: Path) -> None:
    """A window exactly at max_validity_seconds is allowed (boundary)."""
    # Default max_validity = 3h.  issued -5min, expires +2h55min → exactly 3h.
    payload = _make_payload(
        audience="test-client",
        issued_at=_NOW - datetime.timedelta(minutes=5),
        expires_at=_NOW + datetime.timedelta(hours=2, minutes=55),
    )
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    ta = source.get_target_allocation()  # must not raise
    assert ta is not None


def test_custom_max_validity_seconds_enforced(state_file: Path) -> None:
    """A tighter max_validity_seconds rejects an otherwise-valid window."""
    payload = _make_payload(
        audience="test-client",
        issued_at=_NOW - datetime.timedelta(minutes=1),
        expires_at=_NOW + datetime.timedelta(hours=2),  # 2h1min window
    )
    wire = _make_wire(payload)
    http_client = _FakeHttpClient(_default_manifest(), wire)
    source = RemoteSignalSource(
        url="https://haarp.internal/v1/signal",
        client_id="test-client",
        root_key_fingerprint=_ROOT_FP,
        state_path=state_file,
        max_validity_seconds=600,  # 10 minutes — much tighter than the 2h window
        http_client=http_client,
        now_fn=lambda: _NOW,
        local_max_per_asset=0.5,
        local_whitelist={"BTC", "ETH", "SOL"},
    )
    with pytest.raises(SignalRejected, match="[Vv]alidity"):
        source.get_target_allocation()
    assert not state_file.exists()


# ===========================================================================
# F12b (c) — duplicate key_id [revoked, active] resurrection (F4b)
# ===========================================================================


def test_rejects_duplicate_key_id_revoked_then_active(state_file: Path) -> None:
    """Manifest with [revoked, active] entries for the same key_id → rejected.

    A naive last-wins dict would resurrect the revoked key; F4b rejects the
    whole manifest on any duplicate key_id.
    """
    dup_body = _make_manifest_body(
        keys=[
            {"key_id": _SIGNAL_KEY_ID, "public_key": _SIGNAL_PUBKEY_B64, "status": "revoked"},
            {"key_id": _SIGNAL_KEY_ID, "public_key": _SIGNAL_PUBKEY_B64, "status": "active"},
        ]
    )
    dup_manifest = _sign_manifest(dup_body)
    source = _make_source(state_file, manifest_data=dup_manifest)
    with pytest.raises(SignalRejected, match="duplicate"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_key_manifest_duplicate_key_id_raises_on_parse() -> None:
    """KeyManifest constructor itself rejects a duplicate key_id (F4b)."""
    dup_body = _make_manifest_body(
        keys=[
            {"key_id": "k1", "public_key": _SIGNAL_PUBKEY_B64, "status": "active"},
            {"key_id": "k1", "public_key": _SIGNAL_PUBKEY_B64, "status": "active"},
        ]
    )
    dup_raw = _sign_manifest(dup_body)
    with pytest.raises(SignalRejected, match="duplicate"):
        KeyManifest(dup_raw)


# ===========================================================================
# F12b (d) — partial state file {"seq": 50} (no as_of) (F7b)
# ===========================================================================


def test_rejects_partial_state_file_missing_as_of(state_file: Path) -> None:
    """State file with seq but no as_of → SignalRejected; state unchanged."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"seq": 50}))  # missing as_of

    source = _make_source(state_file)
    with pytest.raises(SignalRejected, match="partial"):
        source.get_target_allocation()

    # State must remain the partial file (not overwritten)
    state = json.loads(state_file.read_text())
    assert state == {"seq": 50}


def test_rejects_partial_state_file_missing_seq(state_file: Path) -> None:
    """State file with as_of but no seq → SignalRejected; state unchanged."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"as_of": "2026-05-31"}))  # missing seq

    source = _make_source(state_file)
    with pytest.raises(SignalRejected, match="partial"):
        source.get_target_allocation()
    state = json.loads(state_file.read_text())
    assert state == {"as_of": "2026-05-31"}


# ===========================================================================
# F1b — pin strength validation
# ===========================================================================


def test_init_rejects_short_fingerprint(state_file: Path) -> None:
    """A fingerprint shorter than 16 hex chars raises ValueError at construction."""
    with pytest.raises(ValueError, match="fingerprint"):
        RemoteSignalSource(
            url="https://x/v1/signal",
            client_id="c",
            root_key_fingerprint="deadbeef",  # only 8 chars
            state_path=state_file,
            local_max_per_asset=0.5,
            local_whitelist={"BTC"},
        )


def test_init_rejects_empty_fingerprint(state_file: Path) -> None:
    """An empty fingerprint raises ValueError (would otherwise match many roots)."""
    with pytest.raises(ValueError, match="fingerprint"):
        RemoteSignalSource(
            url="https://x/v1/signal",
            client_id="c",
            root_key_fingerprint="",
            state_path=state_file,
            local_max_per_asset=0.5,
            local_whitelist={"BTC"},
        )


def test_init_rejects_non_hex_fingerprint(state_file: Path) -> None:
    """A non-hex (uppercase / symbols) fingerprint raises ValueError."""
    with pytest.raises(ValueError, match="fingerprint"):
        RemoteSignalSource(
            url="https://x/v1/signal",
            client_id="c",
            root_key_fingerprint="DEADBEEFDEADBEEF",  # uppercase — not lowercase hex
            state_path=state_file,
            local_max_per_asset=0.5,
            local_whitelist={"BTC"},
        )


def test_init_accepts_valid_fingerprint(state_file: Path) -> None:
    """A valid 16-char lowercase-hex fingerprint constructs successfully."""
    src = RemoteSignalSource(
        url="https://x/v1/signal",
        client_id="c",
        root_key_fingerprint=_ROOT_FP,
        state_path=state_file,
        local_max_per_asset=0.5,
        local_whitelist={"BTC"},
    )
    assert src is not None


# ===========================================================================
# F6b — caller-provided as_of must match payload as_of
# ===========================================================================


def test_as_of_arg_must_match_payload(state_file: Path) -> None:
    """get_target_allocation(as_of=X) where payload as_of != X → rejected."""
    source = _make_source(state_file)  # payload as_of is 2026-06-01
    requested = datetime.date(2026, 6, 2)  # different
    with pytest.raises(SignalRejected, match="as_of"):
        source.get_target_allocation(as_of=requested)
    assert not state_file.exists()


def test_as_of_arg_matching_payload_passes(state_file: Path) -> None:
    """get_target_allocation(as_of=X) where payload as_of == X → accepted."""
    source = _make_source(state_file)  # payload as_of is _AS_OF (2026-06-01)
    requested = datetime.date.fromisoformat(_AS_OF)
    ta = source.get_target_allocation(as_of=requested)
    assert ta.as_of == requested


def test_as_of_none_keeps_strict_increase_behavior(state_file: Path) -> None:
    """as_of=None (default) uses only strict-increase replay rule (no equality)."""
    source = _make_source(state_file)
    ta = source.get_target_allocation(as_of=None)
    assert ta is not None
    assert state_file.exists()


# ===========================================================================
# F7c — seq must be an int (reject floats)
# ===========================================================================


def test_rejects_float_seq(state_file: Path) -> None:
    """A non-integer seq (e.g. 100.0001) → SignalRejected; no state persisted."""
    payload = _make_payload(audience="test-client", seq=100)
    payload["seq"] = 100.0001  # float — overwrite after default build
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="seq"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_bool_seq(state_file: Path) -> None:
    """A bool seq (True, an int subclass) is rejected as not a real int."""
    payload = _make_payload(audience="test-client")
    payload["seq"] = True
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected, match="seq"):
        source.get_target_allocation()
    assert not state_file.exists()


def test_seq_persisted_as_int(state_file: Path) -> None:
    """An accepted int seq is persisted as an int in the state file."""
    payload = _make_payload(audience="test-client", seq=7)
    source = _make_source(state_file, payload=payload)
    source.get_target_allocation()
    state = json.loads(state_file.read_text())
    assert state["seq"] == 7
    assert isinstance(state["seq"], int)


# ===========================================================================
# F8b — non-finite server bound rejected
# ===========================================================================


def test_rejects_nan_server_bound(state_file: Path) -> None:
    """bounds.max_per_asset = NaN → SignalRejected; no state persisted.

    Without F8b, ``min(local, nan)`` is order-dependent and could disable the cap.
    """
    payload = _make_payload(
        audience="test-client",
        bounds={"max_per_asset": float("nan"), "max_turnover": 0.5},
    )
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()


def test_rejects_inf_server_bound(state_file: Path) -> None:
    """bounds.max_per_asset = inf → SignalRejected; no state persisted."""
    payload = _make_payload(
        audience="test-client",
        bounds={"max_per_asset": float("inf"), "max_turnover": 0.5},
    )
    wire = _make_wire(payload)
    source = _make_source(state_file, signal_data=wire)
    with pytest.raises(SignalRejected):
        source.get_target_allocation()
    assert not state_file.exists()
