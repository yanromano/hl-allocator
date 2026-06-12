"""
Tests for the ``allow_short`` branch of ``validate_target_allocation`` (CRASH Phase 1, Task A1)
and the schema_version gate + ``allow_short`` consumption in ``RemoteSignalSource``
(CRASH Phase 1, Task A2).

Task A1 covers (spec §3, wire protocol v2):
- ``allow_short=False`` (and kwarg omitted) is BYTE-COMPATIBLE with the deployed
  long-only validator: every one of the 7 existing rejection classes raises
  ``AllocationRejected`` with the EXACT same message text as today.  These
  strings are intentionally hard-coded — they pin deployed-client parity.
- ``allow_short=True`` (v2 semantics): weights finite in [-1, 1];
  gross ``sum(abs(w)) <= 1 + tol``; unity ``sum(abs(w)) + cash == 1 ± tol``;
  per-asset cap on ``abs(w)``; audience/whitelist/finiteness checks unchanged.

Task A2 covers (spec §3 schema gate):
- ``schema_version`` must be in ``{1, 2}``; anything else → ``SignalRejected`` naming the version.
- ``schema_version=2`` on a ``allow_short=False`` source → ``SignalRejected`` mentioning "schema"
  (fails at the gate, NOT at the negative-weight check — fail-loud on misconfiguration).
- ``schema_version=2`` on a ``allow_short=True`` source → accepted; signed weights round-trip.
- v1 payload with a negative weight on an ``allow_short=True`` source → rejected (v1 IS long-only
  by definition; the validator runs with ``allow_short=False`` regardless of source flag).
- Extra coverage: multi-short v2, all-cash v1 and v2.
"""

from __future__ import annotations

import base64
import datetime
import hashlib as _hashlib
from pathlib import Path
from typing import Any

import nacl.signing
import pytest

from automation.allocation.base import (
    AllocationRejected,
    TargetAllocation,
    validate_target_allocation,
)
from automation.allocation.remote_signal import (
    RemoteSignalSource,
    SignalRejected,
    canonical,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = datetime.date(2026, 6, 10)

_DEFAULT_WHITELIST: set[str] = {"BTC", "ETH", "SOL", "TON"}
_DEFAULT_CLIENT = "client-1"
_DEFAULT_MAX_PER_ASSET = 0.6


def _make_ta(
    weights: dict[str, float] | None = None,
    cash: float = 0.0,
    audience: str = _DEFAULT_CLIENT,
    strategy_id: str = "test-strategy",
    model_rev: str = "v1",
    as_of: datetime.date = _TODAY,
    schema_version: int = 1,
) -> TargetAllocation:
    if weights is None:
        weights = {"BTC": 0.5, "ETH": 0.5}
    return TargetAllocation(
        as_of=as_of,
        weights=weights,
        cash=cash,
        strategy_id=strategy_id,
        model_rev=model_rev,
        schema_version=schema_version,
        audience=audience,
    )


def _validate(
    ta: TargetAllocation,
    *,
    max_per_asset: float = _DEFAULT_MAX_PER_ASSET,
    whitelist: set[str] | None = None,
    client_id: str = _DEFAULT_CLIENT,
    **kwargs: bool,
) -> None:
    validate_target_allocation(
        ta,
        max_per_asset=max_per_asset,
        whitelist=whitelist if whitelist is not None else _DEFAULT_WHITELIST,
        client_id=client_id,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Long-only branch: byte-compatible with the deployed validator
# ---------------------------------------------------------------------------

# (name, ta-builder kwargs, EXACT expected message — copied from base.py rendering)
_LONG_ONLY_REJECTIONS: list[tuple[str, dict[str, object], str]] = [
    (
        "audience_mismatch",
        {"weights": {"BTC": 0.5, "ETH": 0.5}, "cash": 0.0, "audience": "other-client"},
        "Allocation audience 'other-client' does not match client_id 'client-1'",
    ),
    (
        "nan_weight",
        {"weights": {"BTC": float("nan"), "ETH": 0.5}, "cash": 0.5},
        "Non-finite weights for coins: ['BTC']",
    ),
    (
        "inf_weight",
        {"weights": {"BTC": float("inf"), "ETH": 0.0}, "cash": 0.0},
        "Non-finite weights for coins: ['BTC']",
    ),
    (
        "nan_cash",
        {"weights": {}, "cash": float("nan")},
        "Non-finite cash value: nan",
    ),
    (
        "negative_weight",
        {"weights": {"BTC": 0.6, "ETH": -0.1}, "cash": 0.5},
        "Negative weights violate long-only contract: ['ETH']",
    ),
    (
        "unknown_coin",
        {"weights": {"BTC": 0.5, "DOGE": 0.5}, "cash": 0.0},
        "Coins not in whitelist: ['DOGE']",
    ),
    (
        "exceeds_max_per_asset",
        {"weights": {"BTC": 0.65, "ETH": 0.35}, "cash": 0.0},
        "Per-asset cap (0.6) breached for: [('BTC', 0.65)]. "
        "Signal is untrusted — executor hard-alerts (spec S-11).",
    ),
    (
        "sum_exceeds_one",
        {"weights": {"BTC": 0.6, "ETH": 0.6}, "cash": 0.0},
        "weights sum 1.200000000 exceeds 1.0 + tol (1.000001000)",
    ),
    (
        "unity_sum_violation",
        {"weights": {"BTC": 0.3, "ETH": 0.3}, "cash": 0.1},
        "sum(weights) + cash = 0.700000000, expected 1.0 ± 1e-06",
    ),
]


@pytest.mark.parametrize(
    ("name", "ta_kwargs", "expected_msg"),
    _LONG_ONLY_REJECTIONS,
    ids=[name for name, _, _ in _LONG_ONLY_REJECTIONS],
)
def test_long_only_branch_byte_compatible(
    name: str, ta_kwargs: dict[str, object], expected_msg: str
) -> None:
    """Every existing rejection class raises with the EXACT deployed message text.

    Checked twice: with ``allow_short=False`` explicit AND with the kwarg omitted
    (default) — both must be byte-identical to today's behavior.
    """
    ta = _make_ta(**ta_kwargs)  # type: ignore[arg-type]

    # Explicit allow_short=False
    with pytest.raises(AllocationRejected) as exc_explicit:
        _validate(ta, allow_short=False)
    assert str(exc_explicit.value) == expected_msg

    # Kwarg omitted — the default must be the long-only branch
    with pytest.raises(AllocationRejected) as exc_default:
        _validate(ta)
    assert str(exc_default.value) == expected_msg


def test_long_only_valid_allocation_still_passes() -> None:
    """A well-formed long-only allocation passes with allow_short=False and omitted."""
    ta = _make_ta(weights={"BTC": 0.5, "ETH": 0.5}, cash=0.0)
    _validate(ta)  # must not raise
    _validate(ta, allow_short=False)  # must not raise


def test_long_only_still_rejects_negative_when_flag_false() -> None:
    """Explicit allow_short=False + negative weight → EXACT long-only rejection text."""
    ta = _make_ta(weights={"TON": -0.4, "BTC": 0.3}, cash=1.1)
    with pytest.raises(AllocationRejected) as exc:
        _validate(ta, max_per_asset=0.9, allow_short=False)
    assert str(exc.value) == "Negative weights violate long-only contract: ['TON']"


# ---------------------------------------------------------------------------
# allow_short=True: v2 semantics (gross / unity / per-asset on |w|)
# ---------------------------------------------------------------------------


def test_allow_short_accepts_negative() -> None:
    """A short leg is accepted: Σ|w| = 0.7 ≤ 1 and Σ|w| + cash = 1.0."""
    ta = _make_ta(weights={"TON": -0.4, "BTC": 0.3}, cash=0.3)
    _validate(ta, max_per_asset=0.9, allow_short=True)  # must not raise


def test_allow_short_gross_cap() -> None:
    """Gross exposure Σ|w| = 1.2 > 1 + tol → rejected (mentions gross / exceeds)."""
    ta = _make_ta(weights={"TON": -0.7, "BTC": 0.5}, cash=0.0)
    with pytest.raises(AllocationRejected) as exc:
        _validate(ta, max_per_asset=0.9, allow_short=True)
    msg = str(exc.value)
    assert "gross" in msg and "exceeds" in msg


def test_allow_short_unity_abs() -> None:
    """Σ|w| + cash = 0.9 ≠ 1 ± tol → rejected (unity check runs on |w|)."""
    ta = _make_ta(weights={"TON": -0.4}, cash=0.5)
    with pytest.raises(AllocationRejected) as exc:
        _validate(ta, max_per_asset=0.9, allow_short=True)
    assert "expected 1.0" in str(exc.value)


def test_allow_short_per_asset_abs() -> None:
    """|w| = 0.95 > max_per_asset 0.9 → rejected (cap applies to magnitude)."""
    ta = _make_ta(weights={"TON": -0.95}, cash=0.05)
    with pytest.raises(AllocationRejected) as exc:
        _validate(ta, max_per_asset=0.9, allow_short=True)
    assert "Per-asset cap" in str(exc.value)


def test_allow_short_weight_below_minus_one() -> None:
    """w = -1.5 is outside the v2 range [-1, 1] → rejected."""
    ta = _make_ta(weights={"TON": -1.5}, cash=0.0)
    with pytest.raises(AllocationRejected) as exc:
        _validate(ta, max_per_asset=0.9, allow_short=True)
    assert "[-1, 1]" in str(exc.value)


def test_allow_short_weight_above_one_rejected() -> None:
    """w = 1.5 is outside the v2 range [-1, 1] → rejected (range is symmetric)."""
    ta = _make_ta(weights={"BTC": 1.5}, cash=0.0)
    with pytest.raises(AllocationRejected) as exc:
        _validate(ta, max_per_asset=0.9, allow_short=True)
    assert "[-1, 1]" in str(exc.value)


def test_allow_short_shared_checks_unchanged() -> None:
    """Finiteness / whitelist / audience checks fire identically under allow_short=True."""
    # NaN weight
    with pytest.raises(AllocationRejected) as exc_nan:
        _validate(
            _make_ta(weights={"TON": float("nan")}, cash=1.0),
            max_per_asset=0.9,
            allow_short=True,
        )
    assert str(exc_nan.value) == "Non-finite weights for coins: ['TON']"

    # Unknown coin
    with pytest.raises(AllocationRejected) as exc_wl:
        _validate(
            _make_ta(weights={"DOGE": -0.5}, cash=0.5),
            max_per_asset=0.9,
            allow_short=True,
        )
    assert str(exc_wl.value) == "Coins not in whitelist: ['DOGE']"

    # Audience mismatch
    with pytest.raises(AllocationRejected) as exc_aud:
        _validate(
            _make_ta(weights={"TON": -0.4, "BTC": 0.3}, cash=0.3, audience="other"),
            max_per_asset=0.9,
            allow_short=True,
        )
    assert str(exc_aud.value) == (
        "Allocation audience 'other' does not match client_id 'client-1'"
    )


def test_allow_short_full_short_sleeve_passes() -> None:
    """A fully-short sleeve (w = -0.9, cash = 0.1) is valid under v2 margin accounting."""
    ta = _make_ta(weights={"TON": -0.9}, cash=0.1)
    _validate(ta, max_per_asset=0.9, allow_short=True)  # must not raise


# ---------------------------------------------------------------------------
# TargetAllocation model: widened to [-1, 1] (validator owns long-only per sleeve)
# ---------------------------------------------------------------------------


def test_model_accepts_negative_weights() -> None:
    """The pydantic model accepts signed weights — the validator layer owns long-only."""
    ta = _make_ta(weights={"TON": -0.4, "BTC": 0.3}, cash=0.3)
    assert ta.weights["TON"] == -0.4


# ===========================================================================
# Task A2 — schema_version gate + allow_short consumption in RemoteSignalSource
# ===========================================================================
#
# The helpers below are minimal copies of those in test_remote_signal.py
# (cross-test-module imports are not the house style).  They are purposely
# self-contained so this module compiles without any test-module dependency.

# ---------------------------------------------------------------------------
# A2: crypto key material (generated once per module load)
# ---------------------------------------------------------------------------

_A2_ROOT_KEY = nacl.signing.SigningKey.generate()
_A2_ROOT_PUBKEY_BYTES = bytes(_A2_ROOT_KEY.verify_key)
_A2_ROOT_FP = _hashlib.sha256(_A2_ROOT_PUBKEY_BYTES).hexdigest()[:16]

_A2_SIGNAL_KEY = nacl.signing.SigningKey.generate()
_A2_SIGNAL_PUBKEY_B64 = base64.b64encode(bytes(_A2_SIGNAL_KEY.verify_key)).decode()
_A2_KEY_ID = "signal-key-a2"

_A2_NOW = datetime.datetime(2026, 6, 10, 12, 0, 0, tzinfo=datetime.UTC)
_A2_ISSUED = _A2_NOW - datetime.timedelta(minutes=5)
_A2_EXPIRES = _A2_NOW + datetime.timedelta(hours=2)
_A2_AS_OF = "2026-06-10"
_A2_SEQ = 200


# ---------------------------------------------------------------------------
# A2: wire-building helpers
# ---------------------------------------------------------------------------


def _a2_make_manifest() -> dict[str, Any]:
    body: dict[str, Any] = {
        "issued_at": "2026-06-10T00:00:00Z",
        "keys": [
            {
                "key_id": _A2_KEY_ID,
                "public_key": _A2_SIGNAL_PUBKEY_B64,
                "status": "active",
            }
        ],
    }
    sig = _A2_ROOT_KEY.sign(canonical(body)).signature
    return {
        **body,
        "signature": base64.b64encode(sig).decode(),
        "root_public_key": base64.b64encode(_A2_ROOT_PUBKEY_BYTES).decode(),
    }


def _a2_make_payload(
    *,
    client_id: str = "a2-client",
    weights: dict[str, float] | None = None,
    cash: float | None = None,
    schema_version: int = 1,
    seq: int = _A2_SEQ,
) -> dict[str, Any]:
    if weights is None:
        weights = {"BTC": 0.4, "ETH": 0.3, "SOL": 0.2}
    if cash is None:
        cash = 1.0 - sum(abs(v) for v in weights.values())
    return {
        "strategy_id": "test-strat",
        "model_rev": "2026-06-10",
        "schema_version": schema_version,
        "audience": client_id,
        "as_of": _A2_AS_OF,
        "weights": weights,
        "cash": cash,
        "bounds": {"max_per_asset": 0.9, "max_turnover": 0.5},
        "seq": seq,
        "issued_at": _A2_ISSUED.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": _A2_EXPIRES.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _a2_make_wire(payload: dict[str, Any]) -> dict[str, Any]:
    sig = _A2_SIGNAL_KEY.sign(canonical(payload)).signature
    return {
        "payload": payload,
        "signature": base64.b64encode(sig).decode(),
        "key_id": _A2_KEY_ID,
    }


class _A2FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def json(self) -> dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        pass


class _A2FakeHttp:
    def __init__(self, manifest: dict[str, Any], wire: dict[str, Any]) -> None:
        self._manifest = manifest
        self._wire = wire

    def get(self, url: str, headers: dict[str, str] | None = None) -> _A2FakeResponse:
        if "keys/manifest" in url:
            return _A2FakeResponse(self._manifest)
        return _A2FakeResponse(self._wire)


def _a2_make_source(
    state_file: Path,
    *,
    payload: dict[str, Any],
    client_id: str = "a2-client",
    allow_short: bool = False,
    local_whitelist: set[str] | None = None,
) -> RemoteSignalSource:
    """Build a RemoteSignalSource with the A2 fake HTTP stack."""
    if local_whitelist is None:
        local_whitelist = {"BTC", "ETH", "SOL", "TON"}
    manifest = _a2_make_manifest()
    wire = _a2_make_wire(payload)
    http = _A2FakeHttp(manifest, wire)
    return RemoteSignalSource(
        url="https://test.internal/v1/signal",
        client_id=client_id,
        root_key_fingerprint=_A2_ROOT_FP,
        state_path=state_file,
        bearer_token="tok",
        http_client=http,
        now_fn=lambda: _A2_NOW,
        local_max_per_asset=0.9,
        local_whitelist=local_whitelist,
        allow_short=allow_short,
    )


# ---------------------------------------------------------------------------
# A2: schema_version gate tests
# ---------------------------------------------------------------------------


class TestSchemaGate:
    """Task A2 — schema_version gate + allow_short wiring in RemoteSignalSource."""

    def test_schema_v2_rejected_on_long_only_source(self, tmp_path: Path) -> None:
        """ALL-POSITIVE v2 payload on a default (allow_short=False) source → SignalRejected
        mentioning 'schema'.  Must fail at the gate — NOT at the negative-weight check
        (the weights here are all positive, so if the gate is absent this would silently pass).
        """
        payload = _a2_make_payload(schema_version=2, weights={"BTC": 0.4}, cash=0.6)
        source = _a2_make_source(
            tmp_path / "state.json",
            payload=payload,
            allow_short=False,
        )
        with pytest.raises(SignalRejected) as exc:
            source.get_target_allocation()
        assert "schema" in str(exc.value).lower()

    def test_schema_v2_accepted_when_allow_short(self, tmp_path: Path) -> None:
        """v2 payload with one negative weight on an allow_short=True source → accepted;
        returned ta.weights carry the signed value through.
        """
        payload = _a2_make_payload(
            schema_version=2,
            weights={"TON": -0.4, "BTC": 0.3},
            cash=0.3,
            client_id="a2-client",
        )
        source = _a2_make_source(
            tmp_path / "state.json",
            payload=payload,
            allow_short=True,
        )
        ta = source.get_target_allocation()
        assert ta.weights["TON"] == pytest.approx(-0.4)
        assert ta.weights["BTC"] == pytest.approx(0.3)

    def test_schema_v3_rejected_everywhere(self, tmp_path: Path) -> None:
        """schema_version=3 → SignalRejected on BOTH allow_short=True and False sources."""
        payload = _a2_make_payload(schema_version=3, weights={"BTC": 0.4}, cash=0.6)

        for allow_short in (False, True):
            state = tmp_path / f"state_v3_{allow_short}.json"
            source = _a2_make_source(state, payload=payload, allow_short=allow_short)
            with pytest.raises(SignalRejected) as exc:
                source.get_target_allocation()
            assert "3" in str(exc.value), (
                f"Expected version '3' in message; got: {exc.value!r}"
            )

    def test_schema_v1_negative_weight_still_rejected_even_on_short_source(
        self, tmp_path: Path
    ) -> None:
        """v1 payload with a negative weight on an allow_short=True source → rejected.

        v1 IS long-only by definition.  The gate passes (sv=1 is supported), but the
        validator runs with allow_short=False, so the negative-weight rejection fires.
        """
        payload = _a2_make_payload(
            schema_version=1,
            weights={"TON": -0.4, "BTC": 0.3},
            cash=1.1,  # cash adjusted for negative: 1 - (0.4+0.3) = 0.3 ... but v1 no abs
            client_id="a2-client",
        )
        # Fix cash so the unity check doesn't fire before the negative-weight check.
        payload["cash"] = 1.1  # intentionally sum-off: negative-weight fires first
        source = _a2_make_source(
            tmp_path / "state.json",
            payload=payload,
            allow_short=True,  # source allows shorts — but v1 payload must still be long-only
        )
        with pytest.raises(SignalRejected) as exc:
            source.get_target_allocation()
        # The long-only rejection text must appear (not a schema error)
        assert "Negative weights" in str(exc.value)

    def test_v2_all_cash_accepted_when_allow_short(self, tmp_path: Path) -> None:
        """v2 payload with empty weights and cash=1.0 is valid (all-cash is neutral)."""
        payload = _a2_make_payload(
            schema_version=2,
            weights={},
            cash=1.0,
            client_id="a2-client",
        )
        source = _a2_make_source(
            tmp_path / "state.json",
            payload=payload,
            allow_short=True,
        )
        ta = source.get_target_allocation()
        assert ta.weights == {}
        assert ta.cash == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A2: extra coverage — validate_target_allocation directly (no HTTP stack)
# ---------------------------------------------------------------------------


def test_allow_short_multi_short_passes() -> None:
    """Multiple shorts summing |w|=0.9 + cash=0.1 passes the allow_short validator."""
    ta = _make_ta(
        weights={"TON": -0.3, "BTC": -0.3, "ETH": -0.3},
        cash=0.1,
    )
    _validate(ta, max_per_asset=0.9, allow_short=True)  # must not raise


def test_long_only_all_cash_passes() -> None:
    """All-cash allocation (no weights) is valid under the long-only validator."""
    ta = _make_ta(weights={}, cash=1.0)
    _validate(ta, allow_short=False)  # must not raise
