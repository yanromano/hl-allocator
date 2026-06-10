"""
Tests for automation.allocation.source_factory.build_source.

The factory dispatches on ``cfg.signal_source.type`` and builds a generic
``AllocationSource`` WITHOUT importing any haarp / systems module.  HAARP is
built separately by the closed injected root (``run_daemon_haarp``).

All tests are OFFLINE: the ``remote`` case is asserted by TYPE only (its
__init__ is network-free — it constructs an httpx.Client but performs no fetch
until ``get_target_allocation`` is called).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from automation.allocation.base import validate_target_allocation
from automation.allocation.examples.equal_weight import EqualWeightSource
from automation.allocation.examples.single_asset import SingleAssetSource
from automation.allocation.remote_signal import RemoteSignalSource
from automation.allocation.source_factory import build_source
from automation.core.config import CapsConfig, Config, SignalSourceConfig


def _cfg(tmp_path: Path, signal_source: SignalSourceConfig, **overrides: Any) -> Config:
    base: dict[str, Any] = {
        "env": "testnet",
        "subaccount_address": "0x" + "0" * 40,
        "signal_source": signal_source,
        "rebalance_utc_time": "00:15",
        "max_per_asset": 1.0,
        "caps": CapsConfig(
            max_notional_per_coin=10_000.0,
            max_total_notional=50_000.0,
            max_turnover_per_cycle=60_000.0,
            max_order_slippage=0.01,
        ),
        "universe_perp_map": {"BTC": "BTC", "ETH": "ETH"},
        "state_path": str(tmp_path / "state.json"),
    }
    base.update(overrides)
    return Config(**base)


# ---------------------------------------------------------------------------
# equal_weight
# ---------------------------------------------------------------------------


def test_equal_weight_builds(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(type="equal_weight", client_id="c1"),
    )
    source = build_source(cfg)
    assert isinstance(source, EqualWeightSource)


def test_equal_weight_sums_to_one(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(type="equal_weight", client_id="c1"),
    )
    source = build_source(cfg)
    ta = source.get_target_allocation()
    assert abs(sum(ta.weights.values()) + ta.cash - 1.0) < 1e-9
    # passes the local safety validator for the configured universe
    validate_target_allocation(
        ta, max_per_asset=1.0, whitelist={"BTC", "ETH"}, client_id="c1"
    )


def test_equal_weight_prefers_tradeable_coins(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(type="equal_weight", client_id="c1"),
        tradeable_coins=["BTC"],
    )
    source = build_source(cfg)
    ta = source.get_target_allocation()
    assert set(ta.weights) == {"BTC"}


# ---------------------------------------------------------------------------
# single_asset
# ---------------------------------------------------------------------------


def test_single_asset_builds(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(type="single_asset", client_id="c1"),
        universe_perp_map={"BTC": "BTC"},
    )
    source = build_source(cfg)
    assert isinstance(source, SingleAssetSource)
    ta = source.get_target_allocation()
    assert ta.weights == {"BTC": 1.0}
    assert ta.cash == 0.0
    assert ta.audience == "c1"


def test_single_asset_uses_first_coin(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(type="single_asset", client_id="c1"),
        tradeable_coins=["ETH", "BTC"],
    )
    source = build_source(cfg)
    ta = source.get_target_allocation()
    assert ta.weights == {"ETH": 1.0}


# ---------------------------------------------------------------------------
# remote
# ---------------------------------------------------------------------------


def _remote_cfg(tmp_path: Path) -> Config:
    return _cfg(
        tmp_path,
        SignalSourceConfig(
            type="remote",
            url="https://signals.example.com/v1/signal",
            client_id="c1",
            root_key_fingerprint_ref="HL_SIGNAL_ROOT_FP",
            token_ref="HL_SIGNAL_BEARER",
        ),
    )


def test_remote_resolves_credentials_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """remote → token_ref / root_key_fingerprint_ref are env var NAMES.

    The factory must resolve the env vars and pass the VALUES to
    RemoteSignalSource — never the literal ref strings (the literal name would
    be sent as the Bearer token and would fail the hex fingerprint check).
    """
    monkeypatch.setenv("HL_SIGNAL_BEARER", "secret-bearer-value")
    monkeypatch.setenv("HL_SIGNAL_ROOT_FP", "0123456789abcdef")
    source = build_source(_remote_cfg(tmp_path))
    assert isinstance(source, RemoteSignalSource)
    assert source._bearer_token == "secret-bearer-value"
    assert source._root_key_fingerprint == "0123456789abcdef"


def test_remote_missing_token_env_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset Bearer-token env var → clear boot-time error naming the env var."""
    monkeypatch.delenv("HL_SIGNAL_BEARER", raising=False)
    monkeypatch.setenv("HL_SIGNAL_ROOT_FP", "0123456789abcdef")
    with pytest.raises(ValueError, match="HL_SIGNAL_BEARER"):
        build_source(_remote_cfg(tmp_path))


def test_remote_missing_fingerprint_env_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset fingerprint env var → clear boot-time error naming the env var."""
    monkeypatch.setenv("HL_SIGNAL_BEARER", "secret-bearer-value")
    monkeypatch.delenv("HL_SIGNAL_ROOT_FP", raising=False)
    with pytest.raises(ValueError, match="HL_SIGNAL_ROOT_FP"):
        build_source(_remote_cfg(tmp_path))


def test_remote_empty_env_value_treated_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty-string env var value → same error as unset."""
    monkeypatch.setenv("HL_SIGNAL_BEARER", "")
    monkeypatch.setenv("HL_SIGNAL_ROOT_FP", "0123456789abcdef")
    with pytest.raises(ValueError, match="HL_SIGNAL_BEARER"):
        build_source(_remote_cfg(tmp_path))


def test_remote_token_ref_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """token_ref is optional — omitting it builds with bearer_token=None."""
    monkeypatch.setenv("HL_SIGNAL_ROOT_FP", "0123456789abcdef")
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(
            type="remote",
            url="https://signals.example.com/v1/signal",
            client_id="c1",
            root_key_fingerprint_ref="HL_SIGNAL_ROOT_FP",
        ),
    )
    source = build_source(cfg)
    assert isinstance(source, RemoteSignalSource)
    assert source._bearer_token is None


# ---------------------------------------------------------------------------
# unknown
# ---------------------------------------------------------------------------


def test_unknown_type_raises(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        SignalSourceConfig(type="equal_weight", client_id="c1"),
    )
    # Bypass the pydantic Literal by mutating after construction.
    object.__setattr__(cfg.signal_source, "type", "bogus")
    with pytest.raises(ValueError, match="signal_source.type|bogus|unknown"):
        build_source(cfg)


def test_factory_imports_no_haarp() -> None:
    """The factory module must not IMPORT any haarp/systems module."""
    import automation.allocation.source_factory as sf

    for line in Path(sf.__file__).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            low = stripped.lower()
            assert "haarp" not in low, f"haarp import leaked: {stripped}"
            assert "systems" not in low, f"systems import leaked: {stripped}"
