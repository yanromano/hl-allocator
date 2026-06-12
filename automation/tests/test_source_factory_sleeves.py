"""
Tests for the per-sleeve construction layer of
``automation.allocation.source_factory.build_sources``.

Task C2 of the CRASH Phase 1 plan.  ``build_sources(cfg)`` returns ONE
``AllocationSource`` per sleeve (``cfg.sleeves()``), keyed by sleeve name.  The
critical invariant (red-team H5) is that each REMOTE sleeve's anti-replay state
file is DISTINCT — two sleeves sharing one filename would collide on the
``seq=YYYYMMDD`` anti-replay sequence and one sleeve would dark-hold EVERY day.

The state-file naming is ``remote_signal_state.{name}.json`` next to
``cfg.state_path``, EXCEPT the singular-adapter sleeve named ``"default"`` which
keeps the LEGACY filename ``remote_signal_state.json`` so deployed daemons do
not orphan their existing anti-replay state.

All tests are OFFLINE: ``RemoteSignalSource.__init__`` is network-free (it
constructs an httpx.Client but performs no fetch until
``get_target_allocation``).  Mirrors the style of ``test_source_factory.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from automation.allocation.remote_signal import RemoteSignalSource
from automation.allocation.source_factory import build_source, build_sources
from automation.core.config import CapsConfig, Config, SignalSourceConfig

# A valid lowercase-hex root-key fingerprint VALUE (>= 16 hex chars).  The YAML
# carries the env-var NAME; the factory resolves it via os.environ at build time.
_VALID_FP = "0123456789abcdef"


def _base(tmp_path: Path) -> dict[str, Any]:
    return {
        "env": "testnet",
        "subaccount_address": "0x" + "0" * 40,
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


def _remote_sleeve(
    name: str, *, budget: float, allow_short: bool
) -> SignalSourceConfig:
    return SignalSourceConfig(
        name=name,
        type="remote",
        url=f"https://signals.example.com/v1/{name}/signal",
        client_id=f"{name}-client",
        budget=budget,
        allow_short=allow_short,
        root_key_fingerprint_ref=f"FP_{name.upper()}",
        token_ref=f"TOK_{name.upper()}",
    )


def _two_sleeve_cfg(tmp_path: Path) -> Config:
    base = _base(tmp_path)
    base["signal_sources"] = [
        _remote_sleeve("haarp", budget=0.5, allow_short=False),
        _remote_sleeve("crash", budget=0.5, allow_short=True),
    ]
    return Config(**base)


def _singular_remote_cfg(tmp_path: Path) -> Config:
    base = _base(tmp_path)
    base["signal_source"] = SignalSourceConfig(
        type="remote",
        url="https://signals.example.com/v1/signal",
        client_id="legacy-client",
        root_key_fingerprint_ref="FP_DEFAULT",
        token_ref="TOK_DEFAULT",
    )
    return Config(**base)


def _set_env(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Set FP_<NAME> to a valid hex fingerprint and TOK_<NAME> to a bearer."""
    for name in names:
        monkeypatch.setenv(f"FP_{name.upper()}", _VALID_FP)
        monkeypatch.setenv(f"TOK_{name.upper()}", f"bearer-{name}")


# ---------------------------------------------------------------------------
# one source per sleeve
# ---------------------------------------------------------------------------


def test_build_sources_returns_one_per_sleeve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2-sleeve cfg → dict {name: AllocationSource} with 2 entries."""
    _set_env(monkeypatch, "haarp", "crash")
    sources = build_sources(_two_sleeve_cfg(tmp_path))
    assert isinstance(sources, dict)
    assert set(sources) == {"haarp", "crash"}
    # both are remote sleeves → concrete RemoteSignalSource (the AllocationSource
    # Protocol is not @runtime_checkable, so assert the concrete type instead)
    assert all(isinstance(s, RemoteSignalSource) for s in sources.values())


# ---------------------------------------------------------------------------
# H5: per-sleeve anti-replay state files MUST be distinct
# ---------------------------------------------------------------------------


def test_per_sleeve_remote_state_files_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two REMOTE sleeves → distinct ``remote_signal_state.{name}.json`` paths.

    The H5 bug: a shared filename collides on ``seq=YYYYMMDD`` so one sleeve
    would dark-hold every day.
    """
    _set_env(monkeypatch, "haarp", "crash")
    sources = build_sources(_two_sleeve_cfg(tmp_path))
    haarp_path = sources["haarp"]._state_path
    crash_path = sources["crash"]._state_path
    parent = Path(tmp_path)
    assert haarp_path == parent / "remote_signal_state.haarp.json"
    assert crash_path == parent / "remote_signal_state.crash.json"
    assert haarp_path != crash_path


# ---------------------------------------------------------------------------
# allow_short forwarded
# ---------------------------------------------------------------------------


def test_allow_short_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sleeve with allow_short=True → its source carries _allow_short=True."""
    _set_env(monkeypatch, "haarp", "crash")
    sources = build_sources(_two_sleeve_cfg(tmp_path))
    assert sources["haarp"]._allow_short is False
    assert sources["crash"]._allow_short is True


# ---------------------------------------------------------------------------
# singular adapter keeps the LEGACY state filename
# ---------------------------------------------------------------------------


def test_singular_config_legacy_state_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SINGULAR cfg → {"default": source} using the LEGACY filename.

    Deployed daemons have anti-replay state under the old
    ``remote_signal_state.json`` name; the singular-adapter ``default`` sleeve
    must NOT migrate to ``remote_signal_state.default.json`` and orphan it.
    """
    _set_env(monkeypatch, "default")
    sources = build_sources(_singular_remote_cfg(tmp_path))
    assert set(sources) == {"default"}
    source = sources["default"]
    assert isinstance(source, RemoteSignalSource)
    assert source._state_path == Path(tmp_path) / "remote_signal_state.json"
    # NOT the per-sleeve name-tagged file
    assert source._state_path != Path(tmp_path) / "remote_signal_state.default.json"


# ---------------------------------------------------------------------------
# per-sleeve wire identity (client_id / url / token / fingerprint)
# ---------------------------------------------------------------------------


def test_per_sleeve_client_id_and_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each source gets its own sleeve's client_id / url / token / fingerprint."""
    _set_env(monkeypatch, "haarp", "crash")
    # distinct fingerprints per sleeve so the resolved VALUE differs too
    monkeypatch.setenv("FP_HAARP", "aaaaaaaaaaaaaaaa")
    monkeypatch.setenv("FP_CRASH", "bbbbbbbbbbbbbbbb")
    monkeypatch.setenv("TOK_HAARP", "bearer-haarp")
    monkeypatch.setenv("TOK_CRASH", "bearer-crash")
    sources = build_sources(_two_sleeve_cfg(tmp_path))

    haarp = sources["haarp"]
    crash = sources["crash"]
    assert haarp._client_id == "haarp-client"
    assert crash._client_id == "crash-client"
    assert haarp._url == "https://signals.example.com/v1/haarp/signal"
    assert crash._url == "https://signals.example.com/v1/crash/signal"
    assert haarp._root_key_fingerprint == "aaaaaaaaaaaaaaaa"
    assert crash._root_key_fingerprint == "bbbbbbbbbbbbbbbb"
    assert haarp._bearer_token == "bearer-haarp"
    assert crash._bearer_token == "bearer-crash"


# ---------------------------------------------------------------------------
# build_source wrapper compatibility (legacy single-source callers)
# ---------------------------------------------------------------------------


def test_build_source_wrapper_returns_sole_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_source`` still returns the lone source for the singular path."""
    _set_env(monkeypatch, "default")
    source = build_source(_singular_remote_cfg(tmp_path))
    assert isinstance(source, RemoteSignalSource)
    assert source._state_path == Path(tmp_path) / "remote_signal_state.json"
