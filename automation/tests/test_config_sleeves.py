"""
Tests for the multi-source (sleeves) layer of automation.core.config.

Task C1 of the CRASH Phase 1 plan: ``signal_sources`` list form + singular
adapter + per-source staleness + ``liquidation_cooldown_days``.

Written before the implementation (TDD RED → GREEN).  Mirrors the
minimal-config builder/yaml style of ``test_config.py`` (load the neutral
example yaml, mutate the raw dict, re-construct).

Invariants pinned here:
* the deployed SINGULAR ``signal_source:`` block keeps validating exactly as
  today AND normalizes to a single ``default`` sleeve (budget 1.0, long-only);
* the new LIST form parses, with per-sleeve budgets/flags/names;
* budgets that exceed unity, duplicate names, and "both forms present" are
  rejected;
* the per-source mainnet-remote rule fires PER source;
* per-source staleness falls back to the global default.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from automation.core.config import Config

# Path to the neutral example config shipped with the package
EX = Path(__file__).resolve().parents[1] / "config" / "automation.example.yaml"


def _raw() -> dict[str, Any]:
    """Fresh raw dict from the neutral example yaml (singular signal_source)."""
    raw: dict[str, Any] = yaml.safe_load(EX.read_text())
    return raw


def _remote_sleeve(name: str, budget: float, allow_short: bool) -> dict:
    """A fully-specified remote sleeve dict (passes the mainnet-remote rule)."""
    return {
        "name": name,
        "type": "remote",
        "url": f"https://signals.example.com/v1/{name}/signal",
        "client_id": f"{name}-client",
        "budget": budget,
        "allow_short": allow_short,
        "root_key_fingerprint_ref": "HL_SIGNAL_ROOT_FP",
        "token_ref": "HL_SIGNAL_BEARER",
    }


# ---------------------------------------------------------------------------
# List form parsing
# ---------------------------------------------------------------------------


def test_list_form_parses() -> None:
    """A signal_sources list of two sources loads; sleeves() returns both."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),
        _remote_sleeve("crash", 0.5, allow_short=True),
    ]
    c = Config(**raw)
    sleeves = c.sleeves()
    assert [s.name for s in sleeves] == ["haarp", "crash"]
    assert [s.budget for s in sleeves] == [0.5, 0.5]
    assert [s.allow_short for s in sleeves] == [False, True]
    assert sleeves[0].client_id == "haarp-client"
    assert sleeves[1].url == "https://signals.example.com/v1/crash/signal"


def test_budgets_sum_at_one_ok() -> None:
    """Budgets summing to exactly 1.0 are accepted (no cash buffer)."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),
        _remote_sleeve("crash", 0.5, allow_short=True),
    ]
    c = Config(**raw)
    assert sum(s.budget for s in c.sleeves()) == pytest.approx(1.0)


def test_budgets_under_one_ok() -> None:
    """Budgets summing to < 1.0 are accepted (remainder is a cash buffer)."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.3, allow_short=False),
        _remote_sleeve("crash", 0.3, allow_short=True),
    ]
    c = Config(**raw)
    assert sum(s.budget for s in c.sleeves()) == pytest.approx(0.6)


def test_budgets_sum_over_one_rejected() -> None:
    """Budgets summing to > 1.0 must raise a ValueError mentioning budget."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.6, allow_short=False),
        _remote_sleeve("crash", 0.6, allow_short=True),
    ]
    with pytest.raises(ValidationError, match="budget"):
        Config(**raw)


def test_duplicate_sleeve_names_rejected() -> None:
    """Two sleeves with the same name must raise (names key per-sleeve state)."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),
        _remote_sleeve("haarp", 0.5, allow_short=True),
    ]
    with pytest.raises(ValidationError, match="name"):
        Config(**raw)


# ---------------------------------------------------------------------------
# Singular adapter (deployed backward compatibility)
# ---------------------------------------------------------------------------


def test_singular_signal_source_normalizes() -> None:
    """The OLD singular signal_source block → exactly one 'default' sleeve."""
    raw = _raw()  # ships singular signal_source (equal_weight, client-local)
    assert "signal_source" in raw
    assert "signal_sources" not in raw
    c = Config(**raw)
    sleeves = c.sleeves()
    assert len(sleeves) == 1
    only = sleeves[0]
    assert only.name == "default"
    assert only.budget == 1.0
    assert only.allow_short is False
    # adapted sleeve carries the singular block's wire fields verbatim
    assert only.type == c.signal_source.type
    assert only.client_id == c.signal_source.client_id
    assert only.url == c.signal_source.url


def test_singular_signal_source_still_populated() -> None:
    """The singular path keeps cfg.signal_source populated for legacy callers."""
    raw = _raw()
    c = Config(**raw)
    assert c.signal_source is not None
    assert c.signal_source.type == "equal_weight"


def test_both_forms_present_rejected() -> None:
    """Both signal_source AND signal_sources set must raise (exactly-one rule)."""
    raw = _raw()  # keeps singular signal_source
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 1.0, allow_short=False),
    ]
    with pytest.raises(ValidationError, match="signal_source"):
        Config(**raw)


def test_neither_form_present_rejected() -> None:
    """Neither signal_source NOR signal_sources set must raise (exactly-one rule)."""
    raw = _raw()
    raw.pop("signal_source", None)
    with pytest.raises(ValidationError, match="signal_source"):
        Config(**raw)


# ---------------------------------------------------------------------------
# Per-source staleness (falls back to the global default)
# ---------------------------------------------------------------------------


def test_per_source_staleness() -> None:
    """A sleeve with max_signal_staleness_days uses it; otherwise the global."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["max_signal_staleness_days"] = 2  # the global fallback
    crash = _remote_sleeve("crash", 0.5, allow_short=True)
    crash["max_signal_staleness_days"] = 1  # per-source override
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),  # no per-source key
        crash,
    ]
    c = Config(**raw)
    haarp_sleeve, crash_sleeve = c.sleeves()
    assert crash_sleeve.max_signal_staleness_days == 1
    assert haarp_sleeve.max_signal_staleness_days is None
    # effective value: per-source override wins, else the global default
    assert c.effective_staleness_days(crash_sleeve) == 1
    assert c.effective_staleness_days(haarp_sleeve) == 2


# ---------------------------------------------------------------------------
# Per-source model_rev pin (each sleeve consumes a DIFFERENT signal)
# ---------------------------------------------------------------------------


def test_per_source_pinned_model_rev() -> None:
    """A sleeve with pinned_model_rev uses it; otherwise the global.

    Two sleeves consume different signals (HAARP NOGATE rev vs CRASH gated rev),
    so a single global pin can never match both — the per-sleeve override is the
    only way to pin both correctly.
    """
    raw = _raw()
    raw.pop("signal_source", None)
    raw["pinned_model_rev"] = "haarp-rev"  # the global fallback
    crash = _remote_sleeve("crash", 0.5, allow_short=True)
    crash["pinned_model_rev"] = "crash-rev"  # per-source override
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),  # no per-source key
        crash,
    ]
    c = Config(**raw)
    haarp_sleeve, crash_sleeve = c.sleeves()
    assert crash_sleeve.pinned_model_rev == "crash-rev"
    assert haarp_sleeve.pinned_model_rev is None
    # effective value: per-source override wins, else the global default
    assert c.effective_pinned_model_rev(crash_sleeve) == "crash-rev"
    assert c.effective_pinned_model_rev(haarp_sleeve) == "haarp-rev"


def test_pinned_model_rev_no_global_no_override_is_none() -> None:
    """No global pin and no per-sleeve pin ⇒ effective pin is None (accept any)."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw.pop("pinned_model_rev", None)
    raw["signal_sources"] = [_remote_sleeve("haarp", 1.0, allow_short=False)]
    c = Config(**raw)
    (sleeve,) = c.sleeves()
    assert c.effective_pinned_model_rev(sleeve) is None


def test_singular_default_sleeve_inherits_global_pin() -> None:
    """The singular signal_source: path (default sleeve) inherits the top-level
    pin via the fallback — preserving today's NOGATE-TEST defense unchanged."""
    raw = _raw()  # keeps the singular signal_source: block
    raw["pinned_model_rev"] = "prod-rev"
    c = Config(**raw)
    (default_sleeve,) = c.sleeves()
    assert default_sleeve.name == "default"
    assert default_sleeve.pinned_model_rev is None
    assert c.effective_pinned_model_rev(default_sleeve) == "prod-rev"


# ---------------------------------------------------------------------------
# Per-source mainnet-remote rule
# ---------------------------------------------------------------------------


def test_mainnet_remote_per_source() -> None:
    """On mainnet, a remote sleeve missing root_key_fingerprint_ref must raise."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["env"] = "mainnet"
    bad = _remote_sleeve("crash", 0.5, allow_short=True)
    del bad["root_key_fingerprint_ref"]  # missing the pinned key fingerprint
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),
        bad,
    ]
    with pytest.raises(ValidationError, match="root_key_fingerprint_ref"):
        Config(**raw)


def test_mainnet_remote_per_source_ok_when_specified() -> None:
    """On mainnet, fully-specified remote sleeves are accepted."""
    raw = _raw()
    raw.pop("signal_source", None)
    raw["env"] = "mainnet"
    raw["signal_sources"] = [
        _remote_sleeve("haarp", 0.5, allow_short=False),
        _remote_sleeve("crash", 0.5, allow_short=True),
    ]
    c = Config(**raw)
    assert c.env == "mainnet"
    assert [s.name for s in c.sleeves()] == ["haarp", "crash"]


# ---------------------------------------------------------------------------
# New root knob: liquidation_cooldown_days
# ---------------------------------------------------------------------------


def test_liquidation_cooldown_default() -> None:
    """liquidation_cooldown_days must default to 1 (additive, backward-compat)."""
    c = Config.from_yaml(EX)
    assert c.liquidation_cooldown_days == 1


def test_liquidation_cooldown_override(tmp_path: Path) -> None:
    """A liquidation_cooldown_days override in yaml must parse."""
    raw = _raw()
    raw["liquidation_cooldown_days"] = 3
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")
    c = Config.from_yaml(yaml_path)
    assert c.liquidation_cooldown_days == 3
