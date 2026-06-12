"""
Tests for automation.core.config.

Written before the implementation (TDD RED → GREEN).
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from automation.core.config import Config

# Path to the neutral example config shipped with the package
EX = Path(__file__).resolve().parents[1] / "config" / "automation.example.yaml"


def test_example_file_exists() -> None:
    """The example yaml must be present and readable."""
    assert EX.exists(), f"Example config not found at {EX}"


def test_loads_example() -> None:
    """Config.from_yaml must load the neutral example without error."""
    c = Config.from_yaml(EX)
    assert c.env == "testnet"
    assert c.threshold_pct == 3.0
    assert c.safety_buffer == 0.03
    # Neutral example ships only BTC + ETH, no real HAARP universe
    assert set(c.universe_perp_map) == {"BTC", "ETH"}


def test_example_defaults() -> None:
    """Spot-check all top-level fields are present with expected types."""
    c = Config.from_yaml(EX)
    assert c.rebalance_utc_time == "00:15"
    assert 0.0 < c.max_per_asset <= 1.0
    assert c.max_signal_staleness_days == 2
    assert isinstance(c.universe_perp_map, dict)


def test_caps_model() -> None:
    """Caps nested model must parse correctly from example yaml."""
    c = Config.from_yaml(EX)
    assert c.caps.max_notional_per_coin == 10_000.0
    assert c.caps.max_total_notional == 50_000.0
    assert c.caps.max_turnover_per_cycle == 60_000.0
    assert 0.0 < c.caps.max_order_slippage < 1.0


def test_signal_source_equal_weight() -> None:
    """Example ships equal_weight signal_source; url must default to None."""
    c = Config.from_yaml(EX)
    assert c.signal_source.type == "equal_weight"
    assert c.signal_source.url is None
    assert c.signal_source.client_id == "client-local"


def test_mainnet_remote_requires_pinned_key() -> None:
    """mainnet + remote must raise if url or root_key_fingerprint_ref is missing."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "mainnet"
    raw["signal_source"] = {"type": "remote", "client_id": "c1"}
    # url and root_key_fingerprint_ref are both absent → must fail validation
    with pytest.raises(ValidationError):
        Config(**raw)


def test_mainnet_remote_valid_when_fully_specified() -> None:
    """mainnet + remote must NOT raise when url and fingerprint_ref are both set."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "mainnet"
    raw["signal_source"] = {
        "type": "remote",
        "url": "https://signals.example.com/allocations",
        "root_key_fingerprint_ref": "HL_SIGNAL_ROOT_FP",
        "client_id": "c1",
    }
    c = Config(**raw)
    assert c.env == "mainnet"
    assert c.signal_source.url == "https://signals.example.com/allocations"


def test_mainnet_non_remote_does_not_require_key() -> None:
    """mainnet + equal_weight must succeed without url/fingerprint_ref."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "mainnet"
    # signal_source stays equal_weight from example
    c = Config(**raw)
    assert c.env == "mainnet"
    assert c.signal_source.type == "equal_weight"


# ---------------------------------------------------------------------------
# G5 / S-12: agent_valid_until_ms field
# ---------------------------------------------------------------------------


def test_agent_valid_until_ms_defaults_none() -> None:
    """agent_valid_until_ms must default to None (additive, backward-compatible)."""
    c = Config.from_yaml(EX)
    assert c.agent_valid_until_ms is None


def test_agent_valid_until_ms_parses_from_yaml(tmp_path: Path) -> None:
    """agent_valid_until_ms must parse from yaml when present."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["agent_valid_until_ms"] = 9_999_999_999_000  # far future ms timestamp
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.agent_valid_until_ms == 9_999_999_999_000


# ---------------------------------------------------------------------------
# G4a: divergence_alert_pct field
# ---------------------------------------------------------------------------


def test_divergence_alert_pct_defaults_5() -> None:
    """divergence_alert_pct must default to 5.0 (additive, backward-compatible)."""
    c = Config.from_yaml(EX)
    assert c.divergence_alert_pct == 5.0


def test_divergence_alert_pct_parses_from_yaml(tmp_path: Path) -> None:
    """divergence_alert_pct must parse from yaml when present."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["divergence_alert_pct"] = 10.0
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.divergence_alert_pct == 10.0


# ---------------------------------------------------------------------------
# D5: spot_routing field + keyspace guard (red-team finding #8)
# ---------------------------------------------------------------------------


def test_spot_routing_defaults_empty() -> None:
    """spot_routing must default to {} (additive, perp-only, backward-compatible)."""
    c = Config.from_yaml(EX)
    assert c.spot_routing == {}
    assert isinstance(c.spot_routing, dict)


def test_spot_routing_parses_from_yaml(tmp_path: Path) -> None:
    """spot_routing must parse a coin→spot-pair map from yaml."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["spot_routing"] = {"HYPE": "@107", "ETH": "@151"}
    # The spot coins must be IDENTITY in universe_perp_map (keyspace guard).
    raw["universe_perp_map"] = {"BTC": "BTC", "ETH": "ETH", "HYPE": "HYPE"}
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.spot_routing == {"HYPE": "@107", "ETH": "@151"}


def test_spot_routing_identity_mapped_passes() -> None:
    """A spot coin that is IDENTITY in universe_perp_map passes the keyspace guard."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["spot_routing"] = {"HYPE": "@107"}
    raw["universe_perp_map"] = {"BTC": "BTC", "ETH": "ETH", "HYPE": "HYPE"}
    c = Config(**raw)
    assert c.spot_routing == {"HYPE": "@107"}


def test_spot_routing_absent_from_perp_map_passes() -> None:
    """A spot coin ABSENT from universe_perp_map is identity by fallback → passes.

    ``universe_perp_map.get(coin, coin) == coin`` is True when the coin is absent,
    so a spot coin that is not listed in the perp map is still identity-routed.
    """
    raw: dict = yaml.safe_load(EX.read_text())
    raw["spot_routing"] = {"HYPE": "@107"}
    raw["universe_perp_map"] = {"BTC": "BTC", "ETH": "ETH"}  # HYPE absent
    c = Config(**raw)
    assert c.spot_routing == {"HYPE": "@107"}


def test_spot_routing_non_identity_mapped_raises() -> None:
    """A spot coin remapped to a DIFFERENT perp name must raise (finding #8).

    If HYPE → HYPE-PERP in universe_perp_map, the sizer keys its order by the
    post-map perp name (HYPE-PERP) while spot_routing keys by HYPE → the executor
    would silently route HYPE to perp.  Reject the config at load time.
    """
    raw: dict = yaml.safe_load(EX.read_text())
    raw["spot_routing"] = {"HYPE": "@107"}
    raw["universe_perp_map"] = {"BTC": "BTC", "ETH": "ETH", "HYPE": "HYPE-PERP"}
    with pytest.raises(ValidationError, match="spot_routing"):
        Config(**raw)


def test_spot_routing_non_identity_from_yaml_raises(tmp_path: Path) -> None:
    """The keyspace guard fires through from_yaml as well as direct construction."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["spot_routing"] = {"HYPE": "@107"}
    raw["universe_perp_map"] = {"BTC": "BTC", "HYPE": "HYPE-PERP"}
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="spot_routing"):
        Config.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# trade_mode field (master-direct routing — fresh accounts with no subaccount)
# ---------------------------------------------------------------------------


def test_trade_mode_defaults_subaccount() -> None:
    """trade_mode must default to 'subaccount' (additive, backward-compatible)."""
    c = Config.from_yaml(EX)
    assert c.trade_mode == "subaccount"


def test_trade_mode_master_direct_parses_from_yaml(tmp_path: Path) -> None:
    """trade_mode: master_direct must parse from yaml."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["trade_mode"] = "master_direct"
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.trade_mode == "master_direct"


def test_trade_mode_master_direct_direct_construction() -> None:
    """trade_mode: master_direct must parse via direct construction too."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["trade_mode"] = "master_direct"
    c = Config(**raw)
    assert c.trade_mode == "master_direct"


def test_trade_mode_invalid_raises() -> None:
    """An invalid trade_mode value must raise (Literal validation)."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["trade_mode"] = "nonsense"
    with pytest.raises(ValidationError):
        Config(**raw)


# ---------------------------------------------------------------------------
# disable_gate field (TEST-ONLY ungated HAARP — testnet functional testing)
# ---------------------------------------------------------------------------


def test_disable_gate_defaults_false() -> None:
    """disable_gate must default to False (additive, production path byte-identical)."""
    c = Config.from_yaml(EX)
    assert c.disable_gate is False


def test_disable_gate_testnet_parses_true(tmp_path: Path) -> None:
    """env: testnet + disable_gate: true must parse (the supported test mode)."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "testnet"
    raw["disable_gate"] = True
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.env == "testnet"
    assert c.disable_gate is True


def test_disable_gate_testnet_direct_construction() -> None:
    """env: testnet + disable_gate: true parses via direct construction too."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "testnet"
    raw["disable_gate"] = True
    c = Config(**raw)
    assert c.disable_gate is True


def test_disable_gate_mainnet_rejected() -> None:
    """env: mainnet + disable_gate: true must raise (ungated mode forbidden on mainnet)."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "mainnet"
    raw["disable_gate"] = True
    with pytest.raises(ValidationError, match="disable_gate"):
        Config(**raw)


def test_disable_gate_mainnet_rejected_from_yaml(tmp_path: Path) -> None:
    """The mainnet disable_gate refusal fires through from_yaml as well."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "mainnet"
    raw["disable_gate"] = True
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="disable_gate"):
        Config.from_yaml(yaml_path)


def test_disable_gate_false_on_mainnet_ok() -> None:
    """env: mainnet + disable_gate: false (the default) must be accepted."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["env"] = "mainnet"
    raw["disable_gate"] = False
    c = Config(**raw)
    assert c.env == "mainnet"
    assert c.disable_gate is False


# ---------------------------------------------------------------------------
# Bounded intra-cycle retry (Task 2 — RetryConfig block)
# ---------------------------------------------------------------------------


def test_retry_config_defaults() -> None:
    """RetryConfig must be instantiable standalone with correct defaults."""
    from automation.core.config import RetryConfig

    rc = RetryConfig()
    assert rc.enabled is True
    assert rc.interval_seconds == 600
    assert rc.window_seconds == 10800
    assert rc.liquidity_safety == 1.0


def test_config_has_retry_block_with_defaults() -> None:
    """Config.retry must exist and carry the correct defaults when not set in yaml."""
    raw: dict = yaml.safe_load(EX.read_text())
    c = Config(**raw)
    assert c.retry.enabled is True
    assert c.retry.interval_seconds == 600
    assert c.retry.window_seconds == 10800


def test_config_loads_retry_block_from_yaml(tmp_path: Path) -> None:
    """A retry: block in the YAML must thread through Config.from_yaml unchanged."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["retry"] = {
        "enabled": True,
        "interval_seconds": 300,
        "window_seconds": 7200,
        "liquidity_safety": 1.2,
    }
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.retry.enabled is True
    assert cfg.retry.interval_seconds == 300
    assert cfg.retry.window_seconds == 7200
    assert cfg.retry.liquidity_safety == 1.2


def test_config_without_retry_block_uses_defaults(tmp_path: Path) -> None:
    """A YAML with NO retry: key must default to the RetryConfig defaults via from_yaml."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw.pop("retry", None)  # ensure the example does not carry a retry block
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.retry.enabled is True
    assert cfg.retry.interval_seconds == 600
    assert cfg.retry.window_seconds == 10800
    assert cfg.retry.liquidity_safety == 1.0


# ---------------------------------------------------------------------------
# Boot leverage knobs (target_leverage / leverage_is_cross / set_leverage_on_boot)
# ---------------------------------------------------------------------------


def test_leverage_knobs_default() -> None:
    """The boot-leverage knobs must default to 1 / False / True (HAARP 1x isolated)."""
    c = Config.from_yaml(EX)
    assert c.target_leverage == 1
    assert c.leverage_is_cross is False
    assert c.set_leverage_on_boot is True


def test_leverage_knobs_override_from_yaml(tmp_path: Path) -> None:
    """A yaml WITH the leverage knobs must override the defaults."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["target_leverage"] = 5
    raw["leverage_is_cross"] = True
    raw["set_leverage_on_boot"] = False
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.target_leverage == 5
    assert c.leverage_is_cross is True
    assert c.set_leverage_on_boot is False


def test_leverage_knobs_absent_uses_defaults(tmp_path: Path) -> None:
    """A yaml WITHOUT the leverage knobs must fall back to the defaults."""
    raw: dict = yaml.safe_load(EX.read_text())
    for k in ("target_leverage", "leverage_is_cross", "set_leverage_on_boot"):
        raw.pop(k, None)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.target_leverage == 1
    assert c.leverage_is_cross is False
    assert c.set_leverage_on_boot is True


# ---------------------------------------------------------------------------
# enforce_leverage_on_drift flag (boot flatten on leverage-TYPE mismatch)
# ---------------------------------------------------------------------------


def test_enforce_leverage_on_drift_defaults_true() -> None:
    """enforce_leverage_on_drift must default to True (additive, backward-compatible)."""
    c = Config.from_yaml(EX)
    assert c.enforce_leverage_on_drift is True


def test_enforce_leverage_on_drift_override_from_yaml(tmp_path: Path) -> None:
    """enforce_leverage_on_drift: false in yaml must override the default."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["enforce_leverage_on_drift"] = False
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.enforce_leverage_on_drift is False


# ---------------------------------------------------------------------------
# PH-1: HealthConfig block
# ---------------------------------------------------------------------------


def test_health_config_defaults() -> None:
    """HealthConfig must be instantiable standalone with correct defaults."""
    from automation.core.config import HealthConfig

    hc = HealthConfig()
    assert hc.enabled is True
    assert hc.funding_drag_pct == 5.0
    assert hc.margin_maint_frac_warn == 0.5
    assert hc.liq_distance_pct_warn == 10.0
    assert hc.spot_collateral_warn_usd == 1.0


def test_config_has_health_block_with_defaults() -> None:
    """Config.health must exist and carry the correct defaults when not set in yaml."""
    raw: dict = yaml.safe_load(EX.read_text())
    c = Config(**raw)
    assert c.health.enabled is True
    assert c.health.funding_drag_pct == 5.0
    assert c.health.margin_maint_frac_warn == 0.5
    assert c.health.liq_distance_pct_warn == 10.0
    assert c.health.spot_collateral_warn_usd == 1.0


def test_config_loads_health_block_from_yaml(tmp_path: Path) -> None:
    """A health: block in the YAML must thread through Config.from_yaml unchanged."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["health"] = {
        "enabled": False,
        "funding_drag_pct": 3.0,
        "margin_maint_frac_warn": 0.75,
        "liq_distance_pct_warn": 15.0,
        "spot_collateral_warn_usd": 50.0,
    }
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.health.enabled is False
    assert cfg.health.funding_drag_pct == 3.0
    assert cfg.health.margin_maint_frac_warn == 0.75
    assert cfg.health.liq_distance_pct_warn == 15.0
    assert cfg.health.spot_collateral_warn_usd == 50.0


def test_config_without_health_block_uses_defaults(tmp_path: Path) -> None:
    """A YAML with NO health: key must default to the HealthConfig defaults via from_yaml."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw.pop("health", None)  # ensure no health block in this yaml
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.health.enabled is True
    assert cfg.health.funding_drag_pct == 5.0
    assert cfg.health.margin_maint_frac_warn == 0.5
    assert cfg.health.liq_distance_pct_warn == 10.0
    assert cfg.health.spot_collateral_warn_usd == 1.0


# ---------------------------------------------------------------------------
# key_source (KeySourceConfig) — env-raw agent key for unattended runs
# ---------------------------------------------------------------------------
def test_key_source_defaults_keystore() -> None:
    """key_source must default to type=keystore / env_var=HL_AGENT_KEY."""
    c = Config.from_yaml(EX)
    assert c.key_source.type == "keystore"
    assert c.key_source.env_var == "HL_AGENT_KEY"


def test_key_source_env_raw_parses_from_yaml(tmp_path: Path) -> None:
    """A yaml with key_source: {type: env_raw, env_var: FOO} must parse."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["key_source"] = {"type": "env_raw", "env_var": "FOO"}
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.key_source.type == "env_raw"
    assert c.key_source.env_var == "FOO"


def test_key_source_absent_block_uses_keystore_default(tmp_path: Path) -> None:
    """A YAML with NO key_source key must default to the keystore KeySourceConfig."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw.pop("key_source", None)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    c = Config.from_yaml(yaml_path)
    assert c.key_source.type == "keystore"
    assert c.key_source.env_var == "HL_AGENT_KEY"


def test_key_source_invalid_type_raises() -> None:
    """An unknown key_source type must be rejected by pydantic."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["key_source"] = {"type": "bogus"}
    with pytest.raises(ValidationError):
        Config(**raw)


# ---------------------------------------------------------------------------
# Task 3: data_dir — volume-mount path prefix for relative state-file paths
# ---------------------------------------------------------------------------


def test_data_dir_prefixes_relative_state_paths() -> None:
    """data_dir must be prepended to every relative state-file path at construction."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["data_dir"] = "/data"
    raw["state_path"] = "state.json"
    raw["nonce_state_path"] = "nonce.state"
    raw["ledger_path"] = "ledger.json"
    raw["audit_path"] = "audit.jsonl"
    cfg = Config(**raw)
    assert cfg.state_path == "/data/state.json"
    assert cfg.nonce_state_path == "/data/nonce.state"
    assert cfg.ledger_path == "/data/ledger.json"
    assert cfg.audit_path == "/data/audit.jsonl"


def test_absolute_state_paths_not_prefixed() -> None:
    """Absolute state-file paths must pass through unchanged even when data_dir is set."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["data_dir"] = "/data"
    raw["state_path"] = "/custom/state.json"
    cfg = Config(**raw)
    assert cfg.state_path == "/custom/state.json"


def test_no_data_dir_leaves_paths_unchanged() -> None:
    """When data_dir is absent, state-file paths must be left exactly as specified."""
    raw: dict = yaml.safe_load(EX.read_text())
    # ensure no data_dir key is present
    raw.pop("data_dir", None)
    raw["state_path"] = "state.json"
    cfg = Config(**raw)
    assert cfg.state_path == "state.json"


def test_data_dir_prefixes_kill_flag_when_relative() -> None:
    """data_dir must also prefix a relative kill_flag_path."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["data_dir"] = "/data"
    raw["kill_flag_path"] = "KILL"
    cfg = Config(**raw)
    assert cfg.kill_flag_path == "/data/KILL"


def test_data_dir_kill_flag_none_is_untouched() -> None:
    """kill_flag_path=None must remain None even when data_dir is set."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["data_dir"] = "/data"
    raw["kill_flag_path"] = None
    cfg = Config(**raw)
    assert cfg.kill_flag_path is None


def test_data_dir_trailing_slash_normalized() -> None:
    """data_dir with a trailing slash must not produce a double-slash in the result."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["data_dir"] = "/data/"
    raw["state_path"] = "state.json"
    cfg = Config(**raw)
    assert cfg.state_path == "/data/state.json"


def test_health_http_defaults() -> None:
    """HealthConfig ships the /healthz liveness defaults (port 8080, 180 s silence)."""
    c = Config.from_yaml(EX)
    assert c.health.http_port == 8080
    assert c.health.heartbeat_max_silence_seconds == 180


def test_health_http_overrides_parse() -> None:
    """A yaml override of the liveness fields must parse onto HealthConfig."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["health"] = {"http_port": 9000, "heartbeat_max_silence_seconds": 300}
    cfg = Config(**raw)
    assert cfg.health.http_port == 9000
    assert cfg.health.heartbeat_max_silence_seconds == 300


# ---------------------------------------------------------------------------
# SP3: SignalRecheckConfig block (no-signal re-fire window)
# ---------------------------------------------------------------------------


def test_defaults_when_block_absent(tmp_path: Path) -> None:
    """A YAML with NO signal_recheck key must default to the SignalRecheckConfig defaults."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw.pop("signal_recheck", None)  # ensure no block in this yaml
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.signal_recheck.enabled is True
    assert cfg.signal_recheck.interval_seconds == 180
    assert cfg.signal_recheck.window_seconds == 7200


def test_block_overrides(tmp_path: Path) -> None:
    """A signal_recheck: block in the YAML must thread through Config.from_yaml unchanged."""
    raw: dict = yaml.safe_load(EX.read_text())
    raw["signal_recheck"] = {
        "enabled": False,
        "interval_seconds": 60,
        "window_seconds": 600,
    }
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.signal_recheck.enabled is False
    assert cfg.signal_recheck.interval_seconds == 60
    assert cfg.signal_recheck.window_seconds == 600
