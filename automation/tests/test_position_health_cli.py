"""automation.tests.test_position_health_cli — the standalone position-health monitor.

The CLI is a READ-ONLY monitor: it gathers a health snapshot, runs the pure
``assess`` checks, prints a per-check report, and routes every finding to the
alert sink (so the webhook fires).  It NEVER signs or trades.

Driven with a ``FakeHLClient`` injected via the ``main(argv, *, client=...)``
seam (no keystore, no network) and a ``--config`` written to disk.  The sink is
monkeypatched onto a recorder so the routed findings can be asserted.

Invariants pinned here:
* a breaching account → every finding routed to the sink + the report printed.
* ``health.enabled=False`` → "health monitor disabled", NO findings emitted.
* READ-ONLY: NO ``update_leverage`` / ``market_close`` / ``submit_ioc`` call.
* the client is ALWAYS closed in a ``finally`` (releases the nonce lock), even
  when the snapshot read raises.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from automation.reporting.alerts import Alert, AlertType
from automation.scripts import position_health
from automation.tests._fakes import FakeHLClient


# ---------------------------------------------------------------------------
# Recording sink + config-on-disk helpers
# ---------------------------------------------------------------------------
class _RecordingSink:
    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


def _write_cfg_yaml(tmp_path: Path, *, health_enabled: bool = True) -> Path:
    raw: dict[str, Any] = {
        "env": "testnet",
        "trade_mode": "master_direct",
        "subaccount_address": "0x" + "0" * 40,
        "signal_source": {"type": "equal_weight", "client_id": "pos-health"},
        "strategy_id": "haarp-multi",
        "rebalance_utc_time": "00:15",
        "threshold_pct": 3.0,
        "max_per_asset": 0.9,
        "safety_buffer": 0.03,
        "max_signal_staleness_days": 2,
        "target_leverage": 1,
        "leverage_is_cross": False,
        "universe_perp_map": {c: c for c in ("BTC", "ETH", "SOL")},
        "tradeable_coins": ["BTC", "ETH", "SOL"],
        "spot_routing": {},
        "caps": {
            "max_notional_per_coin": 10.0,
            "max_total_notional": 50.0,
            "max_turnover_per_cycle": 60.0,
            "max_order_slippage": 0.01,
        },
        "health": {
            "enabled": health_enabled,
            "funding_drag_pct": 5.0,
            "margin_maint_frac_warn": 0.5,
            "liq_distance_pct_warn": 10.0,
            "spot_collateral_warn_usd": 1.0,
        },
        "nonce_state_path": str(tmp_path / "nonce.state"),
        "state_path": str(tmp_path / "state.json"),
        "ledger_path": str(tmp_path / "ledger.json"),
        "audit_path": str(tmp_path / "audit.jsonl"),
    }
    p = tmp_path / "automation.yaml"
    p.write_text(yaml.dump(raw), encoding="utf-8")
    return p


def _breaching_client() -> FakeHLClient:
    """A FakeHLClient whose account/venue/spot breach every health check."""
    return FakeHLClient(
        account_snapshot={
            "equity": 1000.0,
            "maint_margin": 600.0,  # 0.6 > 0.5 → MARGIN_RATIO_LOW
            "positions": {
                "BTC": {
                    "szi": 0.5,
                    "position_value": 30000.0,
                    "unrealized_pnl": 0.0,
                    "liquidation_px": 97.0,  # 3% from mark=100 → CRITICAL margin
                    "funding_since_open": 60.0,  # 6% of equity → FUNDING_DRAG
                    "leverage_type": "cross",
                    "leverage_value": 3,
                },
            },
        },
        marks={"BTC": Decimal("100")},
        venue_meta={},  # BTC absent → COIN_VENUE_DEGRADED
        spot_balances={"USDC": Decimal("5.0")},  # 5 > 1 → COLLATERAL_IN_SPOT
    )


# ---------------------------------------------------------------------------
# 1. breaching account → findings routed + report printed
# ---------------------------------------------------------------------------
def test_breaching_account_routes_all_findings(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    cfg_path = _write_cfg_yaml(tmp_path, health_enabled=True)
    sink = _RecordingSink()
    monkeypatch.setattr(position_health, "_build_alert_sink", lambda _cfg: sink)

    client = _breaching_client()
    rc = position_health.main(["--config", str(cfg_path)], client=client)
    assert rc == 0

    types = {a.type for a in sink.received}
    assert AlertType.FUNDING_DRAG in types
    assert AlertType.MARGIN_RATIO_LOW in types
    assert AlertType.COIN_VENUE_DEGRADED in types
    assert AlertType.COLLATERAL_IN_SPOT in types

    out = capsys.readouterr().out
    assert "position_health" in out
    # a clear per-check report and the findings are printed
    assert "FUNDING" in out.upper()
    assert client.closed is True


def test_breaching_account_is_read_only(tmp_path: Path, monkeypatch: Any) -> None:
    """The monitor NEVER signs: no update_leverage / market_close / submit_ioc."""
    cfg_path = _write_cfg_yaml(tmp_path, health_enabled=True)
    monkeypatch.setattr(position_health, "_build_alert_sink", lambda _cfg: _RecordingSink())

    client = _breaching_client()
    rc = position_health.main(["--config", str(cfg_path)], client=client)
    assert rc == 0
    assert client.leverage_calls == []
    assert client.market_close_calls == []
    assert client.submit_ioc_calls == []


# ---------------------------------------------------------------------------
# 2. disabled → nothing emitted
# ---------------------------------------------------------------------------
def test_disabled_emits_nothing(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    cfg_path = _write_cfg_yaml(tmp_path, health_enabled=False)
    sink = _RecordingSink()
    monkeypatch.setattr(position_health, "_build_alert_sink", lambda _cfg: sink)

    client = _breaching_client()
    rc = position_health.main(["--config", str(cfg_path)], client=client)
    assert rc == 0
    assert sink.received == []
    out = capsys.readouterr().out
    assert "disabled" in out.lower()
    # disabled path is a pure no-op: nothing read, nothing signed.
    assert client.leverage_calls == []
    assert client.market_close_calls == []


# ---------------------------------------------------------------------------
# 3. clean account → no findings emitted
# ---------------------------------------------------------------------------
def test_clean_account_no_findings(tmp_path: Path, monkeypatch: Any) -> None:
    cfg_path = _write_cfg_yaml(tmp_path, health_enabled=True)
    sink = _RecordingSink()
    monkeypatch.setattr(position_health, "_build_alert_sink", lambda _cfg: sink)

    client = FakeHLClient(
        account_snapshot={
            "equity": 1000.0,
            "maint_margin": 100.0,  # 0.1 < 0.5
            "positions": {
                "BTC": {
                    "szi": 0.5,
                    "position_value": 30000.0,
                    "unrealized_pnl": 0.0,
                    "liquidation_px": 50.0,  # 50% from mark=100 → fine
                    "funding_since_open": 1.0,  # 0.1% → fine
                    "leverage_type": "cross",
                    "leverage_value": 3,
                },
            },
        },
        marks={"BTC": Decimal("100")},
        venue_meta={"BTC": {"delisted": False, "only_isolated": False, "max_leverage": 50}},
        spot_balances={"USDC": Decimal("0.1")},
    )
    rc = position_health.main(["--config", str(cfg_path)], client=client)
    assert rc == 0
    assert sink.received == []
    assert client.closed is True


# ---------------------------------------------------------------------------
# 4. client closed even on read error (finally releases nonce lock)
# ---------------------------------------------------------------------------
def test_client_closed_even_on_read_error(tmp_path: Path, monkeypatch: Any) -> None:
    cfg_path = _write_cfg_yaml(tmp_path, health_enabled=True)
    monkeypatch.setattr(position_health, "_build_alert_sink", lambda _cfg: _RecordingSink())

    class _BoomRead(FakeHLClient):
        def account_snapshot(self) -> dict[str, Any]:
            raise RuntimeError("read boom")

    client = _BoomRead()
    with pytest.raises(RuntimeError, match="read boom"):
        position_health.main(["--config", str(cfg_path)], client=client)
    assert client.closed is True


def test_build_parser_requires_config() -> None:
    parser = position_health._build_parser()
    args = parser.parse_args(["--config", "x.yaml"])
    assert args.config == "x.yaml"
    with pytest.raises(SystemExit):
        parser.parse_args([])
