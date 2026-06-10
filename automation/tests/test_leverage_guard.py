"""automation.tests.test_leverage_guard — the standalone leverage-guard safety tool.

Two layers:

1. PURE PLANNER (``plan_leverage``) — the testable classification core. No IO, no
   client, no config: given the held leverage view ``{coin: {"type","value"}}``
   and a target ``(max_leverage, target_is_cross)`` it sorts every OPEN position
   into ``to_lower`` / ``type_mismatch`` / ``ok``.  These tests pin the EXACT
   rule, including the dual-list (over-value AND mode-wrong) case.

2. IO MAIN — driven with a ``FakeHLClient`` injected via the ``main(argv, *,
   client=...)`` seam (no keystore, no network).  Asserts the WRITE surface:
   ``to_lower`` → ``update_leverage`` recorded keeping the CURRENT mode; only the
   ``--enforce-type`` path may ``market_close``; ``--check`` signs NOTHING.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from automation.scripts import leverage_guard
from automation.scripts.leverage_guard import LeveragePlan, plan_leverage
from automation.tests._fakes import FakeHLClient


# ===========================================================================
# 1. PURE PLANNER — plan_leverage
# ===========================================================================
class TestPlanLeverage:
    def test_over_value_and_mode_wrong_goes_to_BOTH_lists(self) -> None:
        """10x CROSS, target (1, isolated) → to_lower AND type_mismatch.

        Over-value → cap in place (to_lower); mode wrong → still needs the boot
        flatten (type_mismatch).  Listed in BOTH.
        """
        held = {"BTC": {"type": "cross", "value": 10}}
        plan = plan_leverage(held, target_leverage=1, target_is_cross=False)
        assert plan.to_lower == ["BTC"]
        assert plan.type_mismatch == ["BTC"]
        assert plan.ok == []

    def test_value_ok_mode_wrong_is_type_mismatch_only(self) -> None:
        """1x CROSS, target (1, isolated) → type_mismatch only (value ok, mode wrong)."""
        held = {"BTC": {"type": "cross", "value": 1}}
        plan = plan_leverage(held, target_leverage=1, target_is_cross=False)
        assert plan.to_lower == []
        assert plan.type_mismatch == ["BTC"]
        assert plan.ok == []

    def test_over_value_mode_ok_is_to_lower_only(self) -> None:
        """5x ISOLATED, target (1, isolated) → to_lower only (value high, mode ok)."""
        held = {"BTC": {"type": "isolated", "value": 5}}
        plan = plan_leverage(held, target_leverage=1, target_is_cross=False)
        assert plan.to_lower == ["BTC"]
        assert plan.type_mismatch == []
        assert plan.ok == []

    def test_value_ok_mode_ok_is_ok(self) -> None:
        """1x ISOLATED, target (1, isolated) → ok."""
        held = {"BTC": {"type": "isolated", "value": 1}}
        plan = plan_leverage(held, target_leverage=1, target_is_cross=False)
        assert plan.to_lower == []
        assert plan.type_mismatch == []
        assert plan.ok == ["BTC"]

    def test_empty_held_all_empty(self) -> None:
        """No open positions → all three lists empty."""
        plan = plan_leverage({}, target_leverage=1, target_is_cross=False)
        assert plan == LeveragePlan(to_lower=[], type_mismatch=[], ok=[])

    def test_below_value_mode_ok_is_ok(self) -> None:
        """value < target (not just ==) with correct mode → ok (we never RAISE)."""
        held = {"BTC": {"type": "isolated", "value": 1}}
        plan = plan_leverage(held, target_leverage=5, target_is_cross=False)
        assert plan.ok == ["BTC"]
        assert plan.to_lower == []

    def test_target_cross_classifies_against_cross(self) -> None:
        """target_is_cross=True flips the mode check: isolated becomes the mismatch."""
        held = {
            "BTC": {"type": "cross", "value": 1},  # mode ok, value ok → ok
            "ETH": {"type": "isolated", "value": 3},  # mode wrong + over → both
        }
        plan = plan_leverage(held, target_leverage=1, target_is_cross=True)
        assert plan.ok == ["BTC"]
        assert plan.to_lower == ["ETH"]
        assert plan.type_mismatch == ["ETH"]

    def test_lists_are_deterministically_ordered_by_held_iteration(self) -> None:
        """Multiple coins land in stable insertion order (dict iteration order)."""
        held = {
            "BTC": {"type": "isolated", "value": 5},  # to_lower
            "ETH": {"type": "isolated", "value": 1},  # ok
            "SOL": {"type": "cross", "value": 8},  # both
        }
        plan = plan_leverage(held, target_leverage=1, target_is_cross=False)
        assert plan.to_lower == ["BTC", "SOL"]
        assert plan.type_mismatch == ["SOL"]
        assert plan.ok == ["ETH"]


# ===========================================================================
# Config helper (master-direct testnet) — minimal, on disk.
# ===========================================================================
def _write_cfg_yaml(
    tmp_path: Path,
    *,
    target_leverage: int = 1,
    leverage_is_cross: bool = False,
) -> Path:
    raw: dict[str, Any] = {
        "env": "testnet",
        "trade_mode": "master_direct",
        "subaccount_address": "0x" + "0" * 40,
        "signal_source": {"type": "equal_weight", "client_id": "lev-guard"},
        "strategy_id": "haarp-multi",
        "rebalance_utc_time": "00:15",
        "threshold_pct": 3.0,
        "max_per_asset": 0.9,
        "safety_buffer": 0.03,
        "max_signal_staleness_days": 2,
        "target_leverage": target_leverage,
        "leverage_is_cross": leverage_is_cross,
        "universe_perp_map": {c: c for c in ("BTC", "ETH", "SOL")},
        "tradeable_coins": ["BTC", "ETH", "SOL"],
        "spot_routing": {},
        "caps": {
            "max_notional_per_coin": 10.0,
            "max_total_notional": 50.0,
            "max_turnover_per_cycle": 60.0,
            "max_order_slippage": 0.01,
        },
        "nonce_state_path": str(tmp_path / "nonce.state"),
        "state_path": str(tmp_path / "state.json"),
        "ledger_path": str(tmp_path / "ledger.json"),
        "audit_path": str(tmp_path / "audit.jsonl"),
    }
    p = tmp_path / "automation.yaml"
    p.write_text(yaml.dump(raw), encoding="utf-8")
    return p


# ===========================================================================
# 2. IO MAIN — injected FakeHLClient
# ===========================================================================
class TestLeverageGuardMain:
    def test_to_lower_calls_update_leverage_keeping_current_mode(
        self, tmp_path: Path
    ) -> None:
        """An over-value position is capped IN PLACE: update_leverage with the
        CURRENT mode (is_cross from held), NO market_close."""
        cfg_path = _write_cfg_yaml(tmp_path, target_leverage=1, leverage_is_cross=False)
        client = FakeHLClient(
            positions_leverage={"BTC": {"type": "cross", "value": 10}},
        )
        rc = leverage_guard.main(["--config", str(cfg_path)], client=client)
        assert rc == 0
        # update_leverage(coin, max_leverage, is_cross=CURRENT mode=cross=True)
        assert client.leverage_calls == [("BTC", 1, True)]
        # Without --enforce-type, the mode-mismatch is REPORTED ONLY — no flatten.
        assert client.market_close_calls == []
        assert client.closed is True

    def test_isolated_over_value_keeps_isolated_mode(self, tmp_path: Path) -> None:
        """5x ISOLATED, target (1, isolated) → update_leverage with is_cross=False."""
        cfg_path = _write_cfg_yaml(tmp_path)
        client = FakeHLClient(
            positions_leverage={"BTC": {"type": "isolated", "value": 5}},
        )
        rc = leverage_guard.main(["--config", str(cfg_path)], client=client)
        assert rc == 0
        assert client.leverage_calls == [("BTC", 1, False)]
        assert client.market_close_calls == []

    def test_type_mismatch_without_enforce_does_NOT_flatten_or_lower(
        self, tmp_path: Path
    ) -> None:
        """1x CROSS, target (1, isolated): value ok, mode wrong → type_mismatch ONLY.

        Without --enforce-type the tool MUST NOT flatten AND MUST NOT call
        update_leverage (the value is already fine; only the mode is wrong)."""
        cfg_path = _write_cfg_yaml(tmp_path)
        client = FakeHLClient(
            positions_leverage={"BTC": {"type": "cross", "value": 1}},
        )
        rc = leverage_guard.main(["--config", str(cfg_path)], client=client)
        assert rc == 0
        assert client.market_close_calls == []
        assert client.leverage_calls == []

    def test_type_mismatch_with_enforce_flattens_reduce_only(
        self, tmp_path: Path
    ) -> None:
        """--enforce-type flattens a mode-wrong position via reduce-only market_close."""
        cfg_path = _write_cfg_yaml(tmp_path)
        client = FakeHLClient(
            positions={"BTC": 0.5},
            positions_leverage={"BTC": {"type": "cross", "value": 1}},
        )
        rc = leverage_guard.main(
            ["--config", str(cfg_path), "--enforce-type"], client=client
        )
        assert rc == 0
        assert len(client.market_close_calls) == 1
        assert client.market_close_calls[0]["coin"] == "BTC"

    def test_check_signs_nothing(self, tmp_path: Path) -> None:
        """--check is read-only: NO update_leverage, NO market_close, exit 0."""
        cfg_path = _write_cfg_yaml(tmp_path)
        client = FakeHLClient(
            positions={"BTC": 0.5},
            positions_leverage={
                "BTC": {"type": "cross", "value": 10},  # over AND mode-wrong
            },
        )
        rc = leverage_guard.main(
            ["--config", str(cfg_path), "--check", "--enforce-type"], client=client
        )
        assert rc == 0
        assert client.leverage_calls == []
        assert client.market_close_calls == []
        assert client.closed is True

    def test_max_leverage_overrides_config_target(self, tmp_path: Path) -> None:
        """--max-leverage caps to the passed value, not cfg.target_leverage."""
        cfg_path = _write_cfg_yaml(tmp_path, target_leverage=1)
        client = FakeHLClient(
            positions_leverage={"BTC": {"type": "isolated", "value": 10}},
        )
        rc = leverage_guard.main(
            ["--config", str(cfg_path), "--max-leverage", "3"], client=client
        )
        assert rc == 0
        # Capped to 3 (not cfg's 1), mode kept isolated.
        assert client.leverage_calls == [("BTC", 3, False)]

    def test_per_coin_update_failure_is_isolated(self, tmp_path: Path) -> None:
        """A failing update_leverage on one coin does not abort the others."""
        cfg_path = _write_cfg_yaml(tmp_path)

        class _BoomOnBTC(FakeHLClient):
            def update_leverage(
                self, coin: str, leverage: int, *, is_cross: bool
            ) -> dict[str, Any]:
                if coin == "BTC":
                    raise RuntimeError("boom")
                resp: dict[str, Any] = super().update_leverage(
                    coin, leverage, is_cross=is_cross
                )
                return resp

        client = _BoomOnBTC(
            positions_leverage={
                "BTC": {"type": "isolated", "value": 5},
                "ETH": {"type": "isolated", "value": 5},
            },
        )
        rc = leverage_guard.main(["--config", str(cfg_path)], client=client)
        assert rc == 0
        # ETH still got its cap despite BTC blowing up.
        assert ("ETH", 1, False) in client.leverage_calls
        assert client.closed is True

    def test_client_closed_even_on_read_error(self, tmp_path: Path) -> None:
        """positions_leverage raising still releases the nonce lock (finally)."""
        cfg_path = _write_cfg_yaml(tmp_path)

        class _BoomRead(FakeHLClient):
            def positions_leverage(self) -> dict[str, dict[str, Any]]:
                raise RuntimeError("read boom")

        client = _BoomRead()
        with pytest.raises(RuntimeError, match="read boom"):
            leverage_guard.main(["--config", str(cfg_path)], client=client)
        assert client.closed is True

    def test_build_parser_exposes_flags(self) -> None:
        """The parser wires --config/--check/--max-leverage/--enforce-type."""
        parser = leverage_guard._build_parser()
        args = parser.parse_args(
            ["--config", "x.yaml", "--check", "--max-leverage", "2", "--enforce-type"]
        )
        assert args.config == "x.yaml"
        assert args.check is True
        assert args.max_leverage == 2
        assert args.enforce_type is True
