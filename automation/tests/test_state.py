"""
Tests for automation.reporting.state — persistent executor state.

Coverage:
- round-trip (save + load returns identical State)
- missing file → fresh defaults
- corrupt file → StateError (never silently reset)
- atomic-replace leaves no .tmp behind
- saved file is valid JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automation.reporting.state import State, StateError, load, save

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    target: dict[str, float] | None = None,
    as_of: str | None = None,
) -> State:
    return State(
        last_rebalanced_target=target,
        last_rebalanced_as_of=as_of,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStateLoad:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """A missing state file yields a fresh State with all defaults."""
        path = tmp_path / "state.json"
        state = load(path)

        assert state.last_rebalanced_target is None
        assert state.last_rebalanced_as_of is None
        assert state.schema_version == 1

    def test_corrupt_file_raises_state_error(self, tmp_path: Path) -> None:
        """A corrupt / unparseable JSON file raises StateError, never silently resets."""
        path = tmp_path / "state.json"
        path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        with pytest.raises(StateError, match="corrupt or unparseable"):
            load(path)

    def test_wrong_type_raises_state_error(self, tmp_path: Path) -> None:
        """JSON with wrong field types raises StateError (pydantic validation failure)."""
        path = tmp_path / "state.json"
        # last_rebalanced_target should be dict[str,float]|None, not a list
        path.write_text(
            json.dumps({"last_rebalanced_target": [1, 2, 3], "schema_version": 1}),
            encoding="utf-8",
        )
        with pytest.raises(StateError):
            load(path)

    def test_round_trip_with_target(self, tmp_path: Path) -> None:
        """save + load round-trips a State with a non-None target correctly."""
        path = tmp_path / "state.json"
        original = _make_state(
            target={"BTC": 0.6, "ETH": 0.4},
            as_of="2026-06-01",
        )
        save(original, path)
        loaded = load(path)

        assert loaded.last_rebalanced_target == {"BTC": 0.6, "ETH": 0.4}
        assert loaded.last_rebalanced_as_of == "2026-06-01"
        assert loaded.schema_version == 1

    def test_round_trip_defaults(self, tmp_path: Path) -> None:
        """save + load round-trips a freshly-constructed State (all defaults)."""
        path = tmp_path / "state.json"
        original = State()
        save(original, path)
        loaded = load(path)

        assert loaded.last_rebalanced_target is None
        assert loaded.last_rebalanced_as_of is None
        assert loaded.schema_version == 1


class TestStateSave:
    def test_atomic_replace_no_tmp_left(self, tmp_path: Path) -> None:
        """After save(), no .tmp file remains on disk."""
        path = tmp_path / "state.json"
        state = _make_state(target={"BTC": 1.0}, as_of="2026-06-01")
        save(state, path)

        tmp = Path(str(path) + ".tmp")
        assert not tmp.exists(), ".tmp file should be cleaned up by os.replace"

    def test_saved_file_is_valid_json(self, tmp_path: Path) -> None:
        """The saved file is parseable as JSON."""
        path = tmp_path / "state.json"
        state = _make_state(target={"ETH": 0.5}, as_of="2026-05-01")
        save(state, path)

        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "last_rebalanced_target" in parsed

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """A second save() correctly overwrites the first."""
        path = tmp_path / "state.json"

        first = _make_state(target={"BTC": 0.5}, as_of="2026-05-01")
        save(first, path)

        second = _make_state(target={"ETH": 0.9}, as_of="2026-06-01")
        save(second, path)

        loaded = load(path)
        assert loaded.last_rebalanced_target == {"ETH": 0.9}
        assert loaded.last_rebalanced_as_of == "2026-06-01"

    def test_save_none_target(self, tmp_path: Path) -> None:
        """Saving a State with None target persists and loads back as None."""
        path = tmp_path / "state.json"
        state = State()  # last_rebalanced_target=None
        save(state, path)
        loaded = load(path)
        assert loaded.last_rebalanced_target is None


class TestLastSuccessfulSignalTs:
    def test_defaults_to_none(self) -> None:
        """Fresh State must have last_successful_signal_ts=None."""
        state = State()
        assert state.last_successful_signal_ts is None

    def test_round_trip_none(self, tmp_path: Path) -> None:
        """A State with last_successful_signal_ts=None round-trips correctly."""
        path = tmp_path / "state.json"
        state = State()
        save(state, path)
        loaded = load(path)
        assert loaded.last_successful_signal_ts is None

    def test_round_trip_set(self, tmp_path: Path) -> None:
        """A State with last_successful_signal_ts set round-trips correctly."""
        path = tmp_path / "state.json"
        ts = "2026-06-01T00:15:00+00:00"
        state = State(last_successful_signal_ts=ts)
        save(state, path)
        loaded = load(path)
        assert loaded.last_successful_signal_ts == ts

    def test_missing_field_in_legacy_file_defaults_to_none(self, tmp_path: Path) -> None:
        """A state file without last_successful_signal_ts loads with None (forward compat)."""
        import json

        path = tmp_path / "state.json"
        # Simulate a legacy state file that predates this field.
        legacy = {
            "last_rebalanced_target": {"BTC": 0.5},
            "last_rebalanced_as_of": "2026-05-01",
            "schema_version": 1,
            "last_scheduled_as_of": "2026-05-01",
        }
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load(path)
        assert loaded.last_successful_signal_ts is None
        assert loaded.last_rebalanced_target == {"BTC": 0.5}


class TestFirstCycleTs:
    def test_defaults_to_none(self) -> None:
        """Fresh State must have first_cycle_ts=None."""
        state = State()
        assert state.first_cycle_ts is None

    def test_round_trip_none(self, tmp_path: Path) -> None:
        """A State with first_cycle_ts=None round-trips correctly."""
        path = tmp_path / "state.json"
        state = State()
        save(state, path)
        loaded = load(path)
        assert loaded.first_cycle_ts is None

    def test_round_trip_set(self, tmp_path: Path) -> None:
        """A State with first_cycle_ts set round-trips correctly."""
        path = tmp_path / "state.json"
        ts = "2026-06-01T00:15:00+00:00"
        state = State(first_cycle_ts=ts)
        save(state, path)
        loaded = load(path)
        assert loaded.first_cycle_ts == ts

    def test_missing_field_in_legacy_file_defaults_to_none(self, tmp_path: Path) -> None:
        """A state file without first_cycle_ts loads with None (forward compat)."""
        import json

        path = tmp_path / "state.json"
        legacy = {
            "last_rebalanced_target": {"BTC": 0.5},
            "last_rebalanced_as_of": "2026-05-01",
            "schema_version": 1,
            "last_scheduled_as_of": "2026-05-01",
            "last_successful_signal_ts": "2026-05-01T00:00:00+00:00",
        }
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load(path)
        assert loaded.first_cycle_ts is None
        assert loaded.last_rebalanced_target == {"BTC": 0.5}


class TestRetryFields:
    def test_state_retry_fields_default_idle(self, tmp_path: Path) -> None:
        """Fresh State must have all retry fields at their IDLE defaults."""
        s = State()
        assert s.retry_as_of is None
        assert s.retry_target is None
        assert s.retry_window_start is None
        assert s.retry_last_attempt is None
        assert s.retry_attempt_count == 0

    def test_state_retry_fields_roundtrip(self, tmp_path: Path) -> None:
        """Retry fields survive a save/load round-trip with full fidelity."""
        p = tmp_path / "state.json"
        s = State(
            retry_as_of="2026-05-17",
            retry_target={"TON": 0.5, "HYPE": 0.075},
            retry_window_start="2026-05-18T00:15:00+00:00",
            retry_last_attempt="2026-05-18T00:15:00+00:00",
            retry_attempt_count=2,
        )
        save(s, p)
        loaded = load(p)
        assert loaded.retry_as_of == "2026-05-17"
        assert loaded.retry_target == {"TON": 0.5, "HYPE": 0.075}
        assert loaded.retry_window_start == "2026-05-18T00:15:00+00:00"
        assert loaded.retry_last_attempt == "2026-05-18T00:15:00+00:00"
        assert loaded.retry_attempt_count == 2


class TestRecheckFields:
    def test_recheck_fields_default_idle(self) -> None:
        """Fresh State must have all recheck fields at their IDLE defaults."""
        s = State()
        assert s.recheck_as_of is None
        assert s.recheck_window_start is None
        assert s.recheck_last_attempt is None
        assert s.recheck_attempt_count == 0
        assert s.recheck_exhausted_as_of is None

    def test_recheck_fields_roundtrip(self, tmp_path: Path) -> None:
        """Recheck fields survive a save/load round-trip with full fidelity."""
        p = tmp_path / "state.json"
        s = State(
            recheck_as_of="2026-06-10",
            recheck_window_start="2026-06-11T00:05:00+00:00",
            recheck_last_attempt="2026-06-11T00:05:00+00:00",
            recheck_attempt_count=3,
            recheck_exhausted_as_of="2026-06-09",
        )
        save(s, p)
        s2 = load(p)
        assert s2.recheck_as_of == "2026-06-10"
        assert s2.recheck_window_start == "2026-06-11T00:05:00+00:00"
        assert s2.recheck_last_attempt == "2026-06-11T00:05:00+00:00"
        assert s2.recheck_attempt_count == 3
        assert s2.recheck_exhausted_as_of == "2026-06-09"

    def test_recheck_fields_default_absent(self, tmp_path: Path) -> None:
        """Old state files (no recheck keys) must load with defaults (forward compat)."""
        p = tmp_path / "state.json"
        # Simulate a legacy state file that predates recheck fields.
        legacy = {
            "last_rebalanced_target": {"BTC": 0.5},
            "last_rebalanced_as_of": "2026-05-01",
            "schema_version": 1,
            "last_scheduled_as_of": "2026-05-01",
            "last_successful_signal_ts": "2026-05-01T00:00:00+00:00",
            "first_cycle_ts": "2026-05-01T00:00:00+00:00",
            "retry_as_of": None,
            "retry_target": None,
            "retry_window_start": None,
            "retry_last_attempt": None,
            "retry_attempt_count": 0,
        }
        p.write_text(json.dumps(legacy), encoding="utf-8")
        s = load(p)
        assert s.recheck_as_of is None
        assert s.recheck_attempt_count == 0


class TestSleeveLedger:
    def test_sleeves_default_empty(self) -> None:
        """Fresh State must have an empty per-sleeve ledger."""
        s = State()
        assert s.sleeves == {}

    def test_sleeves_roundtrip(self, tmp_path: Path) -> None:
        """A populated per-sleeve ledger survives a save/load round-trip with full fidelity."""
        p = tmp_path / "state.json"
        s = State(
            sleeves={
                "haarp": {
                    "last_weights": {"BTC": 0.6, "ETH": 0.4},
                    "last_as_of": "2026-06-10",
                    "last_success_ts": "2026-06-11T00:15:00+00:00",
                    "consecutive_holds": 0,
                },
                "crash": {
                    "last_weights": {"TON": -0.3, "HYPE": -0.2},
                    "last_as_of": "2026-06-10",
                    "last_success_ts": "2026-06-11T00:15:00+00:00",
                    "consecutive_holds": 1,
                },
            },
        )
        save(s, p)
        loaded = load(p)
        assert loaded.sleeves == {
            "haarp": {
                "last_weights": {"BTC": 0.6, "ETH": 0.4},
                "last_as_of": "2026-06-10",
                "last_success_ts": "2026-06-11T00:15:00+00:00",
                "consecutive_holds": 0,
            },
            "crash": {
                "last_weights": {"TON": -0.3, "HYPE": -0.2},
                "last_as_of": "2026-06-10",
                "last_success_ts": "2026-06-11T00:15:00+00:00",
                "consecutive_holds": 1,
            },
        }

    def test_signed_weights_roundtrip(self, tmp_path: Path) -> None:
        """Signed (negative) sleeve weights survive a round-trip without coercion."""
        p = tmp_path / "state.json"
        s = State(
            sleeves={
                "crash": {
                    "last_weights": {"TON": -0.45},
                    "last_as_of": "2026-06-10",
                    "last_success_ts": "2026-06-11T00:00:00+00:00",
                    "consecutive_holds": 2,
                },
            },
        )
        save(s, p)
        loaded = load(p)
        assert loaded.sleeves["crash"]["last_weights"] == {"TON": -0.45}
        assert loaded.sleeves["crash"]["consecutive_holds"] == 2


class TestLiqCooldowns:
    def test_liq_cooldowns_default_empty(self) -> None:
        """Fresh State must have an empty liquidation-cooldown map."""
        s = State()
        assert s.liq_cooldowns == {}

    def test_liq_cooldowns_roundtrip(self, tmp_path: Path) -> None:
        """The coin→ISO-date liquidation-cooldown map survives a round-trip."""
        p = tmp_path / "state.json"
        s = State(liq_cooldowns={"TON": "2026-06-10", "HYPE": "2026-06-11"})
        save(s, p)
        loaded = load(p)
        assert loaded.liq_cooldowns == {"TON": "2026-06-10", "HYPE": "2026-06-11"}


class TestSleeveLedgerLegacyCompat:
    def test_legacy_file_without_sleeves_loads_with_defaults(self, tmp_path: Path) -> None:
        """A legacy state.json with NO sleeves/liq_cooldowns keys loads with {} defaults.

        Red-team H7: there is NO migration — the new fields simply default to
        empty dicts when absent, so an in-flight deploy's existing state.json
        loads unchanged.
        """
        p = tmp_path / "state.json"
        legacy = {
            "last_rebalanced_target": {"BTC": 0.5},
            "last_rebalanced_as_of": "2026-05-01",
            "schema_version": 1,
            "last_scheduled_as_of": "2026-05-01",
            "last_successful_signal_ts": "2026-05-01T00:00:00+00:00",
            "first_cycle_ts": "2026-05-01T00:00:00+00:00",
            "retry_as_of": None,
            "retry_target": None,
            "retry_window_start": None,
            "retry_last_attempt": None,
            "retry_attempt_count": 0,
            "recheck_as_of": None,
            "recheck_window_start": None,
            "recheck_last_attempt": None,
            "recheck_attempt_count": 0,
            "recheck_exhausted_as_of": None,
        }
        p.write_text(json.dumps(legacy), encoding="utf-8")
        s = load(p)
        assert s.sleeves == {}
        assert s.liq_cooldowns == {}
        assert s.last_rebalanced_target == {"BTC": 0.5}
