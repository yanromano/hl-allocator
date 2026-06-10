"""
Tests for automation.reporting.audit.

Coverage:
- Write N records → file has N lines, each valid JSON that round-trips
- Appends don't truncate existing content
- cycle_record maps a CycleReport correctly (with and without exec_report)
- cycle_record with no exec_report → all zeros
- AuditLog creates parent directory if missing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from automation.reporting.audit import AuditLog, cycle_record
from automation.runtime.rebalance import CycleReport

# ---------------------------------------------------------------------------
# Helpers — minimal stubs for CycleReport / ExecReport / CoinResult
# ---------------------------------------------------------------------------


def _make_coin_result(
    coin: str = "BTC",
    filled_size: float = 0.5,
    attempts: int = 1,
    status: str = "filled",
) -> Any:
    """Return a minimal CoinResult-like object."""
    from automation.execution.executor import CoinResult  # noqa: PLC0415

    return CoinResult(
        coin=coin,
        requested_size=filled_size,
        filled_size=filled_size,
        attempts=attempts,
        status=status,  # type: ignore[arg-type]
        cloids=[],
        error=None,
    )


def _make_exec_report(
    n_submitted: int = 1,
    n_filled: int = 1,
    n_error: int = 0,
    n_partial: int = 0,
    aborted: bool = False,
    coins: list[Any] | None = None,
) -> Any:
    from automation.execution.executor import ExecReport  # noqa: PLC0415

    return ExecReport(
        results=coins or [],
        n_submitted=n_submitted,
        n_filled=n_filled,
        n_error=n_error,
        n_partial=n_partial,
        aborted=aborted,
        abort_reason="cap" if aborted else None,
    )


def _make_cycle_report(
    rebalanced: bool = True,
    reason: str = "bootstrap",
    exec_report: Any = None,
    committed: bool = True,
) -> CycleReport:
    return CycleReport(
        rebalanced=rebalanced,
        reason=reason,
        exec_report=exec_report,
        unresolved_pending=[],
        committed=committed,
    )


# ---------------------------------------------------------------------------
# AuditLog — file I/O
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_single_write_creates_one_line(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        log.write({"ts": "2026-06-01T00:15:00", "event": "test"})

        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_written_line_is_valid_json(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        log.write({"ts": "2026-06-01", "rebalanced": True, "n_filled": 2})

        raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip()
        parsed = json.loads(raw)
        assert parsed["rebalanced"] is True
        assert parsed["n_filled"] == 2

    def test_n_writes_produce_n_lines(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        n = 5
        for i in range(n):
            log.write({"i": i, "event": "cycle"})

        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == n

    def test_each_line_round_trips_as_json(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        records = [{"i": i, "val": f"v{i}"} for i in range(3)]
        for r in records:
            log.write(r)

        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["i"] == i
            assert parsed["val"] == f"v{i}"

    def test_appends_do_not_truncate(self, tmp_path: Path) -> None:
        """Opening a new AuditLog on the same path appends, never truncates."""
        path = tmp_path / "audit.jsonl"
        log1 = AuditLog(path)
        log1.write({"record": 1})

        log2 = AuditLog(path)
        log2.write({"record": 2})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["record"] == 1
        assert json.loads(lines[1])["record"] == 2

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "audit.jsonl"
        assert not nested.parent.exists()
        log = AuditLog(nested)
        log.write({"event": "first"})
        assert nested.exists()

    def test_record_is_sorted_keys_compact(self, tmp_path: Path) -> None:
        """Output format: compact JSON (no spaces) with sorted keys."""
        log = AuditLog(tmp_path / "audit.jsonl")
        log.write({"z": 3, "a": 1, "m": 2})
        raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip()
        # Sorted keys: a, m, z
        assert raw == '{"a":1,"m":2,"z":3}'

    def test_decimal_value_serialised_not_dropped(self, tmp_path: Path) -> None:
        """F4: a Decimal (non-JSON-native) value must serialise via default=str,
        not raise TypeError and silently lose the line."""
        from decimal import Decimal  # noqa: PLC0415

        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.write({"equity": Decimal("12345.6789"), "coin": "BTC"})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        # default=str renders the Decimal as a string — line is present, not lost
        assert parsed["equity"] == "12345.6789"
        assert parsed["coin"] == "BTC"


# ---------------------------------------------------------------------------
# cycle_record — pure function
# ---------------------------------------------------------------------------


class TestCycleRecord:
    def test_hold_cycle_no_exec_report(self) -> None:
        report = _make_cycle_report(
            rebalanced=False,
            reason="hold",
            exec_report=None,
            committed=False,
        )
        rec = cycle_record(ts="2026-06-01T00:15:00", as_of="2026-05-31", report=report)

        assert rec["ts"] == "2026-06-01T00:15:00"
        assert rec["as_of"] == "2026-05-31"
        assert rec["rebalanced"] is False
        assert rec["reason"] == "hold"
        assert rec["committed"] is False
        # All numeric fields → zeros
        assert rec["n_submitted"] == 0
        assert rec["n_filled"] == 0
        assert rec["n_error"] == 0
        assert rec["n_partial"] == 0
        assert rec["aborted"] is False
        assert rec["coins"] == []

    def test_rebalanced_cycle_with_exec_report(self) -> None:
        er = _make_exec_report(
            n_submitted=2,
            n_filled=2,
            n_error=0,
            n_partial=0,
            coins=[
                _make_coin_result("BTC", filled_size=0.5, attempts=1, status="filled"),
                _make_coin_result("ETH", filled_size=2.0, attempts=1, status="filled"),
            ],
        )
        report = _make_cycle_report(rebalanced=True, reason="bootstrap", exec_report=er, committed=True)
        rec = cycle_record(ts="2026-06-01T00:15:00", as_of="2026-05-31", report=report)

        assert rec["rebalanced"] is True
        assert rec["committed"] is True
        assert rec["n_submitted"] == 2
        assert rec["n_filled"] == 2
        assert rec["n_error"] == 0
        assert rec["aborted"] is False
        assert len(rec["coins"]) == 2
        assert rec["coins"][0]["coin"] == "BTC"
        assert rec["coins"][1]["coin"] == "ETH"

    def test_aborted_cycle(self) -> None:
        er = _make_exec_report(
            n_submitted=0,
            n_filled=0,
            aborted=True,
        )
        report = _make_cycle_report(rebalanced=True, reason="cap breach", exec_report=er, committed=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert rec["aborted"] is True
        assert rec["n_submitted"] == 0
        assert rec["committed"] is False

    def test_cycle_with_errors(self) -> None:
        er = _make_exec_report(
            n_submitted=3,
            n_filled=2,
            n_error=1,
            n_partial=0,
            coins=[
                _make_coin_result("BTC", status="filled"),
                _make_coin_result("ETH", status="filled"),
                _make_coin_result("SOL", status="error"),
            ],
        )
        report = _make_cycle_report(rebalanced=True, exec_report=er, committed=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert rec["n_error"] == 1
        assert rec["committed"] is False
        assert rec["coins"][2]["coin"] == "SOL"
        assert rec["coins"][2]["status"] == "error"

    def test_as_of_none_is_preserved(self) -> None:
        report = _make_cycle_report(exec_report=None)
        rec = cycle_record(ts="T", as_of=None, report=report)
        assert rec["as_of"] is None

    def test_coin_fields_mapped_correctly(self) -> None:
        coin = _make_coin_result("SOL", filled_size=10.0, attempts=2, status="partial")
        er = _make_exec_report(n_submitted=1, n_filled=0, n_partial=1, coins=[coin])
        report = _make_cycle_report(rebalanced=True, exec_report=er)
        rec = cycle_record(ts="T", as_of="D", report=report)
        c = rec["coins"][0]
        assert c["coin"] == "SOL"
        assert c["filled_size"] == 10.0
        assert c["attempts"] == 2
        assert c["status"] == "partial"


# ---------------------------------------------------------------------------
# G4a: max_divergence + n_unallocated in cycle_record
# ---------------------------------------------------------------------------


class TestCycleRecordG4aFields:
    """cycle_record includes max_divergence and n_unallocated from G4a."""

    def test_max_divergence_default_zero_in_record(self) -> None:
        """A CycleReport with default max_divergence=0.0 → record has 0.0."""
        report = _make_cycle_report(exec_report=None, committed=False, rebalanced=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert "max_divergence" in rec
        assert rec["max_divergence"] == 0.0

    def test_n_unallocated_default_zero_in_record(self) -> None:
        """A CycleReport with default n_unallocated=0 → record has 0."""
        report = _make_cycle_report(exec_report=None, committed=False, rebalanced=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert "n_unallocated" in rec
        assert rec["n_unallocated"] == 0

    def test_max_divergence_propagated_when_set(self) -> None:
        """max_divergence on CycleReport is reflected in the record."""
        from automation.runtime.rebalance import CycleReport  # noqa: PLC0415

        report = CycleReport(
            rebalanced=True,
            reason="bootstrap",
            exec_report=None,
            unresolved_pending=[],
            committed=True,
            max_divergence=0.123456,
            n_unallocated=0,
        )
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert abs(rec["max_divergence"] - 0.123456) < 1e-9

    def test_n_unallocated_propagated_when_set(self) -> None:
        """n_unallocated on CycleReport is reflected in the record."""
        from automation.runtime.rebalance import CycleReport  # noqa: PLC0415

        report = CycleReport(
            rebalanced=True,
            reason="bootstrap",
            exec_report=None,
            unresolved_pending=[],
            committed=True,
            max_divergence=0.0,
            n_unallocated=3,
        )
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert rec["n_unallocated"] == 3

    def test_both_fields_present_in_rebalanced_record(self) -> None:
        """Both max_divergence and n_unallocated appear in a full rebalanced record."""
        er = _make_exec_report(n_submitted=1, n_filled=1)
        report = _make_cycle_report(rebalanced=True, reason="bootstrap", exec_report=er, committed=True)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert "max_divergence" in rec
        assert "n_unallocated" in rec


# ---------------------------------------------------------------------------
# G4b MF-9: deferred + n_throttled in cycle_record
# ---------------------------------------------------------------------------


class TestCycleRecordG4bFields:
    """G4b: cycle_record includes deferred and n_throttled fields (MF-9)."""

    def test_deferred_default_false_in_record(self) -> None:
        """A normal ExecReport (not deferred) → record has deferred=False."""
        er = _make_exec_report(n_submitted=1, n_filled=1)
        report = _make_cycle_report(rebalanced=True, reason="bootstrap", exec_report=er, committed=True)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert "deferred" in rec
        assert rec["deferred"] is False

    def test_n_throttled_default_zero_in_record(self) -> None:
        """A normal ExecReport → record has n_throttled=0."""
        er = _make_exec_report(n_submitted=1, n_filled=1)
        report = _make_cycle_report(rebalanced=True, reason="bootstrap", exec_report=er, committed=True)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert "n_throttled" in rec
        assert rec["n_throttled"] == 0

    def test_deferred_true_when_exec_report_deferred(self) -> None:
        """ExecReport with deferred=True → record has deferred=True."""
        from automation.execution.executor import ExecReport  # noqa: PLC0415

        er = ExecReport(
            results=[],
            n_submitted=0,
            n_filled=0,
            n_error=0,
            n_partial=0,
            deferred=True,
            defer_reason="rate budget 5 < est 8 actions",
        )
        report = _make_cycle_report(rebalanced=True, reason="bootstrap", exec_report=er, committed=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert rec["deferred"] is True
        assert rec["n_submitted"] == 0

    def test_n_throttled_propagated_when_nonzero(self) -> None:
        """ExecReport with n_throttled=2 → record has n_throttled=2."""
        from automation.execution.executor import ExecReport  # noqa: PLC0415

        er = ExecReport(
            results=[],
            n_submitted=2,
            n_filled=0,
            n_error=2,
            n_partial=0,
            n_throttled=2,
        )
        report = _make_cycle_report(rebalanced=True, reason="bootstrap", exec_report=er, committed=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert rec["n_throttled"] == 2

    def test_no_exec_report_deferred_and_throttled_zero(self) -> None:
        """No exec_report (hold cycle) → deferred=False, n_throttled=0."""
        report = _make_cycle_report(rebalanced=False, reason="hold", exec_report=None, committed=False)
        rec = cycle_record(ts="T", as_of="D", report=report)
        assert rec["deferred"] is False
        assert rec["n_throttled"] == 0
