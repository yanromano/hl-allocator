"""
Tests for automation.runtime.intent_ledger — crash-safe PENDING→DONE intent ledger.

Coverage:
- record_pending / mark_done / pending basics
- crash-replay: reconstruct ledger from same path, verify recovery
- reconcile_pending with filled=True → DONE, filled=False → stays PENDING
- corrupt file → LedgerError (never silently resets)
- no .tmp left behind after write
- mark_done with unknown cloid → KeyError
"""

from __future__ import annotations

from pathlib import Path

import pytest

from automation.runtime.intent_ledger import Intent, IntentLedger, LedgerError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    cloid: str = "ord-001",
    coin: str = "BTC",
    target_w: float = 0.5,
    as_of: str = "2026-06-01",
    ts: str = "2026-06-01T00:00:00Z",
    attempt_seq: int = 0,
) -> Intent:
    return Intent(
        cloid=cloid,
        as_of=as_of,
        coin=coin,
        target_w=target_w,
        attempt_seq=attempt_seq,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestIntentLedgerBasics:
    def test_empty_ledger_has_no_pending(self, tmp_path: Path) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        assert ledger.pending() == []

    def test_record_pending_adds_intent(self, tmp_path: Path) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        intent = _make_intent(cloid="ord-001")
        ledger.record_pending(intent)

        pending = ledger.pending()
        assert len(pending) == 1
        assert pending[0].cloid == "ord-001"
        assert pending[0].status == "PENDING"

    def test_mark_done_transitions_status(self, tmp_path: Path) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        intent = _make_intent(cloid="ord-001")
        ledger.record_pending(intent)
        ledger.mark_done("ord-001")

        assert ledger.pending() == []

    def test_mark_done_unknown_cloid_raises_key_error(self, tmp_path: Path) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        with pytest.raises(KeyError, match="ord-999"):
            ledger.mark_done("ord-999")

    def test_record_pending_overwrites_existing(self, tmp_path: Path) -> None:
        """record_pending is idempotent: re-recording the same cloid overwrites."""
        ledger = IntentLedger(tmp_path / "ledger.json")
        intent1 = _make_intent(cloid="ord-001", target_w=0.5)
        intent2 = _make_intent(cloid="ord-001", target_w=0.8, attempt_seq=1)

        ledger.record_pending(intent1)
        ledger.record_pending(intent2)

        pending = ledger.pending()
        assert len(pending) == 1
        assert pending[0].target_w == 0.8
        assert pending[0].attempt_seq == 1

    def test_multiple_intents(self, tmp_path: Path) -> None:
        ledger = IntentLedger(tmp_path / "ledger.json")
        ledger.record_pending(_make_intent(cloid="ord-001", coin="BTC"))
        ledger.record_pending(_make_intent(cloid="ord-002", coin="ETH"))

        assert len(ledger.pending()) == 2
        ledger.mark_done("ord-001")
        pending = ledger.pending()
        assert len(pending) == 1
        assert pending[0].cloid == "ord-002"


# ---------------------------------------------------------------------------
# Crash-replay: construct a new ledger from the same path
# ---------------------------------------------------------------------------


class TestCrashReplay:
    def test_pending_intent_recovered_after_restart(self, tmp_path: Path) -> None:
        """Simulate a crash between record_pending and mark_done.

        A second IntentLedger built from the same path recovers the PENDING intent.
        """
        path = tmp_path / "ledger.json"

        # Ledger A: record and simulate crash (no mark_done called)
        ledger_a = IntentLedger(path)
        intent = _make_intent(cloid="ord-001", coin="BTC")
        ledger_a.record_pending(intent)
        # Process "crashes" here — ledger_a goes out of scope

        # Ledger B: reconstruct from same path
        ledger_b = IntentLedger(path)
        pending = ledger_b.pending()

        assert len(pending) == 1
        assert pending[0].cloid == "ord-001"
        assert pending[0].coin == "BTC"
        assert pending[0].status == "PENDING"

    def test_reconcile_filled_marks_done_and_persists(self, tmp_path: Path) -> None:
        """reconcile_pending with is_filled=True transitions to DONE and persists."""
        path = tmp_path / "ledger.json"

        ledger_a = IntentLedger(path)
        ledger_a.record_pending(_make_intent(cloid="ord-001", coin="BTC"))
        # crash here

        ledger_b = IntentLedger(path)
        remaining = ledger_b.reconcile_pending(is_filled=lambda i: True)

        assert remaining == []

        # Third ledger: verify DONE is persisted
        ledger_c = IntentLedger(path)
        assert ledger_c.pending() == []

    def test_reconcile_unfilled_stays_pending(self, tmp_path: Path) -> None:
        """reconcile_pending with is_filled=False leaves intent as PENDING."""
        path = tmp_path / "ledger.json"

        ledger_a = IntentLedger(path)
        ledger_a.record_pending(_make_intent(cloid="ord-001", coin="ETH"))

        ledger_b = IntentLedger(path)
        remaining = ledger_b.reconcile_pending(is_filled=lambda i: False)

        assert len(remaining) == 1
        assert remaining[0].cloid == "ord-001"
        assert remaining[0].status == "PENDING"

    def test_reconcile_mixed_filled_unfilled(self, tmp_path: Path) -> None:
        """reconcile_pending handles a mix: filled → DONE, unfilled → stays PENDING."""
        path = tmp_path / "ledger.json"

        ledger = IntentLedger(path)
        ledger.record_pending(_make_intent(cloid="ord-001", coin="BTC"))
        ledger.record_pending(_make_intent(cloid="ord-002", coin="ETH"))

        filled_ids = {"ord-001"}
        remaining = ledger.reconcile_pending(
            is_filled=lambda i: i.cloid in filled_ids
        )

        assert len(remaining) == 1
        assert remaining[0].cloid == "ord-002"

        # Reload to verify persistence
        ledger2 = IntentLedger(path)
        assert len(ledger2.pending()) == 1
        assert ledger2.pending()[0].cloid == "ord-002"

    def test_reconcile_resolver_crash_persists_done_so_far(
        self, tmp_path: Path
    ) -> None:
        """F2: resolver returns True for a,b then RAISES on c.

        Per-transition flushing must mean a,b are DONE on disk before c is
        queried.  After the raise propagates and we reload from disk, a,b MUST be
        DONE and only c remains PENDING — so a crash here never replays a,b
        (double-fill).  With the old flush-once-after-loop this test fails: a,b
        would still be PENDING on disk.
        """
        path = tmp_path / "ledger.json"

        ledger = IntentLedger(path)
        ledger.record_pending(_make_intent(cloid="a", coin="BTC"))
        ledger.record_pending(_make_intent(cloid="b", coin="ETH"))
        ledger.record_pending(_make_intent(cloid="c", coin="SOL"))

        class _Boom(RuntimeError):
            pass

        def flaky_resolver(intent: Intent) -> bool:
            if intent.cloid == "c":
                raise _Boom("exchange query failed on c")
            return True  # a, b resolve

        # The raise must propagate (caller decides how to escalate).
        with pytest.raises(_Boom):
            ledger.reconcile_pending(is_filled=flaky_resolver)

        # Reload from disk — simulates the process restarting after the crash.
        reloaded = IntentLedger(path)
        pending = {i.cloid for i in reloaded.pending()}

        # a and b were already DONE and durably persisted BEFORE c was queried.
        assert pending == {"c"}, f"a,b must be DONE on disk; pending={pending}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestIntentLedgerErrors:
    def test_corrupt_file_raises_ledger_error(self, tmp_path: Path) -> None:
        """A corrupt ledger file raises LedgerError at construction time."""
        path = tmp_path / "ledger.json"
        path.write_text("NOT JSON {{{", encoding="utf-8")

        with pytest.raises(LedgerError, match="corrupt or unparseable"):
            IntentLedger(path)

    def test_no_tmp_left_after_record(self, tmp_path: Path) -> None:
        """No .tmp file remains on disk after record_pending."""
        path = tmp_path / "ledger.json"
        ledger = IntentLedger(path)
        ledger.record_pending(_make_intent(cloid="ord-001"))

        tmp = Path(str(path) + ".tmp")
        assert not tmp.exists(), ".tmp should be cleaned up by os.replace"

    def test_no_tmp_left_after_mark_done(self, tmp_path: Path) -> None:
        """No .tmp file remains after mark_done."""
        path = tmp_path / "ledger.json"
        ledger = IntentLedger(path)
        ledger.record_pending(_make_intent(cloid="ord-001"))
        ledger.mark_done("ord-001")

        tmp = Path(str(path) + ".tmp")
        assert not tmp.exists()

    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        """Constructing from a non-existent path gives an empty ledger."""
        ledger = IntentLedger(tmp_path / "nonexistent.json")
        assert ledger.pending() == []
