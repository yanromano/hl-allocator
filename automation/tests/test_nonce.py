"""
Tests for automation.core.nonce.

Covers the four required behaviours:
(a) Monotonically increasing across rapid successive calls.
(b) Restart safety — a new NonceManager(same_path) continues strictly above the
    persisted last nonce.
(c) Backward-clock guard — monkeypatching time.time to jump backward beyond
    tolerance raises ClockError.
(d) Atomic state-file write — the file exists and is parseable immediately after
    next().
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from automation.core.nonce import ClockError, NonceManager

# ---------------------------------------------------------------------------
# (a) Monotonic increasing across calls
# ---------------------------------------------------------------------------


def test_monotonic_increasing(tmp_path: Path) -> None:
    """next() must return strictly increasing values across rapid calls."""
    mgr = NonceManager(tmp_path / "nonce.txt")
    nonces = [mgr.next() for _ in range(20)]
    for i in range(1, len(nonces)):
        assert nonces[i] > nonces[i - 1], (
            f"nonce[{i}]={nonces[i]} not > nonce[{i - 1}]={nonces[i - 1]}"
        )


def test_next_never_goes_backward(tmp_path: Path) -> None:
    """Even if called in a tight loop, nonces never repeat or go backward."""
    mgr = NonceManager(tmp_path / "nonce.txt")
    seen: set[int] = set()
    for _ in range(50):
        n = mgr.next()
        assert n not in seen, f"Duplicate nonce: {n}"
        seen.add(n)


# ---------------------------------------------------------------------------
# (b) Restart safety
# ---------------------------------------------------------------------------


def test_restart_continues_above_last(tmp_path: Path) -> None:
    """A reloaded NonceManager continues strictly above the persisted last."""
    state_file = tmp_path / "nonce.txt"

    mgr1 = NonceManager(state_file)
    last = mgr1.next()
    for _ in range(4):
        last = mgr1.next()
    # last is now the highest nonce issued by mgr1

    # Simulate daemon restart: create a fresh instance from the same path
    mgr2 = NonceManager(state_file)
    first_new = mgr2.next()
    assert first_new > last, (
        f"After restart, first nonce {first_new} must be > last persisted {last}"
    )


def test_restart_from_missing_file_starts_at_current_time(tmp_path: Path) -> None:
    """If the state file is absent, NonceManager starts from current ms time."""
    state_file = tmp_path / "nonexistent" / "nonce.txt"
    before = int(time.time() * 1000)
    mgr = NonceManager(state_file)
    n = mgr.next()
    after = int(time.time() * 1000) + 1
    assert before <= n <= after, f"nonce {n} should be near current time [{before}, {after}]"


# ---------------------------------------------------------------------------
# (c) Backward-clock guard → ClockError
# ---------------------------------------------------------------------------


def test_backward_clock_raises_clock_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patching time.time to jump backward beyond tolerance raises ClockError."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file, backward_tolerance_ms=5000)

    # Establish a high last nonce by directly setting _last
    future_ms = int(time.time() * 1000) + 60_000  # 1 minute in the future
    mgr._last = future_ms

    # Now monkeypatch time.time to return a value well before _last - tolerance
    backward_time = (future_ms - 10_000) / 1000  # 10 seconds before future_ms
    monkeypatch.setattr("automation.core.nonce.time.time", lambda: backward_time)

    with pytest.raises(ClockError):
        mgr.next()


def test_small_backward_clock_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backward step within tolerance must NOT raise; nonce still increases."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file, backward_tolerance_ms=5000)

    now_ms = int(time.time() * 1000)
    mgr._last = now_ms

    # 2 seconds backward — within the 5s tolerance
    slightly_back = (now_ms - 2_000) / 1000
    monkeypatch.setattr("automation.core.nonce.time.time", lambda: slightly_back)

    n = mgr.next()
    # Must still be strictly above _last
    assert n > now_ms, f"nonce {n} should be > last {now_ms} even with small backward clock"


# ---------------------------------------------------------------------------
# (d) Atomic state-file write
# ---------------------------------------------------------------------------


def test_state_file_written_atomically(tmp_path: Path) -> None:
    """The state file must exist and be parseable immediately after next()."""
    state_file = tmp_path / "nonce_state.txt"
    mgr = NonceManager(state_file)

    n = mgr.next()

    assert state_file.exists(), "State file must exist after next()"
    contents = state_file.read_text().strip()
    assert contents, "State file must not be empty"
    persisted = int(contents.splitlines()[-1].strip())
    assert persisted == n, (
        f"Persisted nonce {persisted} must equal returned nonce {n}"
    )


def test_state_file_no_tmp_leftover(tmp_path: Path) -> None:
    """The .tmp file must NOT remain after a successful next()."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file)
    mgr.next()

    tmp_file = state_file.with_suffix(".txt.tmp")
    assert not tmp_file.exists(), "Temporary state file must be cleaned up after atomic replace"


def test_state_file_second_write_overwrites(tmp_path: Path) -> None:
    """Each next() overwrites the state file with the new nonce (not appended)."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file)

    for _ in range(5):
        last = mgr.next()

    lines = state_file.read_text().strip().splitlines()
    # We write a single line per call (overwrite, not append)
    assert len(lines) == 1, f"State file should contain exactly 1 line, got {len(lines)}"
    assert int(lines[0]) == last


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_state_file_handled(tmp_path: Path) -> None:
    """An empty state file does not crash; NonceManager treats it as 0."""
    state_file = tmp_path / "nonce.txt"
    state_file.write_text("")
    before = int(time.time() * 1000)
    mgr = NonceManager(state_file)
    n = mgr.next()
    after = int(time.time() * 1000) + 1
    assert before <= n <= after


def test_thread_safety_monotonic(tmp_path: Path) -> None:
    """Concurrent calls from multiple threads must produce unique, increasing nonces."""
    import threading

    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file)
    results: list[int] = []
    lock = threading.Lock()

    def _worker() -> None:
        n = mgr.next()
        with lock:
            results.append(n)

    threads = [threading.Thread(target=_worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == len(set(results)), "All nonces must be unique under concurrent access"
    assert sorted(results) == results or len(set(results)) == 30, (
        "No duplicate nonces allowed"
    )


# ---------------------------------------------------------------------------
# (e) Forward-clock excursion alarm (F5) — alert, NEVER refuse
# ---------------------------------------------------------------------------


def test_forward_jump_emits_alarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A forward clock jump beyond forward_alarm_ms invokes on_alarm AND still issues."""
    state_file = tmp_path / "nonce.txt"
    alarms: list[str] = []
    mgr = NonceManager(
        state_file,
        forward_alarm_ms=6 * 3600 * 1000,  # 6h
        on_alarm=alarms.append,
    )

    # Establish a baseline last nonce at "now".
    now_ms = int(time.time() * 1000)
    mgr._last = now_ms

    # Jump the clock forward by 1 day — well beyond the 6h alarm threshold.
    forward_time = (now_ms + 24 * 3600 * 1000) / 1000
    monkeypatch.setattr("automation.core.nonce.time.time", lambda: forward_time)

    n = mgr.next()

    # NOT refused: a nonce was issued, strictly above the prior last.
    assert n > now_ms, f"forward jump must still issue a nonce; got {n} <= {now_ms}"
    # Alarm fired exactly once with a non-empty message.
    assert len(alarms) == 1, f"expected exactly 1 alarm, got {len(alarms)}"
    assert alarms[0], "alarm message must be non-empty"
    assert "Forward clock excursion" in alarms[0]


def test_small_forward_jump_no_alarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A forward step within forward_alarm_ms must NOT trigger the alarm."""
    state_file = tmp_path / "nonce.txt"
    alarms: list[str] = []
    mgr = NonceManager(state_file, forward_alarm_ms=6 * 3600 * 1000, on_alarm=alarms.append)

    now_ms = int(time.time() * 1000)
    mgr._last = now_ms

    # 1h forward — within the 6h alarm threshold.
    forward_time = (now_ms + 3600 * 1000) / 1000
    monkeypatch.setattr("automation.core.nonce.time.time", lambda: forward_time)

    n = mgr.next()
    assert n > now_ms
    assert alarms == [], "no alarm should fire for a sub-threshold forward jump"


def test_forward_jump_alarm_not_refused_without_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward jump with no on_alarm callback logs a WARNING but still issues."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file, forward_alarm_ms=6 * 3600 * 1000)  # no callback

    now_ms = int(time.time() * 1000)
    mgr._last = now_ms
    forward_time = (now_ms + 24 * 3600 * 1000) / 1000
    monkeypatch.setattr("automation.core.nonce.time.time", lambda: forward_time)

    # Must not raise; must issue a nonce.
    n = mgr.next()
    assert n > now_ms


def test_no_forward_alarm_on_first_ever_nonce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh manager (last==0) must NOT alarm on its first next() call.

    last==0 means "no prior nonce"; the huge ``now - 0`` delta is not a real
    forward excursion and must not trigger a spurious alarm.
    """
    state_file = tmp_path / "nonce.txt"
    alarms: list[str] = []
    mgr = NonceManager(state_file, on_alarm=alarms.append)
    assert mgr._last == 0

    mgr.next()
    assert alarms == [], "first-ever nonce (last==0) must not fire a forward alarm"


def test_forward_alarm_callback_exception_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising on_alarm callback must NOT block nonce issuance."""
    state_file = tmp_path / "nonce.txt"

    def _boom(_msg: str) -> None:
        raise ValueError("alert sink down")

    mgr = NonceManager(state_file, forward_alarm_ms=6 * 3600 * 1000, on_alarm=_boom)

    now_ms = int(time.time() * 1000)
    mgr._last = now_ms
    forward_time = (now_ms + 24 * 3600 * 1000) / 1000
    monkeypatch.setattr("automation.core.nonce.time.time", lambda: forward_time)

    # Despite the broken callback, next() must still return a nonce.
    n = mgr.next()
    assert n > now_ms


# ---------------------------------------------------------------------------
# (f) reanchor() recovery seam (F5 / G6)
# ---------------------------------------------------------------------------


def test_reanchor_raises_last(tmp_path: Path) -> None:
    """reanchor(min_nonce) raises the persisted last to min_nonce."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file)
    mgr.next()  # establish some baseline last

    target = mgr._last + 1_000_000
    mgr.reanchor(target)

    assert mgr._last == target
    persisted = int(state_file.read_text().strip())
    assert persisted == target, f"reanchor must persist; file has {persisted}, want {target}"


def test_reanchor_then_next_is_above(tmp_path: Path) -> None:
    """After reanchor, the next nonce is strictly above the reanchored value.

    The reanchor floor models a real on-chain nonce: a recent timestamp that is
    slightly *above* the local last but still near ``now`` (within the backward
    tolerance), so a subsequent ``next()`` continues monotonically rather than
    tripping the backward-clock guard.
    """
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file)
    mgr.next()  # establish a baseline last near now

    # Floor a few ms above the current last (mirrors an on-chain nonce that
    # nudged slightly ahead of the local file).
    target = mgr._last + 2
    mgr.reanchor(target)

    n = mgr.next()
    assert n > target, f"next() {n} must be strictly > reanchored last {target}"


def test_reanchor_to_lower_is_noop(tmp_path: Path) -> None:
    """reanchor to a value <= current last must NOT lower last (no regression)."""
    state_file = tmp_path / "nonce.txt"
    mgr = NonceManager(state_file)

    high = int(time.time() * 1000) + 10_000_000
    mgr.reanchor(high)
    assert mgr._last == high

    # Re-anchor to something lower — must be ignored.
    mgr.reanchor(high - 5_000_000)
    assert mgr._last == high, "reanchor must never move last backward"

    # And equal — also a no-op.
    mgr.reanchor(high)
    assert mgr._last == high


def test_reanchor_persists_across_restart(tmp_path: Path) -> None:
    """A reanchored value survives a NonceManager reload from the same path."""
    state_file = tmp_path / "nonce.txt"
    mgr1 = NonceManager(state_file)
    mgr1.next()  # baseline near now

    target = mgr1._last + 2  # modest floor (on-chain-nonce-like, within tolerance)
    mgr1.reanchor(target)

    mgr2 = NonceManager(state_file)
    assert mgr2._last == target
    n = mgr2.next()
    assert n > target


# ---------------------------------------------------------------------------
# (g) Durable mirror + boot recovery (max, never backwards) — Task 4
#
# The manager time-floors next() to now_ms, so losing the primary near "now"
# cannot regress the nonce in practice. The DANGER is a persisted FUTURE value
# (e.g. after reanchor() to an on-chain nonce, or a forward-clock poison): if the
# primary file is then lost, _load() returns 0 and the future floor is gone. The
# mirror makes the seed durable: boot seeds from max(primary, mirror), so the
# nonce can never regress below either file even if the primary disappears.
#
# These tests use FUTURE timestamps in the files so the file value dominates the
# now_ms floor, isolating the FILE-recovery semantics.
# ---------------------------------------------------------------------------

#: One day in ms — far enough above now_ms that a file value at now+_FUTURE
#: dominates the time floor in next().
_FUTURE_MS = 24 * 3600 * 1000


def test_nonce_mirror_written_on_advance(tmp_path: Path) -> None:
    """Advancing the nonce writes BOTH the primary file AND a mirror (<path>.bak).

    The mirror holds a value >= the advanced nonce, in the same on-disk format
    (single line, integer).
    """
    p = tmp_path / "nonce.state"
    nm = NonceManager(p)
    n1 = nm.next()

    mirror = tmp_path / "nonce.state.bak"
    assert mirror.exists(), "mirror file must exist after next()"
    mirror_val = int(mirror.read_text().strip().splitlines()[-1].strip())
    assert mirror_val >= n1, f"mirror {mirror_val} must be >= advanced nonce {n1}"
    # Primary and mirror agree.
    primary_val = int(p.read_text().strip().splitlines()[-1].strip())
    assert mirror_val == primary_val


def test_boot_recovers_from_mirror_when_primary_missing(tmp_path: Path) -> None:
    """If the primary file is lost but the mirror survives, boot recovers from it.

    A FUTURE nonce is persisted (reanchor models an on-chain nonce ahead of now),
    the primary is deleted, and a fresh manager must seed from the mirror so the
    next nonce is still strictly above the lost value — never backwards.
    """
    p = tmp_path / "nonce.state"
    nm = NonceManager(p)
    nm.next()

    # Push last to a FUTURE value so it dominates the now_ms floor.
    hi = int(time.time() * 1000) + _FUTURE_MS
    nm.reanchor(hi)
    assert nm._last == hi

    # Primary lost (mirror survives).
    p.unlink()
    assert not p.exists()
    assert (tmp_path / "nonce.state.bak").exists()

    nm2 = NonceManager(p)  # boot
    # The invariant: the seed is never below the lost value — recovered from the
    # mirror. (We assert on the seed rather than next(): a far-future seed
    # correctly trips the backward-clock guard, which is the reanchor() seam's
    # job to resolve, not this test's.)
    assert nm2._last >= hi, "boot must recover the future value from the mirror"
    assert nm2._last == hi, "seed is exactly the surviving mirror value"


def test_boot_takes_max_of_primary_and_mirror(tmp_path: Path) -> None:
    """Boot seeds from max(primary, mirror) regardless of which is higher."""
    p = tmp_path / "nonce.state"
    mirror = tmp_path / "nonce.state.bak"

    base = int(time.time() * 1000) + _FUTURE_MS
    primary_val = base + 100
    mirror_val = base + 250  # mirror is higher
    p.write_text(f"{primary_val}\n")
    mirror.write_text(f"{mirror_val}\n")

    nm = NonceManager(p)
    # Seed is the max of the two files (mirror here). A far-future seed trips the
    # backward-clock guard on next(), so we assert the recovered seed directly —
    # that is the "never below max(primary, mirror)" invariant.
    assert nm._last == mirror_val, "boot must take the max (mirror) value"


def test_boot_takes_max_when_primary_is_higher(tmp_path: Path) -> None:
    """Boot seeds from the primary when it exceeds a stale mirror."""
    p = tmp_path / "nonce.state"
    mirror = tmp_path / "nonce.state.bak"

    base = int(time.time() * 1000) + _FUTURE_MS
    primary_val = base + 500  # primary is higher (stale mirror)
    mirror_val = base + 250
    p.write_text(f"{primary_val}\n")
    mirror.write_text(f"{mirror_val}\n")

    nm = NonceManager(p)
    assert nm._last == primary_val, "boot must take the max (primary) value"


def test_boot_tolerates_corrupt_mirror(tmp_path: Path) -> None:
    """A corrupt/garbage mirror is treated defensively (as 0), not a crash."""
    p = tmp_path / "nonce.state"
    mirror = tmp_path / "nonce.state.bak"

    val = int(time.time() * 1000) + _FUTURE_MS
    p.write_text(f"{val}\n")
    mirror.write_text("not-an-int\n")  # corrupt mirror

    nm = NonceManager(p)  # must not raise
    assert nm._last == val, "corrupt mirror ignored; primary value used"


def test_boot_tolerates_corrupt_primary_recovers_from_mirror(tmp_path: Path) -> None:
    """A corrupt primary falls back to the (valid) mirror — never backwards."""
    p = tmp_path / "nonce.state"
    mirror = tmp_path / "nonce.state.bak"

    val = int(time.time() * 1000) + _FUTURE_MS
    p.write_text("garbage\n")  # corrupt primary
    mirror.write_text(f"{val}\n")

    nm = NonceManager(p)  # must not raise
    assert nm._last == val, "corrupt primary ignored; mirror value recovered"


def test_mirror_no_tmp_leftover(tmp_path: Path) -> None:
    """The mirror's atomic-write tmp must not remain after next()."""
    p = tmp_path / "nonce.state"
    nm = NonceManager(p)
    nm.next()

    mirror_tmp = tmp_path / "nonce.state.bak.tmp"
    assert not mirror_tmp.exists(), "mirror tmp file must be cleaned up after replace"
