"""
Tests for automation.runtime.scheduler.

Convention being tested
-----------------------
Daily bars close at 00:00 UTC.  When the daemon fires at e.g. 00:15 UTC on
2026-06-02, the most-recently-closed bar has as_of = "2026-06-01" (the UTC
calendar date of that bar's start / open).

should_fire / expected_as_of coverage:
- Before HH:MM on a given day → (False, None)
- After HH:MM, never fired → (True, yesterday_date)
- Already fired same as_of → (False, as_of)
- Next day after HH:MM → (True, new as_of)
- Naive datetime → ValueError
- Exactly at HH:MM → fires (boundary inclusive)
- Midnight edge: 23:59 time, fire at 00:15 → not ready
"""

from __future__ import annotations

import datetime

import pytest

from automation.runtime.scheduler import (
    expected_as_of,
    retry_window_expired,
    should_fire,
    should_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = datetime.UTC


def _dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime.datetime:
    """Construct a tz-aware UTC datetime."""
    return datetime.datetime(year, month, day, hour, minute, tzinfo=UTC)


FIRE_TIME = "00:15"


# ---------------------------------------------------------------------------
# expected_as_of
# ---------------------------------------------------------------------------


class TestExpectedAsOf:
    def test_before_fire_time_returns_none(self) -> None:
        # 2026-06-02 00:10 UTC — before 00:15
        now = _dt(2026, 6, 2, 0, 10)
        result = expected_as_of(now, FIRE_TIME)
        assert result is None

    def test_at_fire_time_returns_yesterday(self) -> None:
        # Exactly at 00:15 — boundary is inclusive
        now = _dt(2026, 6, 2, 0, 15)
        result = expected_as_of(now, FIRE_TIME)
        assert result == "2026-06-01"

    def test_after_fire_time_returns_yesterday(self) -> None:
        # 2026-06-02 01:30 UTC — after 00:15
        now = _dt(2026, 6, 2, 1, 30)
        result = expected_as_of(now, FIRE_TIME)
        assert result == "2026-06-01"

    def test_just_before_fire_time_returns_none(self) -> None:
        # 2026-06-02 00:14 UTC — one minute before 00:15
        now = _dt(2026, 6, 2, 0, 14)
        result = expected_as_of(now, FIRE_TIME)
        assert result is None

    def test_naive_datetime_raises_value_error(self) -> None:
        naive = datetime.datetime(2026, 6, 2, 1, 0)  # no tzinfo
        with pytest.raises(ValueError, match="tz-aware"):
            expected_as_of(naive, FIRE_TIME)

    def test_different_fire_time(self) -> None:
        # Fire at 12:00, now is 2026-06-02 13:00 UTC
        now = _dt(2026, 6, 2, 13, 0)
        result = expected_as_of(now, "12:00")
        assert result == "2026-06-01"

    def test_different_fire_time_not_ready(self) -> None:
        # Fire at 12:00, now is 2026-06-02 11:59 UTC
        now = _dt(2026, 6, 2, 11, 59)
        result = expected_as_of(now, "12:00")
        assert result is None

    def test_as_of_is_always_one_day_behind(self) -> None:
        # Midnight boundary: now = 2026-01-01 00:15, as_of = 2025-12-31
        now = _dt(2026, 1, 1, 0, 15)
        result = expected_as_of(now, "00:15")
        assert result == "2025-12-31"


class TestExpectedAsOfNonUTC:
    """F2: a non-UTC tz-aware ``now`` must be CONVERTED to the UTC instant
    (not relabelled) for BOTH the time comparison AND the date arithmetic."""

    def test_ny_evening_before_utc_fire_does_not_fire(self) -> None:
        """20:10 America/New_York = 00:10 UTC < 00:15 → must NOT fire.

        The pre-fix bug relabelled 20:10 wall-clock as 20:10 UTC, which is past
        00:15, and wrongly returned a date (fired early).
        """
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        ny = ZoneInfo("America/New_York")
        # 2026-06-02 20:10 NY (EDT, UTC-4) = 2026-06-03 00:10 UTC
        now = datetime.datetime(2026, 6, 2, 20, 10, tzinfo=ny)
        result = expected_as_of(now, "00:15")
        assert result is None  # 00:10 UTC is before the 00:15 fire time

    def test_ny_morning_past_utc_fire_does_fire(self) -> None:
        """09:00 America/New_York = 13:00 UTC > 12:00 → must fire.

        The pre-fix bug relabelled 09:00 as 09:00 UTC, which is before 12:00,
        and wrongly returned None (failed to fire).
        """
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        ny = ZoneInfo("America/New_York")
        # 2026-06-02 09:00 NY (EDT, UTC-4) = 2026-06-02 13:00 UTC
        now = datetime.datetime(2026, 6, 2, 9, 0, tzinfo=ny)
        result = expected_as_of(now, "12:00")
        # 13:00 UTC is past 12:00 → fire; as_of = (UTC date 2026-06-02) - 1 day
        assert result == "2026-06-01"

    def test_ny_evening_date_uses_utc_instant(self) -> None:
        """22:00 America/New_York on 2026-06-02 = 02:00 UTC on 2026-06-03.

        The as_of date must be derived from the UTC instant's date (2026-06-03),
        so as_of = 2026-06-02 — NOT the local date 2026-06-02 minus a day.
        """
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        ny = ZoneInfo("America/New_York")
        now = datetime.datetime(2026, 6, 2, 22, 0, tzinfo=ny)  # = 2026-06-03 02:00 UTC
        result = expected_as_of(now, "00:15")
        assert result == "2026-06-02"

    def test_should_fire_respects_utc_conversion(self) -> None:
        """should_fire delegates to expected_as_of — verify end-to-end via NY tz."""
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        ny = ZoneInfo("America/New_York")
        # 20:10 NY = 00:10 UTC < 00:15 → not ready
        now = datetime.datetime(2026, 6, 2, 20, 10, tzinfo=ny)
        fire, as_of = should_fire(now, None, "00:15")
        assert fire is False
        assert as_of is None


# ---------------------------------------------------------------------------
# should_fire
# ---------------------------------------------------------------------------


class TestShouldFire:
    def test_before_fire_time_not_fired(self) -> None:
        now = _dt(2026, 6, 2, 0, 10)  # before 00:15
        fire, as_of = should_fire(now, None, FIRE_TIME)
        assert fire is False
        assert as_of is None

    def test_after_fire_time_never_fired(self) -> None:
        now = _dt(2026, 6, 2, 0, 20)
        fire, as_of = should_fire(now, None, FIRE_TIME)
        assert fire is True
        assert as_of == "2026-06-01"

    def test_already_fired_same_as_of(self) -> None:
        now = _dt(2026, 6, 2, 0, 20)
        fire, as_of = should_fire(now, "2026-06-01", FIRE_TIME)
        assert fire is False
        assert as_of == "2026-06-01"  # echoes the already-fired as_of

    def test_next_day_fires_new_as_of(self) -> None:
        # Day 1 is already fired; now it's day 2 after fire time
        now = _dt(2026, 6, 3, 0, 20)
        fire, as_of = should_fire(now, "2026-06-01", FIRE_TIME)
        assert fire is True
        assert as_of == "2026-06-02"

    def test_naive_datetime_raises_value_error(self) -> None:
        naive = datetime.datetime(2026, 6, 2, 1, 0)
        with pytest.raises(ValueError, match="tz-aware"):
            should_fire(naive, None, FIRE_TIME)

    def test_fires_when_last_as_of_is_older(self) -> None:
        # last_scheduled_as_of is from 2 days ago → should fire
        now = _dt(2026, 6, 5, 1, 0)
        fire, as_of = should_fire(now, "2026-06-03", FIRE_TIME)
        assert fire is True
        assert as_of == "2026-06-04"

    def test_returns_false_and_none_when_fire_time_not_reached(self) -> None:
        # never fired, but fire time hasn't arrived yet
        now = _dt(2026, 6, 2, 0, 5)
        fire, as_of = should_fire(now, None, FIRE_TIME)
        assert fire is False
        assert as_of is None

    def test_idempotent_multiple_calls_same_second(self) -> None:
        """Multiple calls in the same second produce the same result."""
        now = _dt(2026, 6, 2, 1, 0)
        r1 = should_fire(now, "2026-06-01", FIRE_TIME)
        r2 = should_fire(now, "2026-06-01", FIRE_TIME)
        assert r1 == r2

    def test_as_of_arithmetic_matches_signal_service_convention(self) -> None:
        """
        Signal service stamps the last fully-closed daily bar as:
            as_of = (UTC_date_of_fire - 1 day)

        Validate a concrete example:
        - Bar open: 2026-06-01 00:00 UTC
        - Bar close: 2026-06-02 00:00 UTC
        - Signal stamped: as_of = "2026-06-01"
        - Daemon fires: 2026-06-02 00:15 UTC → as_of = "2026-06-01" ✓
        """
        now = _dt(2026, 6, 2, 0, 15)
        _, as_of = should_fire(now, None, "00:15")
        assert as_of == "2026-06-01"


# ---------------------------------------------------------------------------
# should_retry / retry_window_expired
# ---------------------------------------------------------------------------


def _dt_iso(s: str) -> datetime.datetime:
    """Parse an ISO datetime string (with tz offset) into a datetime."""
    return datetime.datetime.fromisoformat(s)


def test_should_retry_idle_returns_false() -> None:
    assert should_retry(
        now=_dt_iso("2026-05-18T01:00:00+00:00"),
        retry_as_of=None, retry_window_start=None, retry_last_attempt=None,
        interval_seconds=600, window_seconds=10800,
        rebalance_utc_time="00:15") is False


def test_should_retry_interval_not_elapsed() -> None:
    assert should_retry(
        now=_dt_iso("2026-05-18T00:20:00+00:00"),
        retry_as_of="2026-05-17",
        retry_window_start="2026-05-18T00:15:00+00:00",
        retry_last_attempt="2026-05-18T00:15:00+00:00",
        interval_seconds=600, window_seconds=10800,
        rebalance_utc_time="00:15") is False


def test_should_retry_interval_elapsed_within_window() -> None:
    assert should_retry(
        now=_dt_iso("2026-05-18T00:26:00+00:00"),
        retry_as_of="2026-05-17",
        retry_window_start="2026-05-18T00:15:00+00:00",
        retry_last_attempt="2026-05-18T00:15:00+00:00",
        interval_seconds=600, window_seconds=10800,
        rebalance_utc_time="00:15") is True


def test_should_retry_window_expired_returns_false() -> None:
    assert should_retry(
        now=_dt_iso("2026-05-18T03:30:00+00:00"),
        retry_as_of="2026-05-17",
        retry_window_start="2026-05-18T00:15:00+00:00",
        retry_last_attempt="2026-05-18T03:10:00+00:00",
        interval_seconds=600, window_seconds=10800,
        rebalance_utc_time="00:15") is False


def test_should_retry_day_rolled_over_returns_false() -> None:
    assert should_retry(
        now=_dt_iso("2026-05-19T00:30:00+00:00"),
        retry_as_of="2026-05-17",
        retry_window_start="2026-05-18T00:15:00+00:00",
        retry_last_attempt="2026-05-18T02:00:00+00:00",
        interval_seconds=600, window_seconds=10800,
        rebalance_utc_time="00:15") is False


def test_retry_window_expired_true_after_window() -> None:
    assert retry_window_expired(
        now=_dt_iso("2026-05-18T03:30:00+00:00"),
        retry_window_start="2026-05-18T00:15:00+00:00",
        window_seconds=10800) is True


def test_retry_window_expired_false_within_window() -> None:
    assert retry_window_expired(
        now=_dt_iso("2026-05-18T02:00:00+00:00"),
        retry_window_start="2026-05-18T00:15:00+00:00",
        window_seconds=10800) is False
