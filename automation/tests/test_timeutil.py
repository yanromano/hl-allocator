"""Tests for automation.core.timeutil.parse_utc."""
import datetime

from automation.core.timeutil import parse_utc


def test_naive_treated_as_utc() -> None:
    dt = parse_utc("2026-06-14T00:03:00")
    assert dt.tzinfo == datetime.UTC
    assert dt == datetime.datetime(2026, 6, 14, 0, 3, tzinfo=datetime.UTC)


def test_z_suffix_normalized() -> None:
    dt = parse_utc("2026-06-14T00:03:00Z")
    assert dt.tzinfo is not None
    assert dt == datetime.datetime(2026, 6, 14, 0, 3, tzinfo=datetime.UTC)


def test_offset_aware_instant_preserved() -> None:
    # +02:00 wall clock 02:03 == 00:03 UTC — same instant, tz preserved.
    dt = parse_utc("2026-06-14T02:03:00+02:00")
    assert dt.astimezone(datetime.UTC) == datetime.datetime(2026, 6, 14, 0, 3, tzinfo=datetime.UTC)


def test_roundtrip_isoformat() -> None:
    src = "2026-06-14T23:59:30+00:00"
    assert parse_utc(src) == datetime.datetime.fromisoformat(src)
