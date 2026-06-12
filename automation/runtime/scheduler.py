"""
automation.runtime.scheduler — Once-per-day, idempotent rebalance scheduler.

Closed-bar convention
---------------------
Hyperliquid perpetual daily bars close at 00:00 UTC.  When the daemon fires at
e.g. ``00:15 UTC`` on ``2026-06-02``, the most-recently-closed daily bar is the
bar whose open was ``2026-06-01 00:00 UTC`` and whose close was
``2026-06-02 00:00 UTC``.  The signal service stamps such a bar as
``as_of = "2026-06-01"`` (the UTC *date* of the bar's open/start).

Therefore:
    ready = now.timetz() >= HH:MM
    as_of = (now.date() - timedelta(days=1)).isoformat()

This matches how the signal service produces its ``as_of`` dates: the last
fully-closed daily bar is always yesterday's UTC date once the UTC day has
turned past the configured fire time.

Idempotency is enforced by ``should_fire`` via ``last_scheduled_as_of``:
once ``state.last_scheduled_as_of == as_of``, the daemon will not re-fire
for that same day regardless of how many times the loop ticks.
"""

from __future__ import annotations

import datetime


def expected_as_of(now: datetime.datetime, rebalance_utc_time: str) -> str | None:
    """Return the ``as_of`` date string the daemon should trade, or ``None``.

    Parameters
    ----------
    now:
        Current time.  MUST be tz-aware (typically UTC).
    rebalance_utc_time:
        Scheduled fire time in ``"HH:MM"`` 24-hour format (e.g. ``"00:15"``).

    Returns
    -------
    str | None
        ``"YYYY-MM-DD"`` of the most-recently-closed daily bar that should be
        traded, or ``None`` if the fire time has not yet passed today.

    Raises
    ------
    ValueError
        If ``now`` is a naive datetime (no tzinfo).

    Notes
    -----
    ``now`` may be in ANY timezone — it is CONVERTED to the UTC instant first
    (``now.astimezone(UTC)``), then BOTH the time-of-day comparison and the
    date arithmetic are derived from that single UTC instant.  This is true
    tz-safety: a ``20:10 America/New_York`` (= ``00:10 UTC``) does NOT fire
    against a ``00:15`` UTC threshold, and a ``09:00 America/New_York``
    (= ``13:00 UTC``) DOES fire against a ``12:00`` UTC threshold.

    The returned date is always ``(now_utc.date() - 1 day)``, i.e. the bar that
    closed at 00:00 UTC this morning.  The signal service stamps that bar's
    ``as_of`` as the UTC calendar date of the bar's START (open), which is
    yesterday in UTC.

    If ``now_utc.time() < HH:MM`` the fire time has not yet arrived today, so we
    return ``None`` (not yet ready for today's cycle).
    """
    if now.tzinfo is None:
        raise ValueError(
            "expected_as_of requires a tz-aware datetime; got naive datetime"
        )
    h, m = rebalance_utc_time.split(":")
    fire_time = datetime.time(int(h), int(m))

    # Convert the instant to UTC ONCE, then derive BOTH the time-of-day
    # comparison and the date from this single UTC value (F2).  Comparing a
    # relabelled local wall-clock against a UTC threshold is a correctness bug.
    now_utc = now.astimezone(datetime.UTC)
    ready = now_utc.time() >= fire_time

    if not ready:
        return None

    # The bar that closed at 00:00 UTC today has as_of = yesterday's UTC date.
    as_of_date = now_utc.date() - datetime.timedelta(days=1)
    return as_of_date.isoformat()


def retry_window_expired(
    now: datetime.datetime,
    retry_window_start: str | None,
    window_seconds: int,
) -> bool:
    """True if the retry window opened at ``retry_window_start`` has elapsed.

    A ``None`` start is treated as expired (no window → nothing to extend).
    ``now`` may be in any tz; it is converted to UTC before comparison.
    """
    if retry_window_start is None:
        return True
    start = datetime.datetime.fromisoformat(retry_window_start)
    elapsed = (now.astimezone(datetime.UTC) - start.astimezone(datetime.UTC)).total_seconds()
    return elapsed >= window_seconds


def should_retry(
    *,
    now: datetime.datetime,
    retry_as_of: str | None,
    retry_window_start: str | None,
    retry_last_attempt: str | None,
    interval_seconds: int,
    window_seconds: int,
    rebalance_utc_time: str,
) -> bool:
    """Pure decision: should the daemon fire a retry attempt on this tick?

    True iff a window is OPEN for the current day's as_of, the inter-attempt
    interval has elapsed, and the window has not expired.  Mirror of
    ``should_fire`` — no side effects, all timing via ``now``.

    Parameters
    ----------
    now:
        Current time.  MUST be tz-aware.
    retry_as_of:
        The ``as_of`` date string the pending retry targets (e.g. ``"2026-05-17"``).
        ``None`` means no retry is pending.
    retry_window_start:
        ISO datetime string of when the retry window opened (i.e. when the
        first attempt fired).  ``None`` means no window is open.
    retry_last_attempt:
        ISO datetime string of the most recent retry attempt.  ``None`` means
        no attempt has been made yet.
    interval_seconds:
        Minimum seconds to wait between consecutive retry attempts.
    window_seconds:
        Total duration of the retry window from ``retry_window_start``; once
        this has elapsed the window is closed and no further retries fire.
    rebalance_utc_time:
        Scheduled fire time in ``"HH:MM"`` format — used to compute the
        current day's expected ``as_of`` via ``expected_as_of()``.
    """
    if retry_as_of is None or retry_window_start is None or retry_last_attempt is None:
        return False
    # The current day's expected as_of must match the pending retry's as_of.
    # expected_as_of returns a str (e.g. "2026-05-17") or None — both compare
    # cleanly to retry_as_of which is also a str.
    if expected_as_of(now, rebalance_utc_time) != retry_as_of:
        return False
    if retry_window_expired(now, retry_window_start, window_seconds):
        return False
    last = datetime.datetime.fromisoformat(retry_last_attempt)
    since = (now.astimezone(datetime.UTC) - last.astimezone(datetime.UTC)).total_seconds()
    return since >= interval_seconds


def should_recheck_signal(
    *,
    now: datetime.datetime,
    recheck_as_of: str | None,
    recheck_window_start: str | None,
    recheck_last_attempt: str | None,
    interval_seconds: int,
    window_seconds: int,
) -> bool:
    """Pure decision: may the daemon RE-FIRE a no-signal re-check now?

    Unlike ``should_retry`` this gates the FIRE branch of ``run_once`` (the
    restore made ``should_fire`` ready again); the as_of match is implied by
    the caller comparing ``as_of == state.recheck_as_of`` (SP3 §5 — gate
    placement).  No side effects; all timing via ``now``.

    Parameters
    ----------
    now:
        Current time.  MUST be tz-aware.
    recheck_as_of:
        The ``as_of`` date string the pending re-check targets (e.g.
        ``"2026-06-10"``).  ``None`` means no re-check is pending.
    recheck_window_start:
        ISO datetime string of when the re-check window opened (i.e. when
        the signal was first found absent).  ``None`` means no window is open.
    recheck_last_attempt:
        ISO datetime string of the most recent re-check attempt.  ``None``
        means no attempt has been made yet.
    interval_seconds:
        Minimum seconds to wait between consecutive re-check attempts.
    window_seconds:
        Total duration of the re-check window from ``recheck_window_start``;
        once this has elapsed the window is closed and no further re-checks
        fire.
    """
    if recheck_as_of is None or recheck_window_start is None or recheck_last_attempt is None:
        return False
    start = datetime.datetime.fromisoformat(recheck_window_start)
    if (now.astimezone(datetime.UTC) - start.astimezone(datetime.UTC)).total_seconds() > window_seconds:
        return False
    last = datetime.datetime.fromisoformat(recheck_last_attempt)
    since = (now.astimezone(datetime.UTC) - last.astimezone(datetime.UTC)).total_seconds()
    return since >= interval_seconds


def should_fire(
    now: datetime.datetime,
    last_scheduled_as_of: str | None,
    rebalance_utc_time: str,
) -> tuple[bool, str | None]:
    """Decide whether the daemon should fire a rebalance cycle right now.

    Parameters
    ----------
    now:
        Current time.  MUST be tz-aware.
    last_scheduled_as_of:
        The ``as_of`` date of the last cycle that was SCHEDULED (from
        ``state.last_scheduled_as_of``).  ``None`` if the daemon has never fired.
    rebalance_utc_time:
        Scheduled fire time in ``"HH:MM"`` format.

    Returns
    -------
    tuple[bool, str | None]
        ``(fire, as_of)`` where ``fire`` is ``True`` iff a new cycle should be
        submitted and ``as_of`` is the date string to use.  When ``fire`` is
        ``False``, ``as_of`` is ``None`` if the fire time hasn't arrived yet, or
        the already-fired ``last_scheduled_as_of`` if we're idempotently holding.

    Raises
    ------
    ValueError
        If ``now`` is a naive datetime.
    """
    exp = expected_as_of(now, rebalance_utc_time)
    if exp is None:
        # Fire time hasn't passed yet today.
        return False, None
    if exp == last_scheduled_as_of:
        # Already fired for this as_of — idempotent hold.
        return False, exp
    return True, exp
