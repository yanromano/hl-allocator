"""Timestamp parsing helpers — one canonical ISO-8601 → tz-aware-UTC parse.

The daemon/rebalance loops repeatedly parsed an ISO timestamp and coerced a
naive value to UTC inline (``fromisoformat(ts)`` + ``if tzinfo is None:
replace(tzinfo=UTC)``).  :func:`parse_utc` is that pattern, once, so the call
sites read intent instead of boilerplate.
"""

from __future__ import annotations

import datetime


def parse_utc(ts: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp into a tz-AWARE UTC ``datetime``.

    Accepts a trailing ``Z`` (normalised to ``+00:00``) and treats a NAIVE
    timestamp as already-UTC (the loop's invariant: every persisted ``*_ts`` is
    a UTC instant).  An offset-aware timestamp keeps its instant unchanged.
    """
    dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt
