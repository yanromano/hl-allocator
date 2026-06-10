"""
automation.reporting.audit — Append-only JSONL audit log.

Every cycle writes exactly ONE compact JSON object followed by a newline.  The
file is fsync'd on each append so records survive a crash between writes.

The audit log is intentionally SEPARATE from the state file and ledger: it is
the human-readable, non-lossy history of every rebalance cycle and is never
read back by the daemon itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from automation.core.redaction import get_logger

if TYPE_CHECKING:
    from automation.runtime.rebalance import CycleReport

_logger = get_logger(__name__)


def cycle_record(
    *,
    ts: str,
    as_of: str | None,
    report: CycleReport,
) -> dict[str, Any]:
    """Build a non-secret audit record from a CycleReport.

    This is a PURE function — it never writes to disk.  Tests can call it
    directly without touching the filesystem.

    Parameters
    ----------
    ts:
        ISO timestamp of the cycle run (injected by caller).
    as_of:
        ISO date of the as_of that was scheduled (the closed-bar date).
    report:
        The CycleReport returned by ``run_cycle``.

    Returns
    -------
    dict
        A flat, non-secret record suitable for JSON serialisation:
        ``{ts, as_of, rebalanced, reason, committed, data_age_days, n_submitted,
           n_filled, n_error, n_partial, aborted, coins:[{coin, status,
           filled_size, attempts}...]}``.  ``data_age_days`` is a FORENSIC STAMP
        (how stale the served bar was, in whole days; ``-1`` on the no-good-signal
        / de-risk path) — observability only, no alert is derived from it.
    """
    er = report.exec_report

    # Aggregate fields — default to zeros when no exec_report
    n_submitted: int = er.n_submitted if er is not None else 0
    n_filled: int = er.n_filled if er is not None else 0
    n_error: int = er.n_error if er is not None else 0
    n_partial: int = er.n_partial if er is not None else 0
    aborted: bool = er.aborted if er is not None else False
    deferred: bool = er.deferred if er is not None else False
    n_throttled: int = er.n_throttled if er is not None else 0

    # Per-coin detail
    coins: list[dict[str, Any]] = []
    if er is not None:
        for r in er.results:
            coins.append(
                {
                    "coin": r.coin,
                    "status": r.status,
                    "filled_size": r.filled_size,
                    "attempts": r.attempts,
                }
            )

    return {
        "ts": ts,
        "as_of": as_of,
        "rebalanced": report.rebalanced,
        "reason": report.reason,
        "committed": report.committed,
        "signal_ok": report.signal_ok,
        "derisked": report.derisked,
        "derisk_incomplete": report.derisk_incomplete,
        "data_age_days": report.data_age_days,
        "max_divergence": report.max_divergence,
        "n_unallocated": report.n_unallocated,
        "n_submitted": n_submitted,
        "n_filled": n_filled,
        "n_error": n_error,
        "n_partial": n_partial,
        "aborted": aborted,
        "deferred": deferred,
        "n_throttled": n_throttled,
        "coins": coins,
    }


class AuditLog:
    """Append-only JSONL audit log.

    Each ``write()`` call appends ONE compact JSON record followed by ``\\n``
    and calls ``flush()`` + ``os.fsync()`` for durability.  The parent
    directory is created automatically if it does not exist.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` audit file.  The file is created on first write.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        """Append ``record`` as a compact JSON line.

        The line is ``json.dumps(record, sort_keys=True, separators=(",",":"))``
        followed by ``"\\n"``.  The file descriptor is flushed and fsync'd
        before returning so the record survives a crash immediately after this
        call returns.

        Parameters
        ----------
        record:
            Any JSON-serialisable dict.  The caller is responsible for ensuring
            no secrets are present — ``cycle_record`` never includes them.
        """
        # default=str ensures a record ALWAYS serialises (F4): a Decimal or any
        # other non-JSON-native value falls back to str() rather than raising
        # TypeError and silently losing the audit line in the caller's try/except.
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        _logger.info("audit.write: appended record", path=str(self._path))
