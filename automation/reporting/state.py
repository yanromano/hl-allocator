"""
automation.reporting.state — Persistent executor state with atomic JSON persistence.

The State object records the FROZEN last-rebalanced target weights, which serve as
the ratchet baseline for the sizer's all-or-nothing threshold check.

Persistence guarantees (MF-7)
------------------------------
- Writes use the tmp+fsync+os.replace pattern: data is NEVER partially visible.
- Corrupt / unparseable files RAISE ``StateError`` (never silently reset) — losing
  the frozen baseline would cause a spurious full rebalance on restart (S-2).
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

import pydantic

from automation.core.redaction import get_logger

_logger = get_logger(__name__)


class StateError(Exception):
    """Raised when the state file exists but cannot be parsed.

    NEVER catch silently — a corrupt baseline means we cannot determine whether
    the last rebalance fired, so execution must halt until human review.
    """


class State(pydantic.BaseModel):
    """Persistent executor state.

    Fields
    ------
    last_rebalanced_target:
        The FROZEN target weights from the last completed rebalance cycle
        (strategy ticker → fraction, matching the TargetAllocation.weights that
        was acted on).  ``None`` when no rebalance has ever completed (cycle-0).
        This is the ratchet baseline: the sizer compares the NEW target against
        this to decide whether threshold is crossed — it is NOT the achieved/live
        weights on Hyperliquid (those come from achieved_weights() in reconciler).
    last_rebalanced_as_of:
        ISO date string (``"YYYY-MM-DD"``) of the TargetAllocation.as_of that
        was acted on, or ``None`` for cycle-0.
    schema_version:
        Monotonically increasing schema revision for forward-compatibility.
    """

    last_rebalanced_target: dict[str, float] | None = None
    last_rebalanced_as_of: str | None = None
    schema_version: int = 1
    last_scheduled_as_of: str | None = None
    """ISO date string of the last ``as_of`` that was SCHEDULED by the daemon
    scheduler, regardless of whether execution succeeded.  Written by
    ``Daemon.run_once`` after a cycle fires (even on failure) to enforce
    once-per-day idempotency.  ``None`` when the daemon has never fired."""
    last_successful_signal_ts: str | None = None
    """ISO timestamp of the last cycle in which ``source.get_target_allocation``
    both succeeded AND passed ``validate_target_allocation``.  Used to drive the
    N-3 staleness clock: when ``now - last_successful_signal_ts`` exceeds
    ``cfg.max_signal_staleness_days``, the executor de-risks to CASH via a
    reduce-only ``flatten_all``.  ``None`` when no good signal has ever been
    received (system has not yet bootstrapped a verified allocation)."""
    first_cycle_ts: str | None = None
    """ISO timestamp of the FIRST ``run_cycle`` invocation for this deployment.
    Written once (when ``None``) at the top of ``run_cycle`` and never updated.
    It is the staleness anchor when no good signal has EVER arrived: a fresh
    deploy whose very first signal fails does NOT flatten immediately — it holds
    until ``max_signal_staleness_days`` elapse since first boot, then de-risks any
    orphan / pre-existing positions to cash.  ``None`` only before the very first
    cycle has run."""
    retry_as_of: str | None = None
    """ISO date of the as_of with an OPEN bounded-retry window, or ``None`` (IDLE).
    When set, the daemon re-attempts the residual toward ``retry_target`` until the
    cycle is clean or the window expires."""
    retry_target: dict[str, float] | None = None
    """The FROZEN target weights the open retry window is driving toward (NOT yet
    committed to ``last_rebalanced_target``).  ``None`` when IDLE."""
    retry_window_start: str | None = None
    """ISO UTC timestamp when the current retry window opened.  Drives expiry."""
    retry_last_attempt: str | None = None
    """ISO UTC timestamp of the last retry attempt (or window-open).  Drives the
    inter-attempt interval gate."""
    retry_attempt_count: int = 0
    """Number of retry attempts made in the current window (telemetry / alert)."""
    recheck_as_of: str | None = None
    """ISO date of the as_of with an OPEN no-signal re-check window (None = IDLE).
    Set on a no-good-signal HOLD when ``signal_recheck.enabled``; the daemon
    re-fires the same as_of on a bounded cadence until a non-hold outcome or
    expiry (SP3 §5)."""
    recheck_window_start: str | None = None
    """ISO UTC timestamp when the re-check window opened (first hold)."""
    recheck_last_attempt: str | None = None
    """ISO UTC timestamp of the most recent re-check attempt (gates cadence)."""
    recheck_attempt_count: int = 0
    """Re-check attempts in the current window (telemetry / alert context)."""
    recheck_exhausted_as_of: str | None = None
    """ISO date whose re-check window EXPIRED (deliberately consumed) — boot
    crash-detection must not WARN for this day."""
    sleeves: dict[str, dict[str, Any]] = pydantic.Field(default_factory=dict)
    """Per-sleeve ledger (CRASH Phase 1, spec §4.2).  Maps each sleeve ``name``
    to its last-serve record:
    ``{"last_weights": dict[str, float], "last_as_of": str, "last_success_ts": str,
    "consecutive_holds": int}``.  ``last_weights`` are SIGNED, sleeve-relative
    weights (negative entries for short sleeves).  Written on each sleeve's
    successful serve; this replaces the single global ``last_successful_signal_ts``
    staleness anchor with a PER-SLEEVE clock (the global field remains for the
    legacy single-source path).  ``consecutive_holds`` counts consecutive daily
    HOLDs since the last fresh serve — a short sleeve de-risks (STALE) after the
    second consecutive HOLD (spec §4.3, red-team H2).  Empty ``{}`` until any
    sleeve has served; LEGACY state files without this key load with ``{}``
    (NO migration — red-team H7)."""
    liq_cooldowns: dict[str, str] = pydantic.Field(default_factory=dict)
    """Post-liquidation re-entry cooldown map (CRASH Phase 1, spec §5/M3).
    Maps a ``coin`` to the ISO date (``"YYYY-MM-DD"``) on which an apparent
    liquidation (position present in the baseline but absent from the live
    snapshot with no committed exit order) was detected.  Same-direction
    re-entries for that coin are skipped until ``liquidation_cooldown_days``
    elapse.  Empty ``{}`` by default; LEGACY state files without this key load
    with ``{}`` (NO migration — red-team H7)."""


def load(path: str | Path) -> State:
    """Load State from *path*.

    Parameters
    ----------
    path:
        Path to the JSON state file.

    Returns
    -------
    State
        A fresh ``State()`` with all defaults when the file does not exist.

    Raises
    ------
    StateError
        When the file exists but cannot be parsed as valid JSON or does not
        conform to the State schema.  NEVER silently returns defaults in this
        case — a corrupt file could mask a missed rebalance.
    """
    p = Path(path)
    if not p.exists():
        _logger.info("state.load: no state file found, returning defaults", path=str(p))
        return State()

    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        state = State(**data)
        _logger.info(
            "state.load: loaded",
            path=str(p),
            last_rebalanced_as_of=state.last_rebalanced_as_of,
        )
        return state
    except Exception as exc:
        _logger.error("state.load: corrupt state file", path=str(p), error=str(exc))
        raise StateError(
            f"State file at {p} is corrupt or unparseable: {exc}"
        ) from exc


def save(state: State, path: str | Path) -> None:
    """Atomically persist *state* to *path* (tmp + fsync + os.replace).

    The write is crash-safe: if the process dies mid-write, *path* retains its
    previous content (the .tmp file is cleaned up by the OS on crash).

    Parameters
    ----------
    state:
        The State object to persist.
    path:
        Destination file path.  The parent directory must already exist.
    """
    p = Path(path)
    tmp = Path(str(p) + ".tmp")

    payload = state.model_dump_json(indent=2)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        _logger.info("state.save: persisted atomically", path=str(p))
    except Exception:
        # Best-effort cleanup of the .tmp file on failure.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
