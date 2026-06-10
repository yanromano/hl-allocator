"""
automation.core.nonce — Persistent, monotonically-increasing nonce for Hyperliquid.

Hyperliquid nonces are millisecond timestamps (per-signer, strictly increasing,
window T−2d to T+1d, server keeps only the 100 highest).  This module ensures:

1. ``next()`` always returns a value strictly greater than the last issued nonce.
2. The nonce is persisted atomically (write-tmp → fsync → os.replace) BEFORE being
   returned, so a crash after persist but before the caller uses it is safe (the
   next call will simply increment above the persisted value).
3. After a daemon restart a new ``NonceManager`` loaded from the same path continues
   strictly above whatever was last persisted.
4. A backward clock step beyond ``backward_tolerance_ms`` raises ``ClockError``
   so the caller can alert and refuse to send stale nonces.

Forward-clock excursion (F5 / G6)
---------------------------------
``next()`` only *guards* against a BACKWARD clock step.  A FORWARD jump (e.g. the
clock briefly reads several days ahead) cannot be distinguished locally from a
legitimate multi-hour/day downtime gap, so refusing on it would block legitimate
recovery.  Instead, a forward jump larger than ``forward_alarm_ms`` is made
VISIBLE (WARNING log + optional ``on_alarm`` callback) but NOT refused.

The danger a forward jump creates: it persists ``last = now + jump``; once the
clock corrects, every subsequent ``next()`` raises a backward ``ClockError`` for
the duration of the jump (the local file is "poisoned"), and the far-future
nonce is outside Hyperliquid's accept window.  FULL protection requires
re-anchoring ``last`` from the on-chain highest-accepted nonce at boot — that
needs a live network call and is handled by B11 (gap G6) calling
:meth:`NonceManager.reanchor`.  This module provides the ``reanchor`` seam and
the forward alert; it does NOT perform the on-chain lookup itself.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from automation.core.redaction import get_logger

logger = get_logger(__name__)

# Default forward-jump alarm threshold: 6 hours in milliseconds.
_DEFAULT_FORWARD_ALARM_MS = 6 * 3600 * 1000


class ClockError(Exception):
    """Raised when the system clock appears to have jumped backward beyond tolerance."""


class NonceManager:
    """Thread-safe, persistent nonce manager for a single Hyperliquid signer.

    Parameters
    ----------
    state_path:
        Path to a plain-text file that stores the last issued nonce as a single
        line.  On each ``next()`` call the file is atomically OVERWRITTEN
        (write-to-tmp → fsync → ``os.replace``) — it is never appended to, so it
        always contains exactly one integer and readers never see a partial
        write.  Only the last line is read on load (tolerant of a legacy
        multi-line file).
    backward_tolerance_ms:
        How many milliseconds the wall clock is allowed to drift backward before
        ``next()`` raises ``ClockError``.  The default (5 000 ms) protects against
        NTP corrections without being overly strict.
    forward_alarm_ms:
        How far forward (ms) ``now`` may exceed the last issued nonce before
        ``next()`` emits a forward-jump alarm (WARNING log + ``on_alarm``
        callback).  Default 6 h.  A forward jump is NEVER refused — full
        protection requires the boot-time :meth:`reanchor` call with the on-chain
        nonce (B11 / G6).
    on_alarm:
        Optional callback invoked with a human-readable message string whenever
        a forward-jump alarm fires.  Use it to surface the event to an alerting
        channel.  Exceptions raised by the callback are caught and logged so a
        broken alert sink never blocks nonce issuance.
    """

    def __init__(
        self,
        state_path: str | Path,
        backward_tolerance_ms: int = 5000,
        forward_alarm_ms: int = _DEFAULT_FORWARD_ALARM_MS,
        on_alarm: Callable[[str], None] | None = None,
    ) -> None:
        self._path = Path(state_path)
        #: Durable mirror of the primary state file. Written on every persist and
        #: read at boot so the seed is never below ``max(primary, mirror)`` — the
        #: nonce cannot regress if the primary file is lost but the mirror survives.
        self._mirror_path = self._path.with_name(self._path.name + ".bak")
        self._tolerance = backward_tolerance_ms
        self._forward_alarm_ms = forward_alarm_ms
        self._on_alarm = on_alarm
        self._lock = threading.Lock()
        self._last: int = self._load()
        logger.debug("NonceManager initialised", path=str(self._path), last=self._last)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next(self) -> int:
        """Return the next nonce, persisting it atomically first.

        Algorithm
        ---------
        1. If ``now_ms < last - tolerance`` → raise ``ClockError`` (clock jumped
           backward beyond acceptable tolerance; do NOT issue a nonce).
        2. If ``now_ms - last > forward_alarm_ms`` → emit a forward-jump alarm
           (WARNING log + ``on_alarm`` callback) but do NOT refuse — a long
           legitimate downtime gap is indistinguishable locally.  Only the
           boot-time :meth:`reanchor` (B11 / G6) can truly self-heal a poisoned
           far-future ``last``.
        3. ``n = max(now_ms, last + 1)`` — always strictly increasing.
        4. Persist ``n`` atomically (tmp file + fsync + os.replace).
        5. Update ``self._last = n``.
        6. Return ``n``.

        Raises
        ------
        ClockError
            If the wall clock appears to have jumped backward by more than
            ``backward_tolerance_ms``.
        """
        with self._lock:
            now_ms = int(time.time() * 1000)

            # Guard: backward clock step beyond tolerance
            if now_ms < self._last - self._tolerance:
                raise ClockError(
                    f"Clock jumped backward: now={now_ms} ms, last={self._last} ms, "
                    f"delta={self._last - now_ms} ms > tolerance={self._tolerance} ms"
                )

            # Alarm (NOT a refusal): forward clock excursion beyond threshold.
            # A forward jump poisons the local file (persists last=now+jump);
            # full protection needs the on-chain reanchor at boot (G6).
            forward_delta = now_ms - self._last
            if self._last > 0 and forward_delta > self._forward_alarm_ms:
                self._emit_forward_alarm(now_ms, forward_delta)

            n = max(now_ms, self._last + 1)
            self._persist(n)
            self._last = n
            return n

    def reanchor(self, min_nonce: int) -> None:
        """Raise the persisted ``last`` to at least ``min_nonce`` (recovery seam).

        Sets ``last = max(current_last, min_nonce)`` and persists atomically.
        This is the boot-time self-heal seam (B11 / G6): the caller passes the
        on-chain highest-accepted nonce so a locally-poisoned far-future ``last``
        is reconciled with what the exchange actually accepted.

        Re-anchoring to a value at or below the current ``last`` is a no-op
        (``last`` never moves backward — that would risk issuing a duplicate
        nonce).  This method performs NO network call; the caller is responsible
        for fetching the on-chain nonce.

        Parameters
        ----------
        min_nonce:
            The floor the persisted ``last`` must meet or exceed.
        """
        with self._lock:
            new_last = max(self._last, min_nonce)
            if new_last == self._last:
                logger.debug(
                    "reanchor no-op (min_nonce <= current last)",
                    min_nonce=min_nonce,
                    last=self._last,
                )
                return
            self._persist(new_last)
            self._last = new_last
            logger.info("NonceManager reanchored", new_last=new_last, min_nonce=min_nonce)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_forward_alarm(self, now_ms: int, forward_delta: int) -> None:
        """Log a WARNING and invoke ``on_alarm`` for a forward clock excursion."""
        msg = (
            f"Forward clock excursion: now={now_ms} ms exceeds last={self._last} ms "
            f"by {forward_delta} ms (> forward_alarm_ms={self._forward_alarm_ms} ms). "
            "Nonce NOT refused; on-chain reanchor required for full protection (G6)."
        )
        logger.warning(
            "Forward clock excursion (nonce not refused)",
            now_ms=now_ms,
            last=self._last,
            forward_delta_ms=forward_delta,
            forward_alarm_ms=self._forward_alarm_ms,
        )
        if self._on_alarm is not None:
            try:
                self._on_alarm(msg)
            except Exception as exc:  # noqa: BLE001 — a broken alert sink must not block issuance
                logger.warning("on_alarm callback raised; ignoring", error=str(exc))

    def _load(self) -> int:
        """Seed ``last`` from ``max(primary, mirror)`` — never below either file.

        Reads BOTH the primary state file and its durable mirror (``<path>.bak``).
        Each is read defensively: a missing, empty, or corrupt file contributes 0
        rather than raising.  The seed is the maximum of the two, so the nonce can
        never regress when the primary is lost but the mirror survives (or vice
        versa).  No time floor is applied here — ``next()`` applies the ``now_ms``
        floor — but the file floor is honoured so a persisted FUTURE value (e.g.
        after :meth:`reanchor` to an on-chain nonce) survives primary loss.
        """
        primary = self._read_one(self._path)
        mirror = self._read_one(self._mirror_path)
        seed = max(primary, mirror)
        if mirror > primary:
            logger.info(
                "Nonce primary below mirror; recovering from mirror",
                primary=primary,
                mirror=mirror,
                seed=seed,
            )
        return seed

    @staticmethod
    def _read_one(path: Path) -> int:
        """Read a single integer nonce from ``path``, or 0 if absent/empty/corrupt."""
        if not path.exists():
            return 0
        try:
            text = path.read_text().strip()
            if not text:
                return 0
            return int(text.splitlines()[-1].strip())
        except (ValueError, OSError) as exc:
            logger.warning(
                "Could not read nonce state file; treating as 0",
                path=str(path),
                error=str(exc),
            )
            return 0

    def _persist(self, n: int) -> None:
        """Atomically write ``n`` to the primary state file AND the mirror.

        Each file is written independently via write-to-tmp + fsync + os.replace
        so neither is ever left partial/corrupt if the process is killed
        mid-write.  The primary is written first; a crash between the two writes
        is safe because boot seeds from ``max(primary, mirror)``.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._path, n)
        self._atomic_write(self._mirror_path, n)

    @staticmethod
    def _atomic_write(path: Path, n: int) -> None:
        """Atomically write ``n`` to ``path`` (write-to-tmp → fsync → os.replace)."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w") as fh:
                fh.write(f"{n}\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError:
            # Clean up tmp if replace failed; re-raise so caller is informed
            tmp.unlink(missing_ok=True)
            raise
