"""
automation.runtime.intent_ledger — Crash-safe PENDING→DONE intent ledger (MF-7).

An "intent" is written to disk BEFORE an order is submitted to the exchange.
On process restart the executor reads all PENDING intents and reconciles them
against on-chain fill state, ensuring every order is accounted for even if the
process crashed between submit and confirmation.

Persistence guarantees (MF-7)
------------------------------
- All mutations (record_pending, mark_done, reconcile_pending) use the
  tmp+fsync+os.replace pattern: the ledger file is never partially visible.
- A corrupt ledger file raises ``LedgerError`` (never silently resets) — losing
  pending intents would skip crash-recovery and double-fill orders.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pydantic

from automation.core.redaction import get_logger

_logger = get_logger(__name__)


class LedgerError(Exception):
    """Raised when the ledger file exists but cannot be parsed.

    NEVER catch silently — a corrupt ledger means pending intents may be lost,
    which could cause double-order submission on restart.
    """


class Intent(pydantic.BaseModel):
    """A single rebalance intent record.

    Written BEFORE the order is submitted; marked DONE after confirmation.

    Fields
    ------
    cloid:
        Client order ID — the primary key.  Unique per order attempt.
    as_of:
        ISO date string of the TargetAllocation that triggered this order.
    coin:
        The perp name being ordered.
    target_w:
        The target weight (fraction of equity) that drove this order.
    attempt_seq:
        Monotonically increasing retry counter (0 = first attempt).
    status:
        ``"PENDING"`` until the order is confirmed filled; then ``"DONE"``.
    ts:
        ISO timestamp of when the intent was created.  Injected by the caller
        (not datetime.now inside) to keep the class deterministic for tests.
    """

    cloid: str
    as_of: str
    coin: str
    target_w: float
    attempt_seq: int = 0
    status: Literal["PENDING", "DONE"] = "PENDING"
    ts: str


class IntentLedger:
    """Crash-safe intent ledger backed by a single JSON file.

    The in-memory representation is ``dict[cloid, Intent]``.  Every mutation
    is atomically flushed to disk before the method returns.

    Parameters
    ----------
    path:
        Path to the ledger JSON file.  Created on first write if absent.
        If the file exists but is corrupt, ``LedgerError`` is raised at
        construction time (fail-fast).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._intents: dict[str, Intent] = {}

        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                data: dict[str, dict[str, Any]] = json.loads(raw)
                for cloid, record in data.items():
                    self._intents[cloid] = Intent(**record)
                _logger.info(
                    "intent_ledger: loaded",
                    path=str(self._path),
                    total=len(self._intents),
                    pending=len(self.pending()),
                )
            except Exception as exc:
                _logger.error(
                    "intent_ledger: corrupt ledger file",
                    path=str(self._path),
                    error=str(exc),
                )
                raise LedgerError(
                    f"Ledger file at {self._path} is corrupt or unparseable: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def record_pending(self, intent: Intent) -> None:
        """Add or overwrite an intent by cloid and persist atomically.

        Parameters
        ----------
        intent:
            The Intent to record.  If a record with the same cloid already
            exists it is overwritten (idempotent re-submission guard).
        """
        self._intents[intent.cloid] = intent
        self._flush()
        _logger.info(
            "intent_ledger: recorded PENDING",
            cloid=intent.cloid,
            coin=intent.coin,
            as_of=intent.as_of,
        )

    def mark_done(self, cloid: str) -> None:
        """Mark an intent as DONE and persist atomically.

        Parameters
        ----------
        cloid:
            The client order ID to mark as done.

        Raises
        ------
        KeyError
            When *cloid* is not in the ledger.
        """
        if cloid not in self._intents:
            raise KeyError(f"Intent {cloid!r} not found in ledger")
        updated = self._intents[cloid].model_copy(update={"status": "DONE"})
        self._intents[cloid] = updated
        self._flush()
        _logger.info("intent_ledger: marked DONE", cloid=cloid)

    def pending(self) -> list[Intent]:
        """Return all intents with status == PENDING.

        Returns
        -------
        list[Intent]
            Intents still awaiting fill confirmation, in insertion order.
        """
        return [i for i in self._intents.values() if i.status == "PENDING"]

    def reconcile_pending(
        self, is_filled: Callable[[Intent], bool]
    ) -> list[Intent]:
        """Resolve pending intents against on-chain fill state.

        For each PENDING intent, calls ``is_filled(intent)`` (injected resolver
        that in production queries the exchange by cloid / fills endpoint).
        If the resolver returns ``True``, the intent is marked DONE.  A single
        atomic flush is performed at the end.

        Parameters
        ----------
        is_filled:
            Callable receiving an Intent and returning ``True`` if the order
            has been confirmed as filled on-chain, ``False`` otherwise.

        Returns
        -------
        list[Intent]
            The intents that remain PENDING after reconciliation (unresolved —
            the executor must re-handle or escalate these).
        """
        # F2: persist DURABLY per-DONE-transition.  If we flushed only once after
        # the loop and the injected resolver RAISED partway (e.g. the exchange
        # query failed on intent N), every already-resolved DONE would be lost on
        # disk — on restart they'd still be PENDING and get REPLAYED (double-fill).
        # Flushing after each transition costs an extra fsync but guarantees that a
        # partial reconcile is durable; the raise then propagates to the caller.
        for cloid, intent in list(self._intents.items()):
            if intent.status != "PENDING":
                continue
            if is_filled(intent):
                self._intents[cloid] = intent.model_copy(update={"status": "DONE"})
                self._flush()  # durable BEFORE the next resolver call can raise
                _logger.info(
                    "intent_ledger: reconcile → DONE",
                    cloid=cloid,
                    coin=intent.coin,
                )
            else:
                _logger.info(
                    "intent_ledger: reconcile → still PENDING",
                    cloid=cloid,
                    coin=intent.coin,
                )

        return self.pending()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Atomically write the entire ledger to disk (tmp+fsync+os.replace)."""
        tmp = Path(str(self._path) + ".tmp")
        payload = json.dumps(
            {cloid: intent.model_dump() for cloid, intent in self._intents.items()},
            indent=2,
        )
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
