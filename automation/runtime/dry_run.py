"""
automation.runtime.dry_run — Offline dry-run pipeline (sends NOTHING).

This module exercises the full snapshot → validate → size → plan pipeline
without touching the ledger, the exchange, or persistent state.  It is the
offline safety harness for:

- Verifying that a new signal would trigger a rebalance (or hold) without
  committing anything.
- Pre-flight inspection before a live run.
- CI / integration tests where no Hyperliquid connection is available.

Guarantees
----------
- ``dry_run_cycle`` NEVER submits an order.
- ``dry_run_cycle`` NEVER writes to the intent ledger.
- ``dry_run_cycle`` NEVER mutates state.
- The returned ``SizePlan`` is the same plan the live executor would act on.
"""

from __future__ import annotations

import datetime
from typing import Any

from automation.allocation.base import AllocationSource, validate_target_allocation
from automation.core.config import Config
from automation.core.redaction import get_logger
from automation.execution import reconciler, sizer
from automation.execution.sizer import SizePlan
from automation.reporting.state import State

_logger = get_logger(__name__)


def dry_run_cycle(
    cfg: Config,
    client: Any,
    source: AllocationSource,
    state: State,
    *,
    as_of: datetime.date | None = None,
) -> SizePlan:
    """Run the full planning pipeline in read-only mode — submits NOTHING.

    Steps
    -----
    1. Fetch an on-chain snapshot via ``reconciler.snapshot(client)``.
    2. Retrieve the target allocation via ``source.get_target_allocation(as_of)``.
    3. Validate the allocation via ``validate_target_allocation``.
    4. Compute the size plan via ``sizer.plan``.
    5. Log the plan summary.
    6. Return the plan — no orders are submitted, no state is mutated.

    Parameters
    ----------
    cfg:
        Executor configuration.
    client:
        Duck-typed client (``HLClient`` or fake) exposing read methods.
    source:
        Allocation source implementing the ``AllocationSource`` protocol.
    state:
        Current persistent executor state (read-only).
    as_of:
        Optional reference date forwarded to ``source.get_target_allocation``.
        If ``None``, the source uses its own internal clock.

    Returns
    -------
    SizePlan
        The deterministic plan that a live executor would act on.

    Raises
    ------
    AllocationRejected
        If the allocation fails local validation (propagates — same behaviour
        as the live executor so dry-run catches signal corruption early).
    """
    # D5: reflect the hybrid spot+perp book in the dry-run snapshot — equity spans
    # both pools and spot coins are venue-aware — so a dry-run mirrors what the live
    # cycle would size.  Empty/absent spot_routing ⇒ None ⇒ unchanged perp-only read.
    # Still sends NOTHING (this is the offline harness).
    snap = reconciler.snapshot(client, spot_routing=cfg.spot_routing or None)

    ta = source.get_target_allocation(as_of)

    whitelist = set(cfg.universe_perp_map) if cfg.universe_perp_map else set(ta.weights.keys())
    client_id = cfg.signal_source.client_id or ""
    validate_target_allocation(
        ta,
        max_per_asset=cfg.max_per_asset,
        whitelist=whitelist,
        client_id=client_id,
    )

    sp = sizer.plan(ta, snap, state, cfg)

    _logger.info(
        "dry_run_cycle: plan computed (DRY RUN — no orders submitted)",
        rebalance=sp.rebalance,
        reason=sp.reason,
        n_orders=len(sp.orders),
        gross_notional=round(sp.gross_notional, 2),
        scaled_down=sp.scaled_down,
        n_skipped=len(sp.skipped),
        equity=round(sp.equity, 2),
    )

    return sp
