"""
automation.execution.executor — IOC order executor with cloid+attempt retry.

Critical invariants:
- MF-7 (write-before-submit): the PENDING intent is recorded in the ledger
  BEFORE the client call is made.  A crash between write and submit leaves a
  PENDING entry that boot-time reconciliation resolves.
- MF-6 (distinct cloids per attempt): ``make_cloid`` hashes the attempt_seq
  AND the cycle nonce (now_ts) into the cloid so partial-fill retries are
  distinguishable on-chain AND re-runs of the same as_of get fresh cloids.
- Per-coin isolation: a single-coin failure (OrderRejected, RateLimited, or
  unexpected exception) logs the error and continues to the next coin — it
  does NOT abort the whole rebalance cycle.
- Full exits (target_size=0, side="sell") use ``market_close`` (reduce-only)
  in a SINGLE shot — reduce-only can't over-close, so we never drive the
  size-residual retry loop for it (avoids double-counting fills).
- Pre-flight cap enforcement: before submitting ANY order, the whole order set
  is checked against ``cfg.caps`` (per-coin / total notional / turnover).  A
  CapBreach aborts the ENTIRE cycle BEFORE any submit (n_submitted=0, aborted).
- Venue routing (D4): each coin is routed to its venue.  Coins listed in the
  ``spot_routing`` map (coin → HL spot pair, e.g. ``{"HYPE": "@107"}``) trade on
  SPOT via ``submit_spot_ioc``; everything else trades on PERP via
  ``submit_ioc`` / ``market_close``.  SPOT has NO reduce-only ``market_close`` —
  a spot reduce OR full exit is just a SELL ``submit_spot_ioc(is_buy=False)``.
  ``spot_routing=None`` is byte-identical to the pre-D4 perp-only path.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from automation.core.config import Config
from automation.core.redaction import get_logger
from automation.core.safety import (
    CapBreach,
    OrderRejected,
    RateLimited,
    enforce_caps,
    parse_order_response,
)
from automation.execution.sizer import OrderPlan, SizePlan
from automation.runtime.intent_ledger import Intent, IntentLedger

_logger = get_logger(__name__)

# Residual size below this fraction of 1 unit is treated as effectively zero
# (avoids infinite retry loops when fill precision exceeds sz_decimals).
_FILL_EPS: float = 1e-8


def make_cloid(
    strategy_id: str,
    as_of: str,
    coin: str,
    attempt_seq: int,
    now_ts: str,
) -> str:
    """Deterministic 128-bit client order ID.

    The cloid is deterministic WITHIN a cycle (same ``now_ts``) so the executor
    can re-derive a cloid for crash recovery, but FRESH across re-runs of the
    same ``as_of`` (different ``now_ts``) so a re-run never reuses a cloid that
    the exchange already saw (which would be a duplicate-cloid rejection) or
    overwrites a prior ledger entry (F-7).

    Parameters
    ----------
    strategy_id:
        Opaque strategy identifier (e.g. ``"haarp-v3"``).
    as_of:
        ISO date string of the TargetAllocation that triggered this order
        (e.g. ``"2026-06-01"``).
    coin:
        Perp name (e.g. ``"BTC"``).
    attempt_seq:
        Monotonically increasing retry counter (0 = first attempt).  Changing
        this produces a distinct cloid so retries are distinguishable on-chain
        (MF-6).
    now_ts:
        ISO timestamp of THIS cycle.  Acts as a per-cycle nonce: two coins in
        the same cycle share ``now_ts`` but differ on ``coin`` (so still
        distinct); a re-run of the same ``as_of`` carries a new ``now_ts`` and
        therefore a fresh cloid (F-7).

    Returns
    -------
    str
        ``"0x"`` followed by 32 lowercase hex characters (16 bytes from SHA-256).
    """
    raw = f"{strategy_id}|{as_of}|{coin}|{attempt_seq}|{now_ts}".encode()
    digest = hashlib.sha256(raw).digest()[:16]
    return "0x" + digest.hex()


@dataclass(frozen=True)
class CoinResult:
    """Execution result for a single coin in the rebalance plan."""

    coin: str
    requested_size: float
    filled_size: float
    attempts: int
    status: Literal["filled", "partial", "error", "skipped"]
    cloids: list[str] = field(default_factory=list)
    error: str | None = None
    resting_anomaly: bool = False
    throttled: bool = False
    """``True`` when this coin's submit raised ``RateLimited`` mid-cycle."""
    liquidity_deferred: bool = False
    """``True`` when the pre-trade liquidity gate DEFERRED this leg: the opposing
    L2 side lacked enough reachable depth within the slippage band, so NO IOC was
    submitted.  A deferred leg is NOT an error — it is retryable next tick."""


@dataclass(frozen=True)
class ExecReport:
    """Aggregate execution report for one rebalance cycle.

    ``aborted`` is True when a pre-flight ``CapBreach`` halted the cycle BEFORE
    any order was submitted — in that case ``n_submitted == 0`` and
    ``results == []`` and the caller MUST NOT commit the frozen baseline.

    ``deferred`` is True when the rate-budget pre-flight (MF-9) determined that
    the exchange rate budget was insufficient to complete the cycle.  In that
    case ``n_submitted == 0`` and NO orders were submitted — the cycle must be
    retried on the next tick.

    ``n_resting_anomaly`` counts coins where an IOC order unexpectedly returned a
    ``resting`` status.  For IOC-only systems this should always be 0; a non-zero
    value indicates an exchange-semantics anomaly and should trigger an alert.

    ``n_throttled`` counts coins where a ``RateLimited`` exception was raised
    mid-cycle (per-coin isolation — other coins still processed).

    ``n_liquidity_deferred`` counts coins whose leg was DEFERRED by the pre-trade
    liquidity gate (insufficient opposing L2 depth within the slippage band).
    A liquidity-deferred leg is retryable later and is NOT counted in ``n_error``.
    """

    results: list[CoinResult]
    n_submitted: int
    n_filled: int
    n_error: int
    n_partial: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    n_resting_anomaly: int = 0
    deferred: bool = False
    defer_reason: str | None = None
    n_throttled: int = 0
    n_liquidity_deferred: int = 0


def _extract_filled_sz(statuses: list[dict[str, Any]]) -> float:
    """Sum the ``totalSz`` from all ``filled`` status dicts returned by
    ``parse_order_response``.

    ``parse_order_response`` returns a list of dicts shaped as either:
    - ``{"filled": {"oid": <int>, "totalSz": "<float_str>", "avgPx": "<float_str>"}}``
    - ``{"resting": {...}}``  (IOC orders should not produce resting; defensive)

    Parameters
    ----------
    statuses:
        List of parsed status dicts (output of ``parse_order_response``).

    Returns
    -------
    float
        Total filled size across all filled entries; 0.0 if none.
    """
    total = 0.0
    for st in statuses:
        if "filled" in st:
            raw = st["filled"].get("totalSz", "0")
            with contextlib.suppress(ValueError, TypeError):
                total += float(raw)
    return total


def _price_of(op: OrderPlan) -> float:
    """Recover the per-unit mark price for an order from its notional fields.

    The sizer only emits an OrderPlan when ``abs(delta_notional) >= $10`` which
    guarantees ``delta_size != 0`` — so ``delta_notional / delta_size`` is the
    signed-cancelled positive price.  We avoid threading the snapshot marks into
    the executor signature: the plan already carries enough to recover price.
    """
    if op.delta_size == 0.0:
        return 0.0
    price: float = abs(op.delta_notional / op.delta_size)
    return price


def _opposing_depth_within_band(
    book: dict[str, list[tuple[float, float]]],
    *,
    is_buy: bool,
    mark: float,
    slippage: float,
) -> float:
    """Sum opposing-side size reachable within the slippage band.

    A buy lifts asks up to ``mark * (1 + slippage)``; a sell hits bids down to
    ``mark * (1 - slippage)``.  Returns the total reachable size (base units).
    """
    if is_buy:
        limit = mark * (1.0 + slippage)
        return sum(sz for px, sz in book.get("asks", []) if px <= limit)
    limit = mark * (1.0 - slippage)
    return sum(sz for px, sz in book.get("bids", []) if px >= limit)


def _liquidity_gate(
    op: OrderPlan,
    client: Any,
    marks: dict[str, Any],
    *,
    slippage: float,
    liquidity_safety: float,
) -> CoinResult | None:
    """Pre-trade liquidity gate for a single leg (Task 5).

    Consults the OPPOSING L2 side for ``op.coin`` and requires that the depth
    reachable within the slippage band is at least
    ``abs(op.delta_size) * liquidity_safety``.  If the book is too thin the leg
    is DEFERRED (no order submitted) and a ``CoinResult(liquidity_deferred=True)``
    is returned; otherwise ``None`` is returned and the caller proceeds to submit.

    A deferred leg is NOT an error: it carries no cloids, zero fills, and is
    retryable on a later tick once the book deepens.

    Fail-safe & per-coin isolation
    ------------------------------
    A missing mark (coin absent from the ``marks()`` snapshot — a new-listing /
    delist race) OR an ``l2_book`` failure CONSERVATIVELY DEFERS this one leg
    rather than crashing the whole cycle.  This preserves the per-coin isolation
    invariant at the top of this module: one coin's data gap never strands every
    other coin's leg.  We defer (not blind-submit) because we cannot confirm the
    book is deep enough — under-trading one leg is the safe failure mode.
    """

    def _deferred() -> CoinResult:
        return CoinResult(
            coin=op.coin,
            requested_size=abs(float(op.delta_size)),
            filled_size=0.0,
            attempts=0,
            status="skipped",
            cloids=[],
            error=None,
            liquidity_deferred=True,
        )

    mark_dec = marks.get(op.coin)
    if mark_dec is None:
        _logger.warning(
            "executor: leg deferred — no mark for liquidity check",
            coin=op.coin,
        )
        return _deferred()

    try:
        book = client.l2_book(op.coin)
        depth = _opposing_depth_within_band(
            book, is_buy=(op.side == "buy"), mark=float(mark_dec), slippage=slippage
        )
    except Exception as exc:  # noqa: BLE001
        # l2_book / depth computation failed for THIS coin only — defer it,
        # keep the rest of the cycle running (per-coin isolation).
        _logger.warning(
            "executor: leg deferred — l2_book error",
            coin=op.coin,
            error=str(exc),
        )
        return _deferred()

    required = abs(float(op.delta_size)) * liquidity_safety
    if depth < required:
        _logger.warning(
            "executor: leg deferred — insufficient opposing depth",
            coin=op.coin,
            side=op.side,
            required=required,
            depth=depth,
        )
        return _deferred()
    return None


def _preflight_caps(plan: SizePlan, cfg: Config) -> None:
    """Run the per-coin / total-notional / turnover cap check over the whole plan.

    Notional sourcing (spec-flagged decision)
    ------------------------------------------
    - per-coin / total caps use the END-STATE notional: ``target_size * price``
      where ``price`` is recovered via ``_price_of(op)`` (delta_notional /
      delta_size).  This is the position value AFTER the order fills — the
      quantity the caps are meant to bound.
    - turnover uses ``abs(delta_notional)`` — the USD actually traded this cycle.

    Raises
    ------
    CapBreach
        On the first cap breach.  The caller aborts the cycle BEFORE submitting.
    """
    cap_orders: list[dict[str, Any]] = []
    for op in plan.orders:
        price = _price_of(op)
        end_state_notional = abs(op.target_size * price)
        cap_orders.append(
            {
                "coin": op.coin,
                "notional": end_state_notional,
                "turnover": abs(op.delta_notional),
            }
        )
    enforce_caps(cap_orders, cfg.caps)


def _preflight_rate_budget(
    client: Any,
    plan: SizePlan,
    max_attempts: int,
) -> str | None:
    """Best-effort rate-budget pre-flight check (MF-9, spec §8 step 5 + §10).

    Queries the exchange for the current rate-limit counters and estimates
    whether the remaining budget is sufficient to complete the cycle.

    Conservative estimate: ``est_actions = len(orders) * (max_attempts + 1)``
    — up to *max_attempts* submits plus a confirm query per order.  We never
    start a cycle we cannot finish (do-not-strand policy).

    Failure policy (best-effort)
    ----------------------------
    If the rate query itself fails (network error, unsupported endpoint, etc.)
    this function returns ``None`` (no defer) and logs a WARNING.  A transient
    query failure MUST NOT block trading — the budget is usually ample and
    stranding the daily rebalance over a quota-fetch error would be worse than
    the risk of an unlikely mid-cycle throttle.

    Parameters
    ----------
    client:
        Duck-typed client with an optional ``user_rate_limit()`` method.
        If the method is absent (legacy fakes) the check is skipped.
    plan:
        The size plan to execute.
    max_attempts:
        Maximum submission attempts per coin (same value passed to
        ``execute()``).  Used to compute the conservative action estimate.

    Returns
    -------
    str | None
        A non-empty defer-reason string when the budget is insufficient;
        ``None`` when the budget is sufficient or the query failed.
    """
    if not hasattr(client, "user_rate_limit"):
        return None

    # No-op guard: a rebalance=True plan with zero orders has nothing to
    # rate-limit (est_actions == 0).  Never defer a no-op cycle — a negative
    # `remaining` (e.g. a transient over-count) must not strand an empty plan.
    est_actions = len(plan.orders) * (max_attempts + 1)
    if est_actions == 0:
        return None

    try:
        rl = client.user_rate_limit()
        remaining = int(rl["nRequestsCap"]) - int(rl["nRequestsUsed"])
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "executor: rate-limit query failed — skipping budget pre-flight (proceeding)",
            error=str(exc),
        )
        return None

    if remaining < est_actions:
        reason = (
            f"rate budget {remaining} < est {est_actions} actions "
            f"({len(plan.orders)} orders × {max_attempts + 1})"
        )
        _logger.warning(
            "executor: rate-budget insufficient — deferring cycle (do-not-strand)",
            remaining=remaining,
            est_actions=est_actions,
            n_orders=len(plan.orders),
        )
        return reason
    return None


def execute(
    plan: SizePlan,
    client: Any,
    cfg: Config,
    ledger: IntentLedger,
    *,
    strategy_id: str,
    as_of: str,
    now_ts: str,
    max_attempts: int = 3,
    spot_routing: dict[str, str] | None = None,
) -> ExecReport:
    """Execute the size plan, submitting IOC orders to the exchange.

    If ``plan.rebalance`` is ``False`` this function is a no-op and returns
    an empty ``ExecReport`` — nothing is submitted, the ledger is untouched.

    Before submitting ANY order the whole plan is checked against ``cfg.caps``
    via ``_preflight_caps``.  A ``CapBreach`` aborts the entire cycle BEFORE any
    submit (returns ``ExecReport(aborted=True, n_submitted=0)``) so a bad plan
    never reaches the exchange.

    Per-coin failures (``OrderRejected``, ``RateLimited``, or unexpected
    exceptions) are caught and recorded as ``status="error"`` — they do NOT
    abort the cycle.  Only programmer errors (AttributeError, etc.) propagate.

    Parameters
    ----------
    plan:
        The size plan produced by ``sizer.plan()``.
    client:
        Duck-typed client with ``submit_ioc``, ``submit_spot_ioc``,
        ``market_close``, and ``query_by_cloid`` methods.
    cfg:
        Executor configuration.  ``cfg.caps.max_order_slippage`` is the
        TIGHT IOC slippage (never the SDK 5 % default); the notional/turnover
        caps are enforced pre-flight.
    ledger:
        Intent ledger for MF-7 write-before-submit.
    strategy_id:
        Opaque strategy identifier forwarded to ``make_cloid``.
    as_of:
        ISO date string of the ``TargetAllocation`` that triggered this cycle.
    now_ts:
        ISO timestamp injected by the caller (kept deterministic for tests).
        Also a per-cycle cloid nonce (F-7).
    max_attempts:
        Maximum submission attempts per coin.  Default 3.
    spot_routing:
        Optional venue map: strategy COIN ticker → HL spot PAIR name
        (e.g. ``{"HYPE": "@107", "ETH": "@151"}``).  A coin present in the map
        is routed to SPOT via ``submit_spot_ioc``; a coin absent (or
        ``spot_routing=None``) keeps the PERP path (``submit_ioc`` /
        ``market_close``).  ``None`` is byte-identical to the pre-D4 behaviour.
        Spot reduce/exit is a SELL — SPOT has no reduce-only ``market_close``.

    Returns
    -------
    ExecReport
        Aggregate report across all coins.
    """
    if not plan.rebalance:
        return ExecReport(results=[], n_submitted=0, n_filled=0, n_error=0, n_partial=0)

    # Pre-flight cap enforcement (F-5): abort the WHOLE cycle BEFORE any submit.
    try:
        _preflight_caps(plan, cfg)
    except CapBreach as breach:
        _logger.error(
            "executor: pre-flight CapBreach — aborting cycle, NO orders submitted",
            reason=str(breach),
        )
        return ExecReport(
            results=[],
            n_submitted=0,
            n_filled=0,
            n_error=0,
            n_partial=0,
            aborted=True,
            abort_reason=str(breach),
        )

    # MF-9: rate-budget pre-flight — DEFER the whole cycle if budget insufficient.
    # "Do-not-strand" policy: submit NOTHING rather than half-finishing a rebalance.
    # The kill-switch flatten path (flatten_all) is EXEMPT — cancels are rate-privileged.
    defer_reason = _preflight_rate_budget(client, plan, max_attempts)
    if defer_reason:
        return ExecReport(
            results=[],
            n_submitted=0,
            n_filled=0,
            n_error=0,
            n_partial=0,
            deferred=True,
            defer_reason=defer_reason,
        )

    results: list[CoinResult] = []
    slippage = cfg.caps.max_order_slippage
    liquidity_safety = cfg.retry.liquidity_safety

    # Pre-trade liquidity gate (Task 5): when armed (liquidity_safety > 0) we
    # consult the L2 book ONCE per cycle.  ``marks`` is read from the client up
    # front so the shared gate below can size each leg's slippage band.  When the
    # gate is OFF (liquidity_safety == 0) we never touch marks/l2 — keeping legacy
    # clients (no l2 book) byte-identical.
    marks: dict[str, Any] = client.marks() if liquidity_safety > 0.0 else {}

    for op in plan.orders:
        # D4 venue routing: a coin in spot_routing trades on its spot pair;
        # absent (or spot_routing=None) → PERP.  ``pair is None`` means PERP.
        pair = spot_routing.get(op.coin) if spot_routing else None

        # Pre-trade liquidity gate — SHARED across the daily AND retry paths
        # (both run through this single dispatch loop).  Just before a coin's
        # first IOC, require the OPPOSING L2 side to carry enough reachable depth
        # within the slippage band; otherwise DEFER the leg (no submit, no error).
        if liquidity_safety > 0.0:
            deferred_leg = _liquidity_gate(
                op, client, marks, slippage=slippage, liquidity_safety=liquidity_safety
            )
            if deferred_leg is not None:
                results.append(deferred_leg)
                continue

        is_full_exit = op.target_size == 0.0 and op.side == "sell"
        if is_full_exit:
            results.append(
                _execute_full_exit(
                    op, client, ledger, slippage,
                    strategy_id=strategy_id, as_of=as_of, now_ts=now_ts,
                    pair=pair,
                )
            )
        else:
            results.append(
                _execute_ioc(
                    op, client, ledger, slippage,
                    strategy_id=strategy_id, as_of=as_of, now_ts=now_ts,
                    max_attempts=max_attempts,
                    pair=pair,
                )
            )

    n_submitted = sum(len(r.cloids) for r in results)
    n_filled = sum(1 for r in results if r.status == "filled")
    n_error = sum(1 for r in results if r.status == "error")
    n_partial = sum(1 for r in results if r.status == "partial")
    n_resting_anomaly = sum(1 for r in results if r.resting_anomaly)
    n_throttled = sum(1 for r in results if r.throttled)
    n_liquidity_deferred = sum(1 for r in results if r.liquidity_deferred)

    _logger.info(
        "executor: cycle complete",
        n_coins=len(results),
        n_submitted=n_submitted,
        n_filled=n_filled,
        n_error=n_error,
        n_partial=n_partial,
        n_resting_anomaly=n_resting_anomaly,
        n_throttled=n_throttled,
        n_liquidity_deferred=n_liquidity_deferred,
    )

    return ExecReport(
        results=results,
        n_submitted=n_submitted,
        n_filled=n_filled,
        n_error=n_error,
        n_partial=n_partial,
        n_resting_anomaly=n_resting_anomaly,
        n_throttled=n_throttled,
        n_liquidity_deferred=n_liquidity_deferred,
    )


def _record_pending(
    ledger: IntentLedger,
    *,
    cloid: str,
    as_of: str,
    coin: str,
    target_w: float,
    attempt_seq: int,
    now_ts: str,
) -> None:
    """MF-7: write the PENDING intent to the ledger BEFORE any submit."""
    ledger.record_pending(
        Intent(
            cloid=cloid,
            as_of=as_of,
            coin=coin,
            target_w=target_w,
            attempt_seq=attempt_seq,
            status="PENDING",
            ts=now_ts,
        )
    )


def _execute_full_exit(
    op: OrderPlan,
    client: Any,
    ledger: IntentLedger,
    slippage: float,
    *,
    strategy_id: str,
    as_of: str,
    now_ts: str,
    pair: str | None = None,
) -> CoinResult:
    """Execute a full exit (target_size==0, side="sell") — SINGLE shot.

    Venue routing (D4)
    ------------------
    - PERP (``pair is None``): ``market_close`` (reduce-only) — takes no size and
      closes the ENTIRE remaining position.
    - SPOT (``pair`` set): SPOT has NO reduce-only close — a full spot exit is a
      plain SELL of the held base size: ``submit_spot_ioc(pair, False,
      abs(delta_size), slippage, cloid)``.  The sizer guarantees a sell never
      exceeds the holding (target_size >= 0), so this can never go short.

    Either way the call is SINGLE-shot: there is nothing to retry against a
    residual.  Driving the size-residual loop here would double-count fills
    across attempts (F-2).  Any short-fall is caught next cycle by the sizer's
    abs-drift backstop (which includes currently-held coins).
    """
    cloid = make_cloid(strategy_id, as_of, op.coin, 0, now_ts)
    cloids = [cloid]
    requested = abs(op.delta_size)

    # MF-7: write-before-submit.
    _record_pending(
        ledger, cloid=cloid, as_of=as_of, coin=op.coin,
        target_w=op.target_size, attempt_seq=0, now_ts=now_ts,
    )

    try:
        if pair is not None:
            # SPOT full exit = SELL the held base size (no reduce-only close).
            resp = client.submit_spot_ioc(pair, False, requested, slippage, cloid)
        else:
            resp = client.market_close(op.coin, slippage, cloid)
    except RateLimited as exc:
        _logger.warning(
            "executor: full-exit rate-limited",
            coin=op.coin, cloid=cloid, error=str(exc),
        )
        ledger.mark_done(cloid)  # clean rejection — nothing landed
        # Tag throttled so the report tallies n_throttled / emits THROTTLE
        # (still status="error", still per-coin isolated, retried next cycle).
        return CoinResult(
            coin=op.coin, requested_size=requested, filled_size=0.0,
            attempts=1, status="error", cloids=cloids, error=str(exc),
            throttled=True,
        )
    except OrderRejected as exc:
        _logger.warning(
            "executor: full-exit rejected",
            coin=op.coin, cloid=cloid, error=str(exc),
        )
        ledger.mark_done(cloid)  # clean rejection — nothing landed
        return CoinResult(
            coin=op.coin, requested_size=requested, filled_size=0.0,
            attempts=1, status="error", cloids=cloids, error=str(exc),
        )
    except Exception as exc:
        # F-4: unexpected exception — the close MIGHT have landed.  Leave the
        # intent PENDING so the next boot's reconcile_pending checks the chain.
        _logger.warning(
            "executor: full-exit unexpected error (intent left PENDING)",
            coin=op.coin, cloid=cloid, error=str(exc),
        )
        return CoinResult(
            coin=op.coin, requested_size=requested, filled_size=0.0,
            attempts=1, status="error", cloids=cloids, error=str(exc),
        )

    try:
        statuses = parse_order_response(resp)
        filled = _extract_filled_sz(statuses)
    except OrderRejected as exc:
        _logger.warning(
            "executor: full-exit parse OrderRejected",
            coin=op.coin, cloid=cloid, error=str(exc),
        )
        ledger.mark_done(cloid)
        return CoinResult(
            coin=op.coin, requested_size=requested, filled_size=0.0,
            attempts=1, status="error", cloids=cloids, error=str(exc),
        )

    with contextlib.suppress(Exception):
        client.query_by_cloid(cloid)

    ledger.mark_done(cloid)

    # reduce-only can't over-close: filled ≤ remaining position.
    if filled >= requested - _FILL_EPS:
        status: Literal["filled", "partial", "error"] = "filled"
    elif filled > _FILL_EPS:
        status = "partial"
    else:
        status = "error"

    return CoinResult(
        coin=op.coin, requested_size=requested, filled_size=filled,
        attempts=1, status=status, cloids=cloids, error=None,
    )


def _execute_ioc(
    op: OrderPlan,
    client: Any,
    ledger: IntentLedger,
    slippage: float,
    *,
    strategy_id: str,
    as_of: str,
    now_ts: str,
    max_attempts: int,
    pair: str | None = None,
) -> CoinResult:
    """Execute a buy / partial-reduce IOC order with residual retry (MF-6).

    Venue routing (D4)
    ------------------
    The ONLY venue-dependent step is the submit call: a coin with ``pair`` set
    submits to SPOT via ``client.submit_spot_ioc(pair, is_buy, residual, ...)``;
    ``pair is None`` keeps the PERP ``client.submit_ioc(op.coin, ...)`` path.
    ``is_buy`` carries the same ``op.side == "buy"`` semantics on both venues —
    a SPOT sell (``is_buy=False``) reduces the spot holding (no reduce-only).
    Everything else (cloid derivation, residual retry, resting-anomaly cancel,
    parse, ledger PENDING-before-submit) is IDENTICAL across venues.

    NOTE (D5, not built here): a SPOT BUY assumes the spot USDC pool is already
    funded.  The perp->spot USDC pre-funding transfer is the NEXT increment
    (D5) — this executor does not move funds; it assumes spot USDC is available.

    Resting-status anomaly handling (peer-issue-2)
    -----------------------------------------------
    IOC orders MUST NOT rest.  If ``parse_order_response`` returns a ``resting``
    status dict, that is an exchange-semantics anomaly.  The executor:

    1. Logs CRITICAL with coin/cloid/attempt.
    2. Cancels the stray resting order via ``client.cancel_by_cloid`` BEFORE the
       residual retry, so the retry cannot re-submit ON TOP of an already-resting
       order (which would double the exposure).
    3. Does NOT count the resting size as filled (it is an unconfirmed resting
       order, not a fill — and we just cancelled it so the retry re-submits the
       full residual cleanly).
    4. Sets ``resting_anomaly=True`` on the returned ``CoinResult`` and increments
       ``ExecReport.n_resting_anomaly``.

    Cancel-failure double-exposure guard
    -------------------------------------
    The cancel must SUCCEED before the residual retry is allowed.  If the cancel
    RAISES, the resting order is STILL on the book — re-submitting the residual
    would let BOTH the resting order AND the new IOC fill (the exact peer
    production failure).  So on cancel failure the executor ABORTS this coin's
    retry (``coin_status="error"``, ``break``) and DOES NOT re-submit.  Under-
    filling is the safe failure mode: the boot-time orphan sweep and the next-
    cycle abs-drift backstop recover the position.
    """
    attempt_seq = 0
    n_submits = 0  # F-6: real submit counter (not attempt_seq+1)
    residual = abs(op.delta_size)
    requested = residual
    filled_total = 0.0
    cloids: list[str] = []
    coin_status: Literal["filled", "partial", "error"] = "error"
    err_msg: str | None = None
    had_resting_anomaly: bool = False

    while attempt_seq < max_attempts and residual > _FILL_EPS:
        cloid = make_cloid(strategy_id, as_of, op.coin, attempt_seq, now_ts)
        cloids.append(cloid)

        # MF-7: write PENDING intent BEFORE submit.
        _record_pending(
            ledger, cloid=cloid, as_of=as_of, coin=op.coin,
            target_w=op.target_size, attempt_seq=attempt_seq, now_ts=now_ts,
        )

        n_submits += 1
        try:
            if pair is not None:
                # D4: route to SPOT.  is_buy carries the same side semantics;
                # a spot sell (is_buy=False) reduces the spot holding.
                resp = client.submit_spot_ioc(
                    pair, op.side == "buy", residual, slippage, cloid
                )
            else:
                resp = client.submit_ioc(op.coin, op.side == "buy", residual, slippage, cloid)
        except RateLimited as exc:
            _logger.warning(
                "executor: order rate-limited mid-cycle",
                coin=op.coin, attempt=attempt_seq, cloid=cloid, error=str(exc),
            )
            err_msg = str(exc)
            ledger.mark_done(cloid)  # clean rejection — nothing landed
            # Tag this coin as throttled so the report can tally n_throttled and
            # the rebalance layer can emit a THROTTLE alert.
            return CoinResult(
                coin=op.coin,
                requested_size=requested,
                filled_size=filled_total,
                attempts=n_submits,
                status="error",
                cloids=cloids,
                error=err_msg,
                resting_anomaly=had_resting_anomaly,
                throttled=True,
            )
        except OrderRejected as exc:
            _logger.warning(
                "executor: order rejected",
                coin=op.coin, attempt=attempt_seq, cloid=cloid, error=str(exc),
            )
            err_msg = str(exc)
            ledger.mark_done(cloid)  # clean rejection — nothing landed
            break
        except Exception as exc:
            # F-4: unexpected exception — order MIGHT have landed.  Leave the
            # intent PENDING for the next boot's reconcile_pending to resolve.
            _logger.warning(
                "executor: unexpected error on submit (intent left PENDING)",
                coin=op.coin, attempt=attempt_seq, cloid=cloid, error=str(exc),
            )
            err_msg = str(exc)
            break

        try:
            statuses = parse_order_response(resp)
            filled_this_attempt = _extract_filled_sz(statuses)
        except OrderRejected as exc:
            _logger.warning(
                "executor: parse_order_response raised OrderRejected",
                coin=op.coin, attempt=attempt_seq, cloid=cloid, error=str(exc),
            )
            err_msg = str(exc)
            ledger.mark_done(cloid)
            break

        # ---------------------------------------------------------------
        # Resting-status anomaly guard (peer-issue-2).
        # IOC orders should NEVER rest.  If any status dict carries a
        # "resting" key, this is an exchange-semantics anomaly.  Cancel
        # the stray order BEFORE the residual retry to prevent double
        # exposure.  The resting size is NOT counted as filled — we
        # cancelled it, so the retry re-submits it cleanly.
        #
        # CRITICAL (double-exposure guard): the cancel must SUCCEED before we
        # allow the residual retry.  If the cancel FAILS, the resting order is
        # STILL on the book — re-submitting the residual would let BOTH the
        # resting order AND the new IOC fill (the exact peer production
        # failure).  So on cancel failure we ABORT this coin's retry (break
        # with status="error").  Under-filling is safe: the boot-time orphan
        # sweep + the next-cycle abs-drift backstop recover the position.
        # ---------------------------------------------------------------
        resting_statuses = [st["resting"] for st in statuses if "resting" in st]
        if resting_statuses:
            had_resting_anomaly = True
            _logger.error(
                "executor: IOC returned a RESTING status (anomaly); "
                "cancelling the resting order before residual retry to avoid double exposure",
                coin=op.coin,
                cloid=cloid,
                attempt=attempt_seq,
            )
            # D4: cancel must target the instrument the order rests on — the
            # spot PAIR for a spot coin, the perp coin name otherwise.
            cancel_target = pair if pair is not None else op.coin
            try:
                client.cancel_by_cloid(cancel_target, cloid)
            except Exception as cancel_exc:
                # CANCEL FAILED → the resting order is STILL on the book.  We
                # MUST NOT re-submit the residual (that would double exposure).
                # Abort this coin: under-fill is safe; the boot-time sweep +
                # next-cycle drift backstop recover.
                err_msg = (
                    "cancel of resting IOC order FAILED — refusing to re-submit "
                    f"(double-exposure guard): {cancel_exc}"
                )
                _logger.error(
                    "executor: resting-order cancel FAILED — aborting coin retry (fail-safe)",
                    coin=op.coin, cloid=cloid, error=err_msg,
                )
                ledger.mark_done(cloid)
                coin_status = "error"
                break  # do NOT re-submit — leave residual unfilled
            # cancel SUCCEEDED → the resting size returns to the residual; safe
            # to retry cleanly on the next iteration.

        with contextlib.suppress(Exception):
            client.query_by_cloid(cloid)

        ledger.mark_done(cloid)

        filled_total += filled_this_attempt
        residual -= filled_this_attempt

        if residual <= _FILL_EPS:
            coin_status = "filled"
            break

        attempt_seq += 1

    # Final status if not already decided as filled.
    if coin_status == "error":
        if residual <= _FILL_EPS:
            coin_status = "filled"
        elif filled_total > _FILL_EPS:
            coin_status = "partial"
        # else remains "error"

    return CoinResult(
        coin=op.coin,
        requested_size=requested,
        filled_size=filled_total,
        attempts=n_submits,  # F-6: real submit count
        status=coin_status,
        cloids=cloids,
        error=err_msg,
        resting_anomaly=had_resting_anomaly,
    )
