"""
automation.execution.reconciler — On-chain snapshot and achieved-weight computation.

The Hyperliquid exchange is the SINGLE SOURCE OF TRUTH for positions (MF-11, S-3).
This module fetches a consistent snapshot of account state and computes the
achieved weight vector from live positions.

The achieved weights are used for:
- The abs-drift backstop in sizer.plan() (S-2): if live positions have drifted
  more than 2×threshold from the frozen target, force a rebalance even when the
  new signal hasn't changed.
- Divergence reporting (S-5): compare intended vs achieved after a cycle.

They are NOT used as the ratchet baseline (that role belongs to
state.last_rebalanced_target — the frozen last-submitted target weights).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from automation.core.redaction import get_logger
from automation.reporting.alerts import Alert, AlertType, Severity

_logger = get_logger(__name__)


@dataclass(frozen=True)
class Snapshot:
    """Consistent on-chain state snapshot.

    All fields are captured in a single conceptual read (three sequential API
    calls — see :func:`snapshot`).  The snapshot is frozen so it can be safely
    passed through the pure planning pipeline without mutation risk.

    Fields
    ------
    equity:
        Total account value (USD) as a ``Decimal``.
    positions:
        Signed position sizes by perp name: ``{"BTC": Decimal("0.5"), ...}``.
        Absent keys ⇒ flat (no position).
    marks:
        Best available price by perp name: ``{"BTC": Decimal("70000"), ...}``.
    sz_decimals:
        Size decimal precision by perp name: ``{"BTC": 5, ...}``.
    """

    equity: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)
    marks: dict[str, Decimal] = field(default_factory=dict)
    sz_decimals: dict[str, int] = field(default_factory=dict)


def snapshot(client: Any, spot_routing: dict[str, str] | None = None) -> Snapshot:
    """Fetch a consistent on-chain snapshot from *client*.

    *client* is duck-typed — in production it is an ``HLClient`` instance; in
    tests it is a fake with the same methods.

    Venue routing (D2 — hole #2 fix)
    --------------------------------
    Some strategy coins may be held on HL **spot** (no funding) while the rest
    stay on **perp**.  ``spot_routing`` maps a strategy COIN ticker → its HL spot
    PAIR name (the ``@NNN`` form), e.g. ``{"HYPE": "@107", "ETH": "@151"}``.  Any
    coin NOT in the map is a perp (the default).

    When ``spot_routing`` is ``None`` or empty, this function is byte-identical
    to the pure-perp behaviour: it reads ONLY the perp pool and never touches the
    spot endpoints.

    When ``spot_routing`` is non-empty, the returned :class:`Snapshot` is HYBRID:

    - **equity** spans BOTH pools — ``client.equity()`` (perp ``accountValue``)
      PLUS the spot pool value (free spot USDC + the value of the routed spot
      holdings).  This is the fix for the understated-equity hole: once any
      capital sits on spot, the pure-perp denominator was too small and every
      weight/size was wrong.
    - **positions / marks / sz_decimals** are venue-aware but keyed by COIN
      ticker in BOTH cases, so the :class:`Snapshot` SHAPE is unchanged and the
      sizer needs ZERO modification.  A spot-routed coin's "position" is its
      owned token quantity (always ``>= 0`` — spot can't be short); its mark is
      the spot pair's mark; its sz_decimals is the spot pair's precision.

    The client MUST be spot-enabled when ``spot_routing`` is non-empty.  If the
    spot reads raise (spot not enabled), the error propagates — we deliberately
    do NOT silently fall back to perp-only, which would re-introduce the
    understated-equity bug.

    Parameters
    ----------
    client:
        Object exposing the perp reads ``.equity() -> Decimal``,
        ``.positions() -> dict[str,Decimal]``, ``.marks() -> dict[str,Decimal]``,
        and ``.sz_decimals() -> dict[str,int]``.  When ``spot_routing`` is
        non-empty it must ALSO expose ``.spot_balances() -> dict[str,Decimal]``,
        ``.spot_marks() -> dict[str,Decimal]`` (keyed by spot pair name), and
        ``.spot_sz_decimals() -> dict[str,int]`` (keyed by spot pair name).
    spot_routing:
        ``{coin_ticker: spot_pair_name}`` for coins routed to spot.  ``None`` or
        ``{}`` ⇒ pure-perp snapshot (unchanged legacy behaviour).

    Returns
    -------
    Snapshot
        Packed snapshot of current account state (hybrid when routing is given).
    """
    # ----- Perp pool (always read) -----------------------------------------
    perp_eq = client.equity()
    perp_pos = client.positions()
    perp_mkts = client.marks()
    perp_szd = client.sz_decimals()

    # ----- Pure-perp path: byte-identical to legacy behaviour --------------
    if not spot_routing:
        snap = Snapshot(
            equity=perp_eq,
            positions=perp_pos,
            marks=perp_mkts,
            sz_decimals=perp_szd,
        )
        _logger.info(
            "reconciler.snapshot: captured",
            equity=str(perp_eq),
            n_positions=len(perp_pos),
            n_marks=len(perp_mkts),
        )
        return snap

    # ----- Hybrid path: read the spot pool too -----------------------------
    # If the client is not spot-enabled, these raise RuntimeError — we let it
    # propagate rather than silently degrade to an understated perp-only equity.
    spot_bal = client.spot_balances()  # {coin: qty>=0, incl. "USDC": free_usdc}
    spot_mkts = client.spot_marks()  # {pair_name: mark}
    spot_szd = client.spot_sz_decimals()  # {pair_name: int}

    spot_coins = set(spot_routing.keys())

    # positions: spot-routed coins use the owned token qty (>=0); perp coins use
    # the signed perp szi.  Combined into one coin-keyed dict.
    positions: dict[str, Decimal] = {
        coin: szi for coin, szi in perp_pos.items() if coin not in spot_coins
    }
    # marks / sz_decimals: same combine, keyed by coin ticker.
    marks: dict[str, Decimal] = {
        coin: mark for coin, mark in perp_mkts.items() if coin not in spot_coins
    }
    sz_decimals: dict[str, int] = {
        coin: szd for coin, szd in perp_szd.items() if coin not in spot_coins
    }

    for coin, pair in spot_routing.items():
        # Spot holding qty (>=0); absent ⇒ flat (zero), mirroring positions().
        positions[coin] = spot_bal.get(coin, Decimal("0"))
        # Spot mark via the pair name; if missing, OMIT the coin's mark — the
        # sizer / achieved_weights already skip unpriceable coins.
        if pair in spot_mkts:
            marks[coin] = spot_mkts[pair]
        if pair in spot_szd:
            sz_decimals[coin] = spot_szd[pair]

    # ----- Total equity = perp accountValue + spot pool value --------------
    # Spot pool value = free spot USDC (×1) + Σ routed-holding qty × spot mark.
    # Idle NON-routed spot tokens are intentionally ignored — for this strategy
    # they shouldn't exist (every spot holding is a routed coin or free USDC).
    spot_pool_value = spot_bal.get("USDC", Decimal("0"))
    for coin, pair in spot_routing.items():
        qty = spot_bal.get(coin, Decimal("0"))
        mark = spot_mkts.get(pair)
        if mark is not None:
            spot_pool_value += qty * mark

    total_equity = perp_eq + spot_pool_value

    snap = Snapshot(
        equity=total_equity,
        positions=positions,
        marks=marks,
        sz_decimals=sz_decimals,
    )
    _logger.info(
        "reconciler.snapshot: captured (hybrid perp+spot)",
        equity=str(total_equity),
        perp_equity=str(perp_eq),
        spot_pool_value=str(spot_pool_value),
        n_positions=len(positions),
        n_marks=len(marks),
        n_spot_routed=len(spot_routing),
    )
    return snap


def achieved_weights(snap: Snapshot) -> dict[str, float]:
    """Compute the current achieved weight vector from live positions.

    For each coin in ``snap.positions``:
        ``weight = float(szi × mark / equity)``

    - Uses ``snap.marks[coin]`` for the price.
    - If a mark is missing for a coin that has a position, that coin is skipped
      (weight treated as 0 for rounding purposes — log a warning).
    - If equity ≤ 0, returns an empty dict (no ZeroDivisionError).

    NOTE: this returns SIGNED weights (negative for short positions), reflecting
    the actual on-chain state.  The long-only guard in sizer.plan() uses the
    sign of ``snap.positions`` directly; achieved_weights is used for drift math.

    KEY-SPACE CONTRACT (F4): the returned dict is keyed by **PERP name** — the
    same key space as ``snap.positions`` and ``snap.marks`` (both populated from
    the Hyperliquid exchange meta, which uses perp names).  sizer.plan() compares
    these keys directly against ``target_w_by_perp`` (also perp-keyed) for the
    abs-drift backstop.  If a future change ever keyed positions/marks by strategy
    ticker instead, the drift comparison ``target_w(p) − achieved(p)`` would equal
    ``target_w(p)`` for every perp (achieved.get(p)==0), spuriously forcing a full
    rebalance every single cycle.  Keep positions/marks PERP-keyed.

    Parameters
    ----------
    snap:
        On-chain snapshot from :func:`snapshot`.

    Returns
    -------
    dict[str, float]
        Perp name → signed weight fraction.  Empty if equity ≤ 0 or no positions.
    """
    if snap.equity <= Decimal("0"):
        _logger.warning("achieved_weights: equity <= 0, returning empty weights")
        return {}

    result: dict[str, float] = {}
    for coin, szi in snap.positions.items():
        mark = snap.marks.get(coin)
        if mark is None:
            _logger.warning(
                "achieved_weights: no mark for coin with open position — skipping",
                coin=coin,
                szi=str(szi),
            )
            continue
        weight = float(szi * mark / snap.equity)
        result[coin] = weight

    return result


def sweep_open_orders(
    client: Any,
    *,
    now_ts: str,
    alert_sink: Any = None,
) -> list[dict[str, Any]]:
    """Cancel any resting open orders found at boot time.

    This system is IOC-only — there should NEVER be a resting order.  Any
    open order found is an orphan (e.g. from a crash mid-submit, or the
    resting-status anomaly handled by the executor).  Each orphan is
    cancelled best-effort and logged.  If any are found, a
    ``CHAIN_LEDGER_DIVERGENCE`` WARNING alert is emitted.

    Parameters
    ----------
    client:
        Duck-typed client exposing ``open_orders() -> list[dict]``,
        ``cancel_by_cloid(coin, cloid) -> dict``, and
        ``cancel_oid(coin, oid) -> dict``.
    now_ts:
        ISO timestamp of this boot event (used in the alert context).
    alert_sink:
        Optional alert sink.  When ``None``, no alert is emitted (no crash).

    Returns
    -------
    list[dict[str, Any]]
        The raw order dicts that were found (and attempted to cancel).
        Returns ``[]`` if no open orders exist.
    """
    orders: list[dict[str, Any]] = client.open_orders()
    if not orders:
        return []

    coins = [o.get("coin", "?") for o in orders]
    _logger.warning(
        "reconciler.sweep_open_orders: orphan resting orders found — cancelling",
        count=len(orders),
        coins=coins,
        now_ts=now_ts,
    )

    for o in orders:
        coin: str = o.get("coin", "")
        cloid: str | None = o.get("cloid")
        oid_raw = o.get("oid")
        if cloid:
            with contextlib.suppress(Exception):
                client.cancel_by_cloid(coin, cloid)
        elif oid_raw is not None:
            with contextlib.suppress(Exception):
                client.cancel_oid(coin, int(oid_raw))

    if alert_sink is not None:
        with contextlib.suppress(Exception):
            alert_sink.emit(
                Alert(
                    type=AlertType.CHAIN_LEDGER_DIVERGENCE,
                    severity=Severity.WARNING,
                    message=(
                        "orphan resting orders found at boot and cancelled — "
                        "IOC-only system should have none"
                    ),
                    context={"count": len(orders), "coins": coins, "now_ts": now_ts},
                )
            )

    return orders
