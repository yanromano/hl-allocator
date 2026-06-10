"""Shared OFFLINE test fakes for the hl-allocator daemon path.

These duck-typed fakes let the daemon, the dry-run pipeline, AND the V6 smoke
harness run end-to-end with ZERO network and ZERO keys.  They were extracted
here so the smoke-harness tests can reuse the SAME fake surface the integration
tests rely on, without importing a test module (pytest test modules are not a
stable import API) and without re-modifying the existing tests' behaviour.

What's here
-----------
* :class:`FakeInfo` — the minimal ``client.info`` surface the V6 venue guard
  reads: ``meta()`` returns ``{"universe": [{"name": ..., "szDecimals": ...}]}``.
  The smoke harness fetches the LIVE perp universe via ``client.info.meta()`` —
  the fake lets us drive that guard with a controlled coin set offline.
* :class:`FakeHLClient` — golden reads + RECORDING writes.  Mirrors the surface
  of ``automation.core.hl_client.HLClient`` the reconciler/daemon/harness need
  (``equity`` / ``positions`` / ``marks`` / ``sz_decimals`` / ``open_orders`` /
  ``user_rate_limit`` / ``info`` / ``close``) plus a fully-filling
  ``submit_ioc`` / ``market_close``.
* :func:`testnet_universe_meta` — the 12-coin testnet perp meta (XRP absent), the
  venue set the master-direct testnet template targets.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

#: The coins live on Hyperliquid TESTNET (XRP absent) — the master-direct
#: template's ``tradeable_coins`` / ``universe_perp_map`` keyset.
TESTNET_PERPS: tuple[str, ...] = (
    "BTC", "ETH", "SOL", "BNB", "XMR", "DOGE",
    "TON", "SUI", "HYPE", "AVAX", "ZEC", "PAXG",
)

#: Per-coin szDecimals used to stamp the fake perp meta (representative values;
#: the harness only reads them for observability, never for sizing assertions).
_SZ_DECIMALS: dict[str, int] = {
    "BTC": 5, "ETH": 4, "SOL": 2, "BNB": 3, "XRP": 0, "XMR": 3,
    "DOGE": 0, "TON": 2, "SUI": 1, "HYPE": 2, "AVAX": 2, "ZEC": 3, "PAXG": 4,
}


def testnet_universe_meta(perps: tuple[str, ...] = TESTNET_PERPS) -> dict[str, Any]:
    """Return a ``meta()``-shaped dict for *perps* (default: the 12-coin testnet set).

    Shape mirrors ``Info.meta()``: ``{"universe": [{"name", "szDecimals"}, ...]}``.
    """
    return {
        "universe": [
            {"name": c, "szDecimals": _SZ_DECIMALS.get(c, 2)} for c in perps
        ]
    }


def universe_meta_missing(drop: str, perps: tuple[str, ...] = TESTNET_PERPS) -> dict[str, Any]:
    """Return a ``meta()`` dict with *drop* REMOVED from the perp universe.

    Used by the venue-guard FAILS-LOUD test to simulate a weighted HAARP winner
    that is absent from the live venue.
    """
    return testnet_universe_meta(tuple(c for c in perps if c != drop))


class FakeInfo:
    """Minimal ``client.info`` surface — only ``meta()`` (the V6 venue guard read)."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self._meta = meta

    def meta(self) -> dict[str, Any]:
        return self._meta


def _ok_resp(total_sz: float, avg_px: float = 50000.0) -> dict[str, Any]:
    """A fully-filled IOC/close response (``totalSz`` == requested ⇒ filled)."""
    return {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"filled": {"oid": 1, "totalSz": str(total_sz), "avgPx": str(avg_px)}}
                ]
            }
        },
    }


class FakeHLClient:
    """Offline duck-typed HL client: golden reads + RECORDING writes + ``.info``.

    Read surface (``equity`` / ``positions`` / ``marks`` / ``sz_decimals`` /
    ``open_orders`` / ``user_rate_limit``) is what ``reconciler.snapshot`` +
    ``daemon.boot`` need.  ``info`` exposes ``meta()`` for the V6 venue guard.
    Write surface (``submit_ioc`` / ``market_close`` / ``cancel_*``) RECORDS every
    call and (by default) fully fills, reflecting the new position so a
    post-execute snapshot shows the book moved toward the target.
    """

    def __init__(
        self,
        *,
        equity: float = 50_000.0,
        positions: dict[str, float] | None = None,
        marks: dict[str, float] | None = None,
        sz_decimals: dict[str, int] | None = None,
        open_orders: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        l2: dict[str, dict[str, list[tuple[float, float]]]] | None = None,
        positions_leverage: dict[str, dict[str, Any]] | None = None,
        account_snapshot: dict[str, Any] | None = None,
        venue_meta: dict[str, dict[str, Any]] | None = None,
        spot_balances: dict[str, Decimal] | None = None,
    ) -> None:
        self._equity = Decimal(str(equity))
        self._positions: dict[str, Decimal] = {
            k: Decimal(str(v)) for k, v in (positions or {}).items()
        }
        self._marks = {k: Decimal(str(v)) for k, v in (marks or {}).items()}
        self._szd = dict(sz_decimals or {})
        self._open_orders = list(open_orders or [])
        self.info = FakeInfo(meta if meta is not None else testnet_universe_meta())
        self._l2: dict[str, dict[str, list[tuple[float, float]]]] = l2 or {}
        self._positions_leverage: dict[str, dict[str, Any]] = positions_leverage or {}
        self._account_snapshot: dict[str, Any] = (
            account_snapshot
            if account_snapshot is not None
            else {"equity": 0.0, "maint_margin": 0.0, "positions": {}}
        )
        self._venue_meta: dict[str, dict[str, Any]] = venue_meta or {}
        self._spot_balances: dict[str, Decimal] = {
            k: Decimal(str(v)) for k, v in (spot_balances or {}).items()
        }
        # ---- recorders ----
        self.submit_ioc_calls: list[dict[str, Any]] = []
        self.market_close_calls: list[dict[str, Any]] = []
        self.cancel_oid_calls: list[tuple[str, int]] = []
        self.cancel_by_cloid_calls: list[tuple[str, str]] = []
        self.leverage_calls: list[tuple[str, int, bool]] = []
        self.open_orders_calls: int = 0
        self.closed: bool = False

    # ---- read surface --------------------------------------------------
    def equity(self) -> Decimal:
        return self._equity

    def positions(self) -> dict[str, Decimal]:
        return dict(self._positions)

    def marks(self) -> dict[str, Decimal]:
        return dict(self._marks)

    def positions_leverage(self) -> dict[str, dict[str, Any]]:
        return {c: dict(v) for c, v in self._positions_leverage.items()}

    def account_snapshot(self) -> dict[str, Any]:
        return self._account_snapshot

    def venue_meta(self) -> dict[str, dict[str, Any]]:
        return {c: dict(v) for c, v in self._venue_meta.items()}

    def spot_balances(self) -> dict[str, Decimal]:
        return dict(self._spot_balances)

    def sz_decimals(self) -> dict[str, int]:
        return dict(self._szd)

    def open_orders(self) -> list[dict[str, Any]]:
        self.open_orders_calls += 1
        return list(self._open_orders)

    def user_rate_limit(self) -> dict[str, int]:
        # Huge budget → rate-budget pre-flight never defers.
        return {"nRequestsCap": 1_000_000, "nRequestsUsed": 0}

    def fills_since(self, *_a: Any, **_kw: Any) -> list[dict[str, Any]]:
        return []

    def l2_book(self, coin: str) -> dict[str, list[tuple[float, float]]]:
        return self._l2.get(coin, {"bids": [], "asks": []})

    # ---- test helpers (opt-in mutators; no-op for existing callers) -----
    def set_l2(
        self,
        coin: str,
        *,
        bids: list[tuple[float, float]] | None = None,
        asks: list[tuple[float, float]] | None = None,
    ) -> None:
        """Set (or replace) the L2 book for *coin* — used to deepen a thin book
        mid-test so a previously liquidity-deferred leg can fill on a retry tick.

        Only the sides passed are written; an omitted side defaults to empty so a
        call with just ``asks=...`` yields a buy-able book with no bids.
        """
        self._l2[coin] = {"bids": list(bids or []), "asks": list(asks or [])}

    # ---- write surface (records; fully fills) --------------------------
    def submit_ioc(
        self, coin: str, is_buy: bool, sz: float, slippage: float, cloid: str
    ) -> dict[str, Any]:
        self.submit_ioc_calls.append(
            {"coin": coin, "is_buy": is_buy, "sz": sz, "cloid": cloid}
        )
        prev = self._positions.get(coin, Decimal("0"))
        delta = Decimal(str(sz)) if is_buy else -Decimal(str(sz))
        self._positions[coin] = prev + delta
        return _ok_resp(sz)

    def market_close(self, coin: str, slippage: float, cloid: str) -> dict[str, Any]:
        size = abs(float(self._positions.get(coin, Decimal("0"))))
        self.market_close_calls.append({"coin": coin, "size": size, "cloid": cloid})
        self._positions.pop(coin, None)  # fully flattened
        return _ok_resp(size)

    def query_by_cloid(self, cloid: str) -> dict[str, Any]:
        return {}

    def cancel_oid(self, coin: str, oid: int) -> dict[str, Any]:
        self.cancel_oid_calls.append((coin, oid))
        return {}

    def cancel_by_cloid(self, coin: str, cloid: str) -> dict[str, Any]:
        self.cancel_by_cloid_calls.append((coin, cloid))
        return {}

    def update_leverage(self, coin: str, leverage: int, *, is_cross: bool) -> dict[str, Any]:
        self.leverage_calls.append((coin, leverage, is_cross))
        return {"status": "ok"}

    def close(self) -> None:
        self.closed = True

    @property
    def sub(self) -> str:
        return "0x" + "0" * 40
