"""
automation.core.hl_client — Hyperliquid client wrapper.

Security-critical routing fact
-------------------------------
There are TWO mutually-exclusive routing modes, selected by ``route_via_vault``:

* **subaccount** (``route_via_vault=True``, default) — to trade on behalf of a
  **subaccount** you MUST build Exchange with::

      Exchange(agent_wallet, base_url, vault_address=SUBACCOUNT_ADDRESS)

  ``vault_address`` is the parameter that routes orders to the subaccount.  In
  this mode ``account_address`` is NOT passed; using it instead of
  ``vault_address`` would silently route orders to the *agent wallet* itself,
  not the subaccount — a hard-to-detect but catastrophically wrong behaviour.

* **master-direct** (``route_via_vault=False``) — to trade the **master**
  account the agent key is approved on (no subaccount; fresh accounts cannot
  create one until ~$100k cumulative volume), you build Exchange with::

      Exchange(agent_wallet, base_url, account_address=MASTER_ADDRESS,
               vault_address=None)

  Here there is NO vault routing (the SDK sends no ``vaultAddress`` in the
  action when ``vault_address`` is ``None``) and ``account_address`` sets the
  act-for context to the master account.  The agent signs FOR the master; the
  master holds the positions.  In this mode ``subaccount_address`` carries the
  MASTER EVM address the agent is approved on.

Reads always query ``self.sub`` (== the constructor's ``subaccount_address``),
which is correct in both modes: the subaccount in vault mode, the master in
master-direct mode — in each case the address that actually holds the positions.

All monetary values returned by this module use ``Decimal`` (never ``float``)
to avoid IEEE-754 rounding errors accumulating across rebalance cycles.

Nonce wiring (MF-10)
---------------------
When a ``NonceManager`` is supplied, every write call (``submit_ioc``,
``market_close``) wraps the underlying SDK call in a per-call context manager
that temporarily swaps ``hyperliquid.exchange.get_timestamp_ms`` with
``self._nonce.next``.  The SDK then picks up that overridden name when it calls
``timestamp = get_timestamp_ms()`` inside ``bulk_orders()``.  The original
function is restored in a ``finally`` block so it is always recovered even if
the SDK call or ``next()`` raises.

**Concurrency constraint**: the override is not re-entrant and assumes NO
concurrent submits within one process.  The spec mandates a single authoritative
signer; the watchdog runs as a SEPARATE ``HLClient`` with its OWN ``NonceManager``
backed by a different state file — distinct nonce spaces, never shared.  Do not
share a ``NonceManager`` across threads or clients.

A ``ClockError`` raised by ``self._nonce.next()`` fires BEFORE the SDK makes any
network call (``get_timestamp_ms()`` is called at the top of ``bulk_orders()``
before signing or posting) and propagates out of ``submit_ioc``/``market_close``
unchanged — the order is refused before send, satisfying MF-10's backward-clock
guard requirement.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path
from typing import Any

import eth_account
import hyperliquid.exchange as _hl_exchange
from filelock import FileLock, Timeout
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from automation.core.nonce import ClockError as ClockError  # re-export for callers  # noqa: F401
from automation.core.nonce import NonceManager
from automation.core.redaction import get_logger

# Empty spot_meta stub — avoids IndexError in Info.__init__ when the SDK tries
# to pre-fetch spot market metadata that may not be needed for perp trading.
_EMPTY_SPOT_META: dict[str, list[Any]] = {"universe": [], "tokens": []}

logger = get_logger(__name__)


class HLClient:
    """Thin, security-aware wrapper around the Hyperliquid Python SDK.

    Parameters
    ----------
    env:
        ``"mainnet"`` or ``"testnet"``.
    subaccount_address:
        EVM address of the account being managed.  When ``route_via_vault`` is
        ``True`` (default) this is the Hyperliquid sub-account and is passed as
        the Exchange ``vault_address`` (the authoritative routing field).  When
        ``route_via_vault`` is ``False`` (master-direct) this is the MASTER EVM
        address the agent is approved on and is passed as ``account_address``
        with ``vault_address=None`` (no vault routing).  In both modes the
        address is stored on ``self.sub`` and used for every read.
    agent_key:
        Hex private key of the agent/API wallet (``0x``-prefixed).  If
        ``None``, the client is read-only (no ``Exchange`` is built).
    route_via_vault:
        Routing-mode selector (default ``True``).  ``True`` → subaccount mode:
        ``Exchange(..., vault_address=subaccount_address)`` (byte-identical to
        the legacy behaviour).  ``False`` → master-direct mode:
        ``Exchange(..., account_address=subaccount_address, vault_address=None)``
        — the agent acts FOR the master account with no vault routing (used when
        the account has no subaccount, e.g. fresh accounts below the ~$100k
        cumulative-volume threshold that gates subaccount creation).
    nonce_manager:
        Optional ``NonceManager`` for persistent, backward-clock-guarded nonces
        (MF-10).  When supplied, ``submit_ioc`` and ``market_close`` override
        ``hyperliquid.exchange.get_timestamp_ms`` for the duration of each SDK
        call so the SDK uses nonces issued by this manager.  When ``None``
        (default), the SDK self-nonces as before — existing behaviour preserved.
    enable_spot:
        Opt-in flag for the SPOT read path (D1 venue routing).  Default
        ``False`` → byte-identical to the perp-only client: ``Info`` (and
        ``Exchange``) are built with the empty spot-meta stub so the SDK makes
        NO spot-meta network call on construction and resolves only perp names.
        When ``True``, ``Info``/``Exchange`` are built WITHOUT the empty stub so
        the SDK fetches real spot metadata at construction time (a network call)
        and spot coin/pair names resolve — required by ``spot_balances``,
        ``spot_marks`` and ``spot_sz_decimals``.  Those three methods raise
        ``RuntimeError`` unless this flag is ``True``.  The perp read/write
        paths are unchanged regardless of this flag.

        Network-on-construction note: ``enable_spot=True`` triggers a real
        ``spotMeta`` POST inside ``Info.__init__`` (and ``Exchange.__init__``),
        so a spot-enabled client must only be constructed where a network round
        trip at build time is acceptable.

    Attributes
    ----------
    info : hyperliquid.info.Info
        SDK Info object wired to the correct base URL.
    exchange : hyperliquid.exchange.Exchange | None
        SDK Exchange object wired with ``vault_address=subaccount_address``
        (subaccount mode) or ``account_address=subaccount_address`` +
        ``vault_address=None`` (master-direct mode).  ``None`` when no
        ``agent_key`` is provided.
    sub : str
        The address passed at construction time (subaccount in vault mode, the
        master in master-direct mode) — used for every read.
    """

    def __init__(
        self,
        env: str,
        subaccount_address: str,
        agent_key: str | None = None,
        nonce_manager: NonceManager | None = None,
        enable_spot: bool = False,
        route_via_vault: bool = True,
    ) -> None:
        if env == "mainnet":
            base_url: str = constants.MAINNET_API_URL
        else:
            base_url = constants.TESTNET_API_URL

        self.sub: str = subaccount_address
        self._base_url: str = base_url
        self._nonce: NonceManager | None = nonce_manager
        self._nonce_lock: FileLock | None = None
        self._spot_enabled: bool = enable_spot

        # spot_meta selector — None lets the SDK fetch REAL spot metadata at
        # construction (opt-in spot path); the empty stub keeps the default
        # perp-only path network-free and byte-identical to before.
        spot_meta_arg: dict[str, list[Any]] | None = (
            None if enable_spot else _EMPTY_SPOT_META
        )

        # Read endpoint — always available
        self.info: Info = Info(base_url, skip_ws=True, spot_meta=spot_meta_arg)

        # Write endpoint — only when an agent key is supplied
        if agent_key is not None:
            wallet = eth_account.Account.from_key(agent_key)
            if route_via_vault:
                # SUBACCOUNT mode (default).  CRITICAL: vault_address=
                # subaccount_address routes orders to the subaccount.  Do NOT
                # pass account_address here — that would silently route orders
                # to the agent wallet instead of the subaccount.
                self.exchange: Exchange | None = Exchange(
                    wallet,
                    base_url,
                    vault_address=subaccount_address,
                    spot_meta=spot_meta_arg,
                )
            else:
                # MASTER-DIRECT mode.  The agent acts FOR the master account
                # with NO vault routing: account_address sets the act-for
                # context to the master, and vault_address=None means the SDK
                # sends no vaultAddress in the action.  Here subaccount_address
                # holds the MASTER EVM address the agent is approved on.
                self.exchange = Exchange(
                    wallet,
                    base_url,
                    account_address=subaccount_address,
                    vault_address=None,
                    spot_meta=spot_meta_arg,
                )
            logger.info(
                "HLClient built (read-write)",
                env=env,
                sub=subaccount_address,
                route_via_vault=route_via_vault,
                nonce_managed=nonce_manager is not None,
            )
        else:
            self.exchange = None
            logger.info("HLClient built (read-only)", env=env, sub=subaccount_address)

    # ------------------------------------------------------------------
    # Factory — live construction seam (B11 / daemon)
    # ------------------------------------------------------------------

    @classmethod
    def for_trading(
        cls,
        env: str,
        subaccount_address: str,
        agent_key: str,
        nonce_state_path: str | Path,
        *,
        backward_tolerance_ms: int = 5000,
        enable_spot: bool = False,
        route_via_vault: bool = True,
    ) -> HLClient:
        """Build a trading-ready HLClient with a wired ``NonceManager`` (MF-10).

        This is the canonical construction seam for the live daemon (B11).
        It creates a ``NonceManager`` backed by *nonce_state_path* and returns
        an ``HLClient`` with that manager wired, so every order submit uses a
        persistent, backward-clock-guarded nonce.

        Parameters
        ----------
        env:
            ``"mainnet"`` or ``"testnet"``.
        subaccount_address:
            EVM address of the account being managed.  In subaccount mode
            (``route_via_vault=True``) this is the Hyperliquid sub-account; in
            master-direct mode (``route_via_vault=False``) this is the MASTER
            EVM address the agent is approved on.
        agent_key:
            Hex private key of the agent/API wallet.
        nonce_state_path:
            Path to the nonce state file.  Parent directories are created
            automatically.  The watchdog kill-agent MUST use a different path
            to keep its nonce space separate (S-4).
        backward_tolerance_ms:
            Passed through to ``NonceManager``.  Default 5 000 ms.
        enable_spot:
            Opt-in spot read path (default ``False``).  Forwarded to
            ``__init__`` — see its docstring for the network-on-construction
            implication.  Default-false keeps the live daemon's perp-only build
            byte-identical to before.
        route_via_vault:
            Routing-mode selector (default ``True``).  Forwarded to ``__init__``:
            ``True`` → subaccount mode (``vault_address=subaccount_address``);
            ``False`` → master-direct mode (``account_address=subaccount_address``
            with ``vault_address=None``).  Default-true keeps the legacy
            subaccount routing byte-identical.

        Returns
        -------
        HLClient
            Read-write client with ``_nonce`` wired to a ``NonceManager``.
        """
        p = Path(nonce_state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(p) + ".lock")
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError(
                "another hl-allocator process already holds the nonce lock for this "
                "agent/state file — refusing to start (single authoritative signer; "
                "see spec §7/§23)"
            ) from None
        nm = NonceManager(p, backward_tolerance_ms=backward_tolerance_ms)
        client = cls(
            env,
            subaccount_address,
            agent_key=agent_key,
            nonce_manager=nm,
            enable_spot=enable_spot,
            route_via_vault=route_via_vault,
        )
        client._nonce_lock = lock
        return client

    # ------------------------------------------------------------------
    # Lock lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the exclusive nonce-file lock acquired by ``for_trading()``.

        Idempotent — safe to call multiple times.  Read-only clients (no agent
        key / no lock) are a no-op.  Called by ``daemon.shutdown()`` so a
        graceful stop releases the lock immediately, allowing a clean restart
        without waiting for OS cleanup.
        """
        if self._nonce_lock is not None:
            with contextlib.suppress(Exception):
                self._nonce_lock.release()
            self._nonce_lock = None

    def __enter__(self) -> HLClient:
        """Context-manager entry — returns ``self``."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context-manager exit — releases the nonce-file lock."""
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _nonce_override(self) -> Generator[None, None, None]:
        """Per-call context manager that swaps ``hyperliquid.exchange.get_timestamp_ms``.

        When ``self._nonce`` is ``None`` this is a no-op (yields immediately).
        Otherwise, for the duration of the ``with`` block, the module-level name
        ``hyperliquid.exchange.get_timestamp_ms`` is replaced with a lambda that
        delegates to ``self._nonce.next()``.  The original function is always
        restored in the ``finally`` clause — even if the SDK call or ``next()``
        raises.

        A ``ClockError`` raised by ``self._nonce.next()`` fires at
        ``timestamp = get_timestamp_ms()`` inside ``bulk_orders()``, which is
        called BEFORE any signature computation or network post.  The order is
        therefore refused before send (MF-10 requirement).

        Re-entrancy guard (F4)
        ----------------------
        The swap mutates a process-global, so a concurrent same-process submit
        would corrupt the save/restore (the inner restore would write back the
        OUTER override's lambda as the "original").  Deployment is a single
        authoritative signer per process, so this never happens in production —
        but we guard it explicitly: the installed lambda is tagged with
        ``_is_hl_nonce_override``; if that tag is already present when we enter,
        we raise ``RuntimeError`` rather than nest.  Because we refuse BEFORE
        swapping, the global is left holding the original outer override and the
        outer ``finally`` still restores the TRUE original.
        """
        if self._nonce is None:
            yield
            return

        nonce: NonceManager = self._nonce  # local non-Optional binding for the closure
        original = _hl_exchange.get_timestamp_ms
        if getattr(original, "_is_hl_nonce_override", False):
            raise RuntimeError(
                "concurrent nonce override — single authoritative signer only"
            )

        def _override() -> int:
            result: int = nonce.next()
            return result

        _override._is_hl_nonce_override = True  # type: ignore[attr-defined]

        _hl_exchange.get_timestamp_ms = _override
        try:
            yield
        finally:
            _hl_exchange.get_timestamp_ms = original

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def equity(self) -> Decimal:
        """Return the total account value (USD) of the subaccount.

        Returns
        -------
        Decimal
            ``marginSummary.accountValue`` as a ``Decimal``.
        """
        state: dict[str, Any] = self.info.user_state(self.sub)
        raw: str = state["marginSummary"]["accountValue"]
        return Decimal(str(float(raw)))

    def positions(self) -> dict[str, Decimal]:
        """Return current perpetual positions as a coin → signed-size mapping.

        Coins with no open position are absent (implicitly flat).

        Returns
        -------
        dict[str, Decimal]
            ``{"BTC": Decimal("0.5"), "ETH": Decimal("-1.2"), ...}``
        """
        state: dict[str, Any] = self.info.user_state(self.sub)
        result: dict[str, Decimal] = {}
        for entry in state.get("assetPositions", []):
            pos = entry["position"]
            coin: str = pos["coin"]
            szi: str = pos["szi"]
            result[coin] = Decimal(str(float(szi)))
        return result

    def positions_leverage(self) -> dict[str, dict[str, Any]]:
        """Return ``{coin: {"type": "cross"|"isolated", "value": int}}`` for every OPEN
        position, read from ``clearinghouseState.assetPositions[].leverage``.  Coins
        with no open position are absent.  Read-only.

        Reads the SAME clearinghouse snapshot ``positions()``/``equity()`` read
        (``self.info.user_state(self.sub)``) so the leverage view is consistent
        with the position view.
        """
        state: dict[str, Any] = self.info.user_state(self.sub)
        out: dict[str, dict[str, Any]] = {}
        for p in state.get("assetPositions", []):
            pos = p.get("position", {})
            coin = pos.get("coin")
            lev = pos.get("leverage") or {}
            if coin and lev:
                value: Any = lev.get("value")
                out[coin] = {"type": lev.get("type"), "value": int(value)}
        return out

    def account_snapshot(self) -> dict[str, Any]:
        """Read-only account snapshot for health monitoring.

        Returns ``{"equity": float, "maint_margin": float, "positions": {coin: {
        "szi": float, "position_value": float, "unrealized_pnl": float,
        "liquidation_px": float | None, "funding_since_open": float,
        "leverage_type": str, "leverage_value": int}}}``.  Parsed from
        clearinghouseState.

        Reads the SAME clearinghouse snapshot ``positions()``/``equity()``/
        ``positions_leverage()`` read (``self.info.user_state(self.sub)``) so the
        health view is consistent with the position/leverage views.  Read-only —
        no agent key required.  All numeric fields are coerced defensively from
        the HL numeric-string representation; ``liquidation_px`` is ``None`` when
        the venue reports it null (position far from liquidation).
        """
        st: dict[str, Any] = self.info.user_state(self.sub)
        ms = st.get("marginSummary", {})
        out_positions: dict[str, dict[str, Any]] = {}
        for ap in st.get("assetPositions", []):
            p = ap.get("position", {})
            coin = p.get("coin")
            if not coin:
                continue
            lev = p.get("leverage") or {}
            liq = p.get("liquidationPx")
            cf = p.get("cumFunding") or {}
            out_positions[coin] = {
                "szi": float(p.get("szi", 0) or 0),
                "position_value": float(p.get("positionValue", 0) or 0),
                "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                "liquidation_px": (float(liq) if liq is not None else None),
                "funding_since_open": float(cf.get("sinceOpen", 0) or 0),
                "leverage_type": lev.get("type"),
                "leverage_value": int(lev.get("value", 0) or 0),
            }
        return {
            "equity": float(ms.get("accountValue", 0) or 0),
            "maint_margin": float(st.get("crossMaintenanceMarginUsed", 0) or 0),
            "positions": out_positions,
        }

    def venue_meta(self) -> dict[str, dict[str, Any]]:
        """Return ``{coin: {"max_leverage": int, "only_isolated": bool, "delisted":
        bool}}`` for every perp in the venue universe.

        Reads ``self.info.meta()`` — the SAME info endpoint the venue-absence
        guard and ``sz_decimals()`` read — but exposes the per-coin health flags
        (max leverage, isolated-only, delisted) the health monitor needs for
        held-coin checks.  Read-only — no agent key required.  ``onlyIsolated``
        may be ``None`` (treated as ``False``) and ``isDelisted`` may be absent
        (treated as ``False``).
        """
        meta: dict[str, Any] = self.info.meta()
        out: dict[str, dict[str, Any]] = {}
        for a in meta.get("universe", []):
            name = a.get("name")
            if not name:
                continue
            out[name] = {
                "max_leverage": int(a.get("maxLeverage", 0) or 0),
                "only_isolated": bool(a.get("onlyIsolated") or False),
                "delisted": bool(a.get("isDelisted") or False),
            }
        return out

    def marks(self) -> dict[str, Decimal]:
        """Return the best available price for every listed perpetual.

        Price priority (first non-None): ``oraclePx`` → ``markPx`` → ``midPx``.

        Returns
        -------
        dict[str, Decimal]
            ``{"BTC": Decimal("70000"), "ETH": Decimal("2500"), ...}``
        """
        mctxs = self.info.meta_and_asset_ctxs()
        meta_dict: dict[str, Any] = mctxs[0]
        ctxs: list[dict[str, Any] | None] = mctxs[1]

        result: dict[str, Decimal] = {}
        for universe_entry, ctx in zip(meta_dict["universe"], ctxs, strict=False):
            coin: str = universe_entry["name"]
            if ctx is None:
                continue
            raw_px: str | None = (
                ctx.get("oraclePx") or ctx.get("markPx") or ctx.get("midPx")
            )
            if raw_px is None:
                continue
            result[coin] = Decimal(str(float(raw_px)))

        return result

    def l2_book(self, coin: str) -> dict[str, list[tuple[float, float]]]:
        """Return the L2 order book for ``coin`` as ``{"bids": [...], "asks": [...]}``.

        Each side is a list of ``(price, size)`` tuples, best-first.  Uses the
        SDK ``Info.l2_snapshot``; an empty/absent side returns ``[]`` (never raises
        for a listed coin).  Read-only — no agent key required.

        Parameters
        ----------
        coin:
            Perp name (e.g. ``"BTC"``).

        Returns
        -------
        dict[str, list[tuple[float, float]]]
            ``{"bids": [(px, sz), ...], "asks": [(px, sz), ...]}``
        """
        snap = self.info.l2_snapshot(coin)
        levels = snap.get("levels", [[], []]) if isinstance(snap, dict) else [[], []]

        def _side(rows: Any) -> list[tuple[float, float]]:
            return [(float(r["px"]), float(r["sz"])) for r in rows]

        bids = _side(levels[0]) if len(levels) > 0 else []
        asks = _side(levels[1]) if len(levels) > 1 else []
        return {"bids": bids, "asks": asks}

    def sz_decimals(self) -> dict[str, int]:
        """Return the size-decimal precision for every listed perpetual.

        Returns
        -------
        dict[str, int]
            ``{"BTC": 5, "ETH": 4, ...}``
        """
        meta: dict[str, Any] = self.info.meta()
        return {entry["name"]: int(entry["szDecimals"]) for entry in meta["universe"]}

    # ------------------------------------------------------------------
    # Spot read methods (D1 venue routing — opt-in via enable_spot=True)
    # ------------------------------------------------------------------
    #
    # These mirror the perp positions()/marks()/sz_decimals() shape but read the
    # spot info endpoints.  They require enable_spot=True so the SDK loaded real
    # spot metadata at construction; otherwise spot names would not resolve.
    # All three raise a clear RuntimeError when the client is not spot-enabled.

    def _require_spot(self) -> None:
        """Guard: the three spot read methods are inert unless opted in."""
        if not self._spot_enabled:
            raise RuntimeError("spot reads require enable_spot=True")

    def spot_balances(self) -> dict[str, Decimal]:
        """Return owned spot token balances as a coin → quantity mapping.

        These are the spot analogue of perp ``positions()``: a spot holding is a
        quantity of OWNED tokens (always ``>= 0``), NOT a signed perp ``szi``.
        Zero balances are SKIPPED — a coin absent from the mapping means a flat
        (zero) spot holding, mirroring how ``positions()`` omits flat perps.

        Requires ``enable_spot=True`` (raises ``RuntimeError`` otherwise).

        Returns
        -------
        dict[str, Decimal]
            ``{"HYPE": Decimal("12.5"), "USDC": Decimal("1000.0"), ...}``
        """
        self._require_spot()
        state: dict[str, Any] = self.info.spot_user_state(self.sub)
        result: dict[str, Decimal] = {}
        for entry in state.get("balances", []):
            coin: str = entry["coin"]
            total = Decimal(str(float(entry["total"])))
            if total == 0:
                continue
            result[coin] = total
        return result

    def spot_marks(self) -> dict[str, Decimal]:
        """Return the best available price for every listed spot pair.

        Price priority (first non-None): ``markPx`` → ``midPx``.

        The spot ``universe`` and the spot ctxs list returned by
        ``spot_meta_and_asset_ctxs()`` can be DIFFERENT lengths and in a
        DIFFERENT order, so ctxs are indexed by their own ``coin`` field (the
        spot pair name) — NEVER positional-zipped against the universe.  A pair
        with no matching ctx, or whose ctx has no usable price, is omitted.

        Requires ``enable_spot=True`` (raises ``RuntimeError`` otherwise).

        Returns
        -------
        dict[str, Decimal]
            ``{"@107": Decimal("35.0"), "PURR/USDC": Decimal("0.42"), ...}``
        """
        self._require_spot()
        mctxs = self.info.spot_meta_and_asset_ctxs()
        spot_meta: dict[str, Any] = mctxs[0]
        ctxs: list[dict[str, Any]] = mctxs[1]

        # Index ctxs by their per-entry pair name — the lists are NOT aligned.
        ctx_by_name: dict[str, dict[str, Any]] = {
            ctx["coin"]: ctx for ctx in ctxs if ctx is not None and "coin" in ctx
        }

        result: dict[str, Decimal] = {}
        for universe_entry in spot_meta["universe"]:
            name: str = universe_entry["name"]
            ctx = ctx_by_name.get(name)
            if ctx is None:
                continue
            raw_px: str | None = ctx.get("markPx") or ctx.get("midPx")
            if raw_px is None:
                continue
            result[name] = Decimal(str(float(raw_px)))

        return result

    def spot_sz_decimals(self) -> dict[str, int]:
        """Return the size-decimal precision for every spot pair.

        Each pair's precision is its BASE token's ``szDecimals``.  In
        ``spot_meta()``, ``universe[i]["tokens"]`` is ``[baseIdx, quoteIdx]`` and
        the token table is ``tokens`` — so the precision is
        ``tokens[baseIdx]["szDecimals"]``.

        Requires ``enable_spot=True`` (raises ``RuntimeError`` otherwise).

        Returns
        -------
        dict[str, int]
            ``{"@107": 2, "PURR/USDC": 0, ...}``
        """
        self._require_spot()
        meta: dict[str, Any] = self.info.spot_meta()
        tokens: list[dict[str, Any]] = meta["tokens"]
        result: dict[str, int] = {}
        for pair in meta["universe"]:
            base_idx: int = pair["tokens"][0]
            result[pair["name"]] = int(tokens[base_idx]["szDecimals"])
        return result

    # ------------------------------------------------------------------
    # Write methods (IOC order submission — require exchange to be set)
    # ------------------------------------------------------------------

    def submit_ioc(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        slippage: float,
        cloid: str,
    ) -> dict[str, Any]:
        """Submit an aggressive-limit IOC order (market open).

        Parameters
        ----------
        coin:
            Perp name (e.g. ``"BTC"``).
        is_buy:
            ``True`` for a long entry; ``False`` for a size reduction.
        sz:
            Absolute size in coin units.
        slippage:
            Maximum slippage fraction (e.g. ``0.005`` = 0.5 %).  Uses the
            TIGHT value from ``cfg.caps.max_order_slippage`` — NOT the SDK
            default of 5 %.
        cloid:
            Client order ID (``"0x"`` + 32 hex chars, 16 bytes).

        Returns
        -------
        dict[str, Any]
            Raw HTTP response from the exchange.
        """
        from hyperliquid.utils.signing import Cloid as _Cloid  # noqa: PLC0415

        assert self.exchange is not None, "submit_ioc requires an agent_key (exchange is None)"
        cloid_obj = _Cloid.from_str(cloid)
        with self._nonce_override():
            return self.exchange.market_open(coin, is_buy, sz, None, slippage, cloid=cloid_obj)  # type: ignore[no-any-return]

    def submit_spot_ioc(
        self,
        pair: str,
        is_buy: bool,
        sz: float,
        slippage: float,
        cloid: str,
    ) -> dict[str, Any]:
        """Submit an aggressive-limit IOC order on the SPOT book (D3 venue routing).

        Thin spot analogue of ``submit_ioc``: it wraps the SAME
        ``Exchange.market_open`` the perp path uses.  The SDK resolves the spot
        asset id (>= 10000) and rounds the aggressive price to 8 decimals for
        spot (vs 6 for perp) INTERNALLY from the spot ``pair`` name — this
        wrapper does NOT reimplement spot pricing or rounding.

        Spot flatten convention
        -----------------------
        ``sz`` is in BASE-TOKEN units (e.g. number of HYPE tokens).
        ``is_buy=True`` spends USDC to BUY the base token; ``is_buy=False``
        SELLS the base token for USDC — selling is how a spot position is
        reduced/flattened, because there is NO reduce-only ``market_close`` on
        spot.

        Nonce wiring
        ------------
        Routes through the SAME per-call ``_nonce_override`` as ``submit_ioc``.
        The nonce space is shared with the perp writes — fine, because it is the
        same agent on the same account (single authoritative signer).

        Parameters
        ----------
        pair:
            HL spot pair name (e.g. ``"@107"`` for HYPE/USDC).
        is_buy:
            ``True`` to BUY the base token (spend USDC); ``False`` to SELL it
            (spot reduce/flatten).
        sz:
            Absolute size in BASE-TOKEN units.
        slippage:
            Maximum slippage fraction (e.g. ``0.01`` = 1 %).
        cloid:
            Client order ID (``"0x"`` + 32 hex chars, 16 bytes).

        Returns
        -------
        dict[str, Any]
            Raw HTTP response from the exchange (parsed by callers with
            ``safety.parse_order_response``).
        """
        from hyperliquid.utils.signing import Cloid as _Cloid  # noqa: PLC0415

        assert self.exchange is not None, "submit_spot_ioc requires an agent_key (exchange is None)"
        cloid_obj = _Cloid.from_str(cloid)
        with self._nonce_override():
            return self.exchange.market_open(pair, is_buy, sz, None, slippage, cloid=cloid_obj)  # type: ignore[no-any-return]

    def market_close(
        self,
        coin: str,
        slippage: float,
        cloid: str,
    ) -> dict[str, Any]:
        """Submit a reduce-only close for a full position exit.

        Uses ``Exchange.market_close`` which is reduce-only — it can NEVER
        flip into a short even if the exchange already filled the position
        to zero before the request arrives.

        Parameters
        ----------
        coin:
            Perp name to close.
        slippage:
            Maximum slippage fraction.
        cloid:
            Client order ID (``"0x"`` + 32 hex chars, 16 bytes).

        Returns
        -------
        dict[str, Any]
            Raw HTTP response from the exchange.
        """
        from hyperliquid.utils.signing import Cloid as _Cloid  # noqa: PLC0415

        assert self.exchange is not None, "market_close requires an agent_key (exchange is None)"
        cloid_obj = _Cloid.from_str(cloid)
        with self._nonce_override():
            return self.exchange.market_close(coin, None, None, slippage, cloid=cloid_obj)  # type: ignore[no-any-return]

    def update_leverage(self, coin: str, leverage: int, *, is_cross: bool) -> dict[str, Any]:
        """Set the per-coin margin leverage (and cross/isolated mode) on the venue.

        Wraps the SDK ``Exchange.update_leverage``.  Requires an agent key
        (``self.exchange``).  Idempotent on the venue side (setting the current
        leverage is a no-op success).
        """
        assert self.exchange is not None, "update_leverage requires an agent_key (exchange is None)"
        with self._nonce_override():
            return self.exchange.update_leverage(leverage, coin, is_cross)  # type: ignore[no-any-return]

    def query_by_cloid(self, cloid: str) -> dict[str, Any]:
        """Query the status of an order by its client order ID.

        Parameters
        ----------
        cloid:
            Client order ID (``"0x"`` + 32 hex chars, 16 bytes).

        Returns
        -------
        dict[str, Any]
            Raw order-status response from the exchange info endpoint.
        """
        from hyperliquid.utils.signing import Cloid as _Cloid  # noqa: PLC0415

        cloid_obj = _Cloid.from_str(cloid)
        return self.info.query_order_by_cloid(self.sub, cloid_obj)  # type: ignore[no-any-return]

    def cancel_by_cloid(self, coin: str, cloid: str) -> dict[str, Any]:
        """Cancel a resting order by its client order ID.

        Parameters
        ----------
        coin:
            Perp name (e.g. ``"BTC"``).
        cloid:
            Client order ID (``"0x"`` + 32 hex chars, 16 bytes).

        Returns
        -------
        dict[str, Any]
            Raw HTTP response from the exchange.
        """
        from hyperliquid.utils.signing import Cloid as _Cloid  # noqa: PLC0415

        assert self.exchange is not None, "cancel_by_cloid requires an agent_key (exchange is None)"
        cloid_obj = _Cloid.from_str(cloid)
        with self._nonce_override():
            return self.exchange.cancel_by_cloid(coin, cloid_obj)  # type: ignore[no-any-return]

    def cancel_oid(self, coin: str, oid: int) -> dict[str, Any]:
        """Cancel a resting order by its exchange order ID.

        Parameters
        ----------
        coin:
            Perp name (e.g. ``"BTC"``).
        oid:
            Exchange-assigned order ID.

        Returns
        -------
        dict[str, Any]
            Raw HTTP response from the exchange.
        """
        assert self.exchange is not None, "cancel_oid requires an agent_key (exchange is None)"
        with self._nonce_override():
            return self.exchange.cancel(coin, oid)  # type: ignore[no-any-return]

    def open_orders(self) -> list[dict[str, Any]]:
        """Return all currently resting open orders for the subaccount.

        Read-only — does not require a nonce.

        Returns
        -------
        list[dict[str, Any]]
            Raw list of open order dicts (each has at least ``coin``, ``oid``,
            ``sz``, ``side``; may carry ``cloid``).
        """
        return self.info.open_orders(self.sub)  # type: ignore[no-any-return]

    def user_rate_limit(self) -> dict[str, Any]:
        """Return the current rate-limit counters for this subaccount.

        Read-only — no nonce required.  Returns the raw dict from
        ``info.user_rate_limit(sub)`` which has the shape::

            {
                "cumVlm": "<str>",
                "nRequestsUsed": <int>,
                "nRequestsCap": <int>,
            }

        Callers compute ``remaining = nRequestsCap - nRequestsUsed`` to decide
        whether the exchange rate budget is sufficient to submit the pending
        cycle's orders (MF-9).

        Returns
        -------
        dict[str, Any]
            Raw rate-limit response from the Hyperliquid info endpoint.
        """
        return self.info.user_rate_limit(self.sub)  # type: ignore[no-any-return]

    def fills_since(self, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]:
        """Return all fills for the subaccount in the given time range.

        Thin wrapper around ``info.user_fills_by_time``.  Tests use a fake
        client implementing this method directly.

        Parameters
        ----------
        start_ms:
            Start of the query window in milliseconds since the Unix epoch
            (inclusive).
        end_ms:
            End of the query window in milliseconds (inclusive).  ``None``
            leaves the upper bound open (returns fills up to the present).

        Returns
        -------
        list[dict[str, Any]]
            Raw fill records as returned by the Hyperliquid info endpoint.
        """
        return self.info.user_fills_by_time(self.sub, start_ms, end_ms)  # type: ignore[no-any-return]
