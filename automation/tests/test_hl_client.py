"""
Tests for automation.core.hl_client.

Network tests hit the Hyperliquid **testnet** (public meta endpoint only — no
account needed, no orders submitted).  If testnet is unreachable they are
skipped via a session-scoped fixture.

Non-network tests (routing assertion) run unconditionally because they need
only a throwaway eth_account key and the SDK constructor — no HTTP call is
made until a read/write method is called.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import eth_account
import pytest

from automation.core.hl_client import HLClient

# ---------------------------------------------------------------------------
# Connectivity guard
# ---------------------------------------------------------------------------


def _testnet_reachable() -> bool:
    """Return True if the Hyperliquid testnet meta endpoint responds."""
    try:
        import json

        import httpx

        r = httpx.post(
            "https://api.hyperliquid-testnet.xyz/info",
            content=json.dumps({"type": "meta"}),
            headers={"Content-Type": "application/json"},
            timeout=8.0,
        )
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def testnet_ok() -> bool:
    return _testnet_reachable()


# ---------------------------------------------------------------------------
# Network tests — marks()
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("testnet_ok")
class TestMarksNetwork:
    def test_marks_returns_nonempty_dict(self, testnet_ok: bool) -> None:
        if not testnet_ok:
            pytest.skip("Hyperliquid testnet unreachable in this environment")

        c = HLClient("testnet", "0x" + "0" * 40)
        marks = c.marks()

        assert len(marks) > 0, "marks() returned an empty dict"

    def test_marks_btc_is_decimal_gt_zero(self, testnet_ok: bool) -> None:
        if not testnet_ok:
            pytest.skip("Hyperliquid testnet unreachable in this environment")

        c = HLClient("testnet", "0x" + "0" * 40)
        marks = c.marks()

        assert "BTC" in marks, f"BTC not in marks(); got keys: {list(marks)[:10]}"
        assert isinstance(marks["BTC"], Decimal), f"marks['BTC'] is {type(marks['BTC'])}"
        assert marks["BTC"] > 0, f"marks['BTC'] = {marks['BTC']} (expected > 0)"

    def test_marks_values_are_all_decimal(self, testnet_ok: bool) -> None:
        if not testnet_ok:
            pytest.skip("Hyperliquid testnet unreachable in this environment")

        c = HLClient("testnet", "0x" + "0" * 40)
        marks = c.marks()

        non_decimal = {k: type(v) for k, v in marks.items() if not isinstance(v, Decimal)}
        assert not non_decimal, f"Non-Decimal values: {non_decimal}"


# ---------------------------------------------------------------------------
# Network tests — sz_decimals()
# ---------------------------------------------------------------------------


class TestSzDecimalsNetwork:
    def test_sz_decimals_btc_is_int(self, testnet_ok: bool) -> None:
        if not testnet_ok:
            pytest.skip("Hyperliquid testnet unreachable in this environment")

        c = HLClient("testnet", "0x" + "0" * 40)
        sd = c.sz_decimals()

        assert "BTC" in sd, f"BTC not in sz_decimals(); got: {list(sd)[:10]}"
        assert isinstance(sd["BTC"], int), f"sz_decimals['BTC'] is {type(sd['BTC'])}"

    def test_sz_decimals_all_int(self, testnet_ok: bool) -> None:
        if not testnet_ok:
            pytest.skip("Hyperliquid testnet unreachable in this environment")

        c = HLClient("testnet", "0x" + "0" * 40)
        sd = c.sz_decimals()

        non_int = {k: type(v) for k, v in sd.items() if not isinstance(v, int)}
        assert not non_int, f"Non-int sz_decimals: {non_int}"


# ---------------------------------------------------------------------------
# Routing test — NO network call, NO order submitted
# ---------------------------------------------------------------------------


class TestRoutingVaultAddress:
    """Confirm that Exchange is wired with vault_address=subaccount_address.

    This is THE most security-critical fact: orders must route to the
    subaccount, not the agent wallet.  The test creates a throwaway key,
    builds HLClient, and inspects exchange.vault_address without making any
    network call or submitting any order.
    """

    def test_exchange_vault_address_equals_subaccount(self) -> None:
        sub_addr = "0x" + "a" * 40  # fake but structurally valid EVM address
        k = eth_account.Account.create().key.hex()

        c = HLClient("testnet", sub_addr, agent_key=k)

        assert c.exchange is not None, "exchange should be built when agent_key is provided"
        assert c.exchange.vault_address == sub_addr, (
            f"ROUTING BUG: exchange.vault_address={c.exchange.vault_address!r} "
            f"but expected {sub_addr!r}.  Orders would route to the wrong account!"
        )

    def test_exchange_is_none_without_key(self) -> None:
        sub_addr = "0x" + "0" * 40
        c = HLClient("testnet", sub_addr)
        assert c.exchange is None, "exchange should be None for read-only client"

    def test_sub_attribute_stored(self) -> None:
        sub_addr = "0x" + "c" * 40
        c = HLClient("testnet", sub_addr)
        assert c.sub == sub_addr

    def test_mainnet_uses_mainnet_url(self) -> None:
        from hyperliquid.utils import constants

        sub_addr = "0x" + "0" * 40
        c = HLClient("mainnet", sub_addr)
        assert c._base_url == constants.MAINNET_API_URL

    def test_testnet_uses_testnet_url(self) -> None:
        from hyperliquid.utils import constants

        sub_addr = "0x" + "0" * 40
        c = HLClient("testnet", sub_addr)
        assert c._base_url == constants.TESTNET_API_URL


# ---------------------------------------------------------------------------
# Routing mode test — master-direct vs subaccount (NO network, NO order)
# ---------------------------------------------------------------------------


class _CapturingExchange:
    """Fake Exchange that records the kwargs it was constructed with.

    Monkeypatched over ``hl_client.Exchange`` so we can assert EXACTLY which
    routing kwargs ``HLClient`` passes for each ``route_via_vault`` mode without
    building the real SDK object or making any network call.
    """

    last_kwargs: dict[str, Any] = {}
    last_args: tuple[Any, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        type(self).last_args = args
        type(self).last_kwargs = kwargs
        # Mirror the real SDK attribute so existing code/tests that read it work.
        self.vault_address: Any = kwargs.get("vault_address")
        self.account_address: Any = kwargs.get("account_address")


class TestRouteViaVault:
    """``route_via_vault`` selects subaccount (vault) vs master-direct (account).

    These tests monkeypatch ``hl_client.Exchange`` with a capturing fake and
    assert the kwargs passed for each mode.  No network call, no order.
    """

    def _build(
        self, monkeypatch: pytest.MonkeyPatch, **client_kw: Any
    ) -> tuple[HLClient, type[_CapturingExchange]]:
        import automation.core.hl_client as hl_client_mod  # noqa: PLC0415

        monkeypatch.setattr(hl_client_mod, "Exchange", _CapturingExchange)
        sub_addr = "0x" + "a" * 40
        k = eth_account.Account.create().key.hex()
        c = HLClient("testnet", sub_addr, agent_key=k, **client_kw)
        return c, _CapturingExchange

    def test_default_routes_via_vault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (route_via_vault unset) → vault_address=sub, NO account_address."""
        sub_addr = "0x" + "a" * 40
        _, ex = self._build(monkeypatch)
        assert ex.last_kwargs.get("vault_address") == sub_addr
        # account_address must be absent or None (subaccount routing only).
        assert ex.last_kwargs.get("account_address") is None

    def test_explicit_true_routes_via_vault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """route_via_vault=True → vault_address=sub, NO account_address (unchanged)."""
        sub_addr = "0x" + "a" * 40
        _, ex = self._build(monkeypatch, route_via_vault=True)
        assert ex.last_kwargs.get("vault_address") == sub_addr
        assert ex.last_kwargs.get("account_address") is None

    def test_false_routes_master_direct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """route_via_vault=False → account_address=addr, vault_address=None."""
        sub_addr = "0x" + "a" * 40
        _, ex = self._build(monkeypatch, route_via_vault=False)
        assert ex.last_kwargs.get("account_address") == sub_addr
        assert ex.last_kwargs.get("vault_address") is None

    def test_sub_attribute_stored_both_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """self.sub == subaccount_address regardless of routing mode."""
        sub_addr = "0x" + "a" * 40
        c_vault, _ = self._build(monkeypatch, route_via_vault=True)
        assert c_vault.sub == sub_addr
        c_master, _ = self._build(monkeypatch, route_via_vault=False)
        assert c_master.sub == sub_addr

    def test_spot_meta_still_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """spot_meta kwarg is forwarded unchanged in both modes (empty stub when perp)."""
        _, ex = self._build(monkeypatch, route_via_vault=False)
        # Perp-only default → empty spot-meta stub, NOT None.
        assert ex.last_kwargs.get("spot_meta") is not None

    def test_for_trading_threads_route_via_vault(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """for_trading(route_via_vault=False) reaches Exchange as master-direct."""
        import automation.core.hl_client as hl_client_mod  # noqa: PLC0415

        monkeypatch.setattr(hl_client_mod, "Exchange", _CapturingExchange)
        sub_addr = "0x" + "f" * 40
        k = eth_account.Account.create().key.hex()
        nonce_path = tmp_path / "nonce.state"
        c = HLClient.for_trading(
            "testnet", sub_addr, k, nonce_path, route_via_vault=False
        )
        try:
            assert _CapturingExchange.last_kwargs.get("account_address") == sub_addr
            assert _CapturingExchange.last_kwargs.get("vault_address") is None
            assert c.sub == sub_addr
        finally:
            c.close()  # release the nonce FileLock

    def test_for_trading_default_routes_via_vault(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """for_trading default → vault routing (vault_address=sub, no account_address)."""
        import automation.core.hl_client as hl_client_mod  # noqa: PLC0415

        monkeypatch.setattr(hl_client_mod, "Exchange", _CapturingExchange)
        sub_addr = "0x" + "9" * 40
        k = eth_account.Account.create().key.hex()
        nonce_path = tmp_path / "nonce.state"
        c = HLClient.for_trading("testnet", sub_addr, k, nonce_path)
        try:
            assert _CapturingExchange.last_kwargs.get("vault_address") == sub_addr
            assert _CapturingExchange.last_kwargs.get("account_address") is None
            assert c.sub == sub_addr
        finally:
            c.close()


# ---------------------------------------------------------------------------
# cancel_by_cloid / cancel_oid / open_orders — offline unit tests
# ---------------------------------------------------------------------------

_FAKE_CLOID = "0x" + "ab" * 16  # 32 hex chars


class FakeExchangeForCancel:
    """Fake exchange that records cancel calls and returns canned responses."""

    def __init__(self) -> None:
        self.vault_address: str = "0x" + "a" * 40
        self.cancel_by_cloid_calls: list[tuple[str, object]] = []
        self.cancel_calls: list[tuple[str, int]] = []
        self.update_leverage_calls: list[tuple[int, str, bool]] = []

    def cancel_by_cloid(self, coin: str, cloid: object) -> dict:
        self.cancel_by_cloid_calls.append((coin, cloid))
        return {"status": "ok", "response": {"type": "cancel"}}

    def cancel(self, coin: str, oid: int) -> dict:
        self.cancel_calls.append((coin, oid))
        return {"status": "ok", "response": {"type": "cancel"}}

    def update_leverage(self, leverage: int, name: str, is_cross: bool = True) -> dict:
        self.update_leverage_calls.append((leverage, name, is_cross))
        return {"status": "ok", "response": {"type": "updateLeverage"}}


class FakeInfoForOrders:
    """Fake Info that returns canned open orders."""

    def __init__(self, orders: list[dict]) -> None:
        self._orders = orders

    def open_orders(self, address: str) -> list[dict]:
        return self._orders


class TestCancelMethods:
    """Offline tests for cancel_by_cloid, cancel_oid, and open_orders."""

    def _client_with_fake_exchange(self) -> tuple[HLClient, FakeExchangeForCancel]:
        import eth_account as _eth  # noqa: PLC0415

        k = _eth.Account.create().key.hex()
        c = HLClient("testnet", "0x" + "a" * 40, agent_key=k)
        fake = FakeExchangeForCancel()
        c.exchange = fake  # type: ignore[assignment]
        return c, fake

    def test_cancel_by_cloid_routes_to_exchange(self) -> None:
        """cancel_by_cloid delegates to exchange.cancel_by_cloid with a Cloid object."""
        client, fake = self._client_with_fake_exchange()

        result = client.cancel_by_cloid("BTC", _FAKE_CLOID)

        assert len(fake.cancel_by_cloid_calls) == 1
        coin_seen, cloid_obj_seen = fake.cancel_by_cloid_calls[0]
        assert coin_seen == "BTC"
        # The cloid must have been converted to a Cloid object (not raw str)
        assert hasattr(cloid_obj_seen, "to_raw"), (
            f"Expected a Cloid object, got {type(cloid_obj_seen)}"
        )
        assert result["status"] == "ok"

    def test_cancel_oid_routes_to_exchange(self) -> None:
        """cancel_oid delegates to exchange.cancel with the coin and int oid."""
        client, fake = self._client_with_fake_exchange()

        result = client.cancel_oid("ETH", 12345)

        assert len(fake.cancel_calls) == 1
        coin_seen, oid_seen = fake.cancel_calls[0]
        assert coin_seen == "ETH"
        assert oid_seen == 12345
        assert result["status"] == "ok"

    def test_cancel_by_cloid_requires_exchange(self) -> None:
        """cancel_by_cloid raises AssertionError when no agent_key (read-only client)."""
        c = HLClient("testnet", "0x" + "0" * 40)  # no agent_key → exchange is None
        try:
            c.cancel_by_cloid("BTC", _FAKE_CLOID)
            raised = False
        except AssertionError:
            raised = True
        assert raised, "cancel_by_cloid must assert when exchange is None"

    def test_cancel_oid_requires_exchange(self) -> None:
        """cancel_oid raises AssertionError when no agent_key (read-only client)."""
        c = HLClient("testnet", "0x" + "0" * 40)
        try:
            c.cancel_oid("BTC", 1)
            raised = False
        except AssertionError:
            raised = True
        assert raised, "cancel_oid must assert when exchange is None"

    def test_cancel_by_cloid_wrapped_in_nonce_override(self, tmp_path: Any) -> None:
        """cancel_by_cloid runs inside _nonce_override when a NonceManager is attached.

        We verify that the nonce issued by the manager is consumed during the
        cancel call (the fake records get_timestamp_ms() at call time, same as
        the nonce-test pattern in test_hl_client_nonce.py).
        """
        import eth_account as _eth  # noqa: PLC0415
        import hyperliquid.exchange as _hl_exchange  # noqa: PLC0415

        from automation.core.nonce import NonceManager  # noqa: PLC0415

        class NonceSensingExchange(FakeExchangeForCancel):
            def __init__(self) -> None:
                super().__init__()
                self.nonces_seen: list[int] = []

            def cancel_by_cloid(self, coin: str, cloid: object) -> dict:
                self.nonces_seen.append(_hl_exchange.get_timestamp_ms())
                return super().cancel_by_cloid(coin, cloid)

        nonce_path = tmp_path / "nonce.txt"
        nm = NonceManager(nonce_path)
        k = _eth.Account.create().key.hex()
        client = HLClient("testnet", "0x" + "a" * 40, agent_key=k, nonce_manager=nm)
        fake_ex = NonceSensingExchange()
        client.exchange = fake_ex  # type: ignore[assignment]

        client.cancel_by_cloid("BTC", _FAKE_CLOID)

        assert len(fake_ex.nonces_seen) == 1
        # The nonce seen by the fake must equal the manager's last issued nonce
        assert fake_ex.nonces_seen[0] == nm._last, (
            f"nonce seen by fake {fake_ex.nonces_seen[0]} != manager's _last {nm._last}"
        )

    def test_open_orders_reads_from_info(self) -> None:
        """open_orders() delegates to info.open_orders(sub) — no exchange call."""
        canned = [{"coin": "BTC", "oid": 1, "sz": "0.5", "side": "B"}]
        c = HLClient("testnet", "0x" + "b" * 40)
        c.info = FakeInfoForOrders(canned)  # type: ignore[assignment]

        orders = c.open_orders()

        assert orders == canned, "open_orders must return info.open_orders(sub) verbatim"

    def test_update_leverage_routes_to_exchange(self) -> None:
        """update_leverage delegates to exchange.update_leverage(leverage, coin, is_cross)."""
        client, fake = self._client_with_fake_exchange()

        result = client.update_leverage("BTC", 1, is_cross=False)

        assert len(fake.update_leverage_calls) == 1
        leverage_seen, name_seen, is_cross_seen = fake.update_leverage_calls[0]
        # SDK arg order is (leverage, name, is_cross)
        assert leverage_seen == 1
        assert name_seen == "BTC"
        assert is_cross_seen is False
        assert result["status"] == "ok"

    def test_update_leverage_cross_true(self) -> None:
        """is_cross=True is forwarded as the SDK's third positional arg."""
        client, fake = self._client_with_fake_exchange()

        client.update_leverage("ETH", 5, is_cross=True)

        assert fake.update_leverage_calls[0] == (5, "ETH", True)

    def test_update_leverage_requires_exchange(self) -> None:
        """update_leverage raises AssertionError when no agent_key (read-only client)."""
        c = HLClient("testnet", "0x" + "0" * 40)  # no agent_key → exchange is None
        try:
            c.update_leverage("BTC", 1, is_cross=False)
            raised = False
        except AssertionError:
            raised = True
        assert raised, "update_leverage must assert when exchange is None"


# ---------------------------------------------------------------------------
# user_rate_limit — offline unit tests
# ---------------------------------------------------------------------------


class FakeInfoForRateLimit:
    """Fake Info that records user_rate_limit calls and returns canned responses."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls: list[str] = []

    def user_rate_limit(self, address: str) -> dict:
        self.calls.append(address)
        return self._response


class TestUserRateLimit:
    """user_rate_limit() delegates to info.user_rate_limit(sub) (MF-9)."""

    def test_user_rate_limit_delegates_to_info(self) -> None:
        """user_rate_limit() calls info.user_rate_limit(self.sub) and returns the result."""
        sub_addr = "0x" + "d" * 40
        canned = {"cumVlm": "1234.56", "nRequestsCap": 1000, "nRequestsUsed": 42}

        c = HLClient("testnet", sub_addr)
        fake_info = FakeInfoForRateLimit(canned)
        c.info = fake_info  # type: ignore[assignment]

        result = c.user_rate_limit()

        assert result == canned, "user_rate_limit must return info.user_rate_limit(sub) verbatim"
        assert len(fake_info.calls) == 1, "info.user_rate_limit must be called exactly once"
        assert fake_info.calls[0] == sub_addr, (
            f"user_rate_limit must pass self.sub={sub_addr!r}; got {fake_info.calls[0]!r}"
        )

    def test_user_rate_limit_passes_correct_sub(self) -> None:
        """Confirm sub address is forwarded correctly (not agent/agent-wallet address)."""
        sub_addr = "0x" + "e" * 40
        c = HLClient("testnet", sub_addr)
        fake_info = FakeInfoForRateLimit({"nRequestsCap": 100, "nRequestsUsed": 0})
        c.info = fake_info  # type: ignore[assignment]

        c.user_rate_limit()

        assert fake_info.calls[0] == sub_addr


# ---------------------------------------------------------------------------
# l2_book — offline unit tests via FakeHLClient
# ---------------------------------------------------------------------------


def test_fake_l2_book_shape() -> None:
    """FakeHLClient.l2_book returns the configured book and falls back to empty."""
    from automation.tests._fakes import FakeHLClient

    c = FakeHLClient(l2={"HYPE": {"bids": [(65.0, 2.0)], "asks": []}})
    book = c.l2_book("HYPE")
    assert book["asks"] == []
    assert book["bids"] == [(65.0, 2.0)]
    # default (unconfigured) coin returns empty book, not KeyError
    assert c.l2_book("BTC") == {"bids": [], "asks": []}


def test_fake_update_leverage_records() -> None:
    """FakeHLClient.update_leverage records (coin, leverage, is_cross) and returns ok."""
    from automation.tests._fakes import FakeHLClient

    c = FakeHLClient()
    assert c.leverage_calls == []
    result = c.update_leverage("BTC", 1, is_cross=False)
    assert result == {"status": "ok"}
    c.update_leverage("ETH", 3, is_cross=True)
    assert c.leverage_calls == [("BTC", 1, False), ("ETH", 3, True)]


# ---------------------------------------------------------------------------
# positions_leverage — per-held-position leverage TYPE + VALUE read
# ---------------------------------------------------------------------------


class FakeInfoForLeverage:
    """Fake Info that returns a canned clearinghouseState user_state."""

    def __init__(self, state: dict) -> None:
        self._state = state
        self.user_state_calls: list[str] = []

    def user_state(self, address: str) -> dict:
        self.user_state_calls.append(address)
        return self._state


def test_positions_leverage_parses_type_and_value() -> None:
    """positions_leverage() maps each OPEN position to {type, value} from the
    clearinghouseState assetPositions[].leverage, read via info.user_state(sub)."""
    state = {
        "assetPositions": [
            {"position": {"coin": "BTC", "szi": "0.5",
                          "leverage": {"type": "cross", "value": 10}}},
            {"position": {"coin": "ETH", "szi": "-1.2",
                          "leverage": {"type": "isolated", "value": 1}}},
        ]
    }
    c = HLClient("testnet", "0x" + "c" * 40)
    fake = FakeInfoForLeverage(state)
    c.info = fake  # type: ignore[assignment]

    out = c.positions_leverage()

    assert out == {
        "BTC": {"type": "cross", "value": 10},
        "ETH": {"type": "isolated", "value": 1},
    }
    # Read from the same clearinghouse path as positions()/marks(): user_state(sub).
    assert fake.user_state_calls == [c.sub]


def test_positions_leverage_absent_position_absent_key() -> None:
    """A coin with no open position (empty assetPositions) is absent from the map."""
    c = HLClient("testnet", "0x" + "d" * 40)
    c.info = FakeInfoForLeverage({"assetPositions": []})  # type: ignore[assignment]

    assert c.positions_leverage() == {}


def test_positions_leverage_skips_entries_without_leverage() -> None:
    """An entry with a coin but no leverage dict is skipped (no partial key)."""
    state = {
        "assetPositions": [
            {"position": {"coin": "BTC", "szi": "0.5",
                          "leverage": {"type": "isolated", "value": 1}}},
            {"position": {"coin": "SOL", "szi": "3.0"}},  # no leverage key
        ]
    }
    c = HLClient("testnet", "0x" + "e" * 40)
    c.info = FakeInfoForLeverage(state)  # type: ignore[assignment]

    assert c.positions_leverage() == {"BTC": {"type": "isolated", "value": 1}}


def test_fake_positions_leverage_default_empty() -> None:
    """FakeHLClient.positions_leverage defaults to {} (no held leverage)."""
    from automation.tests._fakes import FakeHLClient

    assert FakeHLClient().positions_leverage() == {}


def test_fake_positions_leverage_configurable() -> None:
    """FakeHLClient(positions_leverage=...) returns the configured map."""
    from automation.tests._fakes import FakeHLClient

    pl = {"BTC": {"type": "cross", "value": 10}}
    c = FakeHLClient(positions_leverage=pl)
    assert c.positions_leverage() == pl


# ---------------------------------------------------------------------------
# account_snapshot — health-monitor clearinghouse read (PH-3 UNIT A)
# ---------------------------------------------------------------------------


def test_account_snapshot_parses_clearinghouse() -> None:
    """account_snapshot() parses marginSummary + per-position fields from the
    clearinghouseState read via info.user_state(sub).  One position has a null
    liquidationPx (far from liquidation), another a concrete float."""
    state = {
        "marginSummary": {"accountValue": "12345.67"},
        "crossMaintenanceMarginUsed": "234.5",
        "assetPositions": [
            {
                "position": {
                    "coin": "BTC",
                    "szi": "0.5",
                    "positionValue": "35000.0",
                    "unrealizedPnl": "120.25",
                    "liquidationPx": None,
                    "leverage": {"type": "cross", "value": 10},
                    "cumFunding": {
                        "allTime": "9.0",
                        "sinceOpen": "-1.5",
                        "sinceChange": "0.0",
                    },
                }
            },
            {
                "position": {
                    "coin": "ETH",
                    "szi": "-1.2",
                    "positionValue": "3000.0",
                    "unrealizedPnl": "-45.0",
                    "liquidationPx": "4200.5",
                    "leverage": {"type": "isolated", "value": 3},
                    "cumFunding": {
                        "allTime": "2.0",
                        "sinceOpen": "0.75",
                        "sinceChange": "0.0",
                    },
                }
            },
        ],
    }
    c = HLClient("testnet", "0x" + "c" * 40)
    fake = FakeInfoForLeverage(state)
    c.info = fake  # type: ignore[assignment]

    snap = c.account_snapshot()

    assert snap["equity"] == 12345.67
    assert snap["maint_margin"] == 234.5
    # Same clearinghouse path as positions()/equity(): user_state(sub).
    assert fake.user_state_calls == [c.sub]

    btc = snap["positions"]["BTC"]
    assert btc["szi"] == 0.5
    assert btc["position_value"] == 35000.0
    assert btc["unrealized_pnl"] == 120.25
    assert btc["liquidation_px"] is None
    assert btc["funding_since_open"] == -1.5
    assert btc["leverage_type"] == "cross"
    assert btc["leverage_value"] == 10

    eth = snap["positions"]["ETH"]
    assert eth["szi"] == -1.2
    assert eth["liquidation_px"] == 4200.5
    assert isinstance(eth["liquidation_px"], float)
    assert eth["funding_since_open"] == 0.75
    assert eth["leverage_type"] == "isolated"
    assert eth["leverage_value"] == 3


def test_account_snapshot_defensive_on_missing_fields() -> None:
    """Missing marginSummary / assetPositions / sub-fields coerce to safe zeros
    and an empty positions map (HL omits fields when flat)."""
    c = HLClient("testnet", "0x" + "d" * 40)
    c.info = FakeInfoForLeverage({})  # type: ignore[assignment]

    snap = c.account_snapshot()

    assert snap == {"equity": 0.0, "maint_margin": 0.0, "positions": {}}


def test_account_snapshot_skips_entries_without_coin() -> None:
    """A position entry with no coin is skipped (no empty-key garbage)."""
    state = {
        "marginSummary": {"accountValue": "100.0"},
        "assetPositions": [
            {"position": {"szi": "1.0"}},  # no coin
            {
                "position": {
                    "coin": "SOL",
                    "szi": "3.0",
                    "positionValue": "450.0",
                    "unrealizedPnl": "0.0",
                    "leverage": {"type": "cross", "value": 5},
                }
            },
        ],
    }
    c = HLClient("testnet", "0x" + "e" * 40)
    c.info = FakeInfoForLeverage(state)  # type: ignore[assignment]

    snap = c.account_snapshot()

    assert list(snap["positions"].keys()) == ["SOL"]
    sol = snap["positions"]["SOL"]
    # cumFunding absent → funding_since_open defaults to 0.0
    assert sol["funding_since_open"] == 0.0
    assert sol["leverage_value"] == 5


# ---------------------------------------------------------------------------
# venue_meta — per-coin venue flags read (PH-3 UNIT B)
# ---------------------------------------------------------------------------


class FakeInfoForMeta:
    """Fake Info that returns a canned meta() universe."""

    def __init__(self, meta: dict) -> None:
        self._meta = meta
        self.meta_calls: int = 0

    def meta(self) -> dict:
        self.meta_calls += 1
        return self._meta


def test_venue_meta_parses_per_coin_flags() -> None:
    """venue_meta() maps each perp to {max_leverage, only_isolated, delisted}
    from info.meta()['universe'], incl. an only_isolated and a delisted coin."""
    meta = {
        "universe": [
            {"name": "BTC", "maxLeverage": 50, "onlyIsolated": None},
            {"name": "ETH", "maxLeverage": 25, "onlyIsolated": False, "isDelisted": False},
            {"name": "MEME", "maxLeverage": 5, "onlyIsolated": True},
            {"name": "OLD", "maxLeverage": 3, "isDelisted": True},
        ]
    }
    c = HLClient("testnet", "0x" + "a" * 40)
    fake = FakeInfoForMeta(meta)
    c.info = fake  # type: ignore[assignment]

    out = c.venue_meta()

    assert out == {
        "BTC": {"max_leverage": 50, "only_isolated": False, "delisted": False},
        "ETH": {"max_leverage": 25, "only_isolated": False, "delisted": False},
        "MEME": {"max_leverage": 5, "only_isolated": True, "delisted": False},
        "OLD": {"max_leverage": 3, "only_isolated": False, "delisted": True},
    }
    assert fake.meta_calls == 1


def test_venue_meta_skips_entries_without_name() -> None:
    """A universe entry with no name is skipped."""
    meta = {"universe": [{"maxLeverage": 10}, {"name": "BTC", "maxLeverage": 50}]}
    c = HLClient("testnet", "0x" + "b" * 40)
    c.info = FakeInfoForMeta(meta)  # type: ignore[assignment]

    out = c.venue_meta()

    assert list(out.keys()) == ["BTC"]


def test_venue_meta_empty_universe() -> None:
    """An absent/empty universe yields an empty dict."""
    c = HLClient("testnet", "0x" + "c" * 40)
    c.info = FakeInfoForMeta({})  # type: ignore[assignment]

    assert c.venue_meta() == {}


# ---------------------------------------------------------------------------
# FakeHLClient — account_snapshot + venue_meta (PH-3 UNIT C)
# ---------------------------------------------------------------------------


def test_fake_account_snapshot_default_sentinel() -> None:
    """FakeHLClient.account_snapshot defaults to the empty sentinel."""
    from automation.tests._fakes import FakeHLClient

    assert FakeHLClient().account_snapshot() == {
        "equity": 0.0,
        "maint_margin": 0.0,
        "positions": {},
    }


def test_fake_account_snapshot_configurable() -> None:
    """FakeHLClient(account_snapshot=...) returns the configured dict."""
    from automation.tests._fakes import FakeHLClient

    snap = {
        "equity": 1000.0,
        "maint_margin": 50.0,
        "positions": {
            "BTC": {
                "szi": 0.1,
                "position_value": 7000.0,
                "unrealized_pnl": 10.0,
                "liquidation_px": None,
                "funding_since_open": -0.5,
                "leverage_type": "cross",
                "leverage_value": 10,
            }
        },
    }
    c = FakeHLClient(account_snapshot=snap)
    assert c.account_snapshot() == snap


def test_fake_venue_meta_default_empty() -> None:
    """FakeHLClient.venue_meta defaults to {}."""
    from automation.tests._fakes import FakeHLClient

    assert FakeHLClient().venue_meta() == {}


def test_fake_venue_meta_configurable() -> None:
    """FakeHLClient(venue_meta=...) returns the configured map."""
    from automation.tests._fakes import FakeHLClient

    vm = {"BTC": {"max_leverage": 50, "only_isolated": False, "delisted": False}}
    c = FakeHLClient(venue_meta=vm)
    assert c.venue_meta() == vm
