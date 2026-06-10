"""
Tests for the D1 spot READ path on ``automation.core.hl_client.HLClient``.

D1 adds an OPT-IN spot read path (``enable_spot=True``) with three read
methods — ``spot_balances``, ``spot_marks``, ``spot_sz_decimals`` — that mirror
the existing perp ``positions``/``marks``/``sz_decimals`` shape but read from the
spot info endpoints.  This increment is ADDITIVE: with the default
``enable_spot=False`` the client behaves byte-identically to before (the perp
path is untouched and the SDK is still built with the empty spot-meta stub, so
no real spot-meta network call is made on construction).

All tests here are fully OFFLINE: a small ``FakeSpotInfo`` returns canned dicts
mirroring the real SDK shapes (see ``hyperliquid/info.py`` +
``hyperliquid/utils/types.py``), INCLUDING a length mismatch between the spot
``universe`` and the spot ctxs list — this proves ``spot_marks`` indexes ctxs by
their per-entry ``coin`` field (the pair name) and never positional-zips.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from automation.core.hl_client import _EMPTY_SPOT_META, HLClient

# ---------------------------------------------------------------------------
# Fake spot info object — mirrors the real SDK shapes, fully offline.
# ---------------------------------------------------------------------------


class FakeSpotInfo:
    """Canned-response stand-in for ``hyperliquid.info.Info`` spot endpoints.

    Shapes mirror the real SDK (``hyperliquid/info.py``):

    * ``spot_user_state(address)`` → ``{"balances": [{"coin","token","total","hold",...}]}``
    * ``spot_meta_and_asset_ctxs()`` → ``[spot_meta, ctxs]`` where ``ctxs`` is a
      list of ``{"coin","markPx","midPx",...}`` — CRUCIALLY the ``universe`` and
      ``ctxs`` lists are DIFFERENT lengths and in a DIFFERENT order, so a correct
      implementation must index ctxs by their ``coin`` field, not by position.
    * ``spot_meta()`` → ``{"universe": [{"name","tokens":[baseIdx,quoteIdx],"index"}],
      "tokens": [{"name","szDecimals","index"}]}``
    """

    def __init__(self) -> None:
        self.spot_user_state_calls: list[str] = []

    # --- balances ---------------------------------------------------------
    def spot_user_state(self, address: str) -> dict[str, Any]:
        self.spot_user_state_calls.append(address)
        return {
            "balances": [
                {"coin": "HYPE", "token": 1, "total": "12.5", "hold": "0.0"},
                {"coin": "USDC", "token": 0, "total": "1000.0", "hold": "10.0"},
                {"coin": "UETH", "token": 2, "total": "0.0", "hold": "0.0"},  # zero
            ]
        }

    # --- marks (universe / ctxs LENGTH + ORDER mismatch) ------------------
    def spot_meta_and_asset_ctxs(self) -> list[Any]:
        spot_meta: dict[str, Any] = {
            "universe": [
                {"name": "@107", "tokens": [1, 0], "index": 107, "isCanonical": False},
                {"name": "@151", "tokens": [2, 0], "index": 151, "isCanonical": False},
                {"name": "PURR/USDC", "tokens": [3, 0], "index": 0, "isCanonical": True},
            ],
            "tokens": [],
        }
        # ctxs: DIFFERENT length (4 vs 3) AND scrambled order vs universe.
        # The extra "@999" ctx has no matching universe pair (must be ignored),
        # and @151 has markPx=None so the midPx fallback must kick in.
        ctxs: list[dict[str, Any]] = [
            {"coin": "PURR/USDC", "markPx": "0.42", "midPx": "0.421"},
            {"coin": "@999", "markPx": "9.99", "midPx": "9.98"},
            {"coin": "@107", "markPx": "35.0", "midPx": "35.1"},
            {"coin": "@151", "markPx": None, "midPx": "2500.5"},
        ]
        return [spot_meta, ctxs]

    # --- sz decimals ------------------------------------------------------
    def spot_meta(self) -> dict[str, Any]:
        return {
            "universe": [
                {"name": "@107", "tokens": [1, 0], "index": 107, "isCanonical": False},
                {"name": "@151", "tokens": [2, 0], "index": 151, "isCanonical": False},
                {"name": "PURR/USDC", "tokens": [3, 0], "index": 0, "isCanonical": True},
            ],
            "tokens": [
                {"name": "USDC", "szDecimals": 8, "index": 0},
                {"name": "HYPE", "szDecimals": 2, "index": 1},
                {"name": "UETH", "szDecimals": 4, "index": 2},
                {"name": "PURR", "szDecimals": 0, "index": 3},
            ],
        }


def _spot_client() -> tuple[HLClient, FakeSpotInfo]:
    """Build a spot-enabled HLClient with the fake spot info swapped in — OFFLINE.

    We construct with the default ``enable_spot=False`` (the only network-free
    build: it uses the empty spot-meta stub) and then flip ``_spot_enabled`` to
    ``True`` and replace ``c.info`` with the fake.  Constructing directly with
    ``enable_spot=True`` would make the real SDK ``Info.__init__`` POST for live
    spot metadata at build time (the documented network-on-construction
    behaviour) — undesirable in an offline unit test.  Flipping the flag
    afterwards exercises the exact same spot-read code path (the read methods
    gate on ``_spot_enabled``) without any network call.
    """
    c = HLClient("testnet", "0x" + "a" * 40)
    c._spot_enabled = True
    fake = FakeSpotInfo()
    c.info = fake  # type: ignore[assignment]
    return c, fake


# ---------------------------------------------------------------------------
# spot_balances()
# ---------------------------------------------------------------------------


class TestSpotBalances:
    def test_parses_balances_to_decimal(self) -> None:
        c, fake = _spot_client()
        bals = c.spot_balances()

        assert bals["HYPE"] == Decimal("12.5")
        assert bals["USDC"] == Decimal("1000.0")
        assert all(isinstance(v, Decimal) for v in bals.values())

    def test_balances_are_non_negative(self) -> None:
        c, _ = _spot_client()
        bals = c.spot_balances()
        assert all(v >= 0 for v in bals.values())

    def test_zero_balances_are_skipped(self) -> None:
        """UETH has total 0.0 → omitted (mirrors perp positions: flat ⇒ absent)."""
        c, _ = _spot_client()
        bals = c.spot_balances()
        assert "UETH" not in bals

    def test_passes_sub_address(self) -> None:
        c, fake = _spot_client()
        c.spot_balances()
        assert fake.spot_user_state_calls == [c.sub]


# ---------------------------------------------------------------------------
# spot_marks() — proves NON-positional, coin-indexed ctx lookup
# ---------------------------------------------------------------------------


class TestSpotMarks:
    def test_marks_indexed_by_pair_name_not_position(self) -> None:
        """With a 3-pair universe and a 4-ctx (scrambled) list, every pair gets
        its OWN mark — a positional zip would mis-assign or drop pairs."""
        c, _ = _spot_client()
        marks = c.spot_marks()

        assert marks["@107"] == Decimal("35.0")  # markPx
        assert marks["PURR/USDC"] == Decimal("0.42")  # markPx

    def test_midpx_fallback_when_markpx_none(self) -> None:
        """@151 has markPx=None → must fall back to midPx (2500.5)."""
        c, _ = _spot_client()
        marks = c.spot_marks()
        assert marks["@151"] == Decimal("2500.5")

    def test_extra_ctx_without_universe_pair_is_ignored(self) -> None:
        """@999 ctx has no matching universe entry → not in the result."""
        c, _ = _spot_client()
        marks = c.spot_marks()
        assert "@999" not in marks

    def test_marks_keys_match_universe(self) -> None:
        c, _ = _spot_client()
        marks = c.spot_marks()
        assert set(marks) == {"@107", "@151", "PURR/USDC"}

    def test_marks_values_are_all_decimal(self) -> None:
        c, _ = _spot_client()
        marks = c.spot_marks()
        assert all(isinstance(v, Decimal) for v in marks.values())


# ---------------------------------------------------------------------------
# spot_sz_decimals() — pair → BASE token szDecimals
# ---------------------------------------------------------------------------


class TestSpotSzDecimals:
    def test_pair_maps_to_base_token_sz_decimals(self) -> None:
        c, _ = _spot_client()
        sd = c.spot_sz_decimals()

        # @107 base token idx 1 = HYPE (szDecimals 2)
        assert sd["@107"] == 2
        # @151 base token idx 2 = UETH (szDecimals 4)
        assert sd["@151"] == 4
        # PURR/USDC base token idx 3 = PURR (szDecimals 0)
        assert sd["PURR/USDC"] == 0

    def test_all_values_are_int(self) -> None:
        c, _ = _spot_client()
        sd = c.spot_sz_decimals()
        assert all(isinstance(v, int) for v in sd.values())

    def test_keys_match_universe(self) -> None:
        c, _ = _spot_client()
        sd = c.spot_sz_decimals()
        assert set(sd) == {"@107", "@151", "PURR/USDC"}


# ---------------------------------------------------------------------------
# Opt-in gating — default enable_spot=False is byte-identical to before
# ---------------------------------------------------------------------------


class TestSpotDisabledByDefault:
    def test_default_client_is_not_spot_enabled(self) -> None:
        c = HLClient("testnet", "0x" + "0" * 40)
        assert c._spot_enabled is False

    def test_spot_balances_raises_when_disabled(self) -> None:
        c = HLClient("testnet", "0x" + "0" * 40)
        with pytest.raises(RuntimeError, match="enable_spot=True"):
            c.spot_balances()

    def test_spot_marks_raises_when_disabled(self) -> None:
        c = HLClient("testnet", "0x" + "0" * 40)
        with pytest.raises(RuntimeError, match="enable_spot=True"):
            c.spot_marks()

    def test_spot_sz_decimals_raises_when_disabled(self) -> None:
        c = HLClient("testnet", "0x" + "0" * 40)
        with pytest.raises(RuntimeError, match="enable_spot=True"):
            c.spot_sz_decimals()

    def test_default_path_uses_empty_spot_stub_no_network(self) -> None:
        """Default (enable_spot=False) builds Info with the empty spot-meta stub.

        The SDK only fetches real spot meta when ``spot_meta is None`` (see
        ``hyperliquid/info.py`` __init__).  Passing ``_EMPTY_SPOT_META`` means no
        spot-meta network call on construction — the spot ``name_to_coin`` map is
        empty, which is exactly today's perp-only behaviour.  We assert the empty
        stub is in effect by confirming the SDK loaded ZERO spot pairs into its
        ``coin_to_asset`` map for the synthetic '@'-prefixed spot names.
        """
        c = HLClient("testnet", "0x" + "0" * 40)
        # The empty spot stub yields no spot entries in name_to_coin (perp names
        # are added from real meta, but no '@NNN' spot pair names exist).
        spot_names = [k for k in c.info.name_to_coin if k.startswith("@")]
        assert spot_names == [], (
            "default client must build Info with the empty spot stub "
            f"(no spot pairs loaded); found {spot_names[:5]}"
        )

    def test_enable_spot_flag_threads_through_for_trading(self) -> None:
        """``for_trading`` exposes an ``enable_spot`` param, default False."""
        import inspect  # noqa: PLC0415

        sig = inspect.signature(HLClient.for_trading)
        assert "enable_spot" in sig.parameters
        assert sig.parameters["enable_spot"].default is False


# ---------------------------------------------------------------------------
# Sanity: the empty spot-meta constant is unchanged (additive increment).
# ---------------------------------------------------------------------------


def test_empty_spot_meta_constant_unchanged() -> None:
    assert _EMPTY_SPOT_META == {"universe": [], "tokens": []}
