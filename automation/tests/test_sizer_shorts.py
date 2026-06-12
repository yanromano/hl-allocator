"""
Tests for the sizer's SHORT-side money math (red-team C1 + C2, CRITICAL).

All expected values are HAND-COMPUTED — no tautologies.

Seam
----
``plan()`` is called DIRECTLY with a hand-built ``TargetAllocation`` carrying
negative weights.  This is the honest public seam: the ``TargetAllocation``
pydantic model explicitly permits negative weights ("negative values are short
fractions OF THE SLEEVE", allocation/base.py) — the long-only rule lives in the
VALIDATOR layer, not the model, and the sizer must be safe for any allocation
the validator can let through once ``allow_short`` sleeves exist (Task B3).
Nothing in production produces negative targets yet, so these bugs are LATENT;
the tests inject the future inputs at the seam where they will arrive.

C1 — magnitude-based notional sums
----------------------------------
The sizer's gross/cap accumulations were SIGNED sums.  With shorts, longs and
shorts CANCEL: an all-short plan has NEGATIVE "gross" and never trips any cap;
a +0.5/−0.5 book has signed gross EXACTLY 0 while true gross = 1.0·E — silent
over-leverage.  Every cap accumulation must sum ``abs(size × price)``.

C2 — flooring must shrink magnitude, not grow it
------------------------------------------------
``math.floor`` on a NEGATIVE size rounds AWAY from zero (floor(−1.0001) = −2),
GROWING short exposure and violating rounding.py's own "never exceeds equity"
guarantee.  Flooring must be applied on the magnitude with the sign reapplied
(``copysign(floor_size(abs(raw)), raw)``).

NOTE on chosen numbers: the C1 tests use dyadic-exact values (±0.5 weights,
buffer 0.5, powers-of-ten prices) so every floor lands exactly on a lot
boundary for BOTH sign conventions — they isolate C1 from C2 and stay green
between the two fixes.  The C2 tests then deliberately use non-exact sizes
(−1.0001 lot; the plan's +0.6/−0.6 @ 0.97E cap) where sign-broken flooring
diverges.
"""

from __future__ import annotations

import pytest

import automation.execution.sizer as sizer_mod
from automation.execution.sizer import SizerError, plan
from automation.tests.test_sizer import _cfg, _snap, _state, _target

# ---------------------------------------------------------------------------
# C1 — magnitude-based notional sums (Task B1)
# ---------------------------------------------------------------------------


class TestGrossCapOnMagnitude:
    def test_all_short_plan_trips_gross_cap(self) -> None:
        """All-negative targets with Σ|notional| > cap MUST scale down.

        E=100_000, safety_buffer=0.5 → cap = 50_000.
        BTC w=−0.5, px=10_000, szd=1 → raw size −5.0   (|notional| 50_000)
        ETH w=−0.5, px= 2_000, szd=1 → raw size −25.0  (|notional| 50_000)
        Σ|notional| = 100_000 > 50_000 → scale = 0.5:
            BTC → −2.5 (25_000), ETH → −12.5 (25_000) → post gross 50_000 = cap.

        Under SIGNED sums the "gross" is −100_000 < cap → no scaling → the plan
        ships 2× the cap in short exposure.  (All values dyadic-exact: flooring
        is a no-op for both sign conventions — pure C1, no C2 interference.)
        """
        ta = _target({"BTC": -0.5, "ETH": -0.5}, cash=0.0)
        marks = {"BTC": 10_000.0, "ETH": 2_000.0}
        snap = _snap(
            equity=100_000,
            positions={},
            marks=marks,
            sz_decimals={"BTC": 1, "ETH": 1},
        )
        state = _state(last_target=None)  # bootstrap → rebalance fires
        cfg = _cfg(safety_buffer=0.5)

        sp = plan(ta, snap, state, cfg)

        cap = 100_000 * (1 - 0.5)
        assert sp.rebalance is True
        assert sp.scaled_down is True, (
            "scale-down branch did not fire on an all-short over-gross plan "
            "(signed gross sum — C1)"
        )
        assert abs(sp.scale_factor - 0.5) < 1e-12

        # The money assertion: TRUE gross of the planned orders ≤ cap.
        true_gross = sum(abs(o.target_size) * marks[o.coin] for o in sp.orders)
        assert true_gross <= cap * (1 + 1e-9), (
            f"orders' Σ|size·price| = {true_gross:.2f} exceeds cap {cap:.2f}"
        )
        assert sp.gross_notional <= cap * (1 + 1e-9)

        # Hand-computed scaled sizes (sign preserved, magnitude scaled).
        assert abs(sp.target_sizes["BTC"] - (-2.5)) < 1e-12
        assert abs(sp.target_sizes["ETH"] - (-12.5)) < 1e-12

    def test_mixed_book_true_gross_enforced(self) -> None:
        """+0.5 long / −0.5 short: signed gross is EXACTLY 0, true gross = E.

        E=100_000, safety_buffer=0.5 → cap = 50_000.
        BTC w=+0.5, px=10_000, szd=1 → +5.0 ; ETH w=−0.5, px=2_000, szd=1 → −25.0.
        Signed sum: +50_000 − 50_000 = 0 → a signed gross check sees NOTHING
        while the account holds 1.0·E of true exposure (the C1 cancellation in
        its purest form).  Magnitude sum: 100_000 > 50_000 → scale 0.5 →
        BTC +2.5 / ETH −12.5, post gross 50_000 = cap; the post-sizing hard
        guard must NOT raise after a correct scale-down.
        """
        ta = _target({"BTC": 0.5, "ETH": -0.5}, cash=0.0)
        marks = {"BTC": 10_000.0, "ETH": 2_000.0}
        snap = _snap(
            equity=100_000,
            positions={},
            marks=marks,
            sz_decimals={"BTC": 1, "ETH": 1},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.5)

        sp = plan(ta, snap, state, cfg)  # must not raise SizerError

        cap = 100_000 * (1 - 0.5)
        assert sp.scaled_down is True, (
            "mixed book: longs cancelled shorts in the signed gross sum (C1)"
        )
        true_gross = sum(abs(o.target_size) * marks[o.coin] for o in sp.orders)
        assert true_gross <= cap * (1 + 1e-9), (
            f"orders' Σ|size·price| = {true_gross:.2f} exceeds cap {cap:.2f}"
        )
        assert abs(sp.target_sizes["BTC"] - 2.5) < 1e-12
        assert abs(sp.target_sizes["ETH"] - (-12.5)) < 1e-12
        # Long stays a buy, short stays a sell after scaling.
        sides = {o.coin: o.side for o in sp.orders}
        assert sides == {"BTC": "buy", "ETH": "sell"}

    def test_post_gross_guard_uses_abs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The post-sizing HARD guard must measure Σ|notional|, not the signed sum.

        Mutation-style pin of the guard itself: monkeypatch ``floor_size`` with
        a sizing bug that INFLATES magnitudes by 1.5× (standing in for any
        future flooring/scaling defect).  The plan then reaches the guard with
        Σ|notional| ≈ 1.455·E ≫ cap while the SIGNED sum is deeply NEGATIVE
        (all-short) — a signed guard waves it through; the abs guard must
        raise SizerError.
        """
        monkeypatch.setattr(
            sizer_mod, "floor_size", lambda sz, szd: float(sz) * 1.5
        )
        ta = _target({"BTC": -1.0}, cash=0.0)
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 10_000.0},
            sz_decimals={"BTC": 1},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        with pytest.raises(SizerError):
            plan(ta, snap, state, cfg)


# ---------------------------------------------------------------------------
# C2 — flooring must shrink magnitude, never grow it (Task B2)
# ---------------------------------------------------------------------------


class TestFloorTowardZero:
    def test_short_size_floors_toward_zero(self) -> None:
        """A negative raw size must floor TOWARD zero: |planned| ≤ |raw|.

        E=20_002, BTC w=−0.5, px=10_000, szd=0:
            target_notional = −10_001 → raw_size = −1.0001
            math.floor(−1.0001) = −2.0  ← AWAY from zero: the "floor" DOUBLED
            the short exposure ($10_001 wanted → $20_000 planned, red-team C2).
            Correct: copysign(floor(|−1.0001|), −) = −1.0 ($10_000 ≤ wanted).

        safety_buffer=0.0 → cap = 20_002, so even the doubled size stays under
        the cap (gross 20_000) — the C1 guard CANNOT catch this; only
        sign-correct flooring does.
        """
        ta = _target({"BTC": -0.5}, cash=0.5)
        snap = _snap(
            equity=20_002,
            positions={},
            marks={"BTC": 10_000.0},
            sz_decimals={"BTC": 0},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.0)

        sp = plan(ta, snap, state, cfg)

        ts = sp.target_sizes["BTC"]
        raw = -0.5 * 20_002 / 10_000.0  # −1.0001
        assert abs(ts) <= abs(raw), (
            f"flooring GREW the short: |{ts}| > |{raw}| (C2 — floor away from zero)"
        )
        assert ts == -1.0, f"expected −1.0 (floored toward zero), got {ts}"
        # Direction preserved: still a short → sell order.
        assert [o.side for o in sp.orders if o.coin == "BTC"] == ["sell"]

    def test_long_floor_unchanged(self) -> None:
        """Positive path identical to before the C2 fix (pin from test_sizer.py).

        E=10_000, w=0.5, px=63_333, szd=3:
            raw = 5_000 / 63_333 = 0.078947… → floored 0.078
        (same expectation as TestFloorSizing.test_floor_sizing_hand_verified).
        """
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=10_000,
            positions={},
            marks={"BTC": 63_333.0},
            sz_decimals={"BTC": 3},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.0)

        sp = plan(ta, snap, state, cfg)

        btc_ts = sp.target_sizes.get("BTC", -1.0)
        assert abs(btc_ts - 0.078) < 1e-9, f"Expected 0.078, got {btc_ts}"

    def test_mixed_book_realistic_buffer_scale_refloor_toward_zero(self) -> None:
        """The plan-level +0.6/−0.6 @ cap 0.97E scenario — pins the SCALE-LOOP
        floor call site (the second place a signed size is floored).

        E=100_000, safety_buffer=0.03 → cap = 97_000.
        BTC w=+0.6, px=10_000, szd=1 → raw +6.0 ; ETH w=−0.6, px=2_000, szd=1
        → raw −30.0.  Σ|notional| = 120_000 > 97_000 → scale = 0.97/1.2:
            BTC scaled raw = +4.85  → floor → +4.8 (down, fine)
            ETH scaled raw = −24.25 → math.floor → −24.3 (GREW the short, C2)
                                    → correct: −24.2 (toward zero)
        Post gross (fixed): 48_000 + 48_400 = 96_400 ≤ 97_000 — no SizerError.
        """
        ta = _target({"BTC": 0.6, "ETH": -0.6}, cash=0.0)
        marks = {"BTC": 10_000.0, "ETH": 2_000.0}
        snap = _snap(
            equity=100_000,
            positions={},
            marks=marks,
            sz_decimals={"BTC": 1, "ETH": 1},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.03)

        sp = plan(ta, snap, state, cfg)  # must not raise

        cap = 100_000 * (1 - 0.03)
        assert sp.scaled_down is True
        eth_ts = sp.target_sizes["ETH"]
        eth_scaled_raw = -30.0 * 2_000.0 * sp.scale_factor / 2_000.0  # −24.25…
        assert abs(eth_ts) <= abs(eth_scaled_raw), (
            f"scale-loop refloor GREW the short: |{eth_ts}| > |{eth_scaled_raw}| (C2)"
        )
        assert abs(eth_ts - (-24.2)) < 1e-12, f"expected −24.2, got {eth_ts}"
        assert abs(sp.target_sizes["BTC"] - 4.8) < 1e-12
        true_gross = sum(abs(o.target_size) * marks[o.coin] for o in sp.orders)
        assert true_gross <= cap * (1 + 1e-9)


# ---------------------------------------------------------------------------
# B3 — signed targets behind allow_short + conditional long-only guard
# ---------------------------------------------------------------------------


class TestAllowShortPlanning:
    def test_plan_allow_short_opens_short_on_flat_book(self) -> None:
        """allow_short=True + negative target on a FLAT book → one sell-to-open.

        E=100_000, TON w=−0.3, px=6_000, szd=1 (cash=0.7: Σ|w|+cash=1, the
        allow_short unity contract from allocation/base.py):
            target_notional = −30_000 → raw size −5.0 (dyadic-exact, floor
            no-op); buffer 0.03 → cap 97_000 ≫ 30_000 gross → no scaling.
        Expect exactly ONE OrderPlan: target_size −5.0 (SIGNED), delta −5.0
        (flat book → delta == target), notional −30_000, side "sell".
        """
        ta = _target({"TON": -0.3}, cash=0.7)
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        state = _state(last_target=None)  # bootstrap → rebalance fires
        cfg = _cfg()

        sp = plan(ta, snap, state, cfg, allow_short=True)

        assert sp.rebalance is True
        assert len(sp.orders) == 1, f"expected exactly one order, got {sp.orders}"
        o = sp.orders[0]
        assert o.coin == "TON"
        assert o.side == "sell"
        assert o.target_size == -5.0, f"expected signed −5.0, got {o.target_size}"
        assert o.delta_size == o.target_size  # flat book: delta == target
        assert abs(o.delta_notional - (-30_000.0)) < 1e-9
        assert sp.flatten == []

    def test_plan_allow_short_keeps_existing_short(self) -> None:
        """A held short matching the target is KEPT — no flatten, no order.

        Held TON szi=−5.0; target w=−0.3 sizes to exactly −5.0 (E=100_000,
        px=6_000, szd=1 — same dyadic-exact math as the open test).  Delta
        = −5.0 − (−5.0) = 0 → |delta_notional| = 0 < $10 min → skipped as
        below_min_notional.  With allow_short=True the position must NOT be
        swept into flatten — the sleeve owns its short.
        """
        ta = _target({"TON": -0.3}, cash=0.7)
        snap = _snap(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        state = _state(last_target=None)
        cfg = _cfg()

        sp = plan(ta, snap, state, cfg, allow_short=True)

        assert sp.target_sizes["TON"] == -5.0  # sizing matched the held short
        assert [o for o in sp.orders if o.coin == "TON"] == [], (
            "zero-delta short produced an order"
        )
        assert "TON" not in sp.flatten, (
            "allow_short=True still swept the sleeve's own short into flatten"
        )
        # The why-no-order trace: delta 0 lands in skipped/below_min_notional.
        assert ("TON", "below_min_notional") in {
            (e["coin"], e["reason"]) for e in sp.skipped
        }

    def test_plan_default_flattens_shorts_unchanged(self) -> None:
        """allow_short OMITTED → today's long-only guard is byte-identical.

        Pins the default path (same shape as test_sizer.py's
        TestLongOnlyFlatten): a held short (TON szi=−5) on a long-only plan
        goes in the flatten list.  Existing callers pass no kwarg and must
        keep this residue-sweeper behavior.
        """
        ta = _target({"BTC": 0.5})
        snap = _snap(
            equity=100_000,
            positions={"BTC": 1.0, "TON": -5.0},  # TON is a foreign short
            marks={"BTC": 50_000.0, "TON": 6_000.0},
            sz_decimals={"BTC": 4, "TON": 1},
        )
        state = _state(last_target=None)

        sp = plan(ta, snap, state, _cfg())  # no allow_short kwarg

        assert "TON" in sp.flatten
        assert "BTC" not in sp.flatten

    def test_zeroed_by_cap_fires_for_shorts(self) -> None:
        """A wanted SHORT zeroed by scale-down → zeroed_by_cap, NOT below_min.

        The F1 check guarded ``want_notional > 0.0`` — a SIGNED short intent
        (negative notional) fails that comparison, so a short erased by the
        cap would be silently misfiled under below_min_notional (no F1
        warning).  Must be ``abs(want_notional) > 0.0``.

        E=100_000, buffer=0.5 → cap=50_000.  BTC w=+0.9, px=10_000, szd=1 →
        raw +9.0 (90_000); TON w=−0.05, px=5_000, szd=0 → raw −1.0 (5_000);
        cash=0.05 (Σ|w|+cash=1).  Gross 95_000 > 50_000 → scale = 10/19:
            BTC → 4.7368… → floor 4.7 (47_000)
            TON → −0.5263… → floor toward zero → −0.0  ← wanted, ERASED.
        intended_notional[TON] = −5_000 (signed).
        """
        ta = _target({"BTC": 0.9, "TON": -0.05}, cash=0.05)
        snap = _snap(
            equity=100_000,
            positions={},
            marks={"BTC": 10_000.0, "TON": 5_000.0},
            sz_decimals={"BTC": 1, "TON": 0},
        )
        state = _state(last_target=None)
        cfg = _cfg(safety_buffer=0.5)

        sp = plan(ta, snap, state, cfg, allow_short=True)

        assert sp.scaled_down is True  # we ARE on the post-scale path
        assert sp.target_sizes["TON"] == 0.0
        reasons = {(e["coin"], e["reason"]) for e in sp.skipped}
        assert ("TON", "zeroed_by_cap") in reasons, (
            "wanted short erased by cap was not classified zeroed_by_cap "
            "(F1 check compares the SIGNED notional against > 0)"
        )
        assert ("TON", "below_min_notional") not in reasons, (
            "zeroed short was misfiled (or double-filed) under below_min_notional"
        )
        ton_entries = [e for e in sp.skipped if e["coin"] == "TON"]
        assert len(ton_entries) == 1
        assert ton_entries[0]["delta_notional"] == -5_000.0  # signed intent kept

    def test_buy_to_close_short_side(self) -> None:
        """Held short + explicit target 0 → BUY-to-close delta.

        Construction note: ``plan()`` treats target-ABSENT and target-ZERO
        identically (step 7 unions held perps and reads
        ``target_sizes.get(perp, 0.0)``).  The explicit ``{"TON": 0.0}`` form
        is used here because it ALSO pins the zeroed_by_cap exemption: an
        intended notional of exactly 0.0 must not be misread as "wanted but
        erased" once the F1 check moves to ``abs()``.

        Held TON szi=−5.0, target 0, px=6_000 → delta = 0 − (−5) = +5.0,
        notional +30_000 ≥ $10 → one order, side "buy" (sign-aware delta:
        closing a short is a BUY).
        """
        ta = _target({"TON": 0.0}, cash=1.0)
        snap = _snap(
            equity=100_000,
            positions={"TON": -5.0},
            marks={"TON": 6_000.0},
            sz_decimals={"TON": 1},
        )
        state = _state(last_target=None)
        cfg = _cfg()

        sp = plan(ta, snap, state, cfg, allow_short=True)

        ton_orders = [o for o in sp.orders if o.coin == "TON"]
        assert len(ton_orders) == 1
        o = ton_orders[0]
        assert o.side == "buy"
        assert o.delta_size == 5.0
        assert o.target_size == 0.0
        assert abs(o.delta_notional - 30_000.0) < 1e-9
        assert "TON" not in sp.flatten  # the sleeve closes it via the order
        # Explicit-zero target is NOT a cap-erased coin — no F1 entry.
        assert ("TON", "zeroed_by_cap") not in {
            (e["coin"], e["reason"]) for e in sp.skipped
        }
