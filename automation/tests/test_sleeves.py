"""
Tests for automation.allocation.sleeves — the PURE sleeve combination layer.

This module is IO-free and clock-free: it transforms a list of per-sleeve
outcomes into a single combined signed target plus the set of frozen (HOLD)
coins, and it resolves a sleeve's status from a fetch result + staleness policy.

Coverage (spec §4.3):
- combine_sleeves:
  * FRESH + FRESH netting (overlap coin nets signed; budget-scaled)
  * FRESH + FRESH disjoint universes (union, each budget-scaled)
  * FRESH + HOLD (hold coins → frozen_coins, EXCLUDED from combined weights)
  * HOLD coin overlapping a FRESH coin → ValueError frozen-coin conflict
  * FRESH + STALE (stale contributes {} and freezes nothing)
  * never-served sleeve (HOLD, empty weights) → contributes/freezes nothing
  * gross invariant: Σ|combined| ≤ Σ budgets of FRESH sleeves
- resolve_outcome:
  * served_ok → fresh, holds reset to 0
  * failed within window, long → hold, holds += 1
  * failed within window, short, 2nd hold → stale (fast de-risk)
  * failed within window, short, 1st hold → hold (not yet 2)
  * failed past window → stale
"""

from __future__ import annotations

import math

import pytest

from automation.allocation.sleeves import (
    CombinedTarget,
    SleeveOutcome,
    combine_sleeves,
    resolve_outcome,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh(
    name: str,
    weights: dict[str, float],
    budget: float,
    allow_short: bool = False,
) -> SleeveOutcome:
    return SleeveOutcome(
        name=name,
        status="fresh",
        weights=weights,
        budget=budget,
        allow_short=allow_short,
    )


def _hold(
    name: str,
    frozen_weights: dict[str, float],
    budget: float,
    allow_short: bool = False,
) -> SleeveOutcome:
    # A HOLD sleeve carries its frozen coin set via its weights dict KEYS
    # (the daemon passes last_weights so combine knows which coins to freeze).
    return SleeveOutcome(
        name=name,
        status="hold",
        weights=frozen_weights,
        budget=budget,
        allow_short=allow_short,
    )


def _stale(name: str, budget: float, allow_short: bool = False) -> SleeveOutcome:
    return SleeveOutcome(
        name=name,
        status="stale",
        weights={},
        budget=budget,
        allow_short=allow_short,
    )


# ---------------------------------------------------------------------------
# combine_sleeves
# ---------------------------------------------------------------------------


class TestCombineFreshFresh:
    def test_overlap_coin_nets_signed_budget_scaled(self) -> None:
        """Two fresh sleeves overlapping on BTC net signed after budget scaling.

        0.5·{BTC:+0.4} ⊕ 0.5·{BTC:-0.3} → {BTC: 0.5*0.4 + 0.5*-0.3 = +0.05}.
        """
        out = combine_sleeves(
            [
                _fresh("a", {"BTC": 0.4}, budget=0.5),
                _fresh("b", {"BTC": -0.3}, budget=0.5, allow_short=True),
            ]
        )
        assert isinstance(out, CombinedTarget)
        assert out.frozen_coins == frozenset()
        assert out.weights.keys() == {"BTC"}
        assert math.isclose(out.weights["BTC"], 0.05, abs_tol=1e-12)

    def test_disjoint_universes_union_budget_scaled(self) -> None:
        """Disjoint sleeves (HAARP majors + CRASH smallcaps) → union, budget-scaled."""
        out = combine_sleeves(
            [
                _fresh("haarp", {"BTC": 0.6, "ETH": 0.4}, budget=0.7),
                _fresh("crash", {"TON": -0.5, "HYPE": -0.5}, budget=0.3, allow_short=True),
            ]
        )
        assert out.frozen_coins == frozenset()
        assert out.weights.keys() == {"BTC", "ETH", "TON", "HYPE"}
        assert math.isclose(out.weights["BTC"], 0.7 * 0.6, abs_tol=1e-12)
        assert math.isclose(out.weights["ETH"], 0.7 * 0.4, abs_tol=1e-12)
        assert math.isclose(out.weights["TON"], 0.3 * -0.5, abs_tol=1e-12)
        assert math.isclose(out.weights["HYPE"], 0.3 * -0.5, abs_tol=1e-12)

    def test_exact_cancellation_drops_to_zero(self) -> None:
        """When two fresh sleeves net to exactly zero on a coin, it stays present at 0.0."""
        out = combine_sleeves(
            [
                _fresh("a", {"BTC": 0.5}, budget=0.5),
                _fresh("b", {"BTC": -0.5}, budget=0.5, allow_short=True),
            ]
        )
        assert math.isclose(out.weights["BTC"], 0.0, abs_tol=1e-12)


class TestCombineHold:
    def test_hold_coins_frozen_not_in_weights(self) -> None:
        """A HOLD sleeve's coins go to frozen_coins and do NOT appear in weights.

        Freeze-NOTIONAL: the daemon excludes frozen coins from the delta so the
        live positions carry untouched. The combined weights dict must NOT carry
        any contribution for the frozen sleeve (spec §4.3, red-team H3).
        """
        out = combine_sleeves(
            [
                _fresh("a", {"BTC": 0.6, "ETH": 0.4}, budget=0.6),
                _hold("crash", {"TON": -0.3}, budget=0.4, allow_short=True),
            ]
        )
        assert out.frozen_coins == frozenset({"TON"})
        assert "TON" not in out.weights
        assert out.weights.keys() == {"BTC", "ETH"}
        assert math.isclose(out.weights["BTC"], 0.6 * 0.6, abs_tol=1e-12)
        assert math.isclose(out.weights["ETH"], 0.6 * 0.4, abs_tol=1e-12)

    def test_hold_multi_coin_all_frozen(self) -> None:
        """All coin keys of a HOLD sleeve enter frozen_coins."""
        out = combine_sleeves(
            [
                _fresh("a", {"BTC": 1.0}, budget=0.5),
                _hold("crash", {"TON": -0.3, "HYPE": -0.2, "WIF": -0.1}, budget=0.5,
                      allow_short=True),
            ]
        )
        assert out.frozen_coins == frozenset({"TON", "HYPE", "WIF"})
        assert out.weights.keys() == {"BTC"}

    def test_frozen_coin_conflict_raises(self) -> None:
        """A coin both frozen (HOLD A) and traded (FRESH B) → ValueError.

        Cannot both freeze and trade one coin — the combination refuses
        (fail-closed). HAARP×CRASH universes are disjoint in prod, but the
        math must not silently pick one (spec §4.3, red-team).
        """
        with pytest.raises(ValueError, match="frozen-coin conflict: TON"):
            combine_sleeves(
                [
                    _hold("crash", {"TON": -0.3}, budget=0.4, allow_short=True),
                    _fresh("alt", {"TON": 0.5}, budget=0.6),
                ]
            )

    def test_frozen_coin_conflict_order_independent(self) -> None:
        """The conflict fires regardless of sleeve ordering (fresh-then-hold)."""
        with pytest.raises(ValueError, match="frozen-coin conflict"):
            combine_sleeves(
                [
                    _fresh("alt", {"TON": 0.5}, budget=0.6),
                    _hold("crash", {"TON": -0.3}, budget=0.4, allow_short=True),
                ]
            )


class TestCombineStale:
    def test_stale_contributes_nothing(self) -> None:
        """A STALE sleeve contributes {} and freezes nothing → only fresh remains."""
        out = combine_sleeves(
            [
                _fresh("a", {"BTC": 0.6, "ETH": 0.4}, budget=0.6),
                _stale("crash", budget=0.4, allow_short=True),
            ]
        )
        assert out.frozen_coins == frozenset()
        assert out.weights.keys() == {"BTC", "ETH"}
        # Stale sleeve's prior coins simply absent → daemon closes them via delta.

    def test_never_served_hold_empty_weights(self) -> None:
        """A never-served sleeve (HOLD with empty weights) contributes/freezes nothing."""
        out = combine_sleeves(
            [
                _fresh("a", {"BTC": 1.0}, budget=0.5),
                _hold("crash", {}, budget=0.5, allow_short=True),
            ]
        )
        assert out.frozen_coins == frozenset()
        assert out.weights.keys() == {"BTC"}


class TestCombineEdges:
    def test_empty_outcomes(self) -> None:
        """No sleeves → empty combined target."""
        out = combine_sleeves([])
        assert out.weights == {}
        assert out.frozen_coins == frozenset()

    def test_single_fresh_sleeve(self) -> None:
        """A single fresh sleeve is just its budget-scaled weights."""
        out = combine_sleeves([_fresh("a", {"BTC": 0.6, "ETH": 0.4}, budget=1.0)])
        assert math.isclose(out.weights["BTC"], 0.6, abs_tol=1e-12)
        assert math.isclose(out.weights["ETH"], 0.4, abs_tol=1e-12)

    def test_all_hold_or_stale_empty_combined(self) -> None:
        """All sleeves HOLD/STALE → empty combined weights (global hold path upstream)."""
        out = combine_sleeves(
            [
                _hold("a", {"BTC": 0.5}, budget=0.5),
                _stale("crash", budget=0.5, allow_short=True),
            ]
        )
        assert out.frozen_coins == frozenset({"BTC"})
        assert out.weights == {}


class TestGrossInvariant:
    @pytest.mark.parametrize(
        "outcomes",
        [
            [
                _fresh("a", {"BTC": 0.4, "ETH": 0.6}, budget=0.5),
                _fresh("b", {"TON": -0.7, "HYPE": -0.3}, budget=0.5, allow_short=True),
            ],
            [
                _fresh("a", {"BTC": 1.0}, budget=0.3),
                _fresh("b", {"BTC": -1.0}, budget=0.3, allow_short=True),
                _stale("c", budget=0.4, allow_short=True),
            ],
            [
                _fresh("a", {"BTC": 0.5, "ETH": 0.5}, budget=0.7),
                _hold("b", {"TON": -0.4}, budget=0.3, allow_short=True),
            ],
            [
                _fresh("a", {"BTC": 0.2, "ETH": 0.3, "SOL": 0.5}, budget=0.6),
                _fresh("b", {"DOGE": -0.5, "WIF": -0.5}, budget=0.4, allow_short=True),
            ],
        ],
    )
    def test_gross_le_fresh_budgets(self, outcomes: list[SleeveOutcome]) -> None:
        """Σ|combined| ≤ Σ budgets of FRESH sleeves (netting can only reduce gross)."""
        out = combine_sleeves(outcomes)
        fresh_budget_sum = sum(o.budget for o in outcomes if o.status == "fresh")
        gross = sum(abs(w) for w in out.weights.values())
        assert gross <= fresh_budget_sum + 1e-12


# ---------------------------------------------------------------------------
# resolve_outcome
# ---------------------------------------------------------------------------


class TestResolveOutcome:
    def test_served_ok_fresh_resets_holds(self) -> None:
        """A successful serve → fresh status and consecutive_holds resets to 0."""
        outcome, holds = resolve_outcome(
            served_ok=True,
            weights={"BTC": 0.6, "ETH": 0.4},
            allow_short=False,
            budget=0.7,
            name="haarp",
            consecutive_holds=3,
            within_staleness_window=True,
        )
        assert outcome.status == "fresh"
        assert outcome.weights == {"BTC": 0.6, "ETH": 0.4}
        assert outcome.budget == 0.7
        assert outcome.name == "haarp"
        assert holds == 0

    def test_failed_within_window_long_holds(self) -> None:
        """A long sleeve failing within the window → hold, consecutive_holds += 1."""
        outcome, holds = resolve_outcome(
            served_ok=False,
            weights={"BTC": 0.6},
            allow_short=False,
            budget=0.7,
            name="haarp",
            consecutive_holds=4,
            within_staleness_window=True,
        )
        assert outcome.status == "hold"
        assert outcome.weights == {}
        assert holds == 5

    def test_failed_within_window_short_first_hold(self) -> None:
        """A short sleeve's FIRST failure (holds 0→1) → hold (not yet 2nd consecutive)."""
        outcome, holds = resolve_outcome(
            served_ok=False,
            weights={"TON": -0.3},
            allow_short=True,
            budget=0.3,
            name="crash",
            consecutive_holds=0,
            within_staleness_window=True,
        )
        assert outcome.status == "hold"
        assert outcome.weights == {}
        assert holds == 1

    def test_failed_within_window_short_second_hold_derisks(self) -> None:
        """A short sleeve's SECOND consecutive failure (holds 1→2) → stale (fast de-risk).

        Spec §4.3 (red-team H2): holding a dark short for 2 days is the wrong
        fail-safe, so after the second consecutive daily HOLD a short sleeve
        transitions to STALE (de-risk) rather than re-freezing.
        """
        outcome, holds = resolve_outcome(
            served_ok=False,
            weights={"TON": -0.3},
            allow_short=True,
            budget=0.3,
            name="crash",
            consecutive_holds=1,
            within_staleness_window=True,
        )
        assert outcome.status == "stale"
        assert outcome.weights == {}
        assert holds == 2

    def test_failed_past_window_stale(self) -> None:
        """Failing PAST the per-sleeve staleness window → stale regardless of side."""
        outcome, holds = resolve_outcome(
            served_ok=False,
            weights={"BTC": 0.6},
            allow_short=False,
            budget=0.7,
            name="haarp",
            consecutive_holds=0,
            within_staleness_window=False,
        )
        assert outcome.status == "stale"
        assert outcome.weights == {}

    def test_resolve_outcome_hold_returns_empty_weights(self) -> None:
        """resolve_outcome always returns empty weights — even for HOLD.

        Per the plan's policy, resolve_outcome returns ("hold", {}). The HOLD
        sleeve's frozen coin set is NOT carried by resolve_outcome; the daemon
        (C5) rebuilds the HOLD SleeveOutcome with last_weights from the ledger
        before passing it to combine_sleeves, which is what supplies the frozen
        coin keys. This keeps resolve_outcome pure of any ledger lookup.
        """
        outcome, holds = resolve_outcome(
            served_ok=False,
            weights={"TON": -0.3, "HYPE": -0.2},
            allow_short=True,
            budget=0.3,
            name="crash",
            consecutive_holds=0,
            within_staleness_window=True,
        )
        assert outcome.status == "hold"
        assert outcome.weights == {}
        assert holds == 1

    def test_served_ok_overrides_past_window(self) -> None:
        """A successful serve is fresh even if the staleness window had elapsed."""
        outcome, holds = resolve_outcome(
            served_ok=True,
            weights={"BTC": 1.0},
            allow_short=False,
            budget=1.0,
            name="haarp",
            consecutive_holds=9,
            within_staleness_window=False,
        )
        assert outcome.status == "fresh"
        assert holds == 0

    def test_served_ok_preserves_allow_short_flag(self) -> None:
        """The resolved outcome carries the sleeve's allow_short flag through."""
        outcome, _ = resolve_outcome(
            served_ok=True,
            weights={"TON": -0.4},
            allow_short=True,
            budget=0.3,
            name="crash",
            consecutive_holds=0,
            within_staleness_window=True,
        )
        assert outcome.allow_short is True
