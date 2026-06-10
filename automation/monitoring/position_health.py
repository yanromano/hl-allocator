"""automation.monitoring.position_health — pure health checks + read-only adapter.

Design
------
The heart of this module is a set of **pure** functions over an immutable
:class:`HealthSnapshot`.  Each ``check_*`` takes the snapshot plus keyword-only
thresholds and returns a ``list[HealthFinding]`` — no IO, no clock, no client.
``assess`` runs all four checks and concatenates the findings.

All venue/network reads live in exactly one place: :func:`gather_health_snapshot`,
the thin adapter that assembles a ``HealthSnapshot`` from an HLClient.

assess enabled-gate decision
----------------------------
``assess`` does **not** gate on ``cfg.health.enabled``.  It always runs the
checks when called, staying pure over ``(snap, cfg.health thresholds)``.  The
CALLER (the daemon / standalone monitor) is responsible for deciding whether to
invoke ``assess`` at all based on ``cfg.health.enabled``.  This keeps the check
layer free of policy and trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from automation.reporting.alerts import AlertType, Severity

# ---------------------------------------------------------------------------
# Immutable snapshot + finding dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionHealth:
    """A single perp position's health-relevant fields (mark joined in)."""

    coin: str
    position_value: float
    unrealized_pnl: float
    liquidation_px: float | None
    mark: float
    funding_since_open: float
    leverage_type: str | None
    leverage_value: int


@dataclass(frozen=True)
class HealthSnapshot:
    """An immutable, IO-free snapshot of account + venue state for health checks."""

    equity: float
    maint_margin: float
    positions: list[PositionHealth]
    spot_usdc: float
    #: ``{coin: {"delisted": bool, "only_isolated": bool, "max_leverage": int}}``
    venue_coins: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class HealthFinding:
    """A single health concern raised by a check."""

    alert_type: AlertType
    severity: Severity
    message: str
    context: dict[str, Any]


# ---------------------------------------------------------------------------
# Pure checks
# ---------------------------------------------------------------------------


def check_funding(
    snap: HealthSnapshot, *, drag_pct_warn: float
) -> list[HealthFinding]:
    """FUNDING_DRAG when Σ funding-since-open exceeds ``drag_pct_warn`` % of equity.

    Guards on non-positive equity (returns ``[]``).  ``per_coin`` is sorted
    descending by funding so the worst offenders / spot-routing candidates are
    obvious at a glance.
    """
    if snap.equity <= 0:
        return []
    total = sum(p.funding_since_open for p in snap.positions)
    pct = total / snap.equity * 100
    if pct <= drag_pct_warn:
        return []
    per_coin = {
        p.coin: p.funding_since_open
        for p in sorted(
            snap.positions, key=lambda p: p.funding_since_open, reverse=True
        )
    }
    return [
        HealthFinding(
            alert_type=AlertType.FUNDING_DRAG,
            severity=Severity.WARNING,
            message=(
                f"Funding drag {pct:.2f}% of equity exceeds {drag_pct_warn:.2f}%"
            ),
            context={
                "total_funding": total,
                "pct_of_equity": pct,
                "per_coin": per_coin,
            },
        )
    ]


def check_margin(
    snap: HealthSnapshot,
    *,
    maint_frac_warn: float,
    liq_distance_pct_warn: float,
) -> list[HealthFinding]:
    """MARGIN_RATIO_LOW from (a) account maint-margin fraction and (b) per-position
    liquidation-price proximity.

    A liquidation distance below ``liq_distance_pct_warn / 2`` is CRITICAL,
    otherwise WARNING.  Positions with ``liquidation_px is None`` or a
    non-positive mark are skipped.
    """
    findings: list[HealthFinding] = []

    if snap.equity > 0:
        maint_frac = snap.maint_margin / snap.equity
        if maint_frac > maint_frac_warn:
            findings.append(
                HealthFinding(
                    alert_type=AlertType.MARGIN_RATIO_LOW,
                    severity=Severity.WARNING,
                    message=(
                        f"Maintenance margin fraction {maint_frac:.3f} exceeds "
                        f"{maint_frac_warn:.3f}"
                    ),
                    context={
                        "maint_margin": snap.maint_margin,
                        "equity": snap.equity,
                        "maint_frac": maint_frac,
                    },
                )
            )

    for p in snap.positions:
        if p.liquidation_px is None or p.mark <= 0:
            continue
        dist_pct = abs(p.mark - p.liquidation_px) / p.mark * 100
        if dist_pct < liq_distance_pct_warn:
            severity = (
                Severity.CRITICAL
                if dist_pct < liq_distance_pct_warn / 2
                else Severity.WARNING
            )
            findings.append(
                HealthFinding(
                    alert_type=AlertType.MARGIN_RATIO_LOW,
                    severity=severity,
                    message=(
                        f"{p.coin} liquidation {dist_pct:.2f}% from mark "
                        f"(< {liq_distance_pct_warn:.2f}%)"
                    ),
                    context={
                        "coin": p.coin,
                        "mark": p.mark,
                        "liquidation_px": p.liquidation_px,
                        "distance_pct": dist_pct,
                    },
                )
            )

    return findings


def check_venue(snap: HealthSnapshot) -> list[HealthFinding]:
    """COIN_VENUE_DEGRADED for any HELD coin that is absent from the venue or
    flagged delisted.

    Reduce-only is NOT exposed by this HL build, so venue degradation reduces to
    ``absent from venue_meta`` OR ``delisted``.
    """
    findings: list[HealthFinding] = []
    for p in snap.positions:
        coin = p.coin
        if coin not in snap.venue_coins:
            findings.append(
                HealthFinding(
                    alert_type=AlertType.COIN_VENUE_DEGRADED,
                    severity=Severity.WARNING,
                    message=f"{coin} absent from venue meta",
                    context={"coin": coin, "reason": "absent_from_venue"},
                )
            )
        elif snap.venue_coins[coin].get("delisted"):
            findings.append(
                HealthFinding(
                    alert_type=AlertType.COIN_VENUE_DEGRADED,
                    severity=Severity.WARNING,
                    message=f"{coin} is delisted on the venue",
                    context={"coin": coin, "reason": "delisted"},
                )
            )
    return findings


def check_collateral(
    snap: HealthSnapshot, *, spot_usdc_warn: float
) -> list[HealthFinding]:
    """COLLATERAL_IN_SPOT when free spot USDC exceeds ``spot_usdc_warn``.

    Collateral should live in the perp account, not idle as spot USDC.
    """
    if snap.spot_usdc <= spot_usdc_warn:
        return []
    return [
        HealthFinding(
            alert_type=AlertType.COLLATERAL_IN_SPOT,
            severity=Severity.WARNING,
            message=(
                f"Free spot USDC {snap.spot_usdc:.2f} exceeds "
                f"{spot_usdc_warn:.2f} (collateral should be in perp)"
            ),
            context={"spot_usdc": snap.spot_usdc, "threshold": spot_usdc_warn},
        )
    ]


class _HealthThresholds(Protocol):
    # Read-only (property) form → covariant structural matching, so a concrete
    # ``HealthConfig`` (whose plain attributes are invariant) still satisfies the
    # protocol when passed where ``_HasHealth`` is expected (e.g. the daemon /
    # standalone monitor passing a real ``Config``).
    @property
    def funding_drag_pct(self) -> float: ...
    @property
    def margin_maint_frac_warn(self) -> float: ...
    @property
    def liq_distance_pct_warn(self) -> float: ...
    @property
    def spot_collateral_warn_usd(self) -> float: ...


class _HasHealth(Protocol):
    @property
    def health(self) -> _HealthThresholds: ...


def assess(snap: HealthSnapshot, cfg: _HasHealth) -> list[HealthFinding]:
    """Run all four checks with thresholds from ``cfg.health`` and concatenate.

    Pure over ``(snap, cfg.health thresholds)``.  Does NOT gate on
    ``cfg.health.enabled`` — the caller decides whether to invoke ``assess``.
    """
    h = cfg.health
    findings: list[HealthFinding] = []
    findings += check_funding(snap, drag_pct_warn=h.funding_drag_pct)
    findings += check_margin(
        snap,
        maint_frac_warn=h.margin_maint_frac_warn,
        liq_distance_pct_warn=h.liq_distance_pct_warn,
    )
    findings += check_venue(snap)
    findings += check_collateral(snap, spot_usdc_warn=h.spot_collateral_warn_usd)
    return findings


# ---------------------------------------------------------------------------
# Adapter (the ONLY IO)
# ---------------------------------------------------------------------------


class _HealthClient(Protocol):
    """The read-only HLClient surface the adapter needs (all read-only)."""

    def account_snapshot(self) -> dict[str, Any]: ...
    def marks(self) -> dict[str, Decimal]: ...
    def spot_balances(self) -> dict[str, Decimal]: ...
    def venue_meta(self) -> dict[str, dict[str, Any]]: ...


def gather_health_snapshot(client: _HealthClient) -> HealthSnapshot:
    """Read the venue (read-only) and assemble a :class:`HealthSnapshot`.

    Reads: ``client.account_snapshot()``, ``client.marks()``,
    ``client.spot_balances()``, ``client.venue_meta()``.  Coerces ``Decimal`` →
    ``float`` where needed.  This is the single IO boundary; everything
    downstream is pure.
    """
    acct = client.account_snapshot()
    marks = client.marks()
    spot = client.spot_balances()
    venue = client.venue_meta()

    positions: list[PositionHealth] = []
    for coin, raw in acct.get("positions", {}).items():
        liq = raw.get("liquidation_px")
        positions.append(
            PositionHealth(
                coin=coin,
                position_value=float(raw["position_value"]),
                unrealized_pnl=float(raw["unrealized_pnl"]),
                liquidation_px=None if liq is None else float(liq),
                mark=float(marks.get(coin, 0)),
                funding_since_open=float(raw["funding_since_open"]),
                leverage_type=raw.get("leverage_type"),
                leverage_value=int(raw["leverage_value"]),
            )
        )

    return HealthSnapshot(
        equity=float(acct["equity"]),
        maint_margin=float(acct["maint_margin"]),
        positions=positions,
        spot_usdc=float(spot.get("USDC", 0)),
        venue_coins=venue,
    )
