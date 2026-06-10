"""Task 10 — END-TO-END integration test for the bounded intra-cycle retry window.

This is the CAPSTONE of the bounded-retry plan.  Unlike ``test_daemon.py`` (which
monkeypatches ``run_cycle`` with a canned ``CycleReport``) this file drives the
REAL daemon ``run_once`` → real ``run_cycle`` → real ``sizer`` → real ``executor``
(including the pre-trade liquidity gate) → real ``reconciler`` over a real
``FakeHLClient`` whose fills are reflected into ``positions()``.  NOTHING about the
retry path is stubbed: the window opens, holds, fires, and clears (or expires)
through the genuine top-level code path the deploy uses.

The fixture
-----------
A 4-coin fixed target (BTC/ETH/SOL liquid, DOGE = the "thin" coin with an EMPTY
ask book).  At the daily tick the 3 liquid legs FILL and DOGE's buy DEFERS on the
liquidity gate (``liquidity_safety=1.0``) → the cycle is non-clean → a bounded
retry window OPENS.  The two scenarios then drive the window to (a) a clean fill
once DOGE's book deepens and (b) an expiry-driven best-effort commit when it never
does.

Anti-double-order mechanics
---------------------------
``FakeHLClient.submit_ioc`` reflects every filled order into the internal
``_positions`` map that ``positions()`` returns.  So on the retry tick the
reconciler snapshot sees the 3 originally-filled legs ALREADY positioned; the
sizer computes ``delta = target_size - current_position ≈ 0`` for them (below the
$10 MIN_NOTIONAL floor → skipped, no order), and only the (now-liquid) DOGE leg
produces a fresh submit.  We prove each originally-filled leg submits EXACTLY ONCE
by asserting (1) per-coin submit counts == 1 for BTC/ETH/SOL across the whole
sequence and (2) the total submit count == 4 (3 daily + 1 retry).
"""

from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path

from automation.allocation.base import TargetAllocation
from automation.core.config import CapsConfig, Config, RetryConfig, SignalSourceConfig
from automation.reporting.alerts import Alert, AlertType
from automation.reporting.audit import AuditLog
from automation.reporting.state import State
from automation.runtime.daemon import Daemon
from automation.runtime.intent_ledger import IntentLedger
from automation.tests._fakes import FakeHLClient

UTC = datetime.UTC

CLIENT_ID = "retry-itest"
# Liquid legs + the thin coin (DOGE starts with an EMPTY ask book → defers).
LIQUID = ("BTC", "ETH", "SOL")
THIN = "DOGE"
COINS = (*LIQUID, THIN)

# A fixed 4-coin target (sums to 0.80 → 0.20 cash; well under each cap).
TARGET: dict[str, float] = {"BTC": 0.25, "ETH": 0.25, "SOL": 0.20, "DOGE": 0.10}

# The daily bar the daemon trades when it fires at BAR+1day 00:15 UTC.
BAR = datetime.date(2026, 5, 17)
AS_OF = BAR.isoformat()  # "2026-05-17"

EQUITY = 50_000.0
MARKS: dict[str, float] = {"BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0, "DOGE": 0.15}
SZD: dict[str, int] = {"BTC": 5, "ETH": 4, "SOL": 2, "DOGE": 0}


# ---------------------------------------------------------------------------
# Recording alert sink (pattern from test_daemon.py).
# ---------------------------------------------------------------------------
class _CapturingAlertSink:
    def __init__(self) -> None:
        self.received: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.received.append(alert)


# ---------------------------------------------------------------------------
# A frozen fixed-target source (the daily fetch).  ``run_cycle`` serves THIS at
# the daily tick; the retry path builds its own ``_FrozenSource`` from the stashed
# ``retry_target`` (which equals these weights), so the target is identical across
# the daily fetch AND the retry.
# ---------------------------------------------------------------------------
class _FixedSource:
    def __init__(self, weights: dict[str, float], as_of: datetime.date) -> None:
        self._weights = dict(weights)
        self._as_of = as_of

    def get_target_allocation(
        self, as_of: datetime.date | None = None
    ) -> TargetAllocation:
        return TargetAllocation(
            as_of=self._as_of,
            weights=dict(self._weights),
            cash=1.0 - sum(self._weights.values()),
            strategy_id="retry-itest",
            model_rev="itest",
            audience=CLIENT_ID,
        )


def _deep_asks(coin: str) -> list[tuple[float, float]]:
    """A liquid ask book for *coin*: a huge resting size AT the mark (so it is
    always reachable within the slippage band, and ``depth >> required``)."""
    return [(MARKS[coin], 1_000_000.0)]


def _make_config(tmp_path: Path) -> Config:
    return Config(
        env="testnet",
        subaccount_address="0x" + "0" * 40,
        signal_source=SignalSourceConfig(type="equal_weight", client_id=CLIENT_ID),
        rebalance_utc_time="00:15",
        threshold_pct=3.0,
        max_per_asset=0.9,
        safety_buffer=0.03,
        max_signal_staleness_days=2,
        universe_perp_map={c: c for c in COINS},  # identity
        caps=CapsConfig(
            max_notional_per_coin=1e15,
            max_total_notional=1e15,
            max_turnover_per_cycle=1e15,
            max_order_slippage=0.05,
        ),
        # Liquidity gate ARMED — this is the whole point of the capstone.
        retry=RetryConfig(
            enabled=True,
            interval_seconds=600,
            window_seconds=10800,
            liquidity_safety=1.0,
        ),
        state_path=str(tmp_path / "state.json"),
        ledger_path=str(tmp_path / "ledger.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
        strategy_id="retry-itest",
    )


def _make_client(*, thin_liquid: bool) -> FakeHLClient:
    """A fresh flat-book FakeHLClient with all 3 liquid coins deep and DOGE
    either EMPTY (default) or deep (``thin_liquid=True``)."""
    l2: dict[str, dict[str, list[tuple[float, float]]]] = {
        c: {"bids": [], "asks": _deep_asks(c)} for c in LIQUID
    }
    l2[THIN] = {"bids": [], "asks": _deep_asks(THIN) if thin_liquid else []}
    return FakeHLClient(
        equity=EQUITY,
        positions={},  # fresh flat book → all legs are buys
        marks=MARKS,
        sz_decimals=SZD,
        l2=l2,
    )


def _build_daemon(
    cfg: Config, client: FakeHLClient, sink: _CapturingAlertSink
) -> Daemon:
    return Daemon(
        cfg=cfg,
        client=client,
        source=_FixedSource(TARGET, BAR),
        ledger=IntentLedger(cfg.ledger_path),
        state=State(),
        state_path=cfg.state_path,
        audit=AuditLog(cfg.audit_path),
        alert_sink=sink,
        strategy_id=cfg.strategy_id,
    )


def _tick(*, h: int, m: int) -> tuple[datetime.datetime, str]:
    """A daemon tick on BAR+1day at HH:MM UTC (so the scheduler's as_of == BAR)."""
    now = datetime.datetime(2026, 5, 18, h, m, tzinfo=UTC)
    return now, now.isoformat()


def _submit_counter(client: FakeHLClient) -> Counter[str]:
    return Counter(c["coin"] for c in client.submit_ioc_calls)


# ===========================================================================
# Scenario 1 — full window lifecycle: defer → hold → fire → clean → cleared.
# ===========================================================================
def test_full_retry_window_lifecycle_fills_on_liquidity(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    client = _make_client(thin_liquid=False)
    sink = _CapturingAlertSink()
    d = _build_daemon(cfg, client, sink)

    # --- 1) Daily tick @ 00:15: 3 liquid legs fill, DOGE defers → window opens.
    report = d.run_once(*_tick(h=0, m=15))
    assert report is not None
    assert report.rebalanced and not report.committed
    er = report.exec_report
    assert er is not None
    assert er.n_filled == 3  # BTC/ETH/SOL filled
    assert er.n_liquidity_deferred == 1  # DOGE deferred (no order)
    assert er.n_error == 0
    # Window OPENED: as_of frozen, target stashed (includes the thin coin).
    assert d.state.retry_as_of == AS_OF
    assert d.state.retry_target == TARGET
    assert THIN in (d.state.retry_target or {})
    assert d.state.retry_window_start is not None

    submits_after_daily = _submit_counter(client)
    assert submits_after_daily == Counter({"BTC": 1, "ETH": 1, "SOL": 1})  # DOGE: 0
    n_after_daily = len(client.submit_ioc_calls)
    assert n_after_daily == 3
    window_start_after_daily = d.state.retry_window_start

    # --- 2) Tick @ 00:20 (<interval 600s): no retry fires; window untouched.
    r2 = d.run_once(*_tick(h=0, m=20))
    assert r2 is None  # should_retry False (interval not elapsed)
    assert len(client.submit_ioc_calls) == n_after_daily  # no new submits
    assert d.state.retry_as_of == AS_OF  # window unchanged
    assert d.state.retry_window_start == window_start_after_daily

    # --- 3) DOGE becomes liquid (deepen its ask book).
    client.set_l2(THIN, asks=_deep_asks(THIN))

    # --- 4) Tick @ 00:27 (>interval, within window): retry fires, DOGE fills,
    #        cycle clean → window cleared, last_rebalanced_as_of set.
    r4 = d.run_once(*_tick(h=0, m=27))
    assert r4 is not None
    assert r4.committed  # clean retry attempt
    er4 = r4.exec_report
    assert er4 is not None
    assert er4.n_liquidity_deferred == 0
    assert er4.n_error == 0
    # Window CLEARED.
    assert d.state.retry_as_of is None
    assert d.state.retry_target is None
    assert d.state.retry_window_start is None
    assert d.state.last_rebalanced_as_of == AS_OF
    assert d.state.last_rebalanced_target == TARGET

    # --- 5) ANTI-DOUBLE-ORDER: each originally-filled leg submitted EXACTLY ONCE;
    #        the retry added EXACTLY one DOGE submit; total == 4 (3 daily + 1 retry).
    final = _submit_counter(client)
    for coin in LIQUID:
        assert final[coin] == 1, f"{coin} must submit exactly once, got {final[coin]}"
    assert final[THIN] == 1, f"{THIN} must submit exactly once (on the retry)"
    assert len(client.submit_ioc_calls) == 4
    # And the retry tick saw the 3 fills as live positions (delta≈0 → no re-order):
    pos = client.positions()
    for coin in COINS:
        assert float(pos.get(coin, 0)) > 0.0  # all 4 now positioned


# ===========================================================================
# Scenario 2 — window expires UNFILLED → best-effort commit, no re-trade.
# ===========================================================================
def test_retry_window_expires_unfilled_best_effort(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    client = _make_client(thin_liquid=False)  # DOGE stays EMPTY the whole window
    sink = _CapturingAlertSink()
    d = _build_daemon(cfg, client, sink)

    # --- 1) Daily tick: 3 fill, DOGE defers, window opens.
    report = d.run_once(*_tick(h=0, m=15))
    assert report is not None
    assert d.state.retry_as_of == AS_OF
    assert len(client.submit_ioc_calls) == 3
    assert _submit_counter(client) == Counter({"BTC": 1, "ETH": 1, "SOL": 1})

    # --- 2) Tick @ 03:20 (> window 3h): best-effort commit → window cleared,
    #        last_rebalanced_target == the served target, UNFILLED alert emitted,
    #        and the 3 filled legs are NEVER re-traded (submit count stays 3).
    r2 = d.run_once(*_tick(h=3, m=20))
    assert r2 is None  # best-effort path returns None (no attempt fired)
    # Window CLEARED via best-effort commit.
    assert d.state.retry_as_of is None
    assert d.state.retry_target is None
    assert d.state.retry_window_start is None
    # Frozen target committed to baseline (anti baseline-poisoning).
    assert d.state.last_rebalanced_target == TARGET
    assert d.state.last_rebalanced_as_of == AS_OF
    # Best-effort does NOT submit — the 3 filled legs are never re-traded.
    assert len(client.submit_ioc_calls) == 3
    assert _submit_counter(client) == Counter({"BTC": 1, "ETH": 1, "SOL": 1})
    # An UNFILLED WARNING surfaced the residual.
    unfilled = [a for a in sink.received if a.type is AlertType.UNFILLED]
    assert len(unfilled) == 1
    assert unfilled[0].context.get("as_of") == AS_OF


# ===========================================================================
# Guard: the fixture genuinely exercises the liquidity gate (not a no-op).
# ===========================================================================
def test_fixture_thin_leg_actually_defers(tmp_path: Path) -> None:
    """Sanity: with DOGE's book empty the daily cycle must DEFER exactly the thin
    leg (proving the gate fired) and OPEN a window — guarding against a silent
    fixture regression where the gate stopped engaging."""
    cfg = _make_config(tmp_path)
    client = _make_client(thin_liquid=False)
    d = _build_daemon(cfg, client, _CapturingAlertSink())
    report = d.run_once(*_tick(h=0, m=15))
    assert report is not None
    er = report.exec_report
    assert er is not None
    deferred = [r for r in er.results if r.liquidity_deferred]
    assert [r.coin for r in deferred] == [THIN]
    assert not report.committed
    assert d.state.retry_as_of == AS_OF
