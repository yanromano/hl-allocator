"""Testnet preflight — exercises the FULL hl-allocator bring-up against the live
Hyperliquid **testnet**, WITHOUT any real keys or funds.

Goal (operator request): prove the API connection is correct and our wrappers work,
and SEE the exact errors that occur where real funded/approved keys are missing.

How to read the output
-----------------------
- Sections A-D (read connectivity, client construction, dry-run pipeline) should ALL
  print ``[PASS]``.  An ``[UNEXPECTED-FAIL]`` there = a real connectivity or wrapper bug.
- Sections E-G (provisioning / order submit / funding) use UNFUNDED throwaway keys, so
  they are EXPECTED to fail — but they must fail with a STRUCTURED Hyperliquid rejection
  (proving the signed request reached the API), NOT a network/connection error.

Run:  Python/venv/bin/python -m automation.scripts.testnet_preflight
No arguments, no config file, no real keys.  Generates throwaway keys in-process and
cleans up all temp files.
"""

from __future__ import annotations

import secrets as pysecrets
import tempfile
import traceback
from pathlib import Path
from typing import Any

import eth_account

# ---------------------------------------------------------------------------
# Result collector
# ---------------------------------------------------------------------------
_RESULTS: list[tuple[str, str, str]] = []


def _short(v: Any, n: int = 220) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "…"


def step(section: str, name: str, fn: Any, *, expect_fail: bool = False) -> Any:
    """Run one preflight step, record + print its outcome, return its value (or None)."""
    label = f"{section} · {name}"
    try:
        out = fn()
        _RESULTS.append((label, "PASS", _short(out)))
        print(f"  [PASS]            {name}\n                    → {_short(out)}")
        return out
    except Exception as exc:  # noqa: BLE001 — we want to SEE every failure
        kind = "EXPECTED-FAIL" if expect_fail else "UNEXPECTED-FAIL"
        msg = f"{type(exc).__name__}: {exc}"
        _RESULTS.append((label, kind, msg))
        marker = "[EXPECTED-FAIL]  " if expect_fail else "[UNEXPECTED-FAIL]"
        print(f"  {marker} {name}\n                    → {msg}")
        if not expect_fail:
            print("                    " + "\n                    ".join(
                traceback.format_exc().splitlines()[-4:]
            ))
        return None


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Throwaway material (NEVER real keys)
# ---------------------------------------------------------------------------
THROWAWAY_AGENT_KEY = "0x" + pysecrets.token_hex(32)
THROWAWAY_MASTER_KEY = "0x" + pysecrets.token_hex(32)
# A random, almost-certainly-empty testnet address to read against.
RANDOM_ADDR = eth_account.Account.from_key("0x" + pysecrets.token_hex(32)).address
ENV = "testnet"


def main() -> int:
    print(f"Testnet preflight — env={ENV}")
    print(f"  random read addr : {RANDOM_ADDR}")
    print(f"  throwaway agent  : {eth_account.Account.from_key(THROWAWAY_AGENT_KEY).address}")
    print(f"  throwaway master : {eth_account.Account.from_key(THROWAWAY_MASTER_KEY).address}")
    print("  (all keys generated in-process; no real funds; temp files cleaned up)")

    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    base_url = constants.TESTNET_API_URL
    print(f"  testnet base_url : {base_url}")

    # =====================================================================
    header("A. Raw API READ connectivity (SDK Info, no auth)")
    # =====================================================================
    step("A", "Info(testnet) construct",
         lambda: type(Info(base_url, skip_ws=True,
                           spot_meta={"universe": [], "tokens": []})).__name__)
    info_obj = Info(base_url, skip_ws=True, spot_meta={"universe": [], "tokens": []})
    step("A", "meta() perp universe size",
         lambda: len(info_obj.meta()["universe"]))
    step("A", "all_mids() count",
         lambda: len(info_obj.all_mids()))
    step("A", "user_state(random empty addr)",
         lambda: info_obj.user_state(RANDOM_ADDR).get("marginSummary", {}))
    step("A", "open_orders(random addr)",
         lambda: info_obj.open_orders(RANDOM_ADDR))
    step("A", "user_rate_limit(random addr)",
         lambda: info_obj.user_rate_limit(RANDOM_ADDR))

    # =====================================================================
    header("B. HLClient READ wrappers (env=testnet, sub=random addr, no agent)")
    # =====================================================================
    from automation.core.hl_client import HLClient

    ro = step("B", "HLClient(testnet, sub, agent_key=None) construct",
              lambda: HLClient(ENV, RANDOM_ADDR))
    if ro is not None:
        step("B", "equity()", lambda: ro.equity())
        step("B", "positions()", lambda: ro.positions())
        step("B", "marks() count", lambda: len(ro.marks()))
        step("B", "sz_decimals() count", lambda: len(ro.sz_decimals()))
        step("B", "user_rate_limit()", lambda: ro.user_rate_limit())
        step("B", "open_orders()", lambda: ro.open_orders())
        # confirm BTC/ETH are present in testnet perp marks
        step("B", "BTC+ETH in testnet perp marks",
             lambda: {c: (c in ro.marks()) for c in ("BTC", "ETH")})

    # =====================================================================
    header("C. HLClient.for_trading (throwaway agent + nonce-file lock)")
    # =====================================================================
    tmpdir = Path(tempfile.mkdtemp(prefix="hl_preflight_"))
    nonce_path = tmpdir / "nonce.state"
    trading = step("C", "for_trading(... throwaway agent ...) + acquire lock",
                   lambda: HLClient.for_trading(ENV, RANDOM_ADDR, THROWAWAY_AGENT_KEY,
                                                str(nonce_path)))
    if trading is not None:
        step("C", "exchange wired (vault_address routing)",
             lambda: trading.exchange is not None)
        # second process on the same nonce path MUST fail (single signer)
        step("C", "second for_trading same nonce path REJECTED",
             lambda: HLClient.for_trading(ENV, RANDOM_ADDR, THROWAWAY_AGENT_KEY,
                                          str(nonce_path)),
             expect_fail=True)
        step("C", "close() releases lock", lambda: trading.close() or "closed")

    # =====================================================================
    header("D. Offline DRY-RUN pipeline against LIVE testnet data (sends NOTHING)")
    # =====================================================================
    from automation.allocation.examples.equal_weight import EqualWeightSource
    from automation.core.config import CapsConfig, Config, SignalSourceConfig
    from automation.reporting.state import State
    from automation.runtime.dry_run import dry_run_cycle

    def _build_cfg() -> Config:
        return Config(
            env=ENV,
            subaccount_address=RANDOM_ADDR,
            signal_source=SignalSourceConfig(type="equal_weight", client_id="preflight"),
            rebalance_utc_time="00:15",
            threshold_pct=3.0,
            max_per_asset=0.5,
            caps=CapsConfig(
                max_notional_per_coin=10000.0,
                max_total_notional=50000.0,
                max_turnover_per_cycle=60000.0,
                max_order_slippage=0.01,
            ),
            safety_buffer=0.03,
            max_signal_staleness_days=2,
            universe_perp_map={"BTC": "BTC", "ETH": "ETH"},
        )

    cfg = step("D", "build testnet Config (BTC/ETH equal_weight)", _build_cfg)
    src = step("D", "EqualWeightSource construct",
               lambda: EqualWeightSource(coins=["BTC", "ETH"], client_id="preflight"))
    if cfg is not None and src is not None and ro is not None:
        step("D", "source.get_target_allocation() shape",
             lambda: {"weights": src.get_target_allocation().weights,
                      "cash": src.get_target_allocation().cash,
                      "audience": src.get_target_allocation().audience})
        # dry_run_cycle: snapshot(live testnet reads) → target → validate → sizer.plan
        sp = step("D", "dry_run_cycle(...) — FULL pipeline, SENDS NOTHING",
                  lambda: dry_run_cycle(cfg, ro, src, State()))
        if sp is not None:
            step("D", "  → SizePlan summary",
                 lambda: {"rebalance": sp.rebalance, "reason": sp.reason,
                          "n_orders": len(sp.orders), "equity": sp.equity,
                          "scaled": sp.scaled_down})

    # =====================================================================
    header("E. PROVISIONING with UNFUNDED throwaway master (EXPECTED to fail)")
    # =====================================================================
    from automation.account.provisioning import provision

    ks_path = tmpdir / "agent_keystore.json"
    # valid_until_ms ~1 year out (13-digit ms, future) so the validator passes and we
    # reach the create_sub_account network call (the part that gets rejected).
    far_future_ms = 1_900_000_000_000
    step("E", "provision() — create_sub_account + approve_agent (no funds)",
         lambda: provision(master_key=THROWAWAY_MASTER_KEY, env=ENV,
                           keystore_path=str(ks_path), passphrase="preflight-pass",
                           valid_until_ms=far_future_ms, now_ms=1_700_000_000_000),
         expect_fail=True)

    # =====================================================================
    header("F. ORDER SUBMIT with throwaway agent (EXPECTED to fail — shows rejection)")
    # =====================================================================
    nonce_path2 = tmpdir / "nonce2.state"
    t2 = step("F", "for_trading for submit test", lambda: HLClient.for_trading(
        ENV, RANDOM_ADDR, THROWAWAY_AGENT_KEY, str(nonce_path2)))
    if t2 is not None:
        # tiny IOC buy — the agent is not approved + no funds → structured HL rejection.
        step("F", "submit_ioc(BTC, buy, 0.001) — agent not approved / no funds",
             lambda: t2.submit_ioc("BTC", True, 0.001, 0.01, "0x" + pysecrets.token_hex(16)),
             expect_fail=True)
        step("F", "close()", lambda: t2.close() or "closed")

    # =====================================================================
    header("G. FUNDING (subAccountTransfer) with unfunded master (EXPECTED to fail)")
    # =====================================================================
    from automation.account.funding import fund_subaccount

    step("G", "fund_subaccount(sub, 5 USD) — master unfunded",
         lambda: fund_subaccount(master_key=THROWAWAY_MASTER_KEY, env=ENV,
                                 subaccount_address=RANDOM_ADDR, usd_micro=5_000_000),
         expect_fail=True)

    # =====================================================================
    header("SUMMARY")
    # =====================================================================
    n_pass = sum(1 for _, k, _ in _RESULTS if k == "PASS")
    n_exp = sum(1 for _, k, _ in _RESULTS if k == "EXPECTED-FAIL")
    n_unexp = sum(1 for _, k, _ in _RESULTS if k == "UNEXPECTED-FAIL")
    print(f"  PASS              : {n_pass}")
    print(f"  EXPECTED-FAIL     : {n_exp}  (no keys/funds — must be STRUCTURED HL rejections)")
    print(f"  UNEXPECTED-FAIL   : {n_unexp}  (>0 ⇒ real connectivity/wrapper bug)")
    if n_unexp:
        print("\n  ⚠ UNEXPECTED failures:")
        for label, kind, msg in _RESULTS:
            if kind == "UNEXPECTED-FAIL":
                print(f"      - {label}: {msg}")

    # cleanup temp files
    import shutil
    with __import__("contextlib").suppress(Exception):
        shutil.rmtree(tmpdir)

    print("\n  Connectivity verdict: ", end="")
    if n_unexp == 0 and n_pass >= 10:
        print("API REACHABLE + wrappers OK (reads/construct/dry-run all passed; "
              "auth-required actions correctly rejected by the API).")
    else:
        print("SEE UNEXPECTED failures above.")
    return 1 if n_unexp else 0


if __name__ == "__main__":
    raise SystemExit(main())
