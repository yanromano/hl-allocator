"""V5 liquidity/depth probe (PUBLIC mainnet reads — NO keys).

Models the worst-case single-coin IOC market-buy slippage for the HAARP universe
against the LIVE Hyperliquid mainnet L2 order books, and compares it to the
executor's ``max_order_slippage`` cap. Retires the red-team gap "is PAXG (the
defensive sleeve) executable, and do XMR/ZEC/thin coins blow the slippage cap?".

HAARP can allocate up to 100% to a single coin (PAXG gold sleeve, or ETH/BTC),
so the worst case is the ENTIRE account notional hitting one perp in one IOC.

Run:  Python/venv/bin/python -m automation.scripts.mainnet_liquidity_probe
No keys, no submit — only public POST /info {"type":"l2Book"|"meta"}.
"""

from __future__ import annotations

import json
import urllib.request

MAINNET = "https://api.hyperliquid.xyz/info"
SLIPPAGE_CAP = 0.01  # cfg.caps.max_order_slippage default (1%)

# Largest single-coin weight HAARP allocates over the ~1yr golden window
# (from golden_allocations.csv) — the worst-case single-coin trade.
MAX_WEIGHT = {
    "PAXG": 1.00, "ETH": 0.876, "BTC": 0.85, "HYPE": 0.661, "SUI": 0.636,
    "XRP": 0.613, "XMR": 0.50, "BNB": 0.419, "DOGE": 0.25, "SOL": 0.225,
    "ZEC": 0.075, "AVAX": 0.038, "TON": 0.0, "ADA": 0.0,
}
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "XMR", "DOGE", "TON",
         "SUI", "HYPE", "AVAX", "ZEC", "PAXG"]
ACCOUNT_SIZES = [50_000.0, 500_000.0]


def post(body: dict) -> dict | list:
    req = urllib.request.Request(
        MAINNET, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def buy_slippage(asks: list[dict], notional: float) -> tuple[float | None, float]:
    """Walk the ask book filling ``notional`` USD. Return (slippage_frac, filled_usd).

    slippage = (VWAP - best_ask) / best_ask. None if the book can't fill it.
    """
    if not asks:
        return None, 0.0
    best = float(asks[0]["px"])
    spent = 0.0
    cost = 0.0
    sz_filled = 0.0
    for lvl in asks:
        px = float(lvl["px"])
        sz = float(lvl["sz"])
        lvl_notional = px * sz
        if spent + lvl_notional >= notional:
            take_usd = notional - spent
            take_sz = take_usd / px
            cost += take_sz * px
            sz_filled += take_sz
            spent = notional
            break
        spent += lvl_notional
        cost += sz * px
        sz_filled += sz
    if spent < notional - 1e-6:
        return None, spent  # book too thin to fill the trade
    vwap = cost / sz_filled
    return (vwap - best) / best, spent


def capacity_within_cap(asks: list[dict], mid: float, cap: float) -> float:
    """Largest USD notional buyable within ``cap`` slippage (book capacity)."""
    if not asks:
        return 0.0
    best = float(asks[0]["px"])
    limit_px = best * (1 + cap)
    usd = 0.0
    for lvl in asks:
        px = float(lvl["px"])
        if px > limit_px:
            break
        usd += px * float(lvl["sz"])
    return usd


def main() -> int:
    print("V5 mainnet liquidity probe — PUBLIC reads, no keys, no submit\n")
    meta = post({"type": "meta"})
    perps = {u["name"] for u in meta["universe"]}  # type: ignore[index]

    hdr = (f"{'COIN':<6}{'perp?':<7}{'mid':>12}{'maxW':>6}"
           f"{'  $50k IOC':>22}{'  $500k IOC':>22}{'cap<1%':>14}")
    print(hdr)
    print("-" * len(hdr))
    fails: list[str] = []
    for c in COINS:
        if c not in perps:
            print(f"{c:<6}{'NO':<7}{'— not a mainnet perp —':>40}")
            fails.append(f"{c}: no mainnet perp")
            continue
        book = post({"type": "l2Book", "coin": c})
        levels = book["levels"]  # type: ignore[index]
        bids, asks = levels[0], levels[1]
        if not bids or not asks:
            print(f"{c:<6}{'YES':<7}{'— empty book —':>40}")
            fails.append(f"{c}: empty book")
            continue
        mid = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2
        mw = MAX_WEIGHT.get(c, 0.0)
        cells = []
        for a in ACCOUNT_SIZES:
            notional = mw * a
            if notional <= 0:
                cells.append("    (never held)")
                continue
            slip, filled = buy_slippage(asks, notional)
            if slip is None:
                cells.append(f"${notional/1000:.0f}k→THIN({filled/1000:.0f}k)")
                fails.append(f"{c}: ${notional/1000:.0f}k IOC exceeds book depth")
            else:
                flag = "" if slip <= SLIPPAGE_CAP else " ✗"
                cells.append(f"${notional/1000:.0f}k→{slip*100:.2f}%{flag}")
                if slip > SLIPPAGE_CAP:
                    fails.append(f"{c}: ${notional/1000:.0f}k IOC slippage "
                                 f"{slip*100:.2f}% > {SLIPPAGE_CAP*100:.0f}% cap")
        capac = capacity_within_cap(asks, mid, SLIPPAGE_CAP)
        print(f"{c:<6}{'YES':<7}{mid:>12,.4f}{mw*100:>5.0f}%"
              f"{cells[0]:>22}{cells[1]:>22}{'$'+format(capac/1000,'.0f')+'k':>14}")

    print("\nWorst-case single-coin IOC (= maxWeight × account); 'cap<1%' = USD"
          " buyable within 1% slippage (book capacity, one side).")
    print("\n=== VERDICT ===")
    if fails:
        print(f"{len(fails)} concern(s):")
        for f in fails:
            print("  ✗", f)
        print("\nNOTE: a coin over the cap means the executor's IOC would be"
              " rejected (max_order_slippage) — the defensive rotation could be"
              " stranded. Stage the trade across days or cap deploy size for that"
              " coin. PAXG at 100% weight is the canonical concern.")
        return 1
    print("All 14 coins absorb the worst-case single-coin IOC within the 1% cap"
          " at the tested account sizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
