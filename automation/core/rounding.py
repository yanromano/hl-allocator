"""
automation.core.rounding — Price and size rounding for Hyperliquid orders.

Mirrors the Hyperliquid Python SDK rounding logic (§8 of the executor spec):

Price rounding
--------------
- Maximum **5 significant figures** AND at most ``(8 if is_spot else 6) - sz_decimals``
  decimal places.
- Integer prices are always allowed (e.g. 123 456.0 is valid even when >5 sig figs).
- Implementation: ``round(float(px))`` when ``abs(px) >= 1e5``, otherwise
  ``round(float(f"{float(px):.5g}"), max_dec)``.

Size rounding
-------------
- **FLOOR** (never round) to ``sz_decimals`` decimal places.
- Guarantees ``Σ(size × price)`` never exceeds available equity due to rounding up.
"""

from __future__ import annotations

import math
from decimal import Decimal


def round_price(
    px: Decimal | float,
    sz_decimals: int,
    is_spot: bool = False,
) -> float:
    """Round a price to Hyperliquid's tick-size rules.

    .. warning::
        **NOT wired into the live submit path.**  The executor submits IOC orders
        via ``HLClient.submit_ioc`` → SDK ``Exchange.market_open(px=None, slippage=...)``,
        so the **SDK** computes and formats the aggressive limit price (``_slippage_price``
        applies the same 5-sig / (6−szDecimals)dp rule).  This function is therefore
        *unused in production* today and is kept only for a possible future
        explicit-``px`` / Decimal-strict mode (spec §11/§23).  Do NOT wire it into the
        submit path without removing the SDK's own formatting first — applying both
        would double-round.  See the peer-issue-3 audit (spec §23): price formatting is
        owned by the SDK; this is dead code by design.

    Parameters
    ----------
    px:
        Raw price (``Decimal`` or ``float``).
    sz_decimals:
        The asset's ``szDecimals`` field from the exchange meta response.
    is_spot:
        ``True`` for spot markets (allows 8 − sz_decimals decimal places),
        ``False`` (default) for perpetuals (allows 6 − sz_decimals decimal places).

    Returns
    -------
    float
        Price rounded to Hyperliquid spec.

    Examples
    --------
    >>> round_price(123456.7, 2)        # >= 1e5 → integer
    123457.0
    >>> round_price(2.34567, 4)         # 5-sig + ≤ 2 dp → 2.35
    2.35
    """
    max_dec = (8 if is_spot else 6) - sz_decimals
    px_f = float(px)

    if abs(px_f) >= 1e5:
        # Integer prices always allowed for large values
        return float(round(px_f))
    else:
        # 5 significant figures, then clamp to max_dec decimal places
        return round(float(f"{px_f:.5g}"), max_dec)


def floor_size(sz: Decimal | float, sz_decimals: int) -> float:
    """Floor a size to ``sz_decimals`` decimal places (never rounds up).

    Parameters
    ----------
    sz:
        Raw size (``Decimal`` or ``float``).
    sz_decimals:
        Number of decimal places to floor to.

    Returns
    -------
    float
        Size floored to ``sz_decimals`` decimal places.

    Examples
    --------
    >>> floor_size(0.123999, 3)   # floors, not 0.124
    0.123
    >>> floor_size(1.9999, 0)     # floors to integer
    1.0
    """
    factor: int = 10**sz_decimals
    return float(math.floor(float(sz) * factor)) / factor
