"""
automation.allocation.examples.equal_weight — Equal-weight AllocationSource.

Ships an ``EqualWeightSource`` for offline development, testing, and dry-run
simulation when the live HAARP RemoteSignalSource (B5) is not available.

The produced ``TargetAllocation`` is guaranteed to pass
``validate_target_allocation`` for any matching whitelist / client_id /
max_per_asset configuration, provided ``invested_fraction / len(coins)`` does
not exceed ``max_per_asset`` (i.e. the caller's concentration limit is compatible
with the portfolio size).
"""

from __future__ import annotations

import datetime

from automation.allocation.base import TargetAllocation
from automation.core.redaction import get_logger

_logger = get_logger(__name__)


class EqualWeightSource:
    """Equal-weight allocation source.

    Divides ``invested_fraction`` uniformly across all coins and allocates the
    remainder to ``cash``.  The resulting ``TargetAllocation`` satisfies the
    unity-sum constraint exactly (up to floating-point representation).

    Parameters
    ----------
    coins:
        Ordered list of coin symbols to include.  Must be non-empty.
    client_id:
        Executor client identifier.  Written into ``TargetAllocation.audience``
        so that the downstream ``validate_target_allocation`` audience check passes.
    invested_fraction:
        Fraction of the portfolio to invest (in [0, 1]).  The remainder becomes
        ``cash``.  Defaults to 1.0 (fully invested).
    strategy_id:
        Strategy identifier written into the ``TargetAllocation``.
        Defaults to ``"equal-weight"``.
    model_rev:
        Model revision string.  Defaults to ``"example"``.

    Raises
    ------
    ValueError
        If ``coins`` is empty or ``invested_fraction`` is outside [0, 1].
    """

    def __init__(
        self,
        coins: list[str],
        client_id: str,
        invested_fraction: float = 1.0,
        strategy_id: str = "equal-weight",
        model_rev: str = "example",
    ) -> None:
        if not coins:
            raise ValueError("EqualWeightSource requires at least one coin.")
        if not (0.0 <= invested_fraction <= 1.0):
            raise ValueError(
                f"invested_fraction must be in [0, 1], got {invested_fraction!r}"
            )

        self._coins = list(coins)
        self._client_id = client_id
        self._invested_fraction = invested_fraction
        self._strategy_id = strategy_id
        self._model_rev = model_rev

    def get_target_allocation(
        self, as_of: datetime.date | None = None
    ) -> TargetAllocation:
        """Return an equal-weight ``TargetAllocation``.

        Parameters
        ----------
        as_of:
            Optional reference date.  Defaults to ``datetime.date.today()``.

        Returns
        -------
        TargetAllocation
            Allocation with equal weights summing to ``invested_fraction`` and
            ``cash = 1 - invested_fraction``.
        """
        effective_date = as_of if as_of is not None else datetime.date.today()
        per_coin_weight = self._invested_fraction / len(self._coins)
        weights = dict.fromkeys(self._coins, per_coin_weight)
        cash = 1.0 - self._invested_fraction

        _logger.info(
            "EqualWeightSource.get_target_allocation",
            coins=self._coins,
            per_coin_weight=round(per_coin_weight, 6),
            invested_fraction=self._invested_fraction,
            cash=round(cash, 6),
            as_of=str(effective_date),
        )

        return TargetAllocation(
            as_of=effective_date,
            weights=weights,
            cash=cash,
            strategy_id=self._strategy_id,
            model_rev=self._model_rev,
            audience=self._client_id,
        )
