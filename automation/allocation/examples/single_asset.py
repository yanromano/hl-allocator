"""
automation.allocation.examples.single_asset — Single-asset AllocationSource.

Ships a ``SingleAssetSource`` for offline development, testing, and dry-run
simulation: 100 % of the portfolio allocated to a single coin (cash 0).  Useful
for exercising the order path against one perpetual without standing up the live
HAARP RemoteSignalSource.

The produced ``TargetAllocation`` is guaranteed to pass
``validate_target_allocation`` for any matching whitelist / client_id /
max_per_asset >= 1.0 configuration (a single 100 %-weight coin requires the
caller's ``max_per_asset`` to admit a full-weight asset).
"""

from __future__ import annotations

import datetime

from automation.allocation.base import TargetAllocation
from automation.core.redaction import get_logger

_logger = get_logger(__name__)


class SingleAssetSource:
    """100 %-to-one-coin allocation source.

    Allocates the entire portfolio to a single coin (weight 1.0) with cash 0.0.
    The resulting ``TargetAllocation`` satisfies the unity-sum constraint exactly.

    Parameters
    ----------
    coin:
        The coin symbol to hold at 100 %.  Must be non-empty.
    client_id:
        Executor client identifier.  Written into ``TargetAllocation.audience``
        so that the downstream ``validate_target_allocation`` audience check passes.
    strategy_id:
        Strategy identifier written into the ``TargetAllocation``.
        Defaults to ``"single-asset"``.
    model_rev:
        Model revision string.  Defaults to ``"example"``.

    Raises
    ------
    ValueError
        If ``coin`` is empty.
    """

    def __init__(
        self,
        coin: str,
        *,
        client_id: str = "",
        strategy_id: str = "single-asset",
        model_rev: str = "example",
    ) -> None:
        if not coin:
            raise ValueError("SingleAssetSource requires a non-empty coin.")
        self._coin = coin
        self._client_id = client_id
        self._strategy_id = strategy_id
        self._model_rev = model_rev

    def get_target_allocation(
        self, as_of: datetime.date | None = None
    ) -> TargetAllocation:
        """Return a 100 %-to-``coin`` ``TargetAllocation``.

        Parameters
        ----------
        as_of:
            Optional reference date.  Defaults to ``datetime.date.today()``.

        Returns
        -------
        TargetAllocation
            Allocation with ``{coin: 1.0}`` and ``cash = 0.0``.
        """
        effective_date = as_of if as_of is not None else datetime.date.today()
        weights = {self._coin: 1.0}

        _logger.info(
            "SingleAssetSource.get_target_allocation",
            coin=self._coin,
            as_of=str(effective_date),
        )

        return TargetAllocation(
            as_of=effective_date,
            weights=weights,
            cash=0.0,
            strategy_id=self._strategy_id,
            model_rev=self._model_rev,
            audience=self._client_id,
        )
