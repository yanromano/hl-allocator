"""
automation.allocation.source_factory — generic AllocationSource construction.

Builds the configured :class:`AllocationSource` from a :class:`Config` by
dispatching on ``cfg.signal_source.type``.  This module is STRATEGY-FREE and
deliberately imports NO haarp / systems code: the HAARP allocation source is
constructed separately by the closed injected composition root
(``automation.scripts.run_daemon_haarp``) and passed into ``build_daemon`` via
its ``source=`` parameter.

Dispatch
--------
* ``remote``        → :class:`RemoteSignalSource` wired from
  ``signal_source.url / token_ref / root_key_fingerprint_ref / client_id`` plus
  an anti-replay state file derived next to ``cfg.state_path`` and ``now_fn``.
* ``equal_weight``  → :class:`EqualWeightSource` over
  ``cfg.tradeable_coins or cfg.universe_perp_map`` keys.
* ``single_asset``  → :class:`SingleAssetSource` over the first such coin.
* anything else     → ``ValueError``.

In the ``remote`` case ``token_ref`` and ``root_key_fingerprint_ref`` are
**environment-variable NAMES** (e.g. ``HL_SIGNAL_BEARER``); the factory
resolves them via ``os.environ`` at build time and passes the resolved VALUES
to :class:`RemoteSignalSource` — secrets never live in the YAML.  This mirrors
``key_source.type: env_raw`` (:class:`automation.core.secrets.RawEnvKeyProvider`).
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path

from automation.allocation.base import AllocationSource
from automation.allocation.examples.equal_weight import EqualWeightSource
from automation.allocation.examples.single_asset import SingleAssetSource
from automation.allocation.remote_signal import RemoteSignalSource
from automation.core.config import Config


def _resolve_env_ref(env_var: str, field: str) -> str:
    """Resolve a ``signal_source.*_ref`` env-var NAME to its VALUE.

    Mirrors the error style of
    :meth:`automation.core.secrets.RawEnvKeyProvider.load_agent_key` — an
    unset or empty env var is a boot-time configuration error.

    Raises
    ------
    ValueError
        If the environment variable named by *env_var* is unset or empty.
    """
    value = os.environ.get(env_var)
    if not value:
        raise ValueError(
            f"signal_source.{field} names environment variable {env_var!r}, "
            "which is unset or empty.  Set it to the secret value (e.g. as a "
            "Railway Variable) before starting the process."
        )
    return value


def _coins(cfg: Config) -> list[str]:
    """Ordered coin list: ``tradeable_coins`` if set, else ``universe_perp_map`` keys."""
    if cfg.tradeable_coins:
        return list(cfg.tradeable_coins)
    return list(cfg.universe_perp_map)


def build_source(
    cfg: Config,
    *,
    now_fn: Callable[[], datetime.datetime] | None = None,
) -> AllocationSource:
    """Build the configured generic ``AllocationSource`` from *cfg*.

    Dispatches on ``cfg.signal_source.type``.  Never imports haarp/systems.

    Parameters
    ----------
    cfg:
        The validated executor configuration.
    now_fn:
        Optional injectable clock forwarded to sources that have a freshness
        gate (currently only ``RemoteSignalSource``).

    Returns
    -------
    AllocationSource
        A concrete source implementing the ``AllocationSource`` protocol.

    Raises
    ------
    ValueError
        If ``signal_source.type`` is not one of {remote, equal_weight,
        single_asset}, a required field for the chosen type is missing, or an
        environment variable named by ``token_ref`` /
        ``root_key_fingerprint_ref`` is unset or empty.
    """
    ss = cfg.signal_source
    client_id = ss.client_id or ""

    if ss.type == "remote":
        if not ss.url:
            raise ValueError("signal_source.type='remote' requires `url`.")
        if not ss.root_key_fingerprint_ref:
            raise ValueError(
                "signal_source.type='remote' requires `root_key_fingerprint_ref`."
            )
        # token_ref / root_key_fingerprint_ref hold env var NAMES — resolve
        # the secret VALUES here (boot time), never forward the literals.
        fingerprint = _resolve_env_ref(
            ss.root_key_fingerprint_ref, "root_key_fingerprint_ref"
        )
        bearer = _resolve_env_ref(ss.token_ref, "token_ref") if ss.token_ref else None
        # Anti-replay state lives next to the executor state file.
        state_path = Path(cfg.state_path).parent / "remote_signal_state.json"
        whitelist = set(_coins(cfg))
        return RemoteSignalSource(
            ss.url,
            client_id,
            fingerprint,
            state_path,
            bearer_token=bearer,
            now_fn=now_fn,
            local_max_per_asset=cfg.max_per_asset,
            local_whitelist=whitelist,
        )

    if ss.type == "equal_weight":
        coins = _coins(cfg)
        return EqualWeightSource(coins, client_id)

    if ss.type == "single_asset":
        coins = _coins(cfg)
        if not coins:
            raise ValueError(
                "signal_source.type='single_asset' requires at least one coin in "
                "`tradeable_coins` or `universe_perp_map`."
            )
        return SingleAssetSource(coins[0], client_id=client_id)

    raise ValueError(f"unknown signal_source.type: {ss.type!r}")
