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
from automation.core.config import Config, SignalSourceConfig


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


def _remote_state_path(cfg: Config, sleeve: SignalSourceConfig) -> Path:
    """Anti-replay state-file path for a REMOTE *sleeve* (red-team H5).

    Each sleeve gets its OWN file so two remote sleeves never share the
    ``seq=YYYYMMDD`` anti-replay sequence — a shared file would dark-hold one
    sleeve every day.  The file lives next to ``cfg.state_path`` (already
    ``data_dir``-prefixed by the config validators).

    The ``"default"`` sleeve — the singular-adapter form (a deployed long-only
    daemon) — keeps the LEGACY filename ``remote_signal_state.json`` so existing
    on-disk anti-replay state is not orphaned by the migration to per-sleeve
    files.  Every other sleeve is name-tagged ``remote_signal_state.{name}.json``.
    """
    parent = Path(cfg.state_path).parent
    if sleeve.name == "default":
        return parent / "remote_signal_state.json"
    return parent / f"remote_signal_state.{sleeve.name}.json"


def _build_sleeve_source(
    cfg: Config,
    sleeve: SignalSourceConfig,
    *,
    now_fn: Callable[[], datetime.datetime] | None = None,
) -> AllocationSource:
    """Build one generic ``AllocationSource`` for a single *sleeve*.

    Dispatches on ``sleeve.type``.  Never imports haarp/systems.  The REMOTE
    path forwards ``sleeve.allow_short`` and derives a per-sleeve anti-replay
    state file via :func:`_remote_state_path`.  The local whitelist is the
    global universe (per-sleeve whitelists are NOT yet a spec requirement —
    revisit if §4.1 later wants per-sleeve universes).

    Raises
    ------
    ValueError
        If ``sleeve.type`` is not one of {remote, equal_weight, single_asset},
        a required field for the chosen type is missing, or an environment
        variable named by ``token_ref`` / ``root_key_fingerprint_ref`` is unset
        or empty.
    """
    client_id = sleeve.client_id or ""

    if sleeve.type == "remote":
        if not sleeve.url:
            raise ValueError("signal_source.type='remote' requires `url`.")
        if not sleeve.root_key_fingerprint_ref:
            raise ValueError(
                "signal_source.type='remote' requires `root_key_fingerprint_ref`."
            )
        # token_ref / root_key_fingerprint_ref hold env var NAMES — resolve
        # the secret VALUES here (boot time), never forward the literals.
        fingerprint = _resolve_env_ref(
            sleeve.root_key_fingerprint_ref, "root_key_fingerprint_ref"
        )
        bearer = (
            _resolve_env_ref(sleeve.token_ref, "token_ref")
            if sleeve.token_ref
            else None
        )
        state_path = _remote_state_path(cfg, sleeve)
        whitelist = set(_coins(cfg))
        return RemoteSignalSource(
            sleeve.url,
            client_id,
            fingerprint,
            state_path,
            bearer_token=bearer,
            now_fn=now_fn,
            local_max_per_asset=cfg.max_per_asset,
            local_whitelist=whitelist,
            allow_short=sleeve.allow_short,
        )

    if sleeve.type == "equal_weight":
        coins = _coins(cfg)
        return EqualWeightSource(coins, client_id)

    if sleeve.type == "single_asset":
        coins = _coins(cfg)
        if not coins:
            raise ValueError(
                "signal_source.type='single_asset' requires at least one coin in "
                "`tradeable_coins` or `universe_perp_map`."
            )
        return SingleAssetSource(coins[0], client_id=client_id)

    raise ValueError(f"unknown signal_source.type: {sleeve.type!r}")


def build_sources(
    cfg: Config,
    *,
    now_fn: Callable[[], datetime.datetime] | None = None,
) -> dict[str, AllocationSource]:
    """Build one generic ``AllocationSource`` per sleeve, keyed by sleeve name.

    Iterates ``cfg.sleeves()`` (the normalized list — the singular
    ``signal_source`` form surfaces as the lone ``"default"`` sleeve) and
    dispatches each on its ``type`` via :func:`_build_sleeve_source`.  Each
    REMOTE sleeve receives its OWN anti-replay state file
    (``remote_signal_state.{name}.json``, or the legacy
    ``remote_signal_state.json`` for the ``"default"`` sleeve) so the sleeves
    never collide on the ``seq=YYYYMMDD`` anti-replay sequence (red-team H5).

    Never imports haarp/systems.

    Parameters
    ----------
    cfg:
        The validated executor configuration.
    now_fn:
        Optional injectable clock forwarded to each ``RemoteSignalSource``.

    Returns
    -------
    dict[str, AllocationSource]
        One concrete source per sleeve, keyed by ``sleeve.name`` (config
        order preserved; names are unique by config validation).
    """
    return {
        sleeve.name: _build_sleeve_source(cfg, sleeve, now_fn=now_fn)
        for sleeve in cfg.sleeves()
    }


def build_source(
    cfg: Config,
    *,
    now_fn: Callable[[], datetime.datetime] | None = None,
) -> AllocationSource:
    """Build the configured generic ``AllocationSource`` from *cfg*.

    Thin backward-compatible wrapper over :func:`build_sources` for the legacy
    single-source callers (``rebalance`` / ``run_daemon``) that have not yet
    been rewired to the sleeves layer (Task C5).  Returns the ``"default"``
    sleeve's source for the singular path, or the sole source otherwise; raises
    if the config carries more than one sleeve (a multi-sleeve config must use
    :func:`build_sources`).

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
        single_asset}, a required field for the chosen type is missing, an
        environment variable named by ``token_ref`` /
        ``root_key_fingerprint_ref`` is unset or empty, or the config carries
        more than one sleeve (use :func:`build_sources`).
    """
    sources = build_sources(cfg, now_fn=now_fn)
    if "default" in sources:
        return sources["default"]
    if len(sources) == 1:
        return next(iter(sources.values()))
    raise ValueError(
        "build_source() is single-source only; this config has multiple sleeves "
        f"({sorted(sources)}). Use build_sources() instead."
    )
