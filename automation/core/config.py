"""
Configuration loader for the hl-allocator executor.

Loads and validates ``automation.example.yaml`` (or a user-supplied override)
using pydantic v2.  No dependency on Python/systems/haarp or Python/technical.

Usage::

    from automation.core.config import Config

    cfg = Config.from_yaml("/path/to/automation.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator


class SignalSourceConfig(BaseModel):
    """Where the executor fetches allocation weights from.

    One ``SignalSourceConfig`` is a SLEEVE: an independent signed signal with
    its own equity ``budget`` and short capability.  The deployed singular
    ``signal_source:`` block is one sleeve named ``"default"`` (budget 1.0,
    long-only); the new ``signal_sources:`` list holds N of them on one
    account (CRASH Phase 1).
    """

    type: Literal["remote", "equal_weight", "single_asset"]
    """Signal source type.

    * ``remote``        — fetch JSON payload from an authenticated HTTP endpoint.
    * ``equal_weight``  — split equally across all coins in ``universe_perp_map``.
    * ``single_asset``  — 100 % allocated to a single coin (useful for testing).
    """

    url: str | None = None
    """HTTPS endpoint that returns the allocation payload (remote only)."""

    token_ref: str | None = None
    """Name of the environment variable holding the Bearer token used to
    authenticate to ``url`` (e.g. ``"HL_SIGNAL_BEARER"``).  Resolved via
    ``os.environ`` at build time by the source factory — the secret itself
    never lives in the YAML."""

    root_key_fingerprint_ref: str | None = None
    """Name of the environment variable holding the root key fingerprint
    (lowercase hex) used to pin the remote server's signing key (e.g.
    ``"HL_SIGNAL_ROOT_FP"``).  Resolved via ``os.environ`` at build time.
    Required when ``type=="remote"`` on mainnet."""

    client_id: str | None = None
    """Opaque client identifier sent in the ``X-Client-ID`` request header."""

    name: str = "default"
    """Unique sleeve name (snake_case).  Keys per-sleeve state files and audit
    records.  The singular ``signal_source:`` block normalizes to ``"default"``;
    every entry in a ``signal_sources:`` list MUST be uniquely named."""

    budget: float = 1.0
    """Fraction of TOTAL equity this sleeve commands (``0 < budget <= 1``).  The
    sleeves' budgets must sum to ``<= 1``; the remainder is a permanent cash
    buffer.  The singular path is always 1.0 (the whole account)."""

    allow_short: bool = False
    """When True, this sleeve may serve signed (negative) weights — a v2 /
    short-capable sleeve.  Requires ``type=="remote"`` (there is no short
    ``equal_weight``).  The singular path is always long-only (False)."""

    max_signal_staleness_days: int | None = None
    """Per-sleeve override for the maximum signal age (in days).  ``None`` ⇒
    fall back to the top-level :attr:`Config.max_signal_staleness_days`.  Resolve
    the effective value via :meth:`Config.effective_staleness_days`."""

    pinned_model_rev: str | None = None
    """Per-sleeve override for the served-allocation ``model_rev`` pin (the
    NOGATE-TEST defense).  ``None`` ⇒ fall back to the top-level
    :attr:`Config.pinned_model_rev`.  Each sleeve consumes a DIFFERENT signal
    (e.g. HAARP ``…-NOGATE-TEST`` vs CRASH gated rev), so a single global pin
    cannot match both — set this per-sleeve.  Resolve via
    :meth:`Config.effective_pinned_model_rev`."""


class KeySourceConfig(BaseModel):
    """Where the agent key comes from.

    * ``keystore`` (default) — decrypt the encrypted V3 keystore at
      ``agent_keystore_path`` using the configured passphrase backend
      (:class:`automation.core.secrets.KeystoreKeyProvider`).
    * ``env_raw`` — read the *raw* agent private key directly from the
      environment variable named ``env_var``
      (:class:`automation.core.secrets.RawEnvKeyProvider`).  Intended for 24/7
      unattended runs where no passphrase / OS keyring is available.
    """

    type: Literal["keystore", "env_raw"] = "keystore"
    """Key source selector."""

    env_var: str = "HL_AGENT_KEY"
    """Environment variable read by the ``env_raw`` source (ignored otherwise)."""


class RetryConfig(BaseModel):
    """Bounded intra-cycle retry knobs (see 2026-06-08 design spec)."""

    enabled: bool = True
    """When False, a non-clean cycle defers to the next daily rebalance (legacy)."""

    interval_seconds: int = 600
    """Seconds between retry attempts (default 10 min)."""

    window_seconds: int = 10800
    """Total retry window length in seconds (default 3 h)."""

    liquidity_safety: float = 1.0
    """Require opposing L2 depth >= required_size * liquidity_safety (within the
    max_order_slippage band) before firing an IOC; else defer the leg."""


class SignalRecheckConfig(BaseModel):
    """No-signal re-check window (see 2026-06-10 SP3 design spec §5)."""

    enabled: bool = True
    """When False, a no-signal HOLD consumes the day permanently (legacy)."""

    interval_seconds: int = 180
    """Minimum seconds between signal re-checks while the window is open."""

    window_seconds: int = 7200
    """Total re-check window from the first hold; expiry consumes the day."""


class HealthConfig(BaseModel):
    """Position-health monitor thresholds (see 2026-06-08 design spec)."""

    enabled: bool = True
    """When True, the daemon runs the position-health checks each cycle and the
    standalone monitor is active."""
    funding_drag_pct: float = 5.0
    """FUNDING_DRAG alert when Σ(cumFunding.sinceOpen) exceeds this % of equity."""
    margin_maint_frac_warn: float = 0.5
    """MARGIN_RATIO_LOW when crossMaintenanceMarginUsed / equity exceeds this fraction."""
    liq_distance_pct_warn: float = 10.0
    """MARGIN_RATIO_LOW when any liquidationPx is within this % of the mark."""
    spot_collateral_warn_usd: float = 1.0
    """COLLATERAL_IN_SPOT when free spot USDC exceeds this (collateral should be in perp)."""
    http_port: int = 8080
    """Port for the /healthz liveness server."""
    heartbeat_max_silence_seconds: int = 180
    """Mark unhealthy if no daemon tick within this many seconds."""


class CapsConfig(BaseModel):
    """Hard position and order caps.  All monetary values are in USD notional."""

    max_notional_per_coin: float
    """Maximum USD notional held in any single perpetual at end-of-cycle."""

    max_total_notional: float
    """Maximum aggregate USD notional across all positions."""

    max_turnover_per_cycle: float
    """Maximum total USD traded (buys + sells) in a single rebalance cycle."""

    max_order_slippage: float
    """Maximum acceptable slippage fraction per order (e.g. 0.01 = 1 %)."""


class Config(BaseModel):
    """Top-level executor configuration.

    Load from YAML via :meth:`from_yaml`; constructed fields are validated
    by pydantic v2.
    """

    env: Literal["testnet", "mainnet"]
    """Target Hyperliquid environment."""

    subaccount_address: str
    """EVM address of the account to trade from.

    Under ``trade_mode="subaccount"`` (default) this is the Hyperliquid
    sub-account (provisioned in task B8; use the zero address in the example
    config).  Under ``trade_mode="master_direct"`` this field MUST hold the
    MASTER EVM address the agent key is approved on (no on-chain validation is
    possible here — the operator is responsible for setting it correctly)."""

    trade_mode: Literal["subaccount", "master_direct"] = "subaccount"
    """How orders are routed for the agent key.

    * ``subaccount`` (default) — orders route to the sub-account via the
      Exchange ``vault_address`` field; ``subaccount_address`` is the sub.
      Byte-identical to the legacy behaviour.
    * ``master_direct`` — the agent acts FOR the MASTER account with NO vault
      routing (``account_address=<master>``, ``vault_address=None``).  Use this
      when the account has no sub-account — e.g. fresh accounts below the
      ~$100k cumulative-volume threshold that gates sub-account creation.  In
      this mode ``subaccount_address`` MUST hold the MASTER EVM address the
      agent is approved on (no on-chain validation possible)."""

    signal_source: SignalSourceConfig | None = None
    """Singular allocation source (the deployed long-only form).

    Optional ONLY at the pydantic field level: the
    ``_normalize_signal_sources`` validator enforces that EXACTLY ONE of
    ``signal_source`` / ``signal_sources`` is provided (neither-present is
    rejected, restoring the historical "required" semantics).  When the
    singular form is used it stays populated here for legacy callers that read
    ``cfg.signal_source.client_id`` directly — the adapter additionally
    materializes it as the lone ``signal_sources`` sleeve named ``"default"``.
    """

    signal_sources: list[SignalSourceConfig] | None = None
    """Multi-source (sleeves) allocation form (CRASH Phase 1).

    A list of independent signed signals, each with its own ``budget`` /
    ``allow_short``.  Mutually exclusive with ``signal_source`` (exactly one of
    the two forms).  Downstream code reads sleeves via :meth:`sleeves`, NEVER
    this raw field, so the singular path (normalized to one ``default`` sleeve)
    and the list path are uniform from the consumer's perspective."""

    rebalance_utc_time: str
    """UTC wall-clock time for the daily rebalance trigger (``"HH:MM"`` format,
    e.g. ``"00:15"``)."""

    threshold_pct: float = 3.0
    """Minimum absolute allocation change (in percentage points) required to
    trigger an order for a given coin.  Smaller deviations are skipped."""

    max_per_asset: float
    """Maximum allocation fraction for any single asset (0–1, e.g. 0.5 = 50 %)."""

    caps: CapsConfig
    """Hard notional and slippage caps applied before any order is submitted."""

    retry: RetryConfig = RetryConfig()
    """Bounded intra-cycle retry configuration."""

    health: HealthConfig = HealthConfig()
    """Position-health monitor configuration."""

    signal_recheck: SignalRecheckConfig = SignalRecheckConfig()
    """No-signal re-check window configuration."""

    key_source: KeySourceConfig = KeySourceConfig()
    """Where the composition root loads the agent key from (keystore vs env_raw)."""

    safety_buffer: float = 0.03
    """Fraction of total equity kept as free margin (e.g. 0.03 = 3 %).
    Target allocations are scaled to ``1 - safety_buffer`` of equity."""

    max_signal_staleness_days: int = 2
    """Reject and halt if the signal payload is older than this many days.

    This is the GLOBAL fallback; a sleeve may override it via
    :attr:`SignalSourceConfig.max_signal_staleness_days`.  Resolve the
    per-sleeve effective value via :meth:`effective_staleness_days`."""

    liquidation_cooldown_days: int = 1
    """Per-coin post-liquidation re-entry cooldown (in days).

    After a position vanishes from the live snapshot WITHOUT a matching exit
    order in the audit ledger (i.e. an isolated liquidation), the engine arms a
    cooldown for that coin; same-direction re-entries within this many days are
    skipped so the daemon does not re-open into the squeeze (red-team M3).
    Default 1 day.  Consumed in Stage C (run_cycle)."""

    universe_perp_map: dict[str, str] = {}
    """Mapping from strategy coin ticker to Hyperliquid perpetual name.

    Example: ``{"BTC": "BTC", "ETH": "ETH"}``.
    The neutral example ships only BTC and ETH; the real HAARP universe is
    provisioned separately and never committed to the public repository.
    """

    spot_routing: dict[str, str] = {}
    """Mapping from strategy coin ticker to HL **spot** pair name (``@NNN`` form).

    Example: ``{"HYPE": "@107", "ETH": "@151", "BTC": "@142", "SOL": "@156"}``.

    Coins listed here trade on HL **SPOT** (no funding cost); every other coin in
    ``universe_perp_map`` trades on **perp**.  Default empty ⇒ perp-only (the
    legacy behaviour — byte-identical cycle when this is ``{}``).

    Keyspace guard (red-team finding #8)
    ------------------------------------
    Every coin in ``spot_routing`` MUST be **identity** in ``universe_perp_map``
    (i.e. ``universe_perp_map.get(coin, coin) == coin``).  The reconciler and the
    executor key spot coins by the POST-``_map_perp`` coin name; a spot coin that
    is remapped to a different perp name would have its order keyed by the perp
    name while ``spot_routing`` is keyed by the ticker — so it would silently
    route to PERP.  ``_validate_spot_routing_keyspace`` rejects such a config at
    load time (see below).
    """

    divergence_alert_pct: float = 5.0
    """Post-rebalance divergence threshold in percentage points (pp).

    After a rebalance executes, ``run_cycle`` takes a fresh on-chain snapshot and
    computes the maximum absolute deviation between achieved weights and the target
    weights (in the perp key-space).  If ``max |achieved − target| > divergence_alert_pct / 100``
    a ``DIVERGENCE`` alert is emitted (WARNING at 1×, CRITICAL at 2× the threshold).
    Default 5.0 = 5 percentage points.
    """

    agent_valid_until_ms: int | None = None
    """Epoch-millisecond timestamp when the trading agent key expires (UTC).

    The operator copies this value from the ``ProvisionResult.valid_until_ms``
    returned by ``automation.account.provisioning.provision`` (or reads it from
    the non-secret sidecar ``<keystore>.meta.json`` written alongside the
    keystore).

    When set, the daemon's ``boot`` check emits an ``AGENT_EXPIRY`` WARNING
    alert if fewer than 14 days remain, and ``run_once`` emits an
    ``AGENT_EXPIRY`` CRITICAL alert and refuses to trade (fail-safe HOLD) when
    the agent is already expired.  ``None`` means expiry checking is disabled
    (legacy / not-yet-configured deployments).
    """

    # -----------------------------------------------------------------------
    # E1 — composition-root deploy fields.
    #
    # All of these have sensible defaults so existing configs (and the 676
    # existing tests) keep parsing unchanged.  They are consumed ONLY by the
    # composition root (``automation.scripts.run_daemon.build_daemon``) to wire
    # the daemon from config: keystore → KeyProvider → for_trading →
    # HaarpAllocationSource → Daemon.
    # -----------------------------------------------------------------------

    agent_keystore_path: str | None = None
    """Path to the encrypted V3 keystore holding the agent private key.

    Written by ``automation.account.provisioning``.  ``None`` means the
    composition root cannot load a key (the operator must inject a client
    explicitly, e.g. in tests / dry-run with a fake client)."""

    passphrase_backend: Literal["os_keyring", "prompt", "env"] = "os_keyring"
    """Which :class:`automation.core.secrets.PassphraseBackend` the composition
    root constructs to unlock the keystore.

    * ``os_keyring`` — OS secret store (uses ``keyring_service`` / ``keyring_username``).
    * ``prompt``     — interactive getpass (developer workstation / CI with a TTY).
    * ``env``        — read the passphrase from ``passphrase_env_var``.  **DEV /
                       TESTNET ONLY** — rejected on mainnet at the config layer
                       (mirrors the secrets-layer refusal)."""

    passphrase_env_var: str | None = None
    """Environment variable name read by the ``env`` passphrase backend."""

    keyring_service: str | None = None
    """OS-keyring service name for the ``os_keyring`` passphrase backend."""

    keyring_username: str | None = None
    """OS-keyring username for the ``os_keyring`` passphrase backend."""

    nonce_state_path: str = "nonce.state"
    """Path to the persistent nonce state file (MF-10).  The watchdog
    kill-agent MUST use a DIFFERENT path to keep its nonce space separate (S-4)."""

    state_path: str = "state.json"
    """Path to the persistent executor state file (the frozen ratchet baseline)."""

    ledger_path: str = "ledger.json"
    """Path to the crash-safe intent ledger file."""

    audit_path: str = "audit.jsonl"
    """Path to the append-only JSONL audit log."""

    kill_flag_path: str | None = None
    """Tier-1 in-process kill-switch flag file.  ``None`` disables the kill
    (the daemon never polls a flag).  When set, the operator arms the kill by
    creating the file; the daemon flattens to cash and HALTS until it is removed."""

    data_dir: str | None = None
    """When set, RELATIVE state-file paths are resolved under this dir (e.g. a
    Railway volume mount).  Absolute paths pass through unchanged.

    Affected fields: ``state_path``, ``nonce_state_path``, ``ledger_path``,
    ``audit_path``, and ``kill_flag_path`` (when not None).
    """

    pinned_model_rev: str | None = None
    """The parity-passed HAARP ``MODEL_REV`` to pin (S-7/S-9).  Forwarded to
    :class:`HaarpAllocationSource`; if the loaded HAARP code/config has drifted
    from this revision the source fails CLOSED (raises ``SignalRejected``).
    ``None`` disables the pin (dev / offline only)."""

    alert_webhook_url: str | None = None
    """Optional alert webhook.  ``None`` → :class:`LogAlertSink` only.  When set,
    the composition root wires a :class:`MultiSink` of ``LogAlertSink`` +
    ``WebhookAlertSink``."""

    tradeable_coins: list[str] | None = None
    """Coins actually available on the venue (drives the HAARP fold-to-cash gate).
    ``None`` → use the keys of ``universe_perp_map`` (the full configured universe).
    Set this to a SUBSET when a coin in the universe is not yet listed on the
    target venue (e.g. XRP absent on testnet)."""

    strategy_id: str = "haarp-multi"
    """Opaque strategy identifier forwarded to the daemon (cloid generation) and
    the allocation source."""

    target_leverage: int = 1
    """Per-coin leverage to set at boot (1 = unleveraged / spot-like). HAARP is a 1x basket."""

    leverage_is_cross: bool = False
    """Margin mode for the leverage set: False = ISOLATED (default), True = cross."""

    set_leverage_on_boot: bool = True
    """When True, the daemon sets target_leverage on every tradeable coin at boot."""

    enforce_leverage_on_drift: bool = True
    """When True, a held position whose leverage TYPE mismatches the target is
    flattened (reduce-only) at boot so it reopens at the target leverage next
    cycle. When False, only a LEVERAGE_DRIFT alert is emitted."""

    disable_gate: bool = False
    """**TEST-ONLY (default False).**  Neutralize the HAARP BTC macro cash-gate so the
    allocation source builds positions even in a risk-off (all-cash) regime — for
    exercising the order path on TESTNET when production HAARP is in cash.

    Forwarded to :class:`HaarpAllocationSource`; when True the served allocation is
    built ungated AND its ``model_rev`` is stamped ``…-NOGATE-TEST`` so it can never
    masquerade as the parity-locked production signal.

    **FORBIDDEN on mainnet** — rejected at this config layer (mirrors the
    ``passphrase_backend='env'`` refusal) and again by the smoke harness.  Default
    False keeps the production path byte-identical."""

    @model_validator(mode="after")
    def _validate_spot_routing_keyspace(self) -> Config:
        """Enforce that every spot-routed coin is IDENTITY in universe_perp_map.

        Red-team finding #8: the reconciler/executor key a spot coin by the
        post-``_map_perp`` coin name.  A spot coin remapped to a different perp
        name (``universe_perp_map[coin] != coin``) would route to perp silently —
        ``spot_routing`` is keyed by the ticker, the order op by the perp name.
        Reject the config rather than mis-route capital.
        """
        non_identity = [
            coin
            for coin in self.spot_routing
            if self.universe_perp_map.get(coin, coin) != coin
        ]
        if non_identity:
            raise ValueError(
                "spot_routing coins must be IDENTITY in universe_perp_map "
                "(universe_perp_map.get(coin, coin) == coin); these spot coins are "
                f"remapped to a different perp name and would silently route to perp: "
                f"{ {c: self.universe_perp_map[c] for c in non_identity} }"
            )
        return self

    @staticmethod
    def _is_singular_echo(
        singular: SignalSourceConfig,
        sources: list[SignalSourceConfig],
    ) -> bool:
        """True iff ``sources`` is the idempotent echo of the singular adapter.

        Distinguishes a re-validation round-trip (where ``model_dump`` carries
        BOTH the original ``signal_source`` and the ``signal_sources`` list this
        validator previously synthesized) from a genuine operator conflict.  The
        echo is exactly one sleeve named ``"default"`` with budget 1.0 /
        long-only whose wire fields match the singular block.
        """
        if len(sources) != 1:
            return False
        only = sources[0]
        if not (
            only.name == "default"
            and only.budget == 1.0
            and only.allow_short is False
            and only.max_signal_staleness_days is None
        ):
            return False
        # Wire fields must match the singular block (the adapter only overrides
        # the sleeve-identity fields, copying everything else verbatim).
        return (
            only.type == singular.type
            and only.url == singular.url
            and only.token_ref == singular.token_ref
            and only.root_key_fingerprint_ref == singular.root_key_fingerprint_ref
            and only.client_id == singular.client_id
        )

    @model_validator(mode="after")
    def _normalize_signal_sources(self) -> Config:
        """Adapt the singular ``signal_source`` to the ``signal_sources`` list.

        Enforces EXACTLY ONE of the two forms is provided, then materializes a
        uniform ``signal_sources`` list so every downstream consumer (via
        :meth:`sleeves`) sees the same shape:

        * NEITHER present → ValueError (restores the historical "required"
          semantics now that the pydantic field is optional);
        * BOTH present → ValueError (the two forms are mutually exclusive);
        * SINGULAR only → wrap a COPY of the block as one sleeve named
          ``"default"`` with ``budget=1.0`` / ``allow_short=False`` (the whole
          account, long-only).  ``signal_source`` stays populated so legacy
          code reading ``cfg.signal_source.client_id`` keeps working;
        * LIST only → validate ``budget``/name/short constraints and the
          aggregate budget cap.

        This runs BEFORE :meth:`_validate_mainnet_remote` (definition order) so
        that validator can iterate the normalized sleeves uniformly.  It must
        NOT touch the state-file paths — :meth:`_apply_data_dir` (defined later)
        still owns the ``data_dir`` prefixing pass.
        """
        has_singular = self.signal_source is not None
        has_list = self.signal_sources is not None

        if not has_singular and not has_list:
            raise ValueError(
                "exactly one of `signal_source` (singular) or `signal_sources` "
                "(list) must be provided; neither was set"
            )
        if has_singular and has_list:
            # Both fields populated.  This is a CONFLICT *unless* the list is the
            # idempotent echo this very validator wrote on a prior pass — i.e. a
            # re-validation round-trip (``Config.model_validate(cfg.model_dump())``,
            # used by the smoke harness to re-apply mutated fields).  In that case
            # ``signal_sources`` is exactly the one ``default`` sleeve derived from
            # ``signal_source``; accept it (re-derive idempotently below) rather
            # than reject a config that already validated once.  A genuinely
            # operator-supplied list alongside a singular block is rejected.
            assert self.signal_source is not None  # narrowed by has_singular
            assert self.signal_sources is not None  # narrowed by has_list
            if not self._is_singular_echo(self.signal_source, self.signal_sources):
                raise ValueError(
                    "`signal_source` (singular) and `signal_sources` (list) are "
                    "mutually exclusive; provide exactly one form"
                )
            # fall through to re-derive the canonical default sleeve

        if has_singular:
            # Singular adapter: one full-account long-only sleeve named
            # "default".  model_copy(update=...) preserves every wire field
            # (type/url/token_ref/root_key_fingerprint_ref/client_id) while
            # pinning the sleeve identity fields.
            assert self.signal_source is not None  # narrowed by has_singular
            self.signal_sources = [
                self.signal_source.model_copy(
                    update={
                        "name": "default",
                        "budget": 1.0,
                        "allow_short": False,
                        "max_signal_staleness_days": None,
                    }
                )
            ]
            return self

        # LIST form — validate the sleeves.
        assert self.signal_sources is not None  # narrowed by has_list
        sleeves = self.signal_sources

        if not sleeves:
            raise ValueError("`signal_sources` list must not be empty")

        names = [s.name for s in sleeves]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"`signal_sources` sleeve names must be unique; duplicates: {dupes}"
            )

        bad_budgets = [s.name for s in sleeves if s.budget <= 0.0]
        if bad_budgets:
            raise ValueError(
                "every sleeve `budget` must be > 0; offending sleeves: "
                f"{bad_budgets}"
            )

        total_budget = sum(s.budget for s in sleeves)
        if total_budget > 1.0 + 1e-9:
            raise ValueError(
                "sum of sleeve `budget` values must be <= 1.0 (the remainder is "
                f"a permanent cash buffer); got {total_budget}"
            )

        short_non_remote = [
            s.name for s in sleeves if s.allow_short and s.type != "remote"
        ]
        if short_non_remote:
            raise ValueError(
                "`allow_short=true` requires `type: remote`; offending sleeves: "
                f"{short_non_remote}"
            )

        return self

    @model_validator(mode="after")
    def _validate_mainnet_remote(self) -> Config:
        """Enforce mainnet + remote sleeve has both url and key fingerprint.

        Applies the rule PER SOURCE: every ``type=="remote"`` sleeve on mainnet
        must carry both ``url`` and ``root_key_fingerprint_ref``.  Iterates the
        normalized :meth:`sleeves` list so the singular and list forms share one
        code path (the singular block surfaces as the lone ``default`` sleeve).
        """
        if self.env != "mainnet":
            return self
        for sleeve in self.sleeves():
            if sleeve.type != "remote":
                continue
            missing: list[str] = []
            if not sleeve.url:
                missing.append("url")
            if not sleeve.root_key_fingerprint_ref:
                missing.append("root_key_fingerprint_ref")
            if missing:
                raise ValueError(
                    f"mainnet + remote signal source '{sleeve.name}' requires both "
                    f"`url` and `root_key_fingerprint_ref` to be set; missing: {missing}"
                )
        return self

    @model_validator(mode="after")
    def _validate_mainnet_passphrase_backend(self) -> Config:
        """Refuse the plaintext ``env`` passphrase backend on mainnet.

        Mirrors the secrets-layer refusal (``KeystoreKeyProvider._check_backend_policy``
        rejects ``EnvBackend`` on mainnet) at the CONFIG layer too, so a mis-configured
        deploy fails at load time — before any key is ever loaded — rather than only
        when the composition root constructs the backend.
        """
        if self.env == "mainnet" and self.passphrase_backend == "env":
            raise ValueError(
                "passphrase_backend='env' (plaintext env var) is forbidden on mainnet; "
                "use 'os_keyring' (or a KMS/systemd-creds backend) instead."
            )
        return self

    @model_validator(mode="after")
    def _validate_mainnet_disable_gate(self) -> Config:
        """Refuse the ungated HAARP test mode (``disable_gate``) on mainnet.

        ``disable_gate`` neutralizes the BTC macro cash-gate so HAARP builds
        positions even in a risk-off regime — a TESTNET functional-testing knob ONLY.
        Allowing it on mainnet would trade real capital against a deliberately
        un-protected signal.  Mirrors the mainnet ``passphrase_backend='env'`` refusal:
        a mis-configured deploy fails at load time, before any source is built.
        """
        if self.env == "mainnet" and self.disable_gate:
            raise ValueError(
                "disable_gate (ungated HAARP test mode) is forbidden on mainnet; it "
                "neutralizes the BTC macro cash-gate and is for TESTNET functional "
                "testing only. Set disable_gate=false (the default) on mainnet."
            )
        return self

    @model_validator(mode="after")
    def _apply_data_dir(self) -> Config:
        """Prefix RELATIVE state-file paths with ``data_dir`` (volume-mount support).

        When ``data_dir`` is set, every state-file path that does NOT start with
        ``"/"`` is rewritten to ``f"{data_dir.rstrip('/')}/{path}"``.  Absolute
        paths and ``None`` values pass through unchanged.  Composes cleanly with
        the other validators because pydantic v2 runs ``model_validator(mode="after")``
        decorators independently.
        """
        if self.data_dir is None:
            return self

        prefix = self.data_dir.rstrip("/")

        def _prefix(path: str) -> str:
            return path if path.startswith("/") else f"{prefix}/{path}"

        self.state_path = _prefix(self.state_path)
        self.nonce_state_path = _prefix(self.nonce_state_path)
        self.ledger_path = _prefix(self.ledger_path)
        self.audit_path = _prefix(self.audit_path)
        if self.kill_flag_path is not None:
            self.kill_flag_path = _prefix(self.kill_flag_path)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate configuration from a YAML file.

        Args:
            path: Absolute or relative path to the YAML config file.

        Returns:
            A fully-validated :class:`Config` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If any field fails validation.
            ValueError: If a cross-field constraint is violated (e.g. mainnet
                remote without a pinned key fingerprint).
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        raw: dict[str, Any] = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        return cls(**raw)

    def sleeves(self) -> list[SignalSourceConfig]:
        """Return the normalized sleeve list (the ONE accessor downstream uses).

        Always non-empty after validation: the singular ``signal_source`` form
        is materialized into a single ``"default"`` sleeve by
        :meth:`_normalize_signal_sources`, so both config forms present the same
        shape to the source factory / combination layer.
        """
        # _normalize_signal_sources guarantees signal_sources is populated.
        assert self.signal_sources is not None
        return self.signal_sources

    def effective_staleness_days(self, sleeve: SignalSourceConfig) -> int:
        """Resolve the effective max signal age for ``sleeve``.

        The per-sleeve :attr:`SignalSourceConfig.max_signal_staleness_days`
        wins when set; otherwise fall back to the global
        :attr:`max_signal_staleness_days`.
        """
        if sleeve.max_signal_staleness_days is not None:
            return sleeve.max_signal_staleness_days
        return self.max_signal_staleness_days

    def effective_pinned_model_rev(self, sleeve: SignalSourceConfig) -> str | None:
        """Resolve the effective ``model_rev`` pin for ``sleeve``.

        The per-sleeve :attr:`SignalSourceConfig.pinned_model_rev` wins when set;
        otherwise fall back to the global :attr:`pinned_model_rev`.  ``None`` at
        both levels means no pin (any served rev is accepted).  The singular
        ``"default"`` sleeve carries no per-sleeve pin, so it transparently
        inherits the top-level value — preserving today's behaviour.
        """
        if sleeve.pinned_model_rev is not None:
            return sleeve.pinned_model_rev
        return self.pinned_model_rev
