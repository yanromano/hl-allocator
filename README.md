# hl-allocator

Self-hosted execution engine for a signed Hyperliquid allocation signal. It reconciles
**your own** Hyperliquid account to a daily target allocation, with bounded retry,
1x-isolated leverage enforcement, and position-health monitoring. **No strategy code** —
it consumes a signed signal you subscribe to.

## Deploy to Railway (24/7, any OS)

1. **Generate a trade-only agent** in the Hyperliquid UI (API wallet). Copy its private
   key — it can place/cancel orders but **cannot withdraw or transfer**.
2. **Deploy on Railway** (after the repo exists, add the Deploy-on-Railway button
   here; or use `railway up` from a checkout). Railway builds the Dockerfile.
3. **Set env vars** (Railway → Variables): `HL_AGENT_KEY` (the agent key),
   `HL_SIGNAL_BEARER` (your signal token), `HL_SIGNAL_ROOT_FP` (the pinned root
   fingerprint we give you).
4. **Add a volume** mounted at `/data` — Railway supports one volume per service; it
   persists nonce/ledger/state across restarts AND holds your config. Note: Railway
   volumes are root-owned. The simple path is to set the service variable
   `RAILWAY_RUN_UID=0`; alternatively keep the default non-root user, but only if you
   chown the volume first.
5. **Place your config** at `/data/automation.yaml`: copy `automation.example.yaml`,
   fill in your account address + signal `url`/`client_id`, then write it onto the
   volume (e.g. `railway ssh` into the service and paste it, or use a one-off command).
6. Railway runs the daemon 24/7; `/healthz` drives liveness.

## Tools

| Command | Purpose |
|---|---|
| `hl-allocator` | the trading daemon (`--config`, `--once`, `--once --dry-run`) |
| `hl-leverage-guard --config X [--check]` | cap over-target leverage in place (read-only with `--check`) |
| `hl-health --config X` | read-only position-health report + alerts |
| `touch /data/KILL` | kill switch: reduce-only flatten + HALT (remove to resume) |

## Security model

- The agent key is **trade-only** (bounded blast radius). You custody it on your own
  Railway; we never see it.
- The signal is **Ed25519-signed**; the engine verifies it against a pinned root key.
