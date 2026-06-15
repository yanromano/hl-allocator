# hl-allocator-testnet — deploy runbook (remaining steps)

Prepared 2026-06-10. Project `hl-allocator-testnet` (c343b241) exists with:
service `hl-allocator-testnet`, volume at `/data`, vars `HL_SIGNAL_BEARER`,
`HL_SIGNAL_ROOT_FP=13543d381448db13`, `RAILWAY_RUN_UID=0`. The signal server is
live and the consumer path was validated by a local `--once --dry-run`
(allocation accepted; flatten plan computed). Config ready: `automation.railway.yaml`.

## Blocked on operator
1. Generate a **NEW testnet agent key** in the Hyperliquid UI (More → API wallet),
   approved on master `0xcD0D...9aD4`. NEVER reuse the local agent key
   (nonce single-signer invariant). Note the expiry timestamp.

## Then (from ~/repos/hl-allocator-deploy — railway link is per-directory)
```bash
# 1. update agent_valid_until_ms in automation.railway.yaml with the real expiry
# 2. set the key (paste; never commit):
railway variables --service hl-allocator-testnet --set "HL_AGENT_KEY=0x..." --skip-deploys
# 3. first deploy:
railway up --service hl-allocator-testnet --detach
# 4. place the config on the volume — the container crash-loops until it exists;
#    use railway ssh during an up window (or temporarily set the service start
#    command to `sleep 3600` in the Railway UI, ssh, then clear it):
railway ssh --service hl-allocator-testnet -- bash -c 'cat > /data/automation.yaml' < automation.railway.yaml
# 5. redeploy/restart and watch:
railway redeploy --service hl-allocator-testnet -y
railway logs --service hl-allocator-testnet   # expect: key loaded, healthz up, idle until 00:05 UTC
```

## First-cycle expectations (next 00:03 UTC)
- Daemon fetches the signal (server publishes ~00:02), 9 checks + model_rev pin
  (-NOGATE-TEST) pass.
- The server runs UNGATED (HAARP_DISABLE_GATE=true on the service) so the
  testnet E2E actually trades. Current NOGATE allocation: HYPE ~60.6% / cash.
  First cycle: SELL TON/SUI/BNB (stale 06-08 positions) + BUY HYPE
  (validated by local --once --dry-run on 2026-06-11: 4 orders, exact sizes).
- Watch: `railway logs` for `allocation accepted`, order results, committed=True.
- Kill switch: `railway ssh ... -- touch /data/KILL` (flatten + HALT).
