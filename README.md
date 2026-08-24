# Memecoin Shadow Research Desk

Solana-native launch research, prediction, risk, and execution simulation. The
repository defaults to **dry-run** and does not ship a trained production model.
It cannot promise a return, and a target such as 10% per day is not a realistic
or safe operating assumption.

## What is implemented

- Pump.fun bonding-curve and PumpSwap decoders derived from the official IDLs.
- Official-layout pool creation decoders for Raydium AMM v4/CPMM/CLMM, Meteora
  DLMM/Dynamic AMM, and Orca Whirlpools. Unknown instructions are ignored rather
  than inferred from account positions.
- Yellowstone gRPC with vendored official protobuf bindings, handshake
  validation and reconnection. Without Yellowstone, processed Solana WebSocket
  logs decode official Pump/PumpSwap program events directly; confirmed HTTP
  polling is used only while the socket is unavailable, preserving free quota.
- Native SPL and Token-2022 mint-authority, freeze-authority, extension,
  concentration, and Jupiter sell-route checks.
- Point-in-time launch episodes, immutable snapshot timestamps, observed price
  paths, route-feasible outcomes, P50X labels, crash-safe active checkpoints,
  and persistent outcome indices.
- Bounded per-token candidate pipelines re-evaluate at 0/1/3/5/10 seconds;
  quote observation runs separately so research I/O does not stall discovery.
- Budgeted Jupiter round-trip market marks collect price paths even while the
  prediction model is blocked; hourly research leads persist in a JSONL ledger.
- FIFO wallet round-trip scoring, public-chain coordination inference, public
  social/research discovery, creator genealogy, and continuous rug hazard.
- Nested probability correction, disjoint outcome bins, a route-feasible return
  head, net expected-log-wealth, risk-constrained Kelly sizing, live
  equity/SOL-USD inputs, and hard exposure limits.
- Champion/challenger registration, shadow-only research hypotheses, execution
  counterfactuals, persistent schema-bound promotion evidence, partial-exit cost
  basis, profit ratchets, and adaptive exits.
- Correct Solana `VersionedTransaction` signing plus distinct simulated,
  submitted, landed, and filled execution states. Landed wallet deltas—not
  instruction limits—drive token amounts and cost/PnL accounting.
- Block/receipt/decode timestamps and measured balance deltas are retained for
  latency, flow, and transaction-economics research.

Missing data is reported as `DATA_BLOCKED`; it is not replaced with a zero or a
made-up value. Discovery never grants execution authority. A model must first
pass chronological out-of-sample and forward-shadow promotion gates.

## Safety boundary

The Docker image, Compose stack, system service, and documented commands all
launch with `--dry-run`. Dry-run stops before swap construction, signing, Jito
submission, or RPC transaction submission. It uses an ephemeral paper wallet
and does not need a private key.

The source contains a separately gated live code path for audit/testing. It is
not enabled or deployed here. Do not add `SOLANA_PRIVATE_KEY` or
`ALLOW_LIVE_TRADING` to the shadow environment.

## Architecture

```text
official program streams / confirmed RPC fallback
                |
        native transaction decoders
                |
 SPL safety + wallet/social/public-chain intelligence
                |
       point-in-time launch episode lake
                |
 challenger prediction -> net E[log W] -> hard risk gate
                |
       dry execution + counterfactual policies
                |
 observed outcomes -> chronological promotion evidence
```

The MT5 quant should remain a separate service. Share research protocols and
approved experiment artifacts through a narrow versioned bridge; do not share
wallet keys, execution code, datasets, processes, or capital.

## Exact local dry-run commands

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m src.main --dry-run --smoke-test
.venv\Scripts\python.exe -m src.main --dry-run --run-seconds 60
```

Linux/macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.main --dry-run --smoke-test
.venv/bin/python -m src.main --dry-run --run-seconds 60
```

The networked run can use public Solana RPC. For timely and broader collection,
copy `.env.example` to `.env` and add data-provider credentials only.

## Isolated Docker shadow run

```bash
cp .env.example .env
docker compose -p memecoin-shadow build
docker compose -p memecoin-shadow up -d
curl http://127.0.0.1:18080/health
docker compose -p memecoin-shadow logs -f desk
```

The health port binds only to loopback. Persistent data lives in `data/`,
`models/`, and `logs/`; the rest of the container filesystem is read-only.

## VPS shadow deployment

`deploy.sh` installs only `/opt/memecoin-shadow` and the distinct
`memecoin-shadow.service`. It refuses an environment containing a wallet key or
live acknowledgement and does not stop or replace another quant service.

```bash
sudo ./deploy.sh
systemctl status memecoin-shadow
curl http://127.0.0.1:18080/status
journalctl -u memecoin-shadow -f
```

## Data credentials and honest blockers

- `YELLOWSTONE_GRPC_URL` and token: lowest-latency program stream. Without it,
  confirmed RPC polling is slower and provider-rate-limited.
- `HELIUS_API_KEY`: enhanced address transaction history for wallet round trips.
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and comma-separated
  `TELEGRAM_CHANNELS`: read-only monitoring of channels that the dedicated
  Telegram research account can access. Authorize its persistent session once
  with `.venv/bin/python -m src.research.telegram_authorize`; the collector
  never performs global Telegram search.
- `YOUTUBE_API_KEY`: official YouTube Data API monitoring for recent Solana
  launch/rug research videos. Contract strings are evidence signals, never
  endorsements or automatic live orders.
- `X_BEARER_TOKEN`: optional official recent-search and public-account source.
  No unofficial browser/session scraping fallback is included.
- `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`: optional approved Reddit
  application-only OAuth source. Until Reddit approves and issues both values,
  its status remains explicitly `DATA_BLOCKED`.
- `GITHUB_TOKEN`: optional higher public-research API quota. Only repositories
  with an identified compatible license become research leads; all leads remain
  challengers pending evidence.
- A versioned, chronologically validated model bundle is still required before
  prediction status changes from `DATA_BLOCKED`. No constant-score fallback is
  used.
- PumpSwap and the supported Raydium/Meteora/Orca pool-creation instructions use
  native layouts and official program IDs. Pump has a reduced real-mainnet replay
  fixture; equivalent sanitized mainnet fixtures for every AMM remain an evidence
  task and are not represented as complete validation.

## Tests

The suite covers dry-run non-submission, the independent live lock,
VersionedTransaction signatures, native mint checks, nested P2/P5/P10/P50 math,
risk-constrained sizing, partial cost basis, chain-aware RPC health, PumpSwap and
Raydium/Meteora/Orca layouts, FIFO wallet accounting, non-fabricated wallet-regime
labels, point-in-time leakage, counterfactuals,
public coordination, rug hazard, active-episode recovery, promotion-state
recovery, official social-source blocking/ingest, and landed wallet-delta accounting. A reduced fixture from Solana mainnet slot
`441417557` exercises the same Pump.fun inner-instruction decoder used live.

On a Dockerless VPS, `memecoin-shadow-user.service` runs the collector as an
isolated user service. `memecoin-shadow-train.timer` invokes the strict
chronological trainer every six hours. Insufficient samples, class coverage, or
OOS E[log W] remain `DATA_BLOCKED`/`REJECTED`; only passed artifacts are loaded
into forward dry-run shadow evaluation.

## Upstream specifications

- [Pump.fun public IDLs](https://github.com/pump-fun/pump-public-docs)
- [Yellowstone gRPC protobuf](https://github.com/rpcpool/yellowstone-grpc/tree/master/yellowstone-grpc-proto)
- [Risk-constrained Kelly gambling](https://arxiv.org/abs/1603.06183)

Repository licensing has not been specified by the owner. Public research
mining does not change that and does not authorize copying incompatible code.
