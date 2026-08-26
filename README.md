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
- Pump/PumpSwap program events populate first-seconds wallet flow, SOL notional,
  pool-reserve price multiples, and immediate PIT episodes without per-trade
  HTTP requests. Jupiter marks remain the independent executable-route check.
- Native SPL and Token-2022 mint-authority, freeze-authority, extension,
  concentration, and Jupiter sell-route checks.
- Point-in-time launch episodes, immutable snapshot timestamps, observed price
  paths, route-feasible outcomes, P50X labels, crash-safe active checkpoints,
  and persistent outcome indices.
- Bounded per-token candidate pipelines re-evaluate at 0/1/3/5/10 seconds;
  quote observation runs separately so research I/O does not stall discovery.
- Budgeted Jupiter round-trip market marks collect price paths even while the
  prediction model is blocked; hourly research leads persist in a JSONL ledger.
- FIFO wallet round-trip scoring with launch-relative regime classification
  sourced from real detected launch timestamps (never fabricated timing),
  public-chain coordination inference, public social/research discovery,
  creator genealogy, and continuous rug hazard with a chronologically
  validated calibration layer over the leakage-free half of its signals.
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

## Source mesh: verifying endpoints on the node

`config/sources.yaml` is a SEED. Every declaration in it names a lawful public
interface, but a declaration that names an endpoint nobody has checked is worse
than no declaration -- the mesh reports it DEAD, an operator assumes it needs a
key, and the coverage number stays wrong in the flattering direction.

So endpoints are verified on the node the desk runs on, not in a repository:

```bash
.venv/bin/python tools/verify_sources.py                          # what answers here
.venv/bin/python tools/verify_sources.py --out config/sources.verified.yaml
```

The loader reads `config/sources.yaml,config/sources.verified.yaml` and the
overlay WINS on any id it names, so the seed stays under version control and
the verified endpoints stay specific to the host. A sandbox with an egress
allowlist reports almost everything unreachable, and that verdict is about the
sandbox rather than the endpoints -- which is exactly why this file is not
committed from wherever the code happened to be written.

`GET /status` reports the transport layer under `source_mesh.transports`, with
the three reasons a declaration has no transport kept apart, because they have
three different owners:

- `pending_endpoint` -- declared coverage, no endpoint chosen yet (research)
- `unconfigured` -- the credential named in `requires_env` is absent (operator)
- `unsupported` -- no transport exists for that kind (ours)

Transports that need no credential and work out of the box: RSS/Atom (the whole
regional long tail), YouTube per-channel feeds, Mastodon public timelines,
Bluesky Jetstream, Nostr relays, public GitHub repository activity, and official
page change detection. Telegram uses the operator's own registered API
credentials against public channels. Twitch, Discord and token metadata are
push-fed queues: something else in the process produces the records.

## Entity registry: provenance is required

`config/entities.yaml` is EMPTY and stays that way until entries are verified.
An entity declared there asserts that a specific account, domain or wallet
canonically IS a named person or organisation, and a wrong entry does not
degrade gracefully -- it makes an impersonator look verified, which is the most
expensive error this system can make.

So `verified_from` (where the fact was read) and `verified_at` (when) are
required fields. An entry without both is REFUSED at load, not loaded with a
warning: a flag on a record that still confers OFFICIAL_DOMAIN proof is not a
control. A verification older than 180 days is treated as a claim about the
past -- accounts get renamed, sold and abandoned -- so a stale entity keeps
NAME_ONLY and loses every level that authorises a position.

Fill it from what the entity itself publishes, not from memory:

```bash
.venv/bin/python tools/verify_entities.py --domain example.org     --id example-org --name "Example Organisation"     --out config/entities.verified.yaml
```

It fetches the domain's own pages, extracts the profile links those pages
publish, and records each page's URL and content hash. It emits the handles as
COMMENTS: `accounts` holds stable platform ids, and a display handle there
would let a renamed or resold account keep an entity's proof level. Resolving
each handle to its numeric id, and confirming any wallet claim, is a person's
job and deliberately not automated.

`GET /status` reports the registry under `entity_registry`, including which
entities have gone stale, so an empty or ageing registry is visible rather than
silent.

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
  used. The same applies to `ContinuousRugHazardModel`'s calibration layer: it
  runs on its uncalibrated heuristic score until `src.research.hazard_trainer`
  produces a chronologically passed artifact.
- PumpSwap and the supported Raydium/Meteora/Orca pool-creation instructions use
  native layouts and official program IDs. Every Anchor instruction
  discriminator in the decoder (Pump, PumpSwap, Raydium AMM v4/CPMM/CLMM,
  Meteora DLMM/Dynamic AMM, Orca v1/v2) has been independently recomputed from
  its real snake_case instruction name and matches exactly; the CPMM, CLMM, and
  Orca v1 account orderings were checked line-by-line against the current
  official on-chain program source and match exactly. Pump has a reduced
  real-mainnet replay fixture; literal captured-transaction-byte fixtures for
  the other AMMs remain an evidence task -- this environment had no outbound
  access to Solana RPC to capture them, so real (not fabricated) mainnet
  fixtures for those decoders are still not represented as complete
  validation, only as layout-verified.
- Wallet-history regime classification only fires for a token whose launch
  this desk actually observed (`GenealogyGraph.token_launch_times`, populated
  from real detected `pool_created` events) and only for the two regimes pure
  entry timing can honestly support (`ULTRA_EARLY` within 15s, `EARLY_CURVE`
  within 120s). Every other regime, and every trade against a token with no
  known launch time, stays unclassified rather than guessed.

## Tests

The suite covers dry-run non-submission, the independent live lock,
VersionedTransaction signatures, native mint checks, nested P2/P5/P10/P50 math,
risk-constrained sizing, partial-exit cost basis and realized PnL, chain-aware
RPC health, PumpSwap and Raydium/Meteora/Orca layouts (account ordering and
Anchor instruction discriminators independently verified against the current
official on-chain program source), FIFO wallet accounting, launch-relative
wallet-regime classification and its refusal to guess when no real launch
timestamp is known, point-in-time leakage, counterfactuals, public
coordination, rug hazard and its chronological calibration trainer, the full
champion/challenger promotion pipeline end to end (discovered through
shadow/canary to live, plus retirement and decay), distinct Jito
bundle/raw-submission confirmation states (filled, landed-without-fill,
timeout, rejected), active-episode recovery, promotion-state recovery,
official social-source blocking/ingest including read-only Telegram
collection, and landed wallet-delta accounting. A reduced fixture from Solana
mainnet slot `441417557` exercises the same Pump.fun inner-instruction decoder
used live.

On a Dockerless VPS, `memecoin-shadow-user.service` runs the collector as an
isolated user service. `memecoin-shadow-train.timer` invokes the strict
chronological multi-head trainer and the rug-hazard calibration trainer every
six hours. Insufficient samples, class coverage, or OOS E[log W] remain
`DATA_BLOCKED`/`REJECTED`; only passed artifacts are loaded into forward
dry-run shadow evaluation. The hazard calibration trainer only fits on the
leakage-free half of the hazard signal set (trade flow, liquidity, route,
concentration, social velocity, and explicit event tags) -- wallet-reputation
signals are excluded from replay because they are live state, never
point-in-time snapshotted per episode, and calibrating against them would
leak information the model would not have had at that moment.

## Upstream specifications

- [Pump.fun public IDLs](https://github.com/pump-fun/pump-public-docs)
- [Yellowstone gRPC protobuf](https://github.com/rpcpool/yellowstone-grpc/tree/master/yellowstone-grpc-proto)
- [Risk-constrained Kelly gambling](https://arxiv.org/abs/1603.06183)

Repository licensing has not been specified by the owner. Public research
mining does not change that and does not authorize copying incompatible code.
