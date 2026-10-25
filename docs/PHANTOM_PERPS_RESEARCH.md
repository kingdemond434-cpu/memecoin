# Phantom perpetuals: separate public market research

Phantom's [current support documentation](https://help.phantom.com/articles/42292249388307)
identifies Hyperliquid as its underlying perpetuals venue. Contracts are
exposure with separate USDC collateral; spot token swaps and World prediction
outcomes are different products. Phantom also documents regional restrictions;
this research collector does not establish the owner's live eligibility.

The collector only calls Hyperliquid's public information endpoint. It has no
wallet/account queries, signatures, exchange/order methods, deposits or bridge
actions. This is not a live Phantom trading integration.

## What is collected

- Discover the full returned `perpDexs` inventory repeatedly; rotate through
  each DEX's `metaAndAssetCtxs`, including new DEXes without a fixed allowlist.
- Retain exact contract names, delisting flags, precision, leverage metadata,
  margin tables, collateral token IDs, mark/oracle prices, current funding and
  open interest. Resolve collateral identities against the public spot metadata.
- Poll book observations over all retained active contracts. The provider
  returns at most 20 levels per side; displayed depth is not unlimited depth.
- Backfill provider-available funding history using a per-contract durable
  millisecond cursor. Advance only after the canonical miner batch is retained
  and delivered; replay is idempotent. No assertion of complete backfill.

Provider inventory exceeds Phantom's proven UI listings. Crypto/native and
HIP-3 contracts retain their underlying namespace. No new DEX or symbol is
silently certified as available in Phantom. Phantom specifically describes
Trade[XYZ] equity perps; that does not establish every other DEX's support.

The derived SQLite view is `ops_state_dir/phantom_perps.sqlite`; original
observations pass through the existing retained miner-batch ledger. Back up
this SQLite database with SQLite's backup API along with the canonical ledger.
SQLite keeps the latest catalog/book view per observed contract and all
retained funding rows without a lifetime market/source ceiling. Every raw
catalog/book/funding response is retained in the canonical miner-batch archive;
SQLite does not duplicate that entire snapshot history. Product indexing now
runs on the miner worker immediately after canonical retention, before the
bounded main-loop handoff queue. A full queue cannot discard perps indexing,
and the hot-loop product dispatch does no perps SQLite writes. Batch references
are idempotent; malformed records have an explicit `ingestion_errors` receipt
with their canonical reference/index and do not discard valid sibling books.
Transport/database failures still fail the transaction. Retained raw evidence
is the authority for rebuilding rejected derived records.

Public books use four concurrent HTTP slots and a batch sized from the active
universe to aim for a one-minute revisit. This is an objective, not a guarantee:
the global miner concurrency, network, source quotas and retained-work cost can
make it slower. Durable least-recent-attempted scheduling survives restarts and
continues past a failing contract. Successful responses survive another
request's failure in the same pass. Catalog/collateral/book snapshots cannot
rewind when older responses are delivered. Conflicting or future funding rows
cannot advance a history cursor.

All three public perps miners share a durable rolling request-weight budget
and provider cooldown. The collector reserves a cooperative 900 weight/minute
share below Hyperliquid's [documented 1200/minute per-IP limit](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits),
including response-dependent funding weight. Other IP consumers are unknown;
a 429 pauses every local public perps method. These service quanta and provider
quotas do not impose a lifetime market/source ceiling. Status reports fresh
active books separately from historical book count, recent request errors,
derived-record errors, and canonical batches indexed. Zero fresh books remains
visible even when many historical snapshots exist.
The status count `observed_catalog_dexes` counts DEX groups with active observed
contracts. Empty/delisted-only catalog responses remain in the `catalogs`
table and raw archive; that active count alone cannot prove every catalog was
visited or that collection is complete.

## Static costs and causal forward controls

Each valid recent book produces a Decimal-based depth walk for a minimum-size
notional probe, with quantity rounded to the observed contract precision.
Partial depth, crossed/unordered books and stale provider timestamps are not
treated as executable capacity. The diagnostic separately reports spread/depth
cost and an explicitly labeled fee scenario using Phantom's published 9.5bps
baseline per leg. Account tiers and HIP-3 deployer fees are not verified by this
baseline. Future funding, latency, dynamic market impact and liquidation are
explicit missing components.
USDC cost scenarios require observed canonical USDC collateral metadata. Other
collateral IDs remain explicit and blocked for that scenario rather than
assigned an invented 1:1 USDC conversion.

The [perps paper ledger](PERPS_PAPER_LEDGER.md) now opens explicitly synthetic
long/short execution controls using only future observed books. It accounts for
position cost, fees, partial IOC fills, margin, hourly funding liabilities and
observed liquidation/backstop scenarios. `simulated_control_fills` is separate
from real fills and validated strategy fills, which remain zero. These are
independent test accounts, not a pooled portfolio or a demonstrated trading
edge. Authorized account fees and exact billing oracles remain unverified.
World native prediction-market records remain separate and unchanged.

Public keyed WebSocket snapshots supplement REST books/catalogs. One bounded
connection per pass samples `fastAssetCtxs` and a fair rotating group of
`activeAssetCtx` contracts. Frames enter the canonical archive before paper
state. Missing fields never refresh another value's clock, and mark-only
frames never manufacture fresh oracle prices. Connection/subscription budgets
survive restart. This is sampled receipt-time evidence, not continuous
exchange event-time capture or an executable quotation.

## Primary protocol references

- [Hyperliquid public information endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
- [Perpetual metadata, DEXes and funding](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)
- [Phantom product mechanics](https://help.phantom.com/articles/understanding-perp-trading-in-phantom-52933464501907)

An ordinary unauthenticated VPS request to `perpDexs` returned HTTP 200 on
25 October 2026: native plus 10 named DEXes. That is a point-in-time inventory,
not an exhaustive future market claim or a proof that Phantom lists them all.
