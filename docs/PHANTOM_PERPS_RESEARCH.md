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
- Rotate book observations over all retained active contracts. The provider
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
It stores all observed catalogs/books and funding history without a lifetime
market/source ceiling. Polling quanta limit work per pass, not the universe.

## Cost scenarios, not manufactured trades

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

These observations do not open positions, manufacture strategy fills, transfer
collateral, fit a directional model or certify P&L/edge. Both real and simulated
strategy fill counts remain zero. Margin/liquidation execution, authorized
account fees and a complete perpetuals paper position engine remain separate
work. World native prediction-market records remain separate and unchanged.

## Primary protocol references

- [Hyperliquid public information endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
- [Perpetual metadata, DEXes and funding](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)
- [Phantom product mechanics](https://help.phantom.com/articles/understanding-perp-trading-in-phantom-52933464501907)

An ordinary unauthenticated VPS request to `perpDexs` returned HTTP 200 on
25 October 2026: native plus 10 named DEXes. That is a point-in-time inventory,
not an exhaustive future market claim or a proof that Phantom lists them all.
