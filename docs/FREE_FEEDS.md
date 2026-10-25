# Free collection and its actual limits

GMGN is optional. Its API management page observed on 25 October 2026 states
that the free tier requires a minimum $100 balance, with assets refreshed
hourly. No deposit, personal API key or continued-access-after-withdrawal
experiment was performed. Proprietary wallet labels have no established
incremental net return in this bot.

| Question | Existing free path | What this cannot establish |
| --- | --- | --- |
| New Solana launches | Native program streams, public RPC fallbacks, LaunchLab decoder, GeckoTerminal and Dexscreener observations | Universal capture, pre-confirmation timing, or a paid feed entitlement |
| Prices and liquidity | Dexscreener pairs, native reserves/routes, GeckoTerminal pools, regional venue ladders | An indicative aggregator price is not a fill or a depth quote for our size |
| Holder and supply risk | Native mint/Token-2022 parsing, largest accounts and supply, public Rugcheck ladder | Missing native reads are not zero concentration or renounced authority |
| Wallet/creator intelligence | Observed trades, opening buyers, funding/actor graphs, creator episodes and public history | A wallet role or mention is not proven control, insider knowledge or profitable copyability |
| Global breadth | Provider network/DEX enumeration, cross-chain scout, venues, public news/social/onion discovery | Any finite provider index is incomplete; restricted endpoints remain blocked |
| Phantom perps | Hyperliquid public information, independent contract/collateral/book/funding records | The full underlying universe is not a verified Phantom UI listing or a complete paper position engine |
| Phantom World | Separate native discovery and market-identity research | Current firm quotes, complete settlement mapping and real fills remain unavailable |

## Expanding provider discovery

`free_market_discovery` runs keyless on the existing miner worker. It repeatedly
enumerates every accessible network page, admits all valid network identifiers,
and rotates new-pool, trending-pool and DEX pages for each. It does not keep a
fixed network allowlist or a lifetime network/page/record ceiling. Per-request
pacing, provider pagination refusals, outages and finite disk/compute remain.
At one page per 15 seconds, visiting hundreds of networks necessarily takes
time; this does not meet a complete first-second global launch census.

The source response is stored in a durable pending outbox before delivery.
The raw body and all normalized pool rows enter the existing canonical miner
archive before the cursor advances. Network/DEX discoveries then enter the
existing research frontier with explicit provider namespaces and **unverified
native mapping**. A restart replays an unacknowledged original response.
Provider refusals and malformed schemas are retained as gaps, not successful
end-of-history receipts. Foreign-chain rows cannot overwrite Solana curve or
market-regime state. Solana addresses are checked before adding observation
targets; no trading authority is created.

`ops_state_dir/free_market_discovery.sqlite` holds discovery cursors, current
network/DEX/pool views, page reference receipts and the pending outbox. Full
historical raw pages live in the canonical archive. Backup the SQLite database
using its online backup API (or stop the dedicated bot first); preserve the
canonical ledger, frontier and existing framework sidecars together. A future
restore must include this database to retain discovery progress.

The public APIs enforce their own quotas. Helius' published free plan is
1M credits/month and 10 requests/second; that is not unlimited 24/7 history or
firehose capacity. On the VPS, two direct holder/supply diagnostics returned
HTTP 429 on 25 October. This proves a refusal at that time, not its precise
quota cause. Existing provider cooldowns remain; no account/IP rotation is
used to evade them. A full batch of failed holder reads now reports failure
instead of a healthy empty result.

GDELT retains all matching rows in a fetched GKG archive by default. A 404
publication race re-reads the actual manifest once and follows a changed
published archive only. Unknown hosts, redirects and rate-limit refusals are
not followed as alternate access routes. No absent archive is fabricated.

References: [Dexscreener API](https://docs.dexscreener.com/api/reference),
[GeckoTerminal public API](https://api.geckoterminal.com/docs/index.html),
[Helius plans](https://www.helius.dev/pricing),
[GDELT GKG codebook](https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf).
