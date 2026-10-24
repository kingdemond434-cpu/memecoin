# Continuous discovery and independent method research — 24 October 2026

The operator requests expanding discovery, data/cost measurement and execution
research, without adopting the pasted 90-day schedule or new trading gates.
Existing execution authority is unchanged. A finite source list is a starting
catalogue, never proof of complete worldwide coverage.

## Implemented in this change

- Automatic public onion discovery uses a SQLite frontier and canonical outbox,
  replacing the 16/32-host admission ceiling, 512-pending ceiling, fixed page
  count and crawl-depth ceiling. Hosts, URLs and parent provenance survive
  restarts. New child URLs become fetchable only after canonical acknowledgement.
  One paced worker interleaves hosts and revisits; failures and access challenges
  defer work instead of pretending a successful observation. No DNS fallback,
  script execution, login or access-challenge bypass exists.
- Onion directories retain every valid link in each accepted document. Body and
  timeout limits constrain individual requests, not the lifetime frontier. Disk
  exhaustion remains an operational failure; unlimited admission is not infinite
  storage, throughput or an exhaustive onion index. Historical capacity debt is
  retained and cannot reconstruct previously discarded URLs.
- Public Telegram defaults have no candidate/verified-channel admission ceiling.
  Parsing retains all address/handle mentions in each fetched message. Candidate
  verification favors less-attempted channels, preventing an offline first batch
  from permanently hiding later discoveries. Atomic checkpoints retain the book.
- Data miners retain the full batch, deliver every record in yielding chunks,
  and retain explicit retry evidence if the consumer fails. The old per-call
  record count is now a work quantum rather than a discarded/deferred tail.
- Regional/global venue collection rotates across eligible endpoints even when
  the first works. All returned markets are kept by default, rather than only
  BTC/SOL or the first 400 rows. Prices retain venue/currency provenance; listing
  snapshots do not acquire invented prices. RSS ingestion keeps all supplied items.
- Public source/address indexing retains all validated mentions in a record,
  rather than silently stopping at address 100. A mention proves no wallet role,
  ownership, future launch or actionable inside information.
- Research-component updates now commit to independent atomic files before
  publishing the new in-memory value. They do not rewrite the complete growing
  hypothesis/trial framework on each data request. New sidecars overlay legacy
  research values at restore; they cannot replace capital or trial state.
  **Backup/restore must include `<framework.json>.components/` together with
  the framework JSON and existing population SQLite sidecar.** Corrupt component
  files fail restoration visibly. This addresses the observed full-framework
  serialization stall; per-component work and overall loop latency still need
  measurement on the VPS.

## Independent primary-source findings

1. **Fee auctions:** [Jito's documentation](https://docs.jito.wtf/lowlatencytxnsend/)
   publishes a tip-floor REST endpoint and WebSocket percentile stream. Existing
   native code already converts REST tip-floor values from SOL to lamports and
   tracks observed priority-fee percentiles. Extend timestamped cost/landing
   capture, not a second hard-coded fee constant. A tip percentile is not a
   probability of landing or a launch firehose. Provider request limits remain.
2. **Continuity:** [Helius LaserStream](https://www.helius.dev/laserstream)
   documents filtered streams and replay. A raw pre-execution transaction lacks
   execution metadata and is not a confirmed fill. Replay windows and paid-plan
   access must be checked against the actual endpoint; the API key alone does
   not establish those entitlements. No subscription is purchased here.
3. **Sellability:** [Solana simulateTransaction](https://solana.com/docs/rpc/http/simulatetransaction)
   supports unsigned simulation when signature verification is disabled, using
   current committed account state. Two separate simulations do not carry the
   first simulation's account writes into the second. A real round-trip probe
   requires an atomic buy/sell transaction or a stateful fork with exact accounts,
   raw amounts, routing, token program and context slot. Simulation is separate
   from observed execution and cannot guarantee future transferability.
4. **Token semantics:** [SPL's transfer-hook interface](https://www.solana-program.com/docs/transfer-hook-interface)
   permits arbitrary program logic during transfers. The destination already
   parses Token-2022 TLV extensions, including transfer hook, transfer fees,
   permanent delegate and non-transferable flags. The pasted claim that no reader
   exists was incorrect. Static parsing does not replace executable simulation.
5. **Promotion versus attention:** [DEX Screener's API](https://docs.dexscreener.com/api/reference)
   exposes profiles, ads, community takeovers and boosts, with documented request
   limits. [Boosts are purchased visibility](https://docs.dexscreener.com/boosting).
   Treat those as promotion/crowding observations, not organic demand or a census.
6. **Onion discovery:** [Ahmia's operator documentation](https://ahmia.fi/about/)
   describes its public index and an address export. Such indexes can supply
   candidates; they do not certify operator identity, completeness or insider
   truth. Restricted forums and unknown current Dread addresses remain unverified.
7. **Regional enumeration:** [Upbit's Korean reference](https://docs.upbit.com/kr/reference/list-trading-pairs)
   documents its all-market catalogue and `is_details` parameter. The collector's
   obsolete camel-case query spelling is corrected. Supported trading pairs are
   evidence of availability, not evidence of an earlier announcement or a
   pre-listing signal. Snapshot polling is not a pre-announcement feed.

## Method coverage rather than national stereotypes

Expand early-slot wallet coordination, funding ancestry, creator history,
sellability, price/depth/fee/landing measurement, regional venue changes, public
announcement timing, multilingual narratives and information decay across all
relevant regions. The existing actor store, native curve/tradeability math,
coordination detector, replay and cost machinery should be reused. Korean venue
changes are observations, not advance knowledge of a listing. Country/market
tax assumptions in the pasted essay are not adopted as trading rules.

Still required for stronger evidence: authorized native Phantom market mapping,
quotes/rules/settlement; measured first-minute stream completeness and recovery;
an exact stateful buy/sell simulation route; fill-bound markouts/TCA; expanding
source discovery beyond currently supported interfaces. This change does not
claim those are complete or that any measured trading edge exists.
