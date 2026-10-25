# Integration repairs, 25 October 2026

## Failures addressed

- The live miner scheduler awaited every due source as one group. A long
  holder/social call prevented a finished, unrelated source from being polled
  again. Owned per-source tasks now keep independent clocks under the existing
  bounded shared concurrency. The same miner cannot overlap itself through a
  workflow acquisition. Shutdown cancels and drains the owned source tasks.
- Perps SQLite indexing ran on the hot spot/World loop, after an evictable
  queue. It now happens after canonical archive acknowledgement on the miner
  worker, with idempotent batch references and audited per-record parse failures.
- One book every five seconds implied about 28.5 minutes for 342 contracts
  before network/scheduler delays. Adaptive fair batches aim for a one-minute
  universe revisit with shared provider weight and cooldown accounting. No
  measured timing improvement is claimed until deployment observation.
- An older catalog, collateral identity or provider book could replace a newer
  snapshot. Snapshot clocks now prevent rewind; funding conflicts/future rows
  cannot silently advance a cursor.
- Watched-wallet balances always requested up to 25 accounts even when a free
  RPC explicitly accepted only ten. Only a specific JSON-RPC -32602 account
  maximum permits splitting; HTTP/rate/auth failures do not. Every chunk must
  preserve cardinality, raw integer lamports and its own observed context slot.
  A periodic larger probe can release a past bound. This path does not pretend
  multiple RPC contexts describe one atomic snapshot.

## Boundaries still present

The current [Phantom support page](https://help.phantom.com/articles/trade-predictions-in-phantom-47187740851731)
states that the World prediction feature has no public API for external
development. The public World website loads anonymous market information;
that does not grant the VPS an authenticated API or firm executable quotes.
No browser tokens, Turnstile responses, account credentials or region bypass
are imported by these repairs. World research stays separate from Hyperliquid
perps and spot swaps. No direct Kalshi substitution is introduced.

These changes do not add a perpetuals position engine, certify Phantom's
listing of every underlying contract, verify account-specific fees, create an
entry model that passed validation, or unlock live submission. Public feed
rate limits, missing credentials and rejected models remain explicit.

## Verification

Focused tests cover canonical-first indexing and failure, replay, partial HTTP
failures, restart scheduling, shared cooldown and weight exhaustion, snapshot
ordering, funding conflict/future rejection, account alignment/raw units,
independent source clocks, overlap prevention and cancellation/drain. Full
Linux/Windows CI and isolated VPS results belong to the exact tested commit's
receipts; they do not establish forward trading profitability.
