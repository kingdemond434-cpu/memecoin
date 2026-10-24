# Polling breadth and growing-frontier cost

Follow-up live inspection after the continuous-discovery rollout found two
remaining problems: upstream selectors still truncated monitored wallets,
deployers and channel-harvest records, and the main event loop still blocked
inside the growing SQLite research frontier.

## Selection

All discovered tracked-wallet/deployer/pool/name candidates are exposed to
their miners. Per-request work quanta now rotate by next identity instead of
repeatedly taking a fixed prefix. Balance, deployer history, LP supply,
priority-price context and token-safety enrichment can reach later targets.
The full channel-harvest batch is processed. Empty inputs, duplicates and
prepended discoveries do not silently reset a stable target set to its first
few names. Rotation positions are process-local; restart and rapidly expiring
targets remain coverage limitations. Source/raw evidence retention is separate.

The operator-published [Ahmia address inventory](https://ahmia.fi/onions/)
is also added as a refreshing discovery input. Its documented export was
observed through [the operator's about page](https://ahmia.fi/about/).
Directory input supports an explicitly selected 4 MB document while retaining
every valid address in it, with set-based deduplication instead of quadratic
list scanning. A single paced disk-backed frontier handles the resulting
candidate backlog. Index entries are unverified; they are not financial
claims, successful fetches, operator identity proofs or evidence of an edge.

## Frontier persistence and queries

A live stack sample showed `pending_work` scanning the research queue while
the miner worker was merging a growing JSON array of coordinate evidence refs.
The separate framework snapshot measured about 9 MB. The new live stack showed
frontier array merges and pending-work queries, whose costs also grow with
retained history.

Coordinate reference edges now use `coordinate_refs(coordinate_id, ref)` with a
unique key. Each legacy array migrates once, in the same transaction as its
next update; the public `coordinate()` reader merges both representations.
No references are capped or discarded, and observations do not reset TESTED
or hibernated state. Readers should use that API rather than assuming the
legacy JSON column contains the complete reference set. Backups include the
whole SQLite database; downgrade requires a compatible reader or deliberate
reference export, not an older binary reading only the legacy JSON column.

Pending-work queries now use indexed equality seeks for each requested
kind/state and merge only bounded candidate IDs. They hydrate jobs, axes and
tags over one connection instead of opening a connection for every result.
Global cursor order, kinds, states and enabling-change filtering are preserved.
An offline regression exercises a small relevant cursor over 12,000 unrelated
active rows under a SQLite instruction budget.

These are collection/scheduling changes. They add no trading-test schedules,
new capital gates, signing capability or paid data subscription. They do not
prove complete launch coverage or any profitable trading edge.
