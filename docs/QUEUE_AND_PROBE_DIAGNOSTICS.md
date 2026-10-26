# Collection responsiveness and native probe diagnostics

Neural intake now persists the canonical narrative and its neural queue entry
on a worker thread. Claim-link indexing and desk-state mutations remain on
their owner loop. Queue reconciliation, selection, cache reads and atomic
annotation/cache/service-ticket commits also run off the async collection
loop. Cancellation drains initiated database operations before resources may
close. An annotation still requires the existing canonical acknowledgement
before consumers can read it; neither cancellation nor cache reuse grants one.

The owner-loop observation lookup uses a separate read-only SQLite connection
and lock. WAL snapshots let it read committed ACKed rows while the queue writer
holds a transaction or the reporting lock. Uncommitted ACKs remain invisible,
and availability-time filtering still applies. This removes that shared Python
lock from delivery; disk I/O, process scheduling and other owner-loop work can
still delay a request. The reader closes with its component.

Queue reporting has its own read-only connection and lock as well. A single
input grouping replaces separate totals, oldest-pending and lane scans, and
an explicit read transaction keeps all database counters on the same committed
snapshot. Reporting no longer holds the intake/commit or observation Python
locks. It still scans the retained input index, so it is not constant-time.
The report includes read-start and elapsed timing; these are measurements,
not a guarantee of processing capacity or zero backlog. All three connections
close with the component.

`neural_sentiment.last_fetch_profile` separates disk/lock wait, inference await,
total fetch elapsed time and returned windows. Inference await includes pipe
and scheduling delay; it is not pure ONNX compute latency. The model weights,
inference function and semantic cache identity remain unchanged. Existing raw
inputs, pending annotations and replay checkpoints are retained. This change
does not establish that arrival rate is below processing capacity.

Native probe failures now retain the failed phase and safe reason codes.
RPC exhaustion distinguishes no eligible endpoint, provider refusal and
configured allocation exhaustion. HTTP status and numeric JSON-RPC error
codes can be retained without provider URLs or response messages. Previously
recorded outcomes remain unchanged and report `LEGACY_UNCLASSIFIED` when they
lack a reason code. Unknowns remain unknown, not proven honeypots.

An observed migrated DBC curve gets a 3600-second recheck instead of the normal
300-second cadence. Its target, evidence, outcome and coverage denominator
remain retained. Fresh and oldest untried targets continue sharing service
with due retries and least-serviced venues. Migration does not prove a token
cannot trade on its successor venue; successor discovery is separate work.

These changes have no wallet signer, submission or live trading authority.
