# Collection responsiveness and native probe diagnostics

Neural intake now persists the canonical narrative and its neural queue entry
on a worker thread. Claim-link indexing and desk-state mutations remain on
their owner loop. Queue reconciliation, selection, cache reads and atomic
annotation/cache/service-ticket commits also run off the async collection
loop. Cancellation drains initiated database operations before resources may
close. An annotation still requires the existing canonical acknowledgement
before consumers can read it; neither cancellation nor cache reuse grants one.

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
