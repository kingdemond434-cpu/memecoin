# Runtime profile and proposal admission

The deployed paper desk at `4eebbf159b1375c5b2eec11a5aabd94b2cbcb7c5`
was sampled for 45 seconds using py-spy 0.4.2, GIL-only, nonblocking mode.
Only the process owning loopback port 18081 was selected, after checking its
designated `src.main`/production-paper configuration command line. No local
variables, credentials or unrelated processes were captured.

The profiler wheel came from PyPI with SHA-256
`8b06a353c177677e4e1701b288d8c58e2f8d4208ee81a8048d9f72ba800918f8`;
the extracted executable SHA-256 was
`6e83ab095f40453742d01a2719a178f308534015da519ec0659d2584139dc995`.
The retained receipt is `profile-main-4eebbf159b13-20261025.json` on the VPS.

There were 2,606 successful samples and 1,317 sampling errors. Nonblocking
sampling and GIL filtering limit inference; this is not a complete CPU census.
The `asyncio_4` worker had 1,476 samples beneath proposal `_previous`, including
large-prefix canonical JSON parsing and verification. Moving that work to a
thread had not removed its competition for the interpreter lock. No model
provider was locally configured to produce a proposal.

The engine now checks the existing no-network provider configuration report
before allocating a new round. With no locally configured/custom provider it
returns DATA_BLOCKED, no network requests and no evidence scan. It does not
invent a completed round. One configured provider is enough to keep running;
other missing required providers do not suppress useful partial research.
Actual rounds retain the unchanged full-prefix integrity, sealed retry,
point-in-time input and no-trading-authority contracts.

This avoids measured futile work. Its live latency benefit requires a new
deployment/profile; no speedup is inferred from the software checks alone.
Other sampled work includes the main hazard sweep, wallet recomputation and
dataset flush. They remain candidates for further measurement and optimization.

## Attribution throughput

The live orphan outbox was growing while the initial mapper read only eight
pools per pass. It now packs up to Solana's documented 100-account protocol
maximum into each `getMultipleAccounts` call, and deduplicates base-mint reads.
RPC manager admission, provider quotas, pacing and refusal reporting remain.
The next queued pool is retained; this is batching rather than dropping work
or asserting unlimited provider capacity. A regression checks 101 pending
pools across two batches and canonical/delivery acknowledgement separation.

A live public-provider probe then returned JSON-RPC -32602 with an explicit
maximum of 10 accounts. Recovery now learns and persists that stricter bound,
retries only the permitted prefix and leaves the rest queued. Capability is
re-probed hourly so a later endpoint upgrade can expand it. Only explicit
request-shape feedback is used; 429/cooldowns never trigger split retries.
This is an example of a protocol maximum differing from free-provider access.

Protocol reference: https://solana.com/docs/rpc/http/getmultipleaccounts
