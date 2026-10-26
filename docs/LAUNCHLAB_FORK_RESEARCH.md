# LaunchLab sampled-bank sellability

Decoded WSOL LaunchLab launches enter the durable local fork-probe queue after
their canonical append and before episode enrichment. Polling observed episode
mint/pool/quote identities remains a fallback. Source hashes, original recorded
availability and intake time are retained separately from target identity;
repeat intake does not reset arrival or retry state. Failed canonical appends
stop downstream launch admission. Queue-write failures increment the stream
intake-failure counter while the raw journal remains available for recovery.
This handoff does not yet backfill all earlier native journal rows.

The adapter reads the actual pool,
global and platform configuration, vaults, mints and executable images. The
final frame must come from one finalized bank; bootstrap reads do not substitute
for that frame. Its complete archive is hashed before the isolated VM executes.

The reviewed fixed pool layout is 429 bytes, including its discriminator.
Pool and vault PDA derivations, authority bump, program owners, mint identities,
decimals, funding status and vault authority are checked. The supported subset
is SPL or Token-2022 base with legacy WSOL quote. Token-2022 mint TLV is parsed;
transfer-hook accounts remain unsupported rather than being guessed. Migrated
pools, other quote assets and unreviewed layouts remain unknown. A successful
probe does not clear static authority/delegate or other existing risk checks.

Exact-input buy uses virtual WSOL; sell consumes the actual base-token balance
received in the same modified VM. Minimum output one is a local transferability
experiment, not live slippage authorization or a competitive execution quote.
The actual copied program computes curve shape, fees and transfer amounts.
Neither a constant-product approximation nor fabricated reserves are used.
Failures retain their receipts and are not automatically labeled honeypots.

The existing canonical ACK and point-in-time availability contract applies.
Consumers cannot read a result before its linked native sources were available
and ingested, including after a host wall-clock correction.
There is no real wallet, transaction submission, order authority or promotion
write. Bundled SVM feature/loader parity with the sampled validator remains
unverified, and a sampled successful sell cannot guarantee future transferability.

## Reviewed sources

The vendored IDL and its Apache provenance remain in `DONOR_ADAPTATIONS.md`.
The official Raydium SDK revision
`e2e8bcd865189c994682a9477ec8eaf1e2da8aaf` was independently cross-checked:

| Source under `src/raydium/launchpad/` | SHA-256 |
| --- | --- |
| `instrument.ts` | `177d418defd36a173d43fafb29a06019a8754c7abe599365885d0b46c1fd8274` |
| `pda.ts` | `bfa801c13f842c374e27d3e34007a75520094cbb71a82ca24c69882415932522` |
| `layout.ts` | `34b18f1cdb827a586d5f350cd44022a2115f60a4fccac059481ad07ad711e146` |

[Official instruction definitions](https://github.com/raydium-io/raydium-sdk-V2/blob/e2e8bcd865189c994682a9477ec8eaf1e2da8aaf/src/raydium/launchpad/instrument.ts)
specify the 15-account IDL prefix followed by system program, platform claim
vault and creator claim vault when no optional share fee receiver is present.
Both exact-input instructions use three u64 values with share fee zero. Tests
check complete order/flags/data, not merely the discriminator or account count.
Those tests use synthetic state and do not certify an actual mainnet round trip.

The durable native-probe queue allocates two slots to newest untried arrivals,
one to the oldest untried target and one to due retries; empty lanes share their
capacity. Persistent arrival tickets and dispatch state survive restart and do
not rank arrivals by corrected host wall time. Existing jobs migrate without
inventing their original arrival timestamp. New results retain queue age at
attempt and explicitly flag clock regression. Canonical ACK still fences new
work, and no target expires or is removed by this scheduling change.

Reports separate per-venue targets, attempts, untried backlog and acknowledged
dispositions. A supported adapter with no observed targets is explicitly
`NO_TARGETS_OBSERVED`; support is not an executed round trip. Cadence, provider
refusals and backlog can still delay sampling, so the queue has no guaranteed
launch-to-probe deadline or whole-market coverage denominator.

Within each dispatch lane, the least recently serviced eligible venue gets
the next turn. Durable monotone service tickets keep a high-volume launchpad
from monopolizing fresh or historical work. Ties follow that lane's original
arrival/deadline ordering. Empty or not-yet-due venues do not reserve idle
capacity. This is fair service among retained candidates, not a claim that all
venues are decoded or that a provider can supply every requested bank.
