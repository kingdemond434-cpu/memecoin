# Attribution, actor occupancy and source corroboration

The operator's attached assessment was checked against the running VPS. The
negative external-sample coverage, empty hot wallet/creator caches, zero owned
channels and many unmapped PumpSwap trade events were reproduced. The claim
that the repository's figure registry contained zero entries was incorrect:
both the committed registry and the running desk contained 68 figures before
this change. Cache occupancy is not an estimate of operating-capacity utilization.

## Concrete changes

- External-sample coverage now compares unique externally observed tokens
  against the current retained native census. Repeated sightings no longer
  inflate the numerator, and a later native observation can resolve a miss.
  A missing census remains unavailable. This is not global launch coverage.
- PumpSwap program trade events provide a pool but no mint. The monitor only
  learned its map from pool creation, leaving older pools unknown after startup.
  `PoolAttribution` durably captures new orphan events, reads batched pool
  accounts through the existing RPC manager, verifies owner and full layout,
  and reads explicitly owned SPL/Token-2022 mint decimals. Missing decimals are
  retried and never defaulted to zero.
- The raw event and identity proof must enter the canonical archive before the
  map is published. Retained orphans are then canonically derived as late
  observations. Their original time and later availability are separate. They
  can inform research/wallet history; they cannot reseed the old hot price,
  trigger a retroactive entry or repair historical coverage statistics. Earlier
  uncaptured orphan data is not claimed recovered.
- Runtime log invocation scopes identify the actual program emitting a payload.
  A `logsSubscribe` mention filter identifies a transaction, not every emitter
  within it. Nested foreign/failed scopes and truncated scopes cannot supply a
  trade event. Processed observations retain their commitment label and are not
  promoted into confirmed fills.
- Distinct observed early entries and new native creators now populate the
  existing compact hot caches. Their observation counts are separate from
  measured performance DNA; new records are `UNMEASURED`. Filling 20,000 slots
  is not an objective and does not make any wallet profitable or followable.
- Native pool hydration additionally requires the expected owner and base mint.
  It cannot accept a plausible binary layout from a different program/token.
- Native routes now require a SOL quote leg and matching pool/account assets.
  A requested token-to-USDC exit cannot silently build a token-to-SOL sell or
  interpret lamports as USDC raw units in paper accounting. The router must
  actually satisfy the requested pair. SOL-sized entries explicitly spend SOL;
  other-quote pool observations are retained separately from SOL price state.
- Meteora's own documentation links `https://t.me/meteora_dev` as Developer
  Updates. The figure registry retains this provenance and verification clock.
  Three reviewed first-party project domains are loaded separately from the
  still-empty operator-specific account/wallet registry. No numeric platform
  account id, wallet ownership or official-token endorsement is invented.
- Identity corroboration now tracks publication and availability separately,
  refuses knowledge acquired after the launch's registration cutoff, and keeps
  separate mentions when two figures discuss the same mint. One configured
  channel no longer causes the entire figure registry to report fully covered.
  An owned confirmation arriving after creation revises a later observation;
  it cannot leave the current view permanently stuck or rewrite the old feature.

## Retention and recovery

`ops_state_dir/native_pool_attribution.sqlite` contains the orphan outbox,
mapping jobs, reviewed native account identities and recovery counters. It
must be backed up with the canonical ledger, frontier and other state sidecars.
Acknowledged raw and derived history stays in the canonical archive. Pending
events and identities survive restart. RPC refusal, unresolved mint metadata,
disk failure and capture-to-ACK backlog remain visible. Per-pass work quanta
pace requests; there is no lifetime pool/event admission ceiling.

Zero starvation, universal coverage and 100% economically useful exploitation
are not established. Missing native launchpad decoders, provider refusal,
unverified identities, insufficient wallet outcomes and unavailable Phantom
World quotes remain actual gaps. A losing wallet stays rejected; missing
samples and quotes are not filled in to make a utilization percentage green.

The paper/research operating mode remains in force. The operator will explicitly
request any later live Phantom transition after reviewing separate product
results. No positive testing result can independently arm live submission.
