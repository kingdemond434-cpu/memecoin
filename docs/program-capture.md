# Independent program-log capture

`python -m src.runtime.program_capture --config <chain-config> --state-dir <state-directory>`
starts a separate read-only recorder on loopback port 18084. The Windows launcher
is `deploy/windows/start-program-capture.ps1`. Run it under a separate service task
and release directory so changing the trading desk does not stop this process.
It has no order, signing or wallet API. It uses the owner's configured Solana RPC
WebSocket endpoints, and honors provider cooldowns and access refusals. It does
not backfill historical transactions or bypass subscription eligibility.

The default scope is ten configured program IDs shared with the desk: Pump,
PumpSwap, three Raydium programs, two Meteora programs, Orca, LaunchLab and DBC.
Additional public program IDs can be supplied with repeated `--program` arguments.
This is an explicit subscription scope, not proof of all-chain or all-launch
coverage. A provider may refuse some subscriptions; `/status` reports the accepted
IDs and refusal codes without exposing endpoint URLs or credentials.

Every received notification from an accepted subscription retains the complete
log result, including transaction errors and logs that this version cannot decode.
Scoped successful program payloads also retain their exact bytes in base64.
Source transitions, source receipt clocks, capture receipt clocks, slot and
processed commitment are preserved. Processed notifications are provisional:
forks, rollback and repeated observations remain possible. An observed failed
transaction is not a successful trade. Receipt time is not transaction execution
time, and this recorder does not attest to external identity or finality.

`program-events.sqlite` uses WAL transactions and `synchronous=FULL`. Each committed
batch extends a sequence and hash chain, with counters committed in the same
transaction. The bounded in-memory queue applies backpressure and retries failed
batches; it does not impose a history row limit. Retention is subject to actual
disk capacity. Queue admission is not a durable acknowledgment. Only the committed
counter reports retained observations. Cancellation before admission is counted;
an abrupt stop may lose uncommitted or unread network messages. After an unclean
shutdown the unknown loss remains explicit. This does not reconstruct gaps before
the recorder started, or guarantee gap-free recording during network outages.

`/health` distinguishes process/writer health from its `source_status` and retained
program-payload count. A running recorder with no eligible feed is not collection
proof. Acceptance requires recent notifications, increasing committed transaction
counts and a verified committed snapshot. Verify without opening a network feed:

```
python -m src.runtime.program_capture --state-dir <state-directory> --verify-only
```

The checker reads a consistent SQLite snapshot and verifies sequence, hash chain
and committed checkpoint. It verifies local integrity, not external authenticity.
Use SQLite's backup API when making an online backup; copying only the main file
while a WAL writer is active can omit committed rows. Keep the separate state
directory across release changes. The desk does not yet replay this raw tape into
its research corpus automatically; recording and downstream replay are separate
acceptance requirements.

## Segments and archives

One permanent SQLite file grows without bound (98 GB on 2026-11-10). The
recorder therefore seals the active file at `--segment-bytes` (default 2 GiB,
`0` disables) and continues the SAME sequence and hash chain in a new
`program-events.sqlite`. Nothing is dropped and no history limit is imposed.

- A sealed file is `program-events.<first>-<last>.sealed.sqlite`.
- A background archiver, limited to `--archive-duty` of wall time (default 0.1)
  so it cannot starve the recorder inside the role's CPU budget, rewrites each
  sealed file as `program-events.<first>-<last>.jsonl.gz`: the exact row bodies,
  one per line. Every other column is derivable from the body, so this is
  lossless.
- The sealed SQLite file is removed only after the finished archive has been
  read back and its whole chain recomputed against the manifest
  (`...manifest.json`, which also pins the archive's SHA-256).
- Archiving stops and discards its partial output if free space falls to 4 GiB,
  on a stop request, or on any chain failure. In each case the sealed source is
  left untouched and `/status` reports it under `tape.segments`.
- A new active file takes its starting sequence and hash from the newest closed
  segment on disk, so a crash between sealing and creating the successor cannot
  restart the chain.

`--verify-only` verifies every archive, every sealed file and the active file as
one chain. Keep all `program-events.*` files together when moving the state
directory. Duplicate suppression by observation id applies within the active
file.

