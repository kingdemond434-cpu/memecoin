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
