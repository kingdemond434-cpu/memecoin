# Durable live source intake

The live source mesh writes admitted public observations to
`data/state/source_intake.sqlite` before releasing each producer batch. A busy
source no longer evicts older observations from a shared in-memory queue.
SQLite WAL transactions use FULL synchronization and run outside the event
loop; an asynchronous writer gate prevents producers occupying the whole
thread pool while waiting for the same database.

Each eligible source receives a turn. The consumer leases an observation,
retains canonical research work, performs the existing onion-child delivery,
then acknowledges the exact body and generation against that work's evidence
reference. An expired lease can replay after a crash. Old generations,
modified bodies and missing references cannot remove pending data. Failed
canonical intake remains pending; successful intake is not confused with an
upstream transport acknowledgement.

This is **at-least-once delivery**. Retry payloads preserve the first-delivery
clock and canonical intake's existing content identity. Original publication
and observation clocks remain unchanged. `research_available_at` prevents a
late research delivery being backdated to the earlier observation. Payloads
include a queue receipt with accepted time, first-delivery time and body hash.
The receipt comes from the active local lease, never source-supplied metadata.

Acknowledgement clears the queue's full body while retaining its digest and
canonical reference for audit. In-memory secondary indexes are best-effort;
the canonical work remains the recoverable source. Failed neural intake is
still reported separately and is not certified by a source queue ACK.

`source_mesh` reports accepted, pending and acknowledged counts, source ages,
write failures, database/WAL sizes and free disk space. Zero queue evictions
means no eviction of **committed** observations. It does not establish no gaps
at providers, no loss before a producer commit, enough consumer throughput,
complete NLP processing or unlimited storage. Disk failure pauses that
producer on the same batch and records the error type. Invalid event batches
also remain deferred for diagnosis instead of being silently dropped.

The legacy explicit batch API and in-memory test mode retain their previous
semantics. Production continuous collection uses the journal; synchronous
publish/drain calls are rejected in that mode to prevent an accidental bypass.

Deployment backups must include an SQLite online backup of
`source_intake.sqlite` with its integrity check. Stop producers and drain
started writes before closing the database, after the source consumer stops.
