# Finite recovery work admission

`run_fragment_tick` admits at most four existing recovery-step calls, at most
2MiB of new source copies, and a cooperative deadline of at most60seconds.
The default new-copy budget is1MiB. Each driver call still permits at most one
new1MiB window, requires the development2% proof and preserves its pacing,
reserve, two-input and row bounds. A tick has one exclusive sibling journal and
writer lock; it creates no daemon, task or supervisor. No automatic rollover.

The journal is published before work, marks each in-flight call, and records
verified progress pins/cursors and actual copied-byte counts before another
call. Every returned progress pin is read back and its metadata validated.
An admission/time/byte/step/history limit stops further work. Existing journals
are never restarted or overwritten by a new invocation. Each invocation needs
a new journal name and an authenticated current progress pin.

An exception during a step or its checkpoint yields RECONCILIATION_REQUIRED.
The last checkpointed pin is not silently promoted, and the admitted copy upper
bound for an unresolved call is explicit. Check the actual history and retained
attempt artifacts before resuming. A crash may leave a lock or in-flight marker;
there is no automatic stale-lock removal, pin reset, rollback or forced retry.
The caller must retain the returned journal digest and verify it before trusting
its claims. Failed checkpoint writes may leave only the previous journal bytes.

Checkpoint metadata writes can finish after work admission expires so completed
work can be retained. Filesystem calls/parsing are not preemptible, pacing is
logical, and CPU containment is not a RAM/physicalIO cap. Trusted private
ancestors, append-only source and cooperative writers remain assumptions.
The tick reuses existing step verification; it does not authenticate the old
baseline or recursively revalidate history/raw artifacts. Full tail, SQLite
integrity, cross-store coherence/recoverability, tested deployment and paper
producer/consumer acceptance remain separate. No funds, provider calls, source
writes, service/control changes, fills, edge or live authority are introduced.
