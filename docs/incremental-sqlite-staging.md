# Incremental SQLite staging for bounded releases

`ops.incremental_backup.stage_sqlite_backup` stages one of the ten known state
databases through SQLite's online backup API. It includes committed WAL state;
copying only the live database filename would not establish that. This function
never controls tasks, changes CPU limits, modifies a source connection for write,
removes an older backup or certifies a release.

The caller must first verify the existing development CPU job, a private output
directory, single-writer maintenance, source admission and a host-pressure gate.
`max_bytes` and `continue_check` are mandatory. The logical batch rate defaults
to256KiB/s and cannot exceed1MiB/s; steps contain at most16 source pages. Each
successful callback is charged a full quantum, including repeated work after
source changes. Full-copy readback hashing uses64KiB reads through the same
pacer. Busy callbacks, charged bytes, elapsed time and pacing sleep are recorded.
Checks run between small steps and bounded100ms sleep slices: deadline, current
host admission, source size budget and12GiB remaining disk reserve. Explicit
long-operation deadlines may be up to12hours; the default is60seconds. A busy
source can cause repeated backup work and eventual timeout; there is no
throughput guarantee or automatic retry.

This is software pacing of logical work, **not a physical disk IO cap**. A page
batch and hash read occur before their pacing wait. SQLite/cache/metadata work,
source reads plus destination writes and final fsync can cause additional IO or
stall. CPU containment does not bound RAM, WAL growth or device latency. Actual
production use still needs measured host admission and ongoing resource gates;
tests use only small isolated data.

After SQLite completes, the private artifact is fsynced through a writable
handle (required by Windows), then fully reread and hashed. It is published
exclusively as a new `.pending` path; an existing copy is never replaced. Failure
or cancellation closes handles and removes only that invocation's owned unique
partial name. A collision with another partial is preserved. Live source writes
may occur independently; the receipt says no source write was requested, not
that the source stayed unchanged.

The status is `SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING`.
`sqlite_integrity_verified`, `canonical_prefix_verified`,
`cross_store_checkpoint_verified` and `deployment_backup_accepted` all remain
false. No full integrity scan runs here. The caller must not pass this staging
receipt to a deploy gate as a verified backup. Controlled integrity verification,
all required stores, a coherent checkpoint/canonical prefix, retained recovery
receipts, native interpreter, exact tested source/CI/review, unchanged bounded
task fingerprints, safe stop/recovery and fresh kernel/service progress evidence
remain required before deployment.

Isolated tests recover actual WAL-backed rows, commit a concurrent WAL write,
check fixture SQLite integrity, validate pacing with a controlled clock, abort on
host/deadline/space/size gates, preserve existing copies and exercise interruption
cleanup. A separate small real-clock fixture records actual elapsed and pacing
sleep. Fixture recovery is not proof of production backup or full service restore.
