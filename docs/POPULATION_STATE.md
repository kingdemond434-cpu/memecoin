# Incremental population state

The framework checkpoint retains a `population_sqlite_v1` descriptor. Its
same-directory sidecar, `<framework stem>.population.sqlite`, retains all
population observations, measured outcomes, frozen controls, original receipt
children and visibility timestamps. Source text and source facts still belong
to the canonical evidence ledger. This changes persistence, not eligibility,
authority, selection or forward clocks.

On the first start, the complete legacy component is imported in one SQLite
transaction. The framework descriptor is saved only after that commit and an
exact comparison with the legacy state. A crash before descriptor replacement
can retry the identical migration. A missing, conflicting, uninitialized or
damaged referenced sidecar stops initialization; it never creates an empty
replacement for retained research.

Each observation commits its new facts and receipt together with WAL and FULL
synchronization before acknowledging work. Retrying after a crash returns the
original receipt. The accumulated historical component is no longer serialized
and rewritten into the framework for every observation. Independent components
continue to use the existing framework checkpoint.

For a deployment backup, stop the dedicated paper task first. Retain the
framework JSON, canonical evidence, frontier SQLite and this population SQLite
as a set. Use the SQLite backup API for both databases, since copying only the
main database file can miss committed WAL transactions. Verify the canonical
hash chain and `PRAGMA integrity_check` on both snapshots before restarting.
Retain the complete pre-migration backup for rollback. Older releases cannot
read the new descriptor: restoring old code requires restoring its matching
pre-migration state too. Never pair an old framework snapshot with a newer
sidecar or silently discard one to make startup pass.

`tests/test_population_index.py` covers complete migration, crash retry,
atomic conflict rollback, durable receipts, unchanged framework bytes during
new observations, legacy equivalence and fail-closed damage handling. These
are storage correctness tests, not profitable-trading evidence.
