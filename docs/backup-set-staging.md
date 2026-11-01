# Incremental backup-set staging checkpoint

`stage_next_store` advances one of the ten required SQLite stores per call through
the existing paced online-backup primitive. It requires development CPU proof,
explicit host admission and a per-store byte budget. A cooperative exclusive lock
prevents overlapping set writers. The checkpoint binds the exact source-path
inventory and retains earlier receipts without recopying them. Small metadata
writes use flush/fsync and atomic replacement; this is not a cross-file crash
transaction or recovery certificate.

Use a trusted private staging parent separate from source directories. Normalized
source aliases reject. On resume, prior pending files must remain regular with
matching sizes and receipt shapes. Their contents are not rehashed automatically;
`previous_receipts_revalidated` stays false. No completed checkpoint can assert
integrity, canonical prefix, cross-store coherence, recovery or deployment backup
acceptance. All ten files staged is still preparation only.

A copy left after failed checkpoint publication is retained. A later attempt fails
on that existing copy rather than overwriting or deleting it. Reconcile it through
reviewed recovery logic; do not remove a lock owned by another writer or automate
stale-lock deletion. This API neither starts tasks nor changes service actions.

The underlying primitive's logical read/copy pacing and disk reserve still apply,
not physical IO or RAM caps. Host admission must account for those resources.
Production copies have not been accepted using this coordinator. Fixture tests
cover one-store progression, real SQLite fixture recovery, overlapping writers,
publication failure, damaged prior copies, source changes and admission rejection.
