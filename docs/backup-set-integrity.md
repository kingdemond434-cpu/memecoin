# One-copy staged integrity progress

`verify_next_staged_store` requires a complete ten-store staging checkpoint and
binds a separate progress file to its exact bytes. It uses the same cooperative
writer lock as staging, then invokes the existing bounded staged-SQLite checker
for one eligible copy per call. Metadata publication uses flush/fsync and atomic
replacement. A changed staging checkpoint cannot reuse historical results.

The checker retains its8MiB copy limit, VM budget, cooperative deadline, logical
pacing and admission callbacks. Larger reported copies remain explicitly pending
in `blocked_by_copy_budget`; eligible copies can continue without declaring the
large copies checked. No limit is increased and no data is deleted to obtain a pass.

Previous results are historical receipts and are not automatically revalidated.
Even ten recorded quick-check receipts do not certify current full integrity,
canonical prefix, cross-store coherence, recovery or deployment backup acceptance.
All those aggregate flags remain false. The caller must supply a trusted private
serialized staging directory and host admission; no production database may be
passed as a staged copy. CPU containment is not a RAM or physical IO cap.

Tests use isolated real SQLite copies and cover incremental checking, corrupted
copy preservation, changed checkpoint binding, existing writer locks, incomplete
inventory and a synthetic oversized receipt blocking only its own copy. No large
production store has been certified using this API.
