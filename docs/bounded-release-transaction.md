# CPU-bounded release transaction

`ops.bounded_manifest.update_role_action` is a small private-manifest transaction,
not a task installer or complete deployment command. It never registers, starts,
stops or changes a scheduled task. Tests use tiny isolated fixture files.

The deployment caller must independently acquire complete task action/account/
priority snapshots and verify exact tested revision and recoverable state-backup
receipts before calling the transaction. A boolean backup acknowledgement is a
caller contract, not evidence verification. The action must be reviewed paper code;
this library does not infer signing authority from shell command text.

The transaction serializes cooperating writers with an exclusive lock, compares
the expected original digest and fresh task snapshots, stages and fsyncs bounded
JSON, then rechecks bytes and security before replacement. An unchanged underlying
action returns without rewriting the manifest. This is the expected path when a
release updates code within the existing production checkout.

For an actual action change on Windows, only an already protected DACL is
supported. A unique staged file receives that same protected DACL. The staging
security descriptor must match before ReplaceFileW is called with flags zero;
metadata/ACL errors cannot be ignored. The destination and retained original must
then match their expected hashes/security and the task must remain unchanged.
An inheriting live DACL is rejected before replacement: do not protect, weaken or
rewrite a live ACL to get past this gate. The POSIX path exists for CI fixtures.

A failed or uncertain transaction retains its lock and any original produced by
the OS. Inspect those exact files before recovery; do not automatically retry or
remove the lock. A successful action change retains its private original in the
same restricted directory. The fixture recovery drill reuses the same gates to
restore the original action. A cooperating lock does not prevent arbitrary
external administrators from racing a write; deployment must retain a single
writer and its normal private-directory ACLs.

Neither a transaction success nor a Running task proves deployment acceptance.
The release caller still needs preserved state backups and canonical-prefix
verification, exact tested source/tree, service health and paper locks, fresh
kernel receipts after restart, advancing durable counters, recovery checks and
resource measurements. The current production manifest has an unprotected DACL;
it was read only during development. No live manifest, task, ACL or service was
changed by these tests.


Read-only preflight is implemented in `ops.bounded_preflight`. The task collector
exports the actual registered definition through a direct child PowerShell
process. It checks a single LocalService service-account principal, LeastPrivilege,
existing priority7/IgnoreNew, and the exact installed CPU wrapper action. It
fingerprints the entire XML so changes to other settings are also detected by a
later snapshot comparison. No raw task definition or underlying private role
action is returned. Export failure, unsupported arguments or a mismatching
identity blocks the operation. Existing execution-policy text is only matched;
this collector does not execute the exported action or change execution policy.

The revision preflight reads the private isolated validation report, checks its
required pass/native/isolation fields, verifies the actual isolated checkout HEAD
and cleanliness, and compares Git source-tree objects against the proposed
release. An equivalent merge tree is allowed; a changed tree or dirty checkout is
not. Reports must originate in the trusted private validation workflow. This
comparison is not a report signature, CI check or review attestation.

These collectors replace caller assertions for task/revision evidence, but do
not yet form a complete deployment orchestrator. Verified state backups,
canonical-prefix continuity, resource/IO measurements, review/CI gates, safe
stop/restart and recovery, repeated task snapshots, fresh post-start kernel
receipts and advancing service counters remain integration requirements. Do not
run the old task-replacing helper on the strength of this preflight alone.
