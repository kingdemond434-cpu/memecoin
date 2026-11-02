CpuAdmission is an opt-in callback for one operation that cannot alter its CPU
job, task, limits, manifest or process identity. It verifies the current capped
development context before first work, reuses evidence for less than one second,
and refreshes at expiry. The evidence age starts before the native check, so a
slow check does not extend its lifetime. Any failed/error/stale/context/clock
check latches the object closed; callers do not automatically retry it.

The kernel, not this callback, continues to enforce the immutable2%job cap.
This helper changes no limits, principal, priority, permissions, tasks or thread
settings. It does not grant permission or replace independent byte, VM, deadline,
disk, memory, IO or application admission. Cached evidence can age before the
next call and syscalls are not preemptible. Use fresh verification for deployment
transitions/post-start kernel receipts; never use this cache while changing jobs
or actions. Create one instance per operation/process and never persist/transfer
it to another process. Snapshot fields are historical diagnostics, not permission.

A frequent read-only SQLite progress callback can otherwise parse the roles
manifest and query the same unchanged job thousands of times. This callback bounds
those evidence reads without raising the hard CPU cap or any verifier policy
maximum. A caller must explicitly select a supported VM budget and retain prior
budget-stop receipts. Successful quick_check still proves only one staged copy,
not full SQLite integrity, canonical prefix, cross-store coherence or recovery.
