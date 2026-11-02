# Readiness refresh wait boundaries

The existing single coalesced refresh now records process-local waits at
scheduled, frontier_snapshot, projection_dispatch, projection_worker and
projection_owner_resume boundaries. Idle ends/cancels the operation. The
existing dated diagnostic_snapshot carries bounded phase/duration fields and
a separate observation clock; cached projection age remains unchanged.

No extra refresh, executor job, publisher, cadence, provider call or task is
created. The original frontier/worker ordering and exception/cancellation path
remain. The worker wrapper records entry/completion and never changes results.
Offline source-phase diagnostics allowlist these fields; legacy snapshots
without them remain unknown. Persisted timings can themselves be dated.

Frontier includes its await, dispatch includes scheduler/executor wait, worker
includes projection code/locks/IO, and owner-resume includes completion
notification/scheduling. These are not pure loop lag, CPU/IO attribution, source
latency or proof that a particular phase caused an HTTP stall. Production
acceptance requires fresh post-deployment runtime evidence.
