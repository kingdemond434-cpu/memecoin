# Offline source phase diagnostics

read_source_phase_snapshot reads at most1MiB of the existing private readiness
file using64KiB reads and256KiB/s logical pacing. It requires the current capped
development job and caller host admission, checks identity/size/mtime, and has a
cooperative30s deadline. Private ancestors must be trusted. It performs no HTTP,
provider calls, writes or automatic polling and claims no physicalIO/RAM cap.

Output includes only known numeric source-phase fields, paper-lock boolean and
snapshot freshness metadata. Unknown fields/phases, payloads and credentials
are omitted; duplicate keys are rejected and invalid phase rows are identified
without exporting their values. A stale/future-dated sample is explicit.

Counters are cumulative process-local service times; owner-resume includes
completion notification/owner scheduling, worker includes locks/SQLite/commit,
and dispatch includes scheduling/executor wait. These are not source latency,
pureCPU/disk/loop lag, HTTP health or causal proof. Use separately observed
counter deltas and fresh runtime health; restarting can reset these counters.

File freshness and cached projection freshness are separate fields. An actively
republished file can contain an old cached projection while refresh is pending.
The reader computes projection age from its observation timestamp when present;
missing clocks remain unknown. Refresh error presence is exported as a boolean,
never raw error content. A fresh file never upgrades stale/unknown projection
contents or HTTP health.
