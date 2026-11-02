# Own-process recovery measurements

`run_observed_fragment_tick` brackets one existing finite tick with Windows
current-process GetProcessTimes, GetProcessIoCounters and GetProcessMemoryInfo.
It opens no other process and changes no job, priority, permissions or service.
Existing CPU2%, tick/step byte/time/row, reserve and pacing gates still apply.
Linux sampling is explicitly unsupported; there is no fabricated zero fallback.

Samples contain CPU100ns counters, read/write/other operation and transferred
byte counters, working set, lifetime peak working set and private commit bytes.
A delta requires the same PID/creation identity, ordered finite observation
windows and nonregressing CPU/IO counters. Working set can legitimately shrink.
Snapshots are not atomic. Counters include instrumentation and any other work
in this process; they exclude children. Private commit is not resident memory,
and the reported peak covers the process lifetime, not just the tick.

A failed initial sample starts no tick. A failed final sample preserves the
completed tick result and reports missing measurements. Lost tick results are
explicit reconciliation requirements, never automatic retries or pin resets.
Native exception messages are not emitted. Only current-process pseudo handles
are used, with no alternate-rights probe or elevation fallback.

These are observations, not RAM/physicalIO caps, physical storage-byte proofs,
quant-performance measurements, workload admission guarantees or deployment
certificates. Full recoverability, service health, latency and paper execution
remain separate. Receipts stay in the private handoff; no credentials or raw
source bodies are included.

API definitions: [CPU timing](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes),
[IO counters](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-io_counters),
[memory layout](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex).
