# Bounded frontier diagnostic queries

FrontierIndex.diagnostic_report uses a separate mode=ro/query_only connection,
a100ms SQLite busy timeout, disabled mmap and small connection page cache. It
runs the existing aggregates in one read transaction, with default100000VM
steps and cooperative2s deadline (hard policy maxima1millionVM steps/5s).
UnknownOpportunityEngine.frontier_snapshot uses this path for the real index;
legacy frontier adapters without it retain the existing report interface.

Budget exhaustion raises FrontierDiagnosticBudgetExceeded. No partial/zero
report is returned, and a last-good snapshot stays dated with explicit refresh
error. The full report API and every lease/ACK/retry/write path remain available
and unchanged. VM guards do not preempt filesystem/connect/lock waits or cap
physicalIO/RAM; the100ms busy setting is connection-local. This bounds future
diagnostic query work, not the actual production stall cause or throughput.
Runtime acceptance requires tested deployment and fresh health/counter evidence.

If no first snapshot exists, budget failure remains an exception with no invented
counts. The existing health loop catches this specific diagnostic exception,
retains its memory-governor observations and normal cadence, and retries on the
next normal cycle. It publishes nothing for that failed initial projection.
Other failures retain their existing handling.
