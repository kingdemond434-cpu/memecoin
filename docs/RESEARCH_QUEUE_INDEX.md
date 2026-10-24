# Sparse active tag auctions

Production had roughly 265,000 READY jobs. Driving exploration from all active
jobs avoids archived history but still scans unrelated work for rare tags.
The real VPS exceeded a two-second query budget for three of its four tags;
the common tag returned its bounded batch in about 15 ms.

`active_job_tags` is a derived index of costed READY, RETRY and LEASED jobs,
ordered by tag, cost unit, creation time and identity. Single-kind auctions use
the corresponding kind index. The original jobs, tags and transitions remain
the source of truth. No evidence or queued work is discarded.

SQLite triggers update the derived index in the transaction that inserts,
prices, leases, retries, acknowledges, hibernates or wakes a job. Deleting or
changing a tag updates the same index. A legacy database is backfilled once,
with a transactional completion marker; a missing cache table is rebuilt.
The query still checks current due time, coordinate state, kind and resource
budget. Exploration-floor rotation, FIFO order and lease ownership are
unchanged.

Regressions bound SQLite VM work with 12,000 archived tagged jobs and with
12,000 unrelated active jobs. Migration, pricing, hibernation, wake, lease,
acknowledgement and transaction rollback are checked against retained work.
This is a scheduling and availability fix, not an alpha-performance claim.

Queue reporting uses a compact covering index for counts, time-window arrivals
and drains, and unknown-cost work. One aggregation replaces repeated scans of
the large payload table. Sparse indexes retain measured delay values and the
oldest positive-priority work. Zero delay remains a measured zero; missing
delay remains unknown, including for hibernated work. Completed work remains
in the history and arrival/drain windows keep their closed boundaries.
