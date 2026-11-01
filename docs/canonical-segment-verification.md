# Staged segment chain verification

verify_staged_segment reads one trusted private staged segment, at most 1 MiB
and 256 records, with 64 KiB reads and default 256 KiB/s logical pacing. It
checks file identity/size/time, historical digest, strict JSON keys and sequence,
every predecessor/self hash, final cursor and complete newline boundaries.
Development CPU proof and caller host admission are required. Deadline is
cooperative (at most 60 seconds); it cannot preempt filesystem calls.

Caller owns trusted private ancestors and baseline/receipt provenance. This
single-segment result does not authenticate the baseline or certify full-prefix
recovery, cross-store coherence or deployment. No scans, copies, automatic loops,
production writes or physical IO/RAM cap claims are introduced.
