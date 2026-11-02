# Bounded staging and history step

`advance_fragment_step` requires a caller-pinned existing fragment history. It
validates that history before copying, binds its cursor and source identity into
an attempt journal, and stages at most one new 1 MiB raw window per invocation.
Attempt metadata is fsynced and atomically published. Up to two raw windows are
retained for a single row attempt; incomplete rows do not advance history, and a
third window is not silently copied or skipped. No automatic loop is started.

Retries reuse retained receipts before copying more data. If raw publication
succeeded but receipt publication failed, the deterministic orphan filename is
preserved. Adoption reads and compares the exact bounded source range and raw
file, checks identity/size, and fsyncs the retained file without writing its
bytes. A mismatch is rejected without deletion or replacement. Historical
artifacts and attempt journals remain available on failure or later success.
The source must be trusted and append-only; private ancestors and cooperative
writers remain required. This does not attest against malicious in-place source
rewriting or authenticate a baseline.

The driver has a separate writer lock and uses the existing canonical locks for
individual operations. Existing locks are never removed on behalf of another
writer. A crash leaving a lock, or a history update whose new pin the caller did
not retain, requires explicit authenticated reconciliation; the driver will not
reset a pin, roll history back or remove a stale-looking lock automatically.
History rollover beyond 32 steps remains separate and must retain old evidence.

All stages retain the current 2% development job proof, reserve/admission gates,
default logical pacing and a cooperative deadline. These are not hard RAM/IO
caps and cannot preempt filesystem calls or a JSON parse. The driver does not
change services, tasks, wrappers, provider budgets or live authority. Reaching a
captured byte tip is history-only; full-tail reconstruction, baseline and SQLite
integrity, cross-store recovery and tested deployment gates remain outstanding.
