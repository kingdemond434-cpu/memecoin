# Explicit fragment history rollover

`rollover_fragment_history` requires an exact caller-pinned parent with32 valid
steps and an unfinished captured extent. It leaves that file and all raw inputs
intact. A new sibling link pins the parent bytes, original anchor/evidence,
child name, exact parent tip and retained raw suffix. An empty child history
uses the link SHA as its anchor evidence. No new canonical coverage is claimed.

Both canonical locks are acquired in the driver order. CPU2%, private sibling
paths,12GiB reserve and a cooperative30-second deadline are required. Parent
bytes are checked again before each publication. Fsynced exclusive links never
replace existing files. Interrupted publication may leave a link; repeating the
same request accepts only exact matching metadata. An advanced/different child
is rejected rather than reset. Locks left by a crash need explicit authenticated
reconciliation; there is no automatic stale lock removal.

Pass the returned `rollover_link` to `advance_fragment_step`. It freshly checks
that link's pinned bytes and the immediate parent's metadata before selecting
its retained suffix for an empty child. The existing fragment verifier rehashes
raw inputs before advancing. Subsequent child steps use their own prior inputs.
The caller must retain the child pin, anchor, link pin and original provenance.
Ancestry is not recursively scanned; old raw bytes are not verified by rollover.
Full prefix authentication, raw reconstruction, large SQLite integrity,
cross-store coherence, actual recoverability and deployment gates remain
outstanding. This changes no source/service/task/provider/live authority.
Logical pacing and cooperative deadlines do not cap RAM or physical IO and
cannot preempt filesystem calls. Caller-owned private ancestors and cooperative
writers remain assumptions. No automatic rollover loop or history eviction.
