# Bounded canonical segment staging

`stage_chain_segment` verifies one admitted chain window, then copies only its
verified complete byte range into a private pending segment. It re-reads the
source range against the verified digest, flushes/fsyncs the new file, performs
paced readback and publishes with an exclusive hard link. Existing targets and
cooperative writer locks are preserved. Failure removes only the uniquely owned
partial file; source data and other publishers' files are never replaced.

The verification read is at most1MiB. Copy, write and readback add bounded logical
work (at most4MiB payload work including verification), paced across the workflow
with a cooperative deadline, admission callback and12GiB disk reserve. These are
not physical IO/RAM caps or hard preemption of filesystem calls. A single batch
can precede its pacing wait. The default development job remains2%CPU.

Caller-authenticated prefix trust, trusted append-only source and trusted private
destination ancestors are mandatory. One staged segment does not authenticate the
retained prefix, cover unread/partial rows, prove cross-store coherence or certify
recovery/deployment. All those acceptance flags remain false. No automatic sweep,
source write, service restart or task/action change occurs.

Fixture tests reconstruct and verify a real writer's chain from a retained prefix
plus the copied segment. They also cover changed source bytes, corrupt readback,
late publication collision, incomplete rows, disk reserve and existing locks.
Operational evidence is one separately admitted small segment; preserve it and
its cursor receipt for reviewed continuation rather than overwriting/recreating it.
