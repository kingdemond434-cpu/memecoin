# Pinned fragment recovery history

`record_fragment_step` records one freshly verified fragment batch in a private
`.fragment-progress.json` file. It binds an exact anchor cursor, the caller's
SHA-256 of its anchor evidence, a fixed captured extent and up to 32 contiguous
steps. The caller must keep and authenticate that anchor evidence separately;
its hash is a reference, not proof that a baseline is recoverable.

Initial publication requires no existing progress file. Each later call requires
the exact prior progress-file SHA-256. The coordinator checks strict metadata,
parent cursors, raw-range bounds and private-directory references, then invokes
the actual bounded fragment verifier. It never accepts a supplied successful
verification result. Whole fragment digests are rechecked for the new step;
historical artifacts are not revalidated. Only the newly verified byte interval
is recorded, excluding prior prefixes and incomplete overread. Reconstruction
must select those intervals rather than concatenate every raw file wholesale.

One shared canonical writer lock prevents cooperative overlap. Metadata is at
most 64 KiB, temp bytes are fsynced, prior pin is checked again before atomic
replacement and initial publication is exclusive. Existing files, pins and locks
are not reset on failure. Resource admission and the 12 GiB reserve apply; CPU
proof must show the existing 2% development job. A 32-step limit raises an error
without eviction or automatic rollover. There is no automatic loop, source copy,
service/task action, provider access, full-prefix/cross-store recovery certificate
or deployment authority.

Trusted private ancestors and cooperative writers remain required. File fsync
and atomic replacement do not prove cross-store crash consistency or recover the
full canonical tail. Pacing and cooperative deadlines do not cap physical IO or
RAM, interrupt filesystem calls or preempt a single JSON parse. Baseline and all
other deployment gates remain separate and must be verified before deployment.
