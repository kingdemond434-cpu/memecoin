# Native artifact digest gate

`ops.native_artifact_digest.verify_native_digest` compares one private runtime `.pyd`
with a digest and byte count supplied from independently trusted tested-artifact
provenance. Never derive the expected digest from the installed candidate and then
call that a tested artifact. The caller must serialize maintenance and trust the
stage directory and its ancestors. Reparse components beneath that directory are
rejected. Matching bytes do not validate ABI, dependency DLLs, provenance, CI,
backup recovery, task containment or deployment authorization.

Reads use at most 64 KiB blocks, at most 16 MiB total artifact size, a logical rate
of at most 1 MiB/s (default 256 KiB/s), a maximum 60-second cooperative deadline,
and a required admission callback. This is not an OS IO or memory cap; filesystem
calls may block. Identity, size and modification time are checked before and after
reading. It is not a defense against a privileged concurrent file replacement.
No code is loaded, services changed, native files modified or production state
opened. Actual cap-aware deployment orchestration and ABI tests remain pending.
