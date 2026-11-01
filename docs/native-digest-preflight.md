# Digest-bound private native preflight

`probe_digest_bound_runtime` composes the bounded private extension digest verifier
with the existing direct-child location and CPU preflight. It verifies development
CPU admission before reading, compares bytes before import, requires the child to
report the same exact extension path, and repeats the digest check after the child.
A mismatch before import never launches the child. A child failure, different path
or subsequent mismatch cannot produce a successful composed receipt.

Expected size/digest must come from independently trusted tested-artifact
provenance. Do not hash an installed candidate and call that its trusted expected
digest. The caller must serialize maintenance in a trusted stage; this does not
protect against malicious concurrent code swaps. Interpreter and dependency
binary digests are not verified. Each bounded digest pass retains its own logical
read pacing and cooperative deadline; it is not an OS IO/RAM cap. The child has
its existing60-second timeout,2% inherited job and allowlisted environment.

This gate does not perform ABI checks or validate CI/review, coherent backups,
recoverability or post-start collection. It cannot authorize deployment or change
services. Integration with the offline smoke surface and the full cap-aware
deployment transaction remains required. No production binary certification was
attempted without independently trusted expected bytes.
