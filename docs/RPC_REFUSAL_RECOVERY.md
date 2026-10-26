# Shared RPC refusal windows

When `MEMECOIN_RPC_BUDGET_DB` names the owner-local allocation database, RPC
managers also share persistent refusal deadlines through that SQLite file.
The production launcher and diagnostic helpers already use that setting.
No new credential, subscription or real wallet is needed.

A deadline is keyed by the digest of the exact HTTP endpoint and its headers.
Database rows contain only digests and times. Different processes and restarts
therefore respect an observed refusal of that endpoint/credential. They cannot
shorten an existing deadline. A late successful response cannot clear another
process' refusal. Allocation waits recheck the deadline before opening a socket.

Ordinary requests, health probes, batches, broadcast paths and returned
WebSocket candidates observe the same deadline. Already-started requests
cannot be recalled. Separate clients that bypass `RPCManager`, differently
spelled endpoint URLs, other applications and unreported provider limits remain
outside this accounting. Without the configured shared database, the existing
in-process endpoint deadlines still apply.

`Retry-After` supports finite nonnegative seconds and HTTP dates. Genuine
daily/reset windows are honored in full; the old five-minute clipping could
retry too early. Invalid/nonfinite headers use the existing fallback delay.
These are provider refusal windows, not a claim to know plan capacity.
Method-specific 402/403 exclusions retain their existing separate semantics.
Status reports redact URL credentials and show the last HTTP status.

No bypass, alternate path guessing, automatic key rotation or order-authority
change is introduced. Expired windows allow ordinary recovery. A persistent
quota/access refusal still requires appropriate authorized capacity; source
debt remains visible rather than being labeled complete coverage.
