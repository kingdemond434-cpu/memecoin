# Bounded canonical-chain continuation

`verify_chain_window` verifies one forward window from a caller-authenticated
cursor. It reads at most1MiB including its record-boundary byte, in64KiB blocks,
with logical pacing, admission checks and a cooperative deadline. It parses at
most256 complete rows and checks strict sequence types, predecessor hashes,
self hashes and duplicate JSON keys against the existing canonical encoding.

A record limit or trailing partial row never skips bytes: the returned cursor is
immediately after the last verified complete row. If admission fails or any row
is rejected, the original cursor is returned, even if earlier rows were valid.
Empty/incomplete windows are not success. The gate performs no automatic sweep,
prefix copy, SQLite work or source writes and does not return source payloads.

The caller must establish trust in the starting prefix and trusted append-only
storage. The gate does not revalidate a previously certified prefix, defend
against malicious concurrent rewrites or authenticate an arbitrary supplied
anchor. Full-prefix and deployment-backup acceptance stay false. A returned
segment digest covers only the verified bytes, not unread/unparsed trailing data.
Logical pacing is not a physical IO/RAM cap or hard preemption of filesystem calls.
Oversized or unfinished rows remain pending; never discard them to make progress.

Operational sampling is one separately admitted window, not permission for a
history scan. Retained backup certificates must remain valid and recoverability,
cross-store coherence and all release gates still need independent evidence.
