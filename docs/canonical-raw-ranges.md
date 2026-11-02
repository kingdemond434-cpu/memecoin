# Bounded raw canonical ranges

`ops.canonical_raw_chunk.stage_raw_chunk` preserves one contiguous raw byte range
of at most 1 MiB, including a row fragment. It needs a caller-authenticated source,
captured extent, exact byte offset, private separate destination and current 2%
development job proof. It shares the canonical staging writer lock, enforces the
12 GiB reserve, charges read/write/readback at 256 KiB/s by default, fsyncs and
checks the staged SHA-256 before exclusive publication. Existing destinations
and locks are preserved; only its own temporary file is cleaned on failure.

This avoids requiring a complete JSON row to preserve bytes. Callers must retain
contiguous range receipts and reconstruct exactly; this function neither loops
nor advances a verified chain cursor. Raw fragments cannot be passed off as the
verified `.segment.pending` format. The receipt explicitly leaves chain,
record-boundary, baseline, cross-store, recovery and deployment acceptance false.
Record parsing/chain validation across fragments and full backup orchestration
remain separate unfinished requirements. No production stop is authorized by a
raw range receipt.

Trusted append-only storage and trusted ancestors remain assumptions. Identity
and truncation checks do not prove absence of concurrent in-place rewriting.
Logical pacing and cooperative deadlines do not cap physical IO/RAM or interrupt
filesystem calls. A captured extent excludes subsequent appends; a partial final
row is preserved as bytes and is never represented as a complete record.
