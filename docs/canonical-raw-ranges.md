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
Full authenticated reconstruction and backup orchestration
remain separate unfinished requirements. No production stop is authorized by a
raw range receipt.

Trusted append-only storage and trusted ancestors remain assumptions. Identity
and truncation checks do not prove absence of concurrent in-place rewriting.
Logical pacing and cooperative deadlines do not cap physical IO/RAM or interrupt
filesystem calls. A captured extent excludes subsequent appends; a partial final
row is preserved as bytes and is never represented as a complete record.

`ops.canonical_fragmented_row.verify_fragmented_row` can verify exactly one
canonical row from one or two private raw-range receipts (at most 2 MiB total
input). It validates all receipt bounds/contiguity/captured extent, reads each
fragment with pacing and digest/identity checks, rejects aliases and duplicate
JSON keys, and checks sequence, predecessor and canonical self-hash. A complete
row advances only to its exact newline boundary; overread bytes remain explicit.
An incomplete row returns `NO_COMPLETE_FRAGMENTED_ROW` with the original chain
cursor. Rejected/resource-blocked attempts also retain the original cursor.

This is a bounded row verification primitive, not a general streaming parser or
recovery certificate. A row still incomplete after two fragments is unresolved;
callers must not skip it or infer its full size. Larger-row handling, authenticated
baseline and fragment histories, durable row receipts, reconstruction of the full
tail and cross-store recoverability remain unfinished. JSON parsing may allocate
more RAM than raw input size and cannot preempt a filesystem call or a single
parse; CPU containment and input bounds are not a hard RAM/IO cap. Receipts and
private ancestors must be authenticated by the caller. No production writes,
provider requests, full-prefix acceptance or deployment authority are added.

The verifier now accepts a first fragment containing the supplied cursor, even
when that fragment starts earlier. Its entire digest is still rechecked, and a
cursor inside it must follow a newline. `bytes_before_cursor` reports that
preserved prefix; it does not authenticate earlier chain history. Later fragment
receipts must remain exactly contiguous. At a fragment end, callers must select
the next containing fragment rather than keep an exhausted first fragment.

`max_records` defaults to 1 and can be an integer from 1 through 256. A batch
validates only complete rows, reports the exact last verified boundary and leaves
partial/record-limited overread explicit. Any malformed later row, changed file
or resource rejection retains the supplied cursor for the entire batch, even if
earlier rows were checked. This allows reuse of preserved overread without new
production copying. It does not change the 2 MiB input limit, authenticate caller
history, loop automatically, update a recovery manifest or authorize deployment.
