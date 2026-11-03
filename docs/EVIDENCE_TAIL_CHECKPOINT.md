# Bounded evidence-ledger boot: the tail checkpoint

## Baseline (2026-11-03)
After spot restarted at 20:46Z, it stayed in the `prediction` boot phase for more than 40 minutes at its 5% CPU cap. `trade_evidence.jsonl` was 13.54 GB. The protected recovery certificate is dated 10-28, so every row appended since then was JSON-decoded and re-hashed one at a time on every boot. That cost grows with the history.

## Change
The ledger keeps a running SHA-256 over every byte after its verified base. For certified recovery the base is the certificate extent; without a certificate it is 0. Every 30 s it writes `<ledger>.tail.json` atomically, holding `{base_extent, extent, sequence, last_hash, segment_sha256}`.

At boot the ledger accepts that checkpoint only when all of these hold:
- its base equals the verified base;
- streaming SHA-256 of the bytes from the base to its extent matches;
- the extent is a row boundary;
- the boundary row re-hashes to the recorded sequence and hash.

Rows after the extent are still chain-verified one by one, and anything else falls back to the existing path.

## What is unchanged
- The certified prefix is still fully SHA-256 hashed.
- A bad or missing certificate still blocks appends; the checkpoint is never consulted then.
- No history is rewritten, and appends produce byte-identical lines (`os.linesep`).
- `TradeEvidenceLedger.verify` remains the full offline chain audit.

## Integrity argument
Bytes past the base are protected only by the service-written hash chain, which anyone able to write the ledger could recompute. Per-row verification of those bytes therefore detects accidental damage, and nothing more. A SHA-256 over the same bytes, compared with the checkpointed digest, detects the same damage at disk speed instead of JSON-decode speed. The protected certificate stays the only defence against deliberate rewriting, and it is untouched.

## Cost
Boot now costs one sequential hash read of the ledger plus decoding of the rows written in the last ≤30 s. The previous cost was decoding every row since the certificate. The first boot after deploy still pays the old cost once, and then writes a checkpoint.
