# Paper fill proof

The spot desk writes one hash-chained `paper_fill` row to
`<ops_state_dir>/trade_evidence.jsonl` for every simulated buy, scale-in and
sell. Every hour (`paper_fill_proof_seconds`, default 3600) its health loop
re-reads those rows incrementally, verifies the chain they sit in, and writes
`<ops_state_dir>/paper_fill_proof.json` (schema `memecoin_paper_fill_proof/1`).

Verdict `PROVEN` requires, in this order:

| check | meaning |
|---|---|
| `chain_ok` | every scanned evidence row verifies against its predecessor |
| `scan_complete` | every committed byte of the ledger has been verified |
| `round_trip` | at least one paper position bought and fully sold |
| `accounting_identity` | each sell's pnl equals proceeds + native delta - allocated cost, and a closed position's allocations sum to its buy cost |
| `fresh` | a paper fill in the last 24 hours |

Otherwise the verdict is the first failure: `CHAIN_BROKEN`, `CATCHING_UP`,
`NO_FILLS` / `NO_ROUND_TRIP`, `ACCOUNTING_MISMATCH` or `STALE`.

A tick verifies at most 256 MiB at 25% duty, so a large existing ledger is
caught up over several hours (`CATCHING_UP`) without taking the desk's CPU
budget. The checkpoint (`paper_fill_proof.state.json`) belongs to the desk.

Independent audit run, own checkpoint, read-only on the ledger:

    python -m src.research.paper_fill_proof --state-dir <ops_state_dir>

It writes `paper_fill_proof.audit.json` and exits 0 only when `PROVEN`.
