# Native execution-cost units — 24 October 2026

Independent research against [Solana's fee structure](https://solana.com/docs/core/fees/fee-structure)
and [RPC reference](https://solana.com/docs/rpc/http/getrecentprioritizationfees)
found that the recent-priority-fee miner labelled native compute-unit prices
as total lamports. The runtime report and dashboard repeated that label.

The miner/report/UI now explicitly use micro-lamports per compute unit. Raw
per-slot samples are retained with the summary so the calculation is auditable.
These are a node's recent slot minimums, not every competitor's bid or a
probability of landing. They do not supply an observed total transaction cost.

For decoded legacy/v0 ComputeBudget instructions, an explicit valid limit yields
the total priority fee using integer ceiling:
`ceil(unit_price_micro_lamports * requested_CU_limit / 1_000_000)`.
The decoder previously assumed a universal 200,000-unit budget when no limit
was present. Actual defaults depend on the instruction mix/runtime, so the
total now remains unmeasured in that case. Unknown declared budgets are counted
separately from transactions that supplied no priority instruction.

Historical persisted data is not rewritten or relabelled. Jito tips remain
separate SOL/lamport transfers. Network/base fees, failed-transaction fees, rent,
token transfer fees and AMM fees still require their own native receipts;
this correction does not infer them or change execution authority. A synthetic
or unsigned simulation is not an observed fill.
