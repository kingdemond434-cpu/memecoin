# Launch operator research and paper watch plans

This implements original research machinery derived from the user's video
examples: token → recorded creator → funding ancestors/siblings → prior launches
and repeat observed buyers. It does not copy a third-party bot or require a
custodial signing service, paid GMGN account, browser polling or an LLM loop.

`ResearchWorkflows` registers five consumers in the existing durable queue:

| Kind | Input and result |
|---|---|
| `OPERATOR_ANALYSIS` | A wallet/contract or token seed, explicit `as_of`, and canonical references; returns observed funding associations, prior launches, buyer recurrence and descriptive entry-window reports. |
| `OPERATOR_WATCH` | An evidenced funder and reviewed paper preset: native integer amount band, wallet-history condition, creation/buy trigger and optional exit scenarios. |
| `OPERATOR_FUNDING` | A recorded transfer matching that preset arms a wallet at the actual processing time. |
| `OPERATOR_LAUNCH` | A subsequent creation/buy observation makes one globally token-deduplicated **unfilled paper plan**. |
| `OPERATOR_TRADE` | A compact price observation; ordinary-buyer entry classifications require their own evidence references. |

Every request declares `chain_id`, `chain_family` (`SOLANA` or `EVM`),
`observation_id`, `event_at`, `observed_at` and canonical evidence references.
Clocks must satisfy event ≤ observation ≤ processing time. Conflicting reuse of
an observation ID is rejected. SQLite receipts, arm clocks, first-event consumption
and token deduplication survive restart. Process one owner through the research
queue; do not edit its database externally.

Existing canonical actor observations automatically trigger genealogy analysis
at powers of two in repeated funding/deployment counts. Relevant observations
also feed existing armed watches. This starts no additional thread or polling
loop and makes no RPC requests. With no watches, the ordinary actor path does
not open the operator database. A missing transfer amount or history proof
remains unavailable; graph adjacency alone cannot manufacture it.

## Semantics and limits

- Funding paths and recurring buyers establish **associations**, not common
  ownership, authenticated insider status or profitable coordination.
- Funding hubs are excluded by the graph's existing point-in-time rule.
  Native chain/type identifiers separate contracts, wallets and tokens; Solana
  case is preserved and EVM hex case is normalized.
- Solana funding uses integer lamports; EVM uses integer wei. EVM contract tax
  timing, the video's outcome claims and its slippage defaults are not ported
  onto Solana.
- Historical replay cannot arm a watch in the past. A creation before actual
  processing/arming is recorded as missed. First-buy and first-creation switches
  independently consume the first matching **observed** event after arming;
  they do not prove which transaction was first on-chain. Missed observations
  also consume their configured first-event slot.
- Presets store TP/SL/trailing/timeout scenarios without executing them. Plans
  remain `AWAITING_QUOTE_COST_SAFETY_AND_SELL_VALIDATION`. They are not paper
  fills, do not establish an edge and cannot promote themselves into execution.
  Quote, cost, safety, sell validation and prospective experiment acceptance
  still belong to the existing paper-execution pipeline.
- No wallets are invented from truncated video addresses. No private Telegram
  blueprint, wallet login, paid feed or access challenge is bypassed. The user
  supplied a transcript and discussion, not the referenced private MD/PDF.

## Reports and resource use

An analysis walks the existing bounded graph and pages deployment observations
with `page_size` (default 128; maximum 1024), `node_offset` and `edge_offset`.
`next_cursor` retains the deferred tail; these are per-operation resource bounds,
not source admission limits. Token seeds resolve one recorded creator per
`creator_offset`, with `next_creator_offset` for the rest. Reuse the exact `as_of`
across pages and use a new request observation ID for each page. Denominators
count observed deployment edges; they are not a universal or unique-token census.
Traversal truncation and skipped hubs remain explicit.
The canonical consumer emits deterministic continuation children for remaining
deployment and creator pages, so deferred pages stay in the existing queue.

Buyer analysis reads only the indexed earliest 25 **observed** incoming edges
per token; its deferred count is visible. Recurrence is scoped to the report
page and this prefix. It is not an early-slot census or transaction-order proof.
No winning-outcome filter drops known failed launches from the report.

The price benchmark requires a separately evidenced eligible buy. Ambiguous
ordering is unavailable; price peaks over 2/5/10/30/60 seconds are explicitly
**descriptive price excursions, not executable PnL**. Same-quote samples and
availability clocks are enforced. SQL aggregates only the token's initial
60-second window; lifetime tapes are not materialized in Python. Reports expose
elapsed windows and never claim a complete tape, fees, attainable entry/exit or
out-of-sample performance.

Export an actual retained analysis receipt without reading live databases:

```powershell
python tools/operator_play_report.py --receipt analysis-receipt.json --csv operator-launches.csv
```

The CSV contains chain-typed token/deployer IDs, event/availability epochs and UTC times, evidence references,
benchmark status, entry quote and descriptive window excursions. Missing values
remain blank; formula-looking text cells are escaped. It exports the supplied
page only. Request further pages for the rest.

Deployment must preserve `docs/windows-cpu-budgets.md`. The source feature is
not evidence that the VPS is receiving every needed funding/price classification
or executing these plans. Activation and actual producer/consumer receipts must
be verified separately, while live submission remains locked.
