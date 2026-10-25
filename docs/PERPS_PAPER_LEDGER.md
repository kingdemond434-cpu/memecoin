# Perpetuals forward paper accounting

The runtime indexes the paper ledger inside `phantom_perps.sqlite` on the
miner worker, after canonical raw-batch retention. Its fills have no signing,
account-query, deposit or live-order path. Canonical retry references are
idempotent and each paper journal event points to its source batch.

## What runs

Each discovered, metadata-validated USDC contract can produce independent
long/short controls. Every control account starts with a clearly synthetic
100 USDC balance, submits a 10.25 USDC probe at up to 2x leverage and requests
an exit after ten minutes of holding. This is an execution/cost measurement
control, not a directional signal, fixed testing period, capital allocation
or promotion rule. Accounts are separate counterfactual worlds; summing their
balances does not describe deployable capital or simultaneous capacity.
No total-contract ceiling is imposed. Unsupported collateral remains explicit.

An intent matches only a fresh book whose provider timestamp is strictly after
the recorded decision plus modeled latency (300ms). Recorded asks/bids are
walked at the venue's observed size precision. Price limits, available depth,
minimum opening notional, full-request margin admission and IOC cancellation
remain explicit. Reduce-only exits cannot flip a position. Positions and
pending intents survive process restart. Old observations cannot bootstrap
historical fills in a newly created study.

Exact displayed fill notional is retained separately from the possibly
repeating average price. Partial exits allocate remaining entry notional and
the final exit consumes the remainder, preserving cash and fee conservation.
The 9.5bps fee is a frozen scenario input, not verified account/deployer fees.

## Margin and liquidation

Cross and isolated collateral are separate. Observed venue margin tables use
continuous maintenance deductions and the higher tier at an exact boundary.
For published single-tier IDs below 50, the documented ID/leverage convention
applies. Missing tables, stale marks or unpriced past funding leave risk
unmeasured. Opening/increasing risk cannot treat those inputs as zero.

Isolated entries allocate initial margin, a modeled entry-loss buffer and
their fee from free cash. Partial closes keep their residual collateral in
the position until flat. Isolated insolvency cannot consume unallocated cross
cash; deficits are recorded separately, with insurance treatment unverified.
This collateral-allocation policy is explicit in this study and is not a
claim to reproduce every exchange account setting.

Observed mark breaches create a reduce-only liquidation intent for a future
book. Large positions use the documented initial 20%/30-second behavior.
Residual equity below two-thirds of maintenance after an attempted book
liquidation can produce a modeled backstop collateral transfer. That is a
separate journal event, never a manufactured book fill. Cross transfers do
not seize isolated collateral and isolated transfers do not seize cross
cash. Unseen mark/book states between samples remain a measurement gap.

## Funding is not silently zero

Hourly exposure is reconstructed from the fill/transfer history, including
positions closed before the funding-history response arrives. Missing settled
rates or prior oracle samples create durable `UNPRICED` obligations. Today's
mark cannot replace a missing historical billing oracle. Funding rates come
from retained funding history, not the indicative current funding field.
Actual retained settlements are often a few milliseconds after the nominal
hour. Liabilities match that hourly bucket, while exposure and oracle
availability use the actual settlement timestamp. Entries and exits between
those two instants therefore receive the correct treatment.

With a sufficiently recent prior oracle sample, the ledger can calculate a
separate sampled-oracle scenario using signed size × oracle × settled rate.
It is labeled `SAMPLED_ORACLE_FUNDING_SCENARIO_NOT_EXACT_BILLING`; it is not an
exact venue billing receipt. Duplicate responses cannot apply it twice, and
later prices cannot fill earlier oracle gaps. Modeled P&L remains unavailable
when a required cost or mark is missing. Verified net profit is always
unmeasured until actual execution/cost evidence establishes it.

## Inspect and preserve

`/status` → `phantom_perps.paper_ledger` exposes the active study, controls,
positions, simulated fills, order dispositions, funding availability, sample
freshness and bounded account examples. To inspect every account by pages:

```text
python -m tools.perps_paper_report --db data/state/phantom_perps.sqlite --limit 100
python -m tools.perps_paper_report --db data/state/phantom_perps.sqlite --after <next_cursor>
```

The command opens SQLite read-only. No writer is constructed and no wallet
query occurs. Frozen plan/source changes create a new named study while all
prior study rows remain retained. Back up this SQLite database with SQLite's
backup API together with its canonical raw source archive.

## Primary mechanics

- [Funding uses the oracle, paid hourly](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding)
- [Maintenance tiers and deductions](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margin-tiers)
- [Liquidation and backstop behavior](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations)
- [Keyed public mark/context snapshots](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
