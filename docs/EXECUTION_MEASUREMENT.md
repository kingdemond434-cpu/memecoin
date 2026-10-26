# Fill-bound transaction cost analysis and markouts

The owner's quant donor at `921ed183a75b190e52e2eb1cdb67b4b16c7dac8b`
separates execution cost from the subsequent market move. The destination now
implements that method in native raw units, without MT5 point sizes, copied
calendar gates or an assumed fee constant.

`ExecutionEngine._wait_for_fill` retains confirmed owner token-balance changes,
native balance arrays/account keys, signature, slot, block time, network fee
and the local confirmation-receipt clock. Both native and Jupiter results
carry this measurement context into episode execution attempts. Entry,
scale-in and exit attempts retain their decision-intent clock. The arrival
clock is execution-engine entry, before its build/quote/submission work.
Transaction metadata requests accept Solana transaction version 1 as the
stream adapters already do. No signing or live permission changes occur.

Decoded Pump reserve events now retain an exact integer-ratio marginal price
reference, raw quote per raw token, its source event time, local receipt/availability
time and signature/slot provenance. Old source events cannot be refreshed merely
by processing them later. Block timestamps still have their original coarse
resolution. These references are not executable quotes for a
position size, bid/ask spreads or guarantees of sellability. Aggregator prices,
relative multiples, USD-per-token values and other quote assets cannot be
mixed into this tape.

Episode serialization includes `execution_measurement`: signed arrival and
decision shortfall, effective reference deviation and mature 5s/30s/5m/1h
markouts, with available input references. The reference lookup is at-or-before
the requested instant, never nearest/future, and refuses old or unavailable
quotes. Every horizon has a measured, immature or blocked status; unavailable
observations never become zeros. Reference types cannot change across a
markout. Buy and sell costs retain adverse-positive signs.

The live adapter anchors to **confirmation receipt**, because RPC `blockTime`
does not provide precise execution time. Reports explicitly identify this
clock. They must not be advertised as precision fill-time markouts. Paper
attempts and unfilled intents are separate; a dry-run quote cannot establish a
real fill. Wrapped-native input without a verified token balance decrease
remains amount-blocked rather than borrowing the planned input amount.

Fees embedded in actual token balance changes already affect the fill price.
Network costs, rent and separately confirmed bundle tips need independent
reconciliation; cross-asset fees need a contemporaneous FX reference. The
initial native adapter reports these as incomplete. Gross measurements may
be available while net markout remains unavailable. A normalized caller can
supply explicitly reconciled external costs in the same quote unit; absence
never means free execution. Spread, own impact and timing are not falsely
separated using forecast values alone.

To audit a retained corpus without writing into it:

```powershell
python -m tools.execution_measurements --episodes C:\corpus --output C:\receipts\measurement-new.jsonl
```

The output must be fresh and outside the corpus. The tool streams episodes,
retains blocked-file reasons and writes a summary. It creates no fills,
positions, promotion receipts or profitability evidence. Horizon coverage
still depends on actual tape continuing long enough after each execution.

The audit also groups cost-complete markout returns by quote mint, raw unit,
reference kind and horizon. It normalizes each net markout by its actual
observed fill notional and deduplicates identical fill observations. Conflicting
duplicates fail the audit. These are marginal reference markouts, not realized
strategy returns or independently executable exits.

`sample_measurement.py` adapts the owner's quant DSR and sample-geometry
methods. Raw rows, distinct launches and maximum disjoint observation windows
are reported separately. Their minimum is an independence **upper bound**;
serial, common-factor, entity and regime dependence remain unmeasured, so an
effective sample size is not invented. No regime discount constants are copied.

Per-observation Sharpe and an IID moment approximation to PSR require finite,
nonconstant returns and enough observations to estimate sample kurtosis. DSR
additionally requires the complete selection trial count, observed cross-trial
Sharpe variance and a provenance reference, supplied together with
`--trial-count`, `--trial-sharpe-variance` and `--selection-evidence-ref`.
These inputs must describe the selection universe, not just the winning
candidate. Missing selection history produces an unavailable DSR, not one
assumed trial. The tool does not verify a supplied external register or certify
independence. It reports conditional statistics and never grants execution
authority, sets a calendar waiting period or declares profitability.
