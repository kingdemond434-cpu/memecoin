# User review requirements: evidence and implementation gaps

This records the complete attached review's scope without treating tests or
architectural breadth as proof of returns. All live-capital gates remain in
force. The October 20 verification is a dated baseline, not current live PnL.

| Requested capability | Existing implementation / verification surface | Remaining evidence or work |
| --- | --- | --- |
| Broad launch census, including rejection | `launch_census`, chain observers, discovery reports | Coverage denominator per chain; unsupported sources cannot be counted |
| Public prelaunch and actor intelligence | `prelaunch_intent`, funding, genealogy, Telegram and world miners | Verified source endpoints, measured independent delayed follow outcomes |
| Calibrated extreme-tail prediction | Multihead targets through 500x, calibration book | Positive rare-event evidence and untouched calibration; this change adds descriptive Wilson intervals rather than extrapolation authority |
| Adversarial and catalyst intelligence | Authenticity, public coordination, social claims, insider-event and pre-event anomaly research | Verified identities/catalysts, causal holdouts and measured incremental value |
| Autonomous alpha discovery | Frozen exact candidates, mechanism labs, machine search, L0–L5 receipts | A native fill/cost corpus, historical survivors, untouched forward cohorts |
| Execution throughput and competition | Native routes, transaction kernel, bid curves, feed race and latency tools | Sustained node-specific bursts, real landing/cost evidence; source counts are not throughput benchmarks |
| Counterfactual actions and exits | Action-value policy/corpus, exit policy trainer, reentry, allocator displacement | After-cost paired policy comparisons on identical eligible cohorts; legacy price-only replay is insufficient |
| Growth and extreme-upside sleeves | Opportunity allocator and budget-limited asymmetric-tail sleeve | Independent comparisons against cash, no-displacement and ranking baselines |
| Target-reaching probability and survival | Wealth-target simulator and ticket-size diagnostic | Existing IID/survival-curve model is scenario analysis; measured chronology, dependence, elapsed-time targets and executable exit-policy comparisons remain necessary |
| Capacity-aware scaling | Native depth/capture, escape, wallet and chain budgets | Actual size-dependent realized costs and independent regime validation |
| Cross-market/chain operation | Chain opportunity research and native EVM observation/quote parity | EVM quote parity is not trading authority; prediction-market and exchange execution sleeves are not established |
| Missed exceptional-winner audit | Census missed-monster report and executable opportunity-cost attribution | Per-day executable 50x–1000x reconstruction with full decision-time availability and native exit evidence |
| Marginal research-value controller | Decision ablations, contribution and source substitution reports | Realized paired benefit minus source/model/route cost; decision influence alone is insufficient |
| Continuous verification and promotion | Durable canonical evidence, frozen receipts, promotion ledger, calibration and strategy decay | Minimum stage time/volume, real fills, positive lower bound and gauntlet survivors; no backdating |
| Public onion/Dread research | Dedicated SOCKS5 remote-DNS transport, bounded crawl, persistent outbox, canonical narrative intake | Current public Dread/other endpoint identity and source-specific observations; missing URLs and challenges remain explicit blocks |

The public-onion implementation has a standalone real-Tor connectivity receipt
from Tor Project's published onion page. That receipt establishes the network
path only. The Dread example remains unavailable until its current public
endpoint is verified. No universal onion coverage or financial performance is
asserted.

Production profiling also found the existing research loop inside the tagged
job lease query. This change drives auctions from indexed active costed jobs,
keeps archived work/evidence intact, skips an unnecessary priority scan when a
floor already fills the pass, and bounds due-kind existence checks. A regression
test imposes a SQLite work budget with 12,000 retained tagged jobs.

The VPS subsequently showed another main-loop stall in framework state writes.
The population component had grown to roughly 75 MB of the 80 MB checkpoint.
Population observations and replay receipts now commit incrementally in a
single SQLite transaction, retaining the complete legacy data and leaving a
small framework descriptor. See `docs/POPULATION_STATE.md` for migration,
backup and rollback requirements. Availability must still be measured after
deployment; this change does not establish trading performance.

VPS validation on release `83bd3e2` passed 3,388 tests, with 106 skips and 14
subtests passed, in a separate checkout. Linux CI additionally exercised the
native extension: 3,485 tests passed, nine skipped and 14 subtests passed.
These counts describe software checks, not earned shadow time or actual fills.
See `docs/VPS_VALIDATION.md` for isolation and the retained recovery audit from
the earlier interrupted shared-checkout run.
