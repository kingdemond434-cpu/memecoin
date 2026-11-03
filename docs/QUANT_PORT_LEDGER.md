# Quant -> memecoin port ledger

This file tracks every mechanism the MT5 quant has proven useful, and whether it is ported to the memecoin desk. Status is one of PRESENT, PORT, PORTING or SKIP. Evidence is a file path in this repo; each status carries its reason. Started 2026-11-03.

Ranking is by expected E[log W] impact on the memecoin goals: capturable early breakouts, survival, and the budgeted tail sleeve.

| # | Quant mechanism | Status | Evidence / reason |
|---|---|---|---|
| 1 | Gauntlet + counted trials + DSR/PBO promotion bar | PRESENT | `src/research/promotion_gate.py`, `src/research/sample_measurement.py`. The trial count must include every shadow hypothesis family, so audit that. |
| 2 | E[log W] / risk-constrained Kelly sizing | PRESENT | `src/research/calibration.py`, net expected-log-wealth head in `src/main.py` |
| 3 | Empirical-Bayes shrinkage of per-strategy edge | PORT | Absent: no shrinkage of family/strategy means toward the pooled prior before Kelly. Memecoin samples are tiny and fat-tailed, so this matters more here than in FX. |
| 4 | Loss-learning loop (every closed loss gets a cause and a rule proposal; gauntlet judges) | PORT | Absent. Fold it into #6 as one post-trade/miss auditor. |
| 5 | Weak-signal combiner (many net-of-cost weak edges, crisis-correlation sizing) | PARTIAL | `src/strategies/multihead_predictor.py` ensembles heads. There is no per-producer net-of-cost ingestion ledger and no auto-feed of new producers. |
| 6 | Missed-trade watchdog -> breakout miss audit (coverage/latency/attribution/decision/execution/exit) | PORT (top) | `src/runtime/discovery.py` records misses. The brief's per-breakout classification, with same-cohort rugs for survivorship, is not built. |
| 7 | Decay monitor (regime-aware, fast retire) | PRESENT | `src/research/promotion_gate.py`, `src/runtime/source_intelligence.py` |
| 8 | Drift monitor | PRESENT | `src/research/shadow_trainer.py`, `src/runtime/reporting.py` |
| 9 | Meta-labelling (secondary model: take/skip a primary signal) | PORT | Absent. A cheap filter on the rug/sellability head for the sniper path. |
| 10 | Fill-cost / TCA model, markouts | PRESENT (unproven live) | `src/research/execution_measurement.py`. Needs real paper fills (brief step 2). |
| 11 | Position manager: BE, trail, partials, market-state exits | PRESENT | Profit ratchet, partial-exit cost basis and adaptive exits in `src/main.py`. A BE stop after a cost-recovery partial is equivalent. |
| 12 | Worldwide strategy hunter / unknown-unknowns | PRESENT (framework) | Shadow-only research hypotheses, `src/research/population_hunter.py` |
| 13 | Dataset hunt (free public sources) | PRESENT | `docs/FREE_FEEDS.md`, source mesh |
| 14 | Producer-yield ledger (raw -> candidate -> plan -> fill -> PnL per producer) | PORT | Source lead time is measured. Conversion to paper PnL per producer is not. |
| 15 | Hourly pass monitoring | PORTED | quant item 39 reads `deploy/windows/memecoin-health.ps1` output (#144) |
| 16 | Status-write cannot kill a capped service | PORTED | #143 |
| 17 | Event-driven discovery (world event -> related new tokens) | PORT | zuck 11-03. Social/narrative mesh exists, but there is no event -> ticker/name matcher on fresh launches. |
| 18 | Tail (lottery) sleeve with an explicit loss budget | PRESENT | `tests/test_asymmetric_tail.py`. The sizing must be checked against #3. |
| 19 | Clock law / PIT availability | PRESENT | `research_available_at`, PIT episodes |
| 20 | Prop-firm rules, MT5 gateway, FTMO parity | SKIP | Not applicable to on-chain spot. |
| 21 | Registry merge path, sealed signing | SKIP | This repo merges through GitHub CI; no registry here. |
| 22 | quant #585 loss-learner v2 (losses + missed trades -> tweak trials) | PORT | Same as #4/#6; the trials go through promotion_gate with counted trials |
| 23 | quant #610 loss-learner drift watch | PORT | It rides with #22 |
| 24 | quant #581 ntfy digest alerts | SKIP | The hourly pass relays memecoin REDs from MEMECOIN_HEALTH.json, and a second alert path would duplicate it |
| 25 | quant #612 replay prop lane | SKIP | It is prop-firm specific; `src/research/candidate_replay.py` covers replay |
| 26 | quant #579 deterministic reviewer pattern | SKIP | This repo gates through GitHub CI plus the Auditor |
| 27 | PSI/KS feature-drift early warning on live inputs (idea 278) | PRESENT/CHECK | Drift exists (`shadow_trainer.py`); confirm it is computed per input feature, not only on the score |
| 28 | Horizon fan-out as charged trials (idea 279) | PORT | Hold horizons of 10s-24h per family, each counted in the DSR trial total |
| 29 | MFE/MAE + stop path per trade (idea 280) | PORT | It feeds the exit research; `paper_exit_facts.py` is the place |
| 30 | James-Stein shrinkage before Kelly (idea 281) | PORT | The same item as #3 |
| 31 | Lawful discovery: Common Crawl CDX, GDELT, RSS (idea 263) | PORT | GDELT is the event feed for #17 event-driven sniping |
| 32 | Item 38 exit set (BE, trail, partials, time, regime/news/liquidity exits) | PRESENT/PORT | Ratchet, partials and adaptive exits exist; the liquidity-drain and news exits get added with #17 |
| 33 | Account parity | PRESENT (n/a) | A single paper book; recheck when Phantom/perps books go live |
| 34 | Combiner auto-enrol (#604/#618) | PORT | The same item as #5 |
| 35 | quant #630 CI account-parity gate | PORT (later) | Once Phantom/perps books exist, add a CI gate that every book runs the same veto, sizing and exit mechanisms |
| 36 | quant #637 signed release attestations | SKIP | This repo releases through GitHub CI receipts plus the cap-aware bounded release |
| 37 | quant #622 loss-learner min-lot read | SKIP | There are no lot minimums on-chain; the analogue is the dust/min-notional guard already in sizing |
| 38 | quant #628 exit overlay grid (liquidity-drain exit) | PORT | Same as row 32; memecoin depth collapses faster than FX, so test a drain exit against the ratchet |

Build order after spot is restored and paper accounting is proven: 6+4, then 3, then 17, then 14, then 9.
