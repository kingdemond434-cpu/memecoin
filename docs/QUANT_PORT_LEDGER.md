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

Build order after spot is restored and paper accounting is proven: 6+4, then 3, then 17, then 14, then 9.
