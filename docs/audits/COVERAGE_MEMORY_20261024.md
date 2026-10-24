# Coverage, research memory and remaining requirements

This release extends the existing paper/public-research sidecar. It introduces
no signer, live authority, executable strategy scheduler or promotion path.

## Implemented behavior

- The main desk exports sanitized source health and the retained-launch
  positive-peak review at `/intelligence-coverage`. Every positive observed
  peak can become a review lead, including moves smaller than 10x. Entered
  launches are excluded from missed-entry leads. Unresolved observations stay
  unknown. Evicted historical launch details are explicitly outside this view.
- The paper sidecar records source gaps, hindsight review leads and bounded
  planning rounds in its existing SQLite database. Eight proposals per round
  reserve two slots for rotating blind-spot domains. These are advisory research
  proposals requiring verified inputs; they never execute themselves or change
  code, capital policies or model weights. Existing research/promotion gates
  remain authoritative.
- `config/intelligence_scope.yaml` declares intended domains, actor classes,
  chains, languages and regions. Declarations never count as observations.
  Main snapshots expire after 180 seconds. Dynamically discovered sources are
  reported separately from declared targets. Healthy polling and actual event
  observation are separate fields. Recovered-source history remains retained
  without consuming current repair proposals.
  Proposal selection seeks a bounded indexed window with a durable cursor;
  it does not reload the complete accumulated gap history into memory.
- Public wallet analysis recalls exact prior work only for the identical native
  venue, wallet, ordered input IDs and algorithm hash. Changed inputs, venue or
  algorithm trigger fresh work. Result checksums and conflict rejection guard
  reuse. Operational compute reuse is measured; economic research ROI remains
  unmeasured. Missing labels are never recorded as economic failures.
- The paired operator page displays coverage freshness, proposals and memory
  measurements. Health fails if paper, wallet-research or coverage workers die.
- A Windows suite exposed a transient access-denied error during atomic
  promotion-state replacement. The same write retries up to five times only
  for Windows sharing/access error codes. Persistent denial raises and leaves
  the previous authoritative file intact. Promotion criteria are unchanged.
- Two VPS stack samples found wallet valuation inside the continuous hazard
  sweep while HTTP health timed out. Wallet rankings and reports now compute
  their common population prior once per call, preserving exact value/rank
  results. Each hazard sweep shares one contemporaneous wallet ranking and
  yields between tokens; tokens evicted during a yield are skipped.

## Requirements that software alone cannot complete

| User objective | Current implementation | Remaining acceptance evidence |
| --- | --- | --- |
| Global launch/actor/insider-claim coverage | Existing world, chain, Telegram and public-onion miners; public claim/address links; explicit scope and gap reports | Verified endpoints and permitted access per source; measured events and independent denominator. Neither all onion sites nor all global launches is enumerable here |
| Broad Phantom/Solana execution | Native Pump/PumpSwap/Jupiter/Jito routes and wallet bridge | Wallet-owner browser session and native fill/cost/landing evidence; Phantom signatures require owner approval |
| Prediction trading continuously | Public Polymarket/Kalshi discovery, paper books/fees/depth, controls and complete-set candidate | Fresh server deployment, unattended main availability, independently evaluated resolved after-cost outcomes |
| Quant machinery transfer | Inspected transfer map in `QUANT_TRANSFER_20261024.md`; validation, forensics, exact-work reuse and existing native machinery | Native market-specific labels, prospective comparisons and validated incremental contribution. Donor weights and MT5 mandates are not evidence for these venues |
| Self-improvement | Existing candidate/frontier/validation system plus durable gap proposals and research memory | Real candidate outcomes, compute costs, chronological validation and untouched forward evidence before promotion |
| Miss no profitable opportunity | Positive-peak and existing missed-monster audits with explicit unknowns | Causal entry/exit availability, size and all costs; hindsight peaks cannot establish executable missed profits |
| Maximum growth and prediction accuracy | Geometric-growth allocator, tail heads, local typed-model framework, strict gates | Calibrated native labels, after-cost comparisons, enough rare outcomes and actual elapsed forward time; no profitable edge established |

## Deployment boundary

At the start of this release, GitHub main contains PR29 at
`730b34deabfb10dc5d48ad15dd81fcbfc4fab428`. The last verified main VPS deployment
is PR28 merge `02145a517d9f44f0ac1b5ccbe2cb33b0616a9b89`; its separately deployed
paper sidecar remains on release `14e7a52968e54a663eee3dd2a3e71e02c9af1661`.
The expanded release was prepared on the VPS, but SSH is timing out and its
final isolated test receipt has not been retrieved. RDP reachability does not
confirm the bot's health. Do not label this release deployed there.

The expanded experiment is temporarily running locally in a separate state
directory. This release leaves `prediction_markets.py` and its frozen paper
specification unchanged, permitting a restart with the same experiment ledger.
Local operation requires the host and process to remain running. The VPS main
task still depends on the interactive Administrator session; the approved
LocalService prediction task is separate. No live trading is enabled.
