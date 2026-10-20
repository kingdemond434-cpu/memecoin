# Production research readiness

The research engine remains in paper mode. Native quote parity, immutable
replay inputs, historical survival, untouched forward results and actual
trading authority are separate capabilities.

## Observed data inventory, 20 October 2026

The isolated server deployment contained 888 compressed observed episodes at
capture, including 583 resolved outcomes. The preserved older training tree
contained 31,247 compressed episodes. Neither tree contained a single
`replay_execution` native fill/cost record; the current deployment also had no
execution attempts. Their price paths cannot be promoted into executable
replay by assigning assumed fees or marking quotes as fills.

The server environment had no Anthropic, OpenAI or DeepSeek API credential,
and no Helius, Alchemy or Yellowstone credential. Presence was inspected
without printing values. Existing public RPC and public-source collection
continue independently. Provider configuration now reports disabled/missing
model and credential environment names before a proposal round starts, with
zero authentication or paid API requests.

## Preparing the actual replay corpus

`python -m tools.prepare_research_corpus --episodes <directory>` reports
missing native clocks, routes, depth, latency-specific fills/costs, unresolved
outcomes and duplicate launch identities. It does not modify the source.

For a complete native corpus, declare explicit past `[start,end]` windows for
`train`, `validation` and `test` in JSON. The gaps between windows are the purge
intervals. Run with `--windows <json> --output <fresh-directory>`. Every launch
whose creation falls inside a window must pass admission; a difficult,
unresolved or partition-overlapping launch blocks the complete cohort. Return
or safety labels never select the cohort. The report retains source hashes,
all window assignments and admission failures even when blocked.

A successful seal copies the immutable observations, records compressed file
hashes and the existing canonical content hash, and freezes the split clock.
Configure the existing `research_replay` adapter against that directory and
its `split.json` **before constructing candidate variants**. Later byte
changes, unlisted episodes and missing files are refused by the actual replay
handler. A seal is input preparation, not a passing gauntlet verdict.

`config/research-production.example.yaml` describes replay resource bounds
and provider environment declarations. It contains no secret values and does
not enable providers or replay implicitly.

## Native EVM probes

`global.evm_native_probes` accepts at most four explicit Uniswap V2 routes.
Each declaration supplies the registry chain, factory/router, input/output
token addresses, exact raw input amount and official deployment provenance.
The existing collection loop rotates through at most two probes per pass;
the existing registry owns transport startup, retries and shutdown. There is
no additional scheduler, signer or ledger.

Each probe verifies chain identity, safe/finalized block, native contract
code, factory/getPair relationships, token ordering and reserve ABI widths.
All contract reads are pinned to the canonical block hash using EIP-1898.
Exact integer reserve quotes in both directions must match the native router
at that block. Solidity overflow, malformed ABI, changed block or disagreement
blocks the receipt. The quoted exit uses the same observed state; it does not
pretend that future exit liquidity or the entry's state impact was observed.

An explicit public wallet address optionally enables **independent** entry
and exit `eth_call` probes with real balance/allowance checks. Gas estimates
remain estimates. Calls do not persist their state, so two successful calls
are never described as a sequential round trip, a real fill or future exit
reliability. No approval, transaction signing or submission occurs.

The desk retains successful and blocked probe receipts in its canonical
ledger. Successful raw output measurements flow through `DATA_OBSERVATION`
into the existing typed data catalogue, with actual receipt availability and
the `forward` partition. The diagnostic `tools.probe_native_routes` writes a
fresh standalone report and never races the running canonical ledger writer.

The Ethereum USDC address and Arbitrum V2 factory/router declarations were
corrected against [Circle's contract inventory](https://developers.circle.com/stablecoins/usdc-contract-addresses)
and [Uniswap's deployment inventory](https://developers.uniswap.org/docs/protocols/v2/deployments).
Math/ABI references are retained in the implementation and each successful
receipt. See [EIP-1898](https://eips.ethereum.org/EIPS/eip-1898) for canonical
block-hash state queries.

## Remaining real-world requirements

- Actual provider credentials, selected model IDs and explicit operating
  budgets must exist before paid proposal calls can run.
- Native measured fill/cost data and frozen chronological corpus are needed
  before historical experiments. Legacy price-only episodes stay blocked.
- Uniswap V2 read-only probes do not establish EVM trading authority, other
  protocol math, other virtual machines, tax-free transfers or future exits.
  Further ecosystems require measured opportunity and their own verified
  semantics, as the implementation sequence requires.
- Historical survival must precede prospective registrations. Untouched
  forward cohorts require new events and actual elapsed observation time;
  they cannot be supplied by a synthetic test or relabeled history.
- The existing Windows task still uses the interactive Administrator session.
  RDP disconnection is supported; logoff stops that deployment. Switching to
  an unattended account was rejected by automatic approval review and was
  not performed.

Starting real public EVM backfills exposed long initial observation passes.
Each scouting/backfill call now has a 20-second deadline, and the same loop
drains research work between source calls and native probes. Partial native
backfills retain their original checkpoint/evidence; cancellation releases the
actual observer's durable lease. This is a network work bound, not a claim
about decision-path latency or completed historical coverage.

All of these conditions remain explicit blocks, without synthetic performance
claims or bypassing the existing promotion and live-submission locks.
