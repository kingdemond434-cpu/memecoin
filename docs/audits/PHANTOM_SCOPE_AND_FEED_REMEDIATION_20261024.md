# Phantom scope and feed assessment remediation

The requested prediction venue is Phantom's native World prediction markets.
Polymarket and standalone Kalshi are not substitutes for that request.

## Verified operational baseline

PR30 passed 3,494 isolated Windows VPS tests (106 skipped), plus 3,591 Linux
tests exercising the native extension. All 563,698 retained canonical records
passed hash-chain verification before the guarded main deployment. PR31 then
fixed an orphaned prediction writer by reusing the existing Windows job-object
owner; two real Windows launch/stop/restart checks verified both descendants
terminated and the same study ledger survived. The Python/config/native/test
trees were unchanged by that launcher-only follow-up.

The main Solana service was subsequently restarted successfully as LocalService
with an AtStartup trigger, limited privileges and its existing locked dry-run
launcher. It no longer requires an interactive Administrator login. This is a
verified account/startup configuration, not an observed machine reboot.

Following the explicit venue clarification, the standalone Kalshi/Polymarket
task was disabled and stopped. Both original study directories are retained;
the paused state-v2 backup passed SQLite integrity. Their paper outcomes and
losses must never be presented as Phantom-native evidence.

## Assessment findings and action

| Finding | Remediation / acceptance boundary |
| --- | --- |
| Missing native RPC access | Operator supplied an RPC credential. It is stored outside Git under restricted ACLs and passed getVersion/getLatestBlockhash probes. A private, literal-data connection loader supports the service account; no wallet secret or live acknowledgement is accepted |
| Retry storms and finite RPC allowance | Shared durable HTTP reservations default to five requests/second, 25,000 allocation units/day and 750,000/month. Archival-heavy requests reserve conservatively; sendTransaction is separately limited. Counters survive restarts and separate processes. These are allocation units, not verified provider billing; streaming and other applications remain unmeasured |
| All providers cooling still selected | Previously the fallback immediately retried a cooling endpoint. Selection now respects its refusal window even if no endpoint remains available; missing data remains blocked rather than generating another request |
| Credential privacy | Health output exposes origins only. Transport exceptions no longer export URL text. Credentials are loaded as allowlisted literal strings, never evaluated as shell code or loaded into the Git checkout |
| Every judged token has authority vetoes | Native SPL COption tags, account owner and timestamp must be inspected per mint. No risk flag or veto is disabled merely to obtain entries. A current account read cannot retroactively prove a historical authority observation false |
| Predictor rejected / negative expected growth | Rejection is retained. Native labels, costs, chronology, controls and untouched outcomes are still required. Feed access does not establish predictive skill and must not force promotion |
| Empty follow_candidates | This file is pending wallet-follow outcome work, not the creator database. Existing actor/genealogy stores retain creator relationships; the separate donor audit maps creator-history techniques to those stores |
| Phantom prediction mismatch | Gateway defaults to a new Phantom-only research store. Legacy venues require an explicit separate mode and cannot reuse the native store or be relabeled. Production legacy experiments are paused |
| Phantom-native execution access | Phantom's current World support article states external development/public API is unsupported and the feature is mobile-only. World's official frontend names its current catalog host; an ordinary unauthenticated request returned HTTP 401, while operator docs require an access code. No login challenge, access gate or private endpoint is bypassed |
| Native World research | Bounded public metadata access probes and independent public-chain program-candidate observations are retained. Executable account presence is not verified market identity. Native market mapping, authorized quotes, rules, depth/cost and settlement evidence remain required before executable paper fills |
| Claimed insider/social/dark-web gaps | Existing public Telegram, regional, source-claim and onion research remain separate from authenticated streams. Missing source-specific access is reported explicitly. Public claims never become verified insider identities or live authority |

## Source proof for the actual requested provider

- Phantom's World feature and API boundary:
  https://help.phantom.com/articles/trade-predictions-in-phantom-47187740851731
- Provider disclosure: https://phantom.com/prediction-market-disclosures
- Official frontend entry inspected: `https://world.xyz/assets/index-BfLG9oFW.js`
  SHA256 `92ec43ecdc069336810c93ccff81354aba113d19b713a1d241065ca5fe0ea708`.
  It imports `https://world.xyz/assets/api-CcCaG8Rc.js`, which names the
  metadata host used by the new read-only observer. The current response was
  HTTP 401. This does not establish authorized trading API access.
- Independent candidate-program research:
  https://github.com/chainstacklabs/world-xyz-research . This is not an operator
  identity certificate, IDL guarantee or quote/fee/settlement receipt.
- Helius plan/credit information: https://www.helius.dev/pricing . A credential
  alone does not establish its plan, remaining project credits or the usage of
  other applications sharing it; configured reservations retain that distinction.

The native study reports zero simulated and real fills while executable
inputs are blocked. A documented frontend fee formula is exposed only as a
scenario calculation, explicitly not an observed quote or measured cost.

## Current native probes, distinct from historical decisions

The supplied credential returned successful native version/blockhash reads.
A public program-candidate account read at slot 461001091 observed executable
upgradeable-loader state with account SHA256
`b71836a736614dc552be40d65b25bc92cb4778dfea30a7d86bfe58414ef07e57`.
That verifies account presence, not operator/market identity or tradability.

On the VPS, five currently retained authority-vetoed mints were read together
at finalized slot 461001396. All five currently had both authorities revoked.
Their retained rejection timestamps were from several days earlier; this is
not a reconstruction of those accounts at rejection time. Historical veto
records are retained. Cumulative census totals must not be labeled a recent
ten-hour window without a separate timestamped window census.
