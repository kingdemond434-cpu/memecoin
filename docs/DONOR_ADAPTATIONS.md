# Donor audit and adaptations — 25 October 2026

Repository popularity, a quote, a provider label or a paper result does not
establish net trading returns. No audited source establishes a profitable
strategy for this desk. No installers, binaries or donor signing examples
were executed. The destination retains its own isolated signer and authority.

## Sources inspected

| Source / pinned revision | License evidence | Useful methods and disposition |
| --- | --- | --- |
| [GMGNAI/gmgn-skills](https://github.com/GMGNAI/gmgn-skills/tree/b0f4ac71355bb565b592c48061ecb2663549fe4b) | MIT LICENSE | Provider token/security/holder/wallet/creator observations; authenticated read API. Provider “insider”, KOL, bundler and smart-money labels remain claims. Hand-weighted scores are not measured alpha. No remote trade authority or private-key import. |
| [chainstacklabs/pumpfun-bonkfun-bot](https://github.com/chainstacklabs/pumpfun-bonkfun-bot/tree/b5d0fd7761cac9db5d61442f16d346467ae0f73a) | Apache-2.0 LICENSE | Native LaunchLab discovery is adapted below. CPI/ALT support, quote-asset identity, token-program and transfer-fee facts matter. Blind fixed-amount entry and plaintext local signer are not adopted. RPC/Geyser service is still required. |
| [vibeforge1111/dexscreener-cli-mcp-tool](https://github.com/vibeforge1111/dexscreener-cli-mcp-tool/tree/177295c5ad3704c8647ee3eb4a3e9598e9224e5f) | README says MIT; no LICENSE artifact at this revision | Independently implement raw windowed volume, transaction imbalance and liquidity ratios; do not copy code or import arbitrary ranking weights. Existing Dexscreener discovery is retained. Paid boosts are attention evidence, not quality. |
| [vineetdev02/Memcoin-Sniper-Bot](https://github.com/vineetdev02/Memcoin-Sniper-Bot/tree/bb2e6fc8250347a6e4e863e4127b7b9d08cf63c8) | MIT LICENSE | Filter attribution and explainable observed facts are useful. Paper-only engine and synthetic screenshots do not establish live execution. Its quality plan explicitly records skipped LP checks, unknown creator history, incomplete insider links and quote-only sell checks; those are not upgrades over our native checks. |
| [sendaifun/solana-agent-kit](https://github.com/sendaifun/solana-agent-kit/tree/1254fe550872ee48d42f168fc8ca854c87ab6eff) | Apache-2.0 LICENSE | Typed protocol adapters, oracle observations and independent program verification. Launch/LP/creator-fee actions are not trading alpha. Existing native routes, oracle collectors and isolated signer take precedence over adding another wallet/LLM execution stack. |
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade/tree/150a10bc2b2c49959975693fafd668f7fb37de7d) | GPL-3.0 LICENSE | Prefix-versus-full-history feature comparison and startup sensitivity are useful diagnostic methods. No GPL code copied. Exchange candle fills are not AMM/bonding-curve fills. No additional calendar, trade-count or promotion mandates adopted. |
| [AL-THE-BOT-FATHER/pump_fun_py](https://github.com/AL-THE-BOT-FATHER/pump_fun_py/tree/104c84ac20de44a84d2b9667ee5a911f8c325e7e) (redirect from Allen-Taylor) | No LICENSE artifact | Small native Pump account/curve examples; reference only. Existing pinned official IDLs and measured routes take precedence. No unlicensed code copied or old constants promoted as current proof. |

[Chameleon](https://chameleonscope.com/) and
[CopyFomo](https://t.me/copyfomo_bot) expose marketing/product material, not
reproducible engines or an audited return series. “Quantum” is presented as a
coin narrative/theme; it supplies no evidence of quantum computation. Useful
product concepts—after-cost forecasts, calibration, wallet histories and
reason attribution—already have native counterparts. The Chameleon prediction
venue and settlement claims concern Polymarket and cannot define Phantom
World outcomes. Screenshots/handles do not prove wallet control. No commercial
bot deposit, key export, installation or trade was performed.

Earlier donor sources (the user's quant repository, Dexter, hexnome and the
other repositories named in the research notes) remain subject to the same
module-by-module provenance and wiring review. This audit is not a claim that
all proposed quant measurement ports are complete.

## Native LaunchLab adaptation

The original destination registry declared the program with no reviewed mint
position, and RPC/Geyser wiring omitted it. The new `LaunchLabMonitor` is
registered before stream startup on primary RPC, primary Geyser and racing RPC.
All platform configuration accounts are accepted, including newly appearing
platforms. Platform names are not guessed from metadata.

Vendored verbatim from Chainstack at the revision above:

| Destination | Upstream | SHA-256 |
| --- | --- | --- |
| `idl/raydium_launchlab.json` | `idl/raydium_launchlab_idl.json` | `4949983335d0a345c91b76f88e91e1e9680230e8190be5e5b9fe537df927668f` |
| `tests/fixtures/launchlab_launches.json` | `tests/regression/raw_stonkfun_launches_from_gettransaction.json` | `599ff4b61939d455d87ff8460c823c044b6e5d60cf44a198f109a5d3404a7ead` |

The full upstream Apache license is in `idl/licenses/CHAINSTACK-APACHE-2.0.txt`.
These public transaction fixtures are attributed examples, not our own forward
observations. Their original signature, slot, block time and metadata remain
unchanged. Parser/runtime integration code is independently implemented here.

Cross-check: official [Raydium SDK instrument definitions](https://github.com/raydium-io/raydium-sdk-V2/blob/e2e8bcd865189c994682a9477ec8eaf1e2da8aaf/src/raydium/launchpad/instrument.ts)
agree on initializer discriminators, account positions and Borsh layouts. The
SDK serializes ten zero padding bytes after an absent Token-2022 fee option;
the decoder permits that specific reviewed representation. Legacy initialize
is decoded for historical evidence; the SDK marks it deprecated. Current
discovery supports initialize_v2 and initialize_with_token_2022.

The callback archives native launches before admission, starts episodes with
their actual pool and quote asset, and records creator/funding graph edges.
It does not seed Pump reserves, assume quote decimals, warm Pump accounts or
assert LaunchLab curve execution. Native curve quotes, trade tape, graduation
tracking and buy/sell builders remain separate work. A generic router can be
consulted after the existing registry observation requirement, with the desk's
existing execution checks intact. Discovery proof is not execution proof.

## V1 transaction observations

The RPC feed opts into transaction version 1. As documented by [Solana](https://solana.com/developers/cookbook/transactions/versions),
V1 carries priority fees as total lamports in message config; old ComputeBudget
instructions are no-ops. JSON and the vendored Geyser config schema read those
fields directly. An ambiguous old protobuf packet without version/config
evidence is unpriced, not assigned a guessed fee. This does not change the
versions this desk's signer can build or approve.

## Market-window adaptation

The existing Dexscreener pair miner now retains all returned m5/h1/h6/h24
transaction, volume and price-change windows. It records transaction imbalance,
buy fraction, transactions/volume per minute and volume relative to liquidity
with explicit missing denominators. These are independently implemented
measurements, not the donor's hand-weighted score. Transactions are not unique
people, reported volume is not verified organic volume, paid boosts remain
separate attention evidence, and indicative prices are not firm fills.

The pair, holder-structure and mint-metadata miners rotate through all retained
targets instead of repeatedly taking the first batch. Provider batch sizes and
polling work quanta remain; they no longer define the target universe. Targets
that disappear before their turn or provider outages remain coverage gaps.

## External integrations left inactive

Automatic approval review initially rejected GMGN requests that disclose
internally selected public wallet/deployer addresses. The owner then explicitly
authorized that connection after being told what it sends; the renewed patch
was accepted. Eight key-gated read miners now cover token metadata/security,
pools, holders, traders, wallet activity/statistics and creator launches.
They use only fixed reviewed GET routes and refuse redirects, remote signing,
wallet-key loading and order methods. No `GMGN_API_KEY` is currently provisioned,
so registration reports that dependency and makes no request. The Helius key
is never reused. Provider payloads remain unverified claims with observation
time and pagination incompleteness; native safety facts are not overwritten.

SendAI's Pyth helper was inspected: it selects a feed by symbol and formats a
price without carrying the confidence interval and publication freshness.
That implementation is not imported. An ordinary request to the documented
public Hermes feed inventory returned HTTP 403 from the local environment;
no bypass or claim of a working oracle feed is made. Independent oracle
cross-checks require an accessible, verified feed and remain unavailable.

## Claims deliberately not promoted to facts

“0.3 seconds” is not a measured end-to-end confirmed fill on our VPS; “100% of
launches” has no independent denominator here. Provider rate limits, missing
keys, unavailable endpoints and inaccessible market rules remain coverage
gaps. No donor score, source count or collection breadth certifies net ROI.
