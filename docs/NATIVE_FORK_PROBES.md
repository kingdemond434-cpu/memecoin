# Sequential native fork research

The registered `chain:local_fork_roundtrip` miner takes public finalized account
snapshots and executes setup, buy, then sell in **one local LiteSVM bank**. The
sell uses the actual token balance received by the buy. This closes a different
gap from independent unsigned RPC simulations, which do not retain post-buy
state. There is no network transaction submission or real wallet signer.

Implemented adapters: the reviewed Pump `buy_v2`/`sell_v2` WSOL route and
PumpSwap WSOL pools. Other launchpads remain unsupported/unknown. These are
implementation capabilities, not assertions that every token or mainnet program
version has passed. The launch registry's global simulation evidence fields
are deliberately not populated with a generic implementation filename.

PumpSwap's Anchor IDL describes only the fixed account prefix. The current
legacy buy/sell runtime also requires ordered remaining accounts: cashback
quote ATA (plus the accumulator on sells), pool-v2 for nonzero coin creators,
and the buyback recipient/quote ATA even when its fee is zero. The adapter
follows the official `@pump-fun/pump-swap-sdk@2.2.0` source, `offlinePumpAmm.ts`
and `pda.ts`. Its npm archive SHA-512 is
`YGHDcLexyL6/N8lDcLMmFBX6o5l7X0Nxqf2kzjRJ96plLatFZ9LXNJZErdl9yySe3dpKpO+/nzQHL4dHtL7nmQ==`.
Only the IDL prefix is compared with the Rust builder; runtime tails have
separate conditional-order tests. Historical frames lacking those accounts
remain historical failures; new bank snapshots must be acquired for new proof.

Additional owner-provided Solana RPCs are read from private environment fields
`SOLANA_RESEARCH_RPC_URL`, `SOLANA_RESEARCH_RPC_URL_1`, etc., with corresponding
optional `SOLANA_RESEARCH_WS_URL` suffixes. These append without shifting YAML
HTTP/WS pairs. Remote connections require HTTPS/WSS; explicit loopback local
validators may use HTTP/WS. Values are literal data, never evaluated or printed.
They share normal provider health, cooldown and accounting behavior; adding an
endpoint neither evades quota nor proves uninterrupted coverage.

The worker verifies mint owner/layout, route identities and token-account
authorities, reads quote-mint and mode flags using Pump's documented trailing
field defaults, and
loads executable ELF bytes from the sampled immutable/upgradeable loader
accounts. ProgramData pointers, owners, metadata and upgrade slots are checked.
Every final account group must have exactly the same context slot, including
the Clock sysvar. Explicit provider account-count refusals can produce concurrent
smaller groups; rate limits do not. Inconsistent banks remain unknown.

Only the public, deterministic virtual payer is funded locally. Its existing
on-chain account or ATAs make a probe unavailable; no existing holder balances,
pool reserves, authorities, extension flags or program state are patched to
force success. Both adapters create real ATAs through program instructions.
Pump v2 receives WSOL as an interface mint but uses
**native payer lamports** for SOL-paired curves, whose stored quote mint is
`Pubkey::default()`. Only PumpSwap wraps quote funding into a WSOL account.
The initial scheduling size is 1,000,000 quote lamports.
The job schema supports other representable sizes; a small successful probe
does **not** establish transferability at a larger intended order size.

`native_fork_probe.sqlite` retains immutable, hashed input frames, deduplicated
compressed executable blobs, durable jobs and results. VM execution reads the
archived, hash-checked frame. A raw result must be canonically retained before
its ACK makes it available to episode research. The canonical receipt binds
the archive's input hash; preserve this SQLite database with canonical backups.
ACK compares the archived body after removing only the six declared miner
transport stamp fields; additional or modified body fields remain refused.
Availability is the ACK time, never backdated to the sampled slot. Restart
retries unacknowledged results. New targets cannot leapfrog an older backlog.
New results bind a manifest of the worker, both route builders, curve decoder,
IDL resolver, vendored IDLs and recipient table, plus its digest and Python
version. The older worker-only hash does not identify a changed dependency.
Historical results keep their original metadata; this manifest is local
reproducibility evidence rather than external software attestation.
There is no top-N or lifetime token ceiling; cadence, provider quota and VM
time impose real throughput limits. A backlog is not complete launch coverage.

The child process receives no API keys or wallet environment, accepts a bounded
snapshot, limits compute/log output and is killed and reaped on timeout or
cancellation. It uses the bundled mainnet feature set and reconstructed loader
accounts. **Exact production-validator parity is unverified.** Missing CPI
accounts, unsupported hooks/programs, loader/ABI changes and VM failures can
prevent a valid real-world sell. A failed local sell is recorded as
`FORK_SELL_FAILED`, not a proven honeypot. Setup/buy/environment failures are
unknown. A successful round trip records positive returned WSOL and zero
remaining purchased units; it is not a future sell guarantee or profit proof.
Native-SOL proceeds measure payer lamport change plus the modeled transaction
fee and may include rent refunds. The receipt labels that basis and retains
quote balances around both legs; it does not assert pure sale principal or
cost-adjusted profit.
It neither overrides existing safety checks nor unlocks live trading.

The native fee adapter's standalone RPC simulations remain separately useful
for exact-intent fee observations. A local VM fee is labeled `vm_fee_lamports`;
it is not a paid on-chain fee or evidence of landing, priority fees or tips.

Validation includes synthetic exchange fixtures using actual bundled SPL and
ATA SBF: a post-buy sell succeeds, an oversell fails atomically, and no base
balance is prefilled. Adapter identity, loader versions, same-bank enforcement,
provider refusals, archive integrity, canonical order and restart/PIT behavior
are checked separately. These fixtures do not certify Pump/PumpSwap mainnet
round trips. Runtime receipts must establish actual observed coverage.

Primary interfaces:

- [Solders LiteSVM](https://kevinheavey.github.io/solders/tutorials/litesvm.html)
- [Solana getMultipleAccounts](https://solana.com/docs/rpc/http/getmultipleaccounts)
- [Pump quote-marker and trailing-field compatibility](https://github.com/pump-fun/pump-public-docs/blob/main/docs/PUMP_PROGRAM_README.md)
- [Pump v2 native-SOL account semantics](https://github.com/pump-fun/pump-public-docs/blob/main/docs/instructions/SELL.md)
- Reviewed vendored `idl/pump.json`, `idl/pump_amm.json`, `idl/pump_fees.json`
