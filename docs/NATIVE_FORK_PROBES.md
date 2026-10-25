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

The worker verifies mint owner/layout, route identities and token-account
authorities, reads quote-mint and mode flags rather than guessing them, and
loads executable ELF bytes from the sampled immutable/upgradeable loader
accounts. ProgramData pointers, owners, metadata and upgrade slots are checked.
Every final account group must have exactly the same context slot, including
the Clock sysvar. Explicit provider account-count refusals can produce concurrent
smaller groups; rate limits do not. Inconsistent banks remain unknown.

Only the public, deterministic virtual payer is funded locally. Its existing
on-chain account or ATAs make a probe unavailable; no existing holder balances,
pool reserves, authorities, extension flags or program state are patched to
force success. Both adapters create real ATAs and wrap virtual SOL through
program instructions. The initial scheduling size is 1,000,000 quote lamports.
The job schema supports other representable sizes; a small successful probe
does **not** establish transferability at a larger intended order size.

`native_fork_probe.sqlite` retains immutable, hashed input frames, deduplicated
compressed executable blobs, durable jobs and results. VM execution reads the
archived, hash-checked frame. A raw result must be canonically retained before
its ACK makes it available to episode research. The canonical receipt binds
the archive's input hash; preserve this SQLite database with canonical backups.
Availability is the ACK time, never backdated to the sampled slot. Restart
retries unacknowledged results. New targets cannot leapfrog an older backlog.
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
- Reviewed vendored `idl/pump.json`, `idl/pump_amm.json`, `idl/pump_fees.json`
