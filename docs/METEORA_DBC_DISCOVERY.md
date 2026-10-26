# Meteora DBC native discovery

The official SDK IDL is pinned at commit
`3a965ec0c0a527c2be3c48b8b479366705bf4926`:
https://github.com/MeteoraAg/dynamic-bonding-curve-sdk/blob/3a965ec0c0a527c2be3c48b8b479366705bf4926/packages/dynamic-bonding-curve/src/idl/dynamic-bonding-curve/idl.json

The unchanged IDL SHA-256 is
`63035b9a4a04b6a7f8fb9f81778275049c129a70d6f8bf6e08866516cf3da4cc`.
Its MIT license is retained in `idl/METEORA_SDK_LICENSE`. No execution code
was copied from the separately licensed on-chain program repository.

The decoder covers all three published SPL/Token-2022/transfer-hook creation
instructions, across all config identities. Creator and sponsoring payer are
separate. Arbitrary quote mint identity is retained; decimals, fees, curve
liquidity and transferability are unknown until independently read. A hook
program appearing in initialization is not a successful buy/sell test.

`tests/fixtures/meteora_dbc_launches.json` is an actual successful confirmed
public-chain transaction captured on 25 October 2026, slot 461296968,
signature `5FgG29GHzWobtbdoyqYfgpxm6ZSG4rpqwRNNTUQmAeTBg66Z5ti9A7MDWU36CbuNggGrjhqyXHTHKhnRACTjoEJ4`.
It was observed using Solana's public logs subscription and retrieved from
PublicNode's public `getTransaction` method, with no signing or paid key.
It exercises the SPL initializer and a non-SOL quote. Token-2022/hook account
layouts are additionally schema-tested, not presented as captured live fixtures.

The monitor shares LaunchLab's successful-transaction, ALT/CPI and timestamp
envelope. Both Yellowstone and primary/racing RPC subscriptions are wired;
the fallback's creation-log selector recognizes every initializer. The registry
still requires distinct actual runtime observations before venue promotion.
Live native DBC curve execution remains explicitly unavailable. The separate
local-bank research adapter is documented below; neither router support nor
a sampled-bank pass establishes every quote asset, config or future exit.

## Sequential local-bank research

`meteora_dbc_route.py` independently reads the pinned SDK's complete 424-byte
`VirtualPool` and 1048-byte `PoolConfig` layouts, including their explicit
padding. It binds pool/config/mint/vault identities and PDA derivations, token
owners, decimals and initialized vault states. The reviewed subset is an
ordinary unmigrated pool with SPL or Token-2022 base and SPL WSOL quote.
Transfer-hook pools/configs/mints, foreign layouts and other quote assets stay
unknown. Discovery continues retaining those launches.

The unsigned `swap` instructions use actual input token accounts in each
direction, an absent-referral program-ID placeholder and the transaction
instructions sysvar needed by rate-limited/first-swap configurations. Account
order and flags come from the unchanged IDL; the optional placeholder is
read-only. The amount/minimum struct is checked against `SwapParameters`.
No curve quote, fee assumption or source-account top-up replaces execution.

The shared fork worker snapshots the actual pool/config/vaults, mints, program
images and remaining dependencies in one final bank. It buys with virtual
WSOL, then sells the **actual received token units in that modified local VM**.
Protocol fee, activation, rate-limiter and curve behavior are evaluated by the
captured program, not guessed from a constant-product formula. Transfer-fee
effects therefore enter observed balances. Probe minimum=1 is a transferability
diagnostic, not a live slippage policy. Results retain input and implementation
hashes, source availability and explicit local/validator-parity limitations.

Canonical DBC/WSOL initialization events enter the durable research queue
before slower enrichment, using the same source-time-bound handoff as LaunchLab.
The certified sealed-journal recovery tool also recognizes that subset. This
recovers candidate identities for current-bank research, never historical banks
or historical fills. Four-venue service fairness prevents a large Pump backlog
from monopolizing all eligible probe turns. Queued/adapted is not a successful
round trip; runtime observations must be verified separately.

Interface references at the pinned MIT SDK revision:

- [Swap construction](https://github.com/MeteoraAg/dynamic-bonding-curve-sdk/blob/3a965ec0c0a527c2be3c48b8b479366705bf4926/packages/dynamic-bonding-curve/src/services/pool.ts)
- [Pool, vault and authority PDAs](https://github.com/MeteoraAg/dynamic-bonding-curve-sdk/blob/3a965ec0c0a527c2be3c48b8b479366705bf4926/packages/dynamic-bonding-curve/src/helpers/pda.ts)
- [Instructions-sysvar remaining account](https://github.com/MeteoraAg/dynamic-bonding-curve-sdk/blob/3a965ec0c0a527c2be3c48b8b479366705bf4926/packages/dynamic-bonding-curve/src/helpers/swap.ts)

Synthetic layout and adversarial-state tests are not live DBC execution proof.
The existing captured non-SOL initializer independently checks PDA derivations
but is deliberately not relabelled as a successful executable WSOL bank.
