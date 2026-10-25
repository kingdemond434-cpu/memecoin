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
Native DBC curve execution remains explicitly unavailable. Router support does
not establish that every quote asset, config or transfer hook is sellable.
