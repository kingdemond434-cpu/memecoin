# Provenance

Every file in this directory is a verbatim copy from Pump's public
documentation repository. Nothing here is transcribed, summarised, or recalled.

    source:   https://github.com/pump-fun/pump-public-docs
    commit:   9c82f61cb711b044a17f770ab8ce9f9bdf78f333
    fetched:  2026-08-25T19:11:08Z
    files:    idl/pump.json, idl/pump_amm.json, idl/pump_fees.json,
              docs/FEE_RECIPIENTS.md

They are vendored rather than read from the network because an entry path that
depends on a fetch is an entry path that can be slow or unavailable at exactly
the wrong moment, and because a transaction must be built from a byte-identical
copy of what was reviewed rather than from whatever the network returns today.

Refreshing them is a deliberate act: replace the files, record the new commit
above, and re-run the suite. The account-list tests read these files directly,
so an upstream change that alters an instruction's shape fails loudly here
instead of silently producing transactions against the old layout.

## Why the IDL and not the markdown tables

The prose tables in docs/instructions/BUY.md and SELL.md are readable and, on
three flags, wrong relative to the program: they present `fee_recipient` and
`buyback_fee_recipient` as non-writable, `global_volume_accumulator` as
writable, and `sharing_config` as a Pump PDA when it is derived under the
Pump Fees program. An account list built from the prose is a transaction that
fails on chain. The IDL is what the program was compiled against, so the
account lists in this codebase are generated from it.
