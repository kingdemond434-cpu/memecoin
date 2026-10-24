# Retained native replay identities

The production episode builder stores the configured chain display name and
the decoded source program ID. The exact replay adapter formerly required the
literal strings `solana` and `pump`, so it rejected real Pump episodes before
even examining their execution evidence.

Replay now recognizes `solana`, `Solana` and `Solana Mainnet`, paired with
`pump`, `pump.fun` or the exact Pump program ID from the reviewed vendored IDL.
All map to the existing Solana/Pump adapter. Program IDs remain case sensitive.
Unknown chains and programs, including other Solana AMMs, remain unsupported.

This is a read-time mapping. Original episodes, source hashes, frozen corpus
inventories and candidate hashes are unchanged. It adds no native fills,
transaction costs, resolved outcomes or executable exits to an episode. Those
requirements remain independent blocks. The production corpus audit found
984 episodes: 962 Pump, 16 Pump AMM, five Whirlpool and one Raydium CPMM, all
missing required native replay fill/cost evidence. Identity recognition alone
cannot make that corpus eligible for historical validation.
