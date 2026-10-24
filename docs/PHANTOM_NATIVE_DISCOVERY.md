# Native World research

The Phantom collector observes the candidate `prediCt` program through finalized
Solana RPC independently of World HTTP metadata access. `getProgramAccounts`
enumerates currently existing 320-byte candidates; it does not reconstruct closed
historical markets. Every response is retained with its context slot. Mint
authorities, Token-2022 metadata and collateral vault relationships are checked in
100-account RPC batches. Pending batches survive restarts; batch size is not a
total coverage limit. A changed vault authority remains an explicit discrepancy.

`/status` reports collection freshness, progress, relationship discrepancies and
separate execution blockers. The paired, read-only
`/api/phantom-predictions/markets?limit=100&after=<address>` endpoint exposes all
current candidates by cursor. Pagination may span a new census; clients should
restart pagination if `census_slot` changes. Dashboard labels distinguish native
candidates from executable book snapshots. No candidate becomes a fill, a verified
Phantom contract, a settlement or permission to trade.

World's public frontend also advertises a PayBox MCP connector at
`https://api.paybox.sh/mcp?app=world`. PayBox requires owner OAuth authorization.
Discovery neither creates that authorization nor accesses selected wallets.
HTTP 401/403/429 responses back off; no authentication or jurisdiction bypass is
attempted. Native observations continue while that metadata path is blocked.

Changing the collector or native decoder changes the frozen research specification.
Keep the previous study and use a fresh state directory; never relabel old venue
experiments as Phantom-native evidence.

References: [Phantom feature](https://help.phantom.com/articles/47187740851731),
[Solana RPC](https://solana.com/docs/rpc/http/getprogramaccounts),
[independent inferred layout](https://github.com/chainstacklabs/world-xyz-research/blob/main/docs/reference.md),
[PayBox authorization](https://api.paybox.sh/).
