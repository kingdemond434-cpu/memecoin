# World native RPC transport recovery

The World research runner previously constructed only an owner-key Helius
endpoint. An absent key or exhausted Helius allowance therefore disabled public
chain research even though the main desk had configured keyless Solana repair
providers. The runner now uses the reviewed Solana configuration and the same
RPC manager's endpoint/method failover and shared Helius credit accounting.
Unresolved credential placeholders and public/demo-key URLs do not participate.
No browser, World account or wallet credentials are imported.

World relationship validation previously sent up to 100 accounts even when a
free RPC explicitly accepted only ten. A specific JSON-RPC -32602 account
maximum now reduces each complete market frame (CASH, Yes mint, No mint and
vault). The bound survives restart and periodically permits a larger probe.
Rate/auth failures never trigger split retries. Every accepted response still
has a finalized slot at least as new as the candidate census, exact address
cardinality and the existing native owner/mint/vault checks. There is no total
market ceiling. Providers accepting fewer than four accounts cannot supply a
complete relationship frame and remain explicit failures.

Changing transport code changes the frozen World research specification. Keep
the previous study and deploy this version to a new state directory. Do not
overwrite its specification hash or relabel older venue paper fills as World.
This repair does not remove World HTTP 401/403, establish Phantom listing
identity, obtain firm executable quotes or verify market rules/settlement.
Research candidates still have no trading authority or manufactured fills.
