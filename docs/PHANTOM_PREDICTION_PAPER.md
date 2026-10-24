# Phantom, prediction venues and local typed decisions

The operator surface runs on loopback port 18082 beside the memecoin desk on
18081. Its paper ledger is independent of production evidence and wallet
balances. It sends no orders to Polymarket or Kalshi. Phantom owns its keys;
neither a seed phrase nor an exported Phantom private key is required.

## Phantom

The self-contained browser application in `dashboard/wallet` detects the official
Phantom injected Solana provider, connects, and verifies a one-use, 120-second
ownership challenge. Account changes and disconnects cancel pending approvals.
Each transaction requires an explicit wallet-owner click and Phantom approval.
The bridge verifies the exact message and Ed25519 signature, expected payer,
program allowlist, transfer ceiling, queue/rate limits, signing halt and live
authority. A 45-second timeout refuses the request. It never broadcasts.

The existing Solana transaction builder can select `PhantomSignerClient` with
`MEMECOIN_WALLET_MODE=phantom`, `SOLANA_WALLET_ADDRESS`,
`MEMECOIN_PHANTOM_URL=http://127.0.0.1:18082` and
`MEMECOIN_PHANTOM_TOKEN_FILE`. The wallet address must match the connected
account. There is no local-key fallback. All existing execution and promotion
gates remain necessary. The deployed paper gateway always denies live signing;
its environment cannot arm it. A later live gateway must independently bind
its `live_authority` callback to reviewed earned promotion evidence. This
release does not provide a switch that bypasses that requirement.

Use an SSH tunnel to access the VPS UI from the computer with Phantom installed.
The gateway token is generated in its private state directory. A local operator
launcher passes it in a URL fragment, the browser exchanges it for an HttpOnly,
SameSite=Strict cookie, and removes the fragment. Tokens never enter URL query
strings or access logs. API mutations require authentication and same-origin
checks; browser sessions cannot manufacture signing requests. Loopback Host
validation blocks DNS rebinding. The page has no third-party scripts/CDNs.

Real owner interaction cannot be certified by an automated mock or an empty
wallet session. Paper testing runs while Phantom is disconnected. This is an
interactive Solana integration, not an assertion that Phantom can sign
unattended or that a Solana signature authenticates a prediction exchange.

## Prediction-market forward paper experiment

Official public Polymarket Gamma/CLOB and Kalshi REST APIs provide discovery,
market rules, order books, fee schedules and resolution. Catalog pages rotate;
held and pending markets are revisited before they can disappear from a page.
Only binary YES/NO CTF Polymarket positions and binary Kalshi markets are
currently supported. Unsupported position systems, closed/challenged markets,
unknown fee models and unavailable APIs remain explicit exclusions.

The experiment freezes a specification and implementation hash before collecting
orders. Separate simulated $1,000 accounts per venue hold YES controls, NO
controls and an experimental same-market complete-set strategy. This does not
assert a predictive model or profit. Complete sets require a 0.5% after-fee
buffer and both simulated legs commit together; real atomic execution is not
inferred. Cross-venue rule equivalence is not guessed from matching headlines.

Orders require a retained book observed at least one second after the decision,
prices at asks for buys and bids for sells, at most 10% of displayed depth,
bounded age, known fees, available cash, no shorts, position caps and at most
$25 cost per market/account. FOK fills both size and limit or nothing; FAK can
partially fill. Reused IDs cannot double spend, consumed snapshot depth cannot
be reused in one strategy, and comparison accounts remain independent.
Controls seek a depth-based exit after 300 seconds; failed exits remain open
and are retried. Official final results settle positions once. Closed or
near-one market prices alone never establish resolution. Cash, cost basis,
fees, results and original public responses persist atomically in SQLite WAL
with FULL synchronization. Paper returns remain explicitly simulated.

The loop observes broad public categories rather than filtering only for
memecoin headlines. Market count and catalog pages are bounded to protect the
memecoin runtime. `/status` exposes coverage, snapshots, orders, source errors,
positions and P&L. No synthetic fills enter the memecoin promotion ledger.

## Local alternative to typed decision APIs

`src/research/local_decisions.py` provides Choice, Score and Noul outputs
locally, with no paid model calls. Transparent rules immediately annotate
insider/launch claims, guaranteed-return claims, wallet-secret requests and
settlement ambiguity. These rules do not invent calibrated probabilities.
Both the memecoin narrative intake and prediction paper loop retain their
shadow annotations. They cannot approve trades or assert that a claim is true.

The learned path uses a hashed word/bigram logistic classifier, validation-only
temperature selection and a later chronological holdout against class-prior
log loss. Campaign/market groups and duplicate inputs cannot cross folds;
labels unresolved at split boundaries are purged. JSON model weights are
checksummed and shape-validated; executable pickle files are never loaded.
Low-confidence or poorly supported answers abstain. No model is installed
until evidenced labels and untouched holdouts exist. General text accuracy is
not proof of positive financial returns.

Training input is JSONL with `text`, `label`, `decision_at`,
`label_observed_at`, `group_id`, and `source_ref`. Example training command:

```
python tools/train_local_decisions.py --labels reviewed-labels.jsonl --question source_quality --primitive noul --options no yes --output data/research/local_decisions/source_quality.json
```

Growth/outcome questions additionally need executable after-cost labels at the
declared horizon, including losers and rejected candidates. Do not relabel
wallet claims or price-only replay as realized trading evidence. The local
model is our small domain classifier, not Jev's architecture or weights.

## Continuous operation and existing Solana breadth

The main desk retains launch census, native Pump/PumpSwap research, wallet and
public-source intelligence, Jito routing, fee bids, leader/landing models and
adverse-execution analysis. These are separate from a claim of profitable MEV
or universal Solana coverage. Phantom does not supply launch data or private
order flow. Unsupported venues require their own adapters and validation.

Run the sidecar with `deploy/windows/start-prediction-paper.ps1`. Keep code
read-only to its task account and its private state writable. Restart policies
and login dependencies must be verified on the VPS; an Interactive task cannot
promise collection across Windows logout. Do not reuse production directories
for software tests. Run the full suite with `tools/validate_checkout.py` in a
new isolated checkout, with credentials removed.

Sources checked 24 October 2026:
- https://docs.phantom.com/solana/detecting-the-provider
- https://docs.phantom.com/solana/establishing-a-connection
- https://docs.phantom.com/solana/sending-a-transaction
- https://docs.polymarket.com/market-data/market-details
- https://docs.polymarket.com/trading/fees
- https://docs.kalshi.com/getting_started/orderbook_responses
- https://docs.kalshi.com/api-reference/events/get-event-fee-changes
- https://docs.kalshi.com/getting_started/api_environments
