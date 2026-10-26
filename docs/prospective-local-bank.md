# Prospective local-bank controls

In paper mode the existing fork-probe lane also runs a separate, unvalidated
infrastructure control. It is not the historically validated strategy lane and
does not manufacture historical survivors, real fills or Phantom World orders.

The study is registered durably before enrollment. Only targets first queued
after registration enter; existing targets are not retroactively enrolled. Each
target has its own explicit **2 SOL virtual seed**, with a **0.001 SOL maximum
buy budget**. These independent accounts are not a shared portfolio or real
capital. The first usable public bank drives entry, rather than an invented
launch-time price. Coverage depends on the existing sampler and RPC availability.

Entry runs actual local program instructions and retains the actual received
token units, payer, token accounts and allowlisted user-specific volume state.
A later snapshot supplies all market accounts, executable code and the clock.
Only the virtual trader's accounts are overlaid. No reserve, mint, pool, creator
vault or global accumulator can be carried over as trader inventory.

The exit rule is the first sampled, strictly later bank at least 60 seconds
after entry. The ordinary retry lane prioritizes open controls; fresh-target
lanes retain their existing share. Delayed observations do not become fills at
the nominal timeout. Failed exits retain actual token balances and VM fee
debits for subsequent attempts. Failed entries are terminal costed trials,
not free retries with replenished capital.

Inventory and the result outbox commit together in `native_fork_probe.sqlite`.
A restart redelivers the same pending result. A later execution must match its
retained inventory to the preceding canonically acknowledged result. Public
status counts only acknowledged controls, with pending results reported apart.
The new counters appear under `native_fork_probe.prospective_local_bank`.

Conditional net liquid PnL is final virtual payer SOL plus WSOL, minus the
declared seed. It includes actual local instruction effects and VM fee debits;
fees are not subtracted a second time. Unreturned rent is conservatively
expensed. No unobserved account closure or refund is assumed. Results remain
SOL-denominated; there is no fabricated dollar conversion.

This uses an **exogenous market overlay**: the hypothetical entry's effect is
not propagated into the later observed market. Landing, priority fees, tips,
targeted freezes, participant reactions and exact validator parity remain
unmodelled. The reporting fields therefore retain `live_costs_complete: false`,
`promotion_eligible: false`, `real_fills: 0` and `verified_live_profit: null`.
Synthetic SPL test fixtures verify the machinery, not production DEX profits.
