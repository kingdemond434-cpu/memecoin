# Runbook

The shadow desk, from a fresh box to a canary. Nothing here enables live
capital: that requires a deliberate, separate acknowledgement, and the gate
below decides whether it would be justified.

## The one-line answer to "is shadow running?"

```bash
systemctl --user is-active memecoin-shadow.service
curl -s localhost:18080/health
```

If the first says `inactive` or `unknown`, it has never been started. Every
other unit -- health checks, audit pack, shadow trainer -- declares
`After=memecoin-shadow.service`, so with the desk down they are watching a
service nobody started, and the forward-evidence ledger reads zero decisions
no matter how long the code has existed.

## 1. Install and start

```bash
bash deploy/install_shadow.sh
```

It installs to `~/.local/opt/memecoin-shadow`, builds the virtualenv, builds
the native extension when cargo is present (and runs on the Python reference
path when it is not), installs the units, enables linger so the desk survives
logout, and starts everything in DRY RUN.

`ALLOW_LIVE_TRADING` and `SOLANA_PRIVATE_KEY` are cleared in the unit AFTER
the environment file is read, so a stale env cannot promote a shadow run into
a live one by accident.

The status endpoint binds loopback. It serves the desk's whole interior --
open positions, watched wallets, model reports -- so putting it on a public
interface publishes all of that to anything that can reach the box.

## 2. Credentials and coverage

Put credentials in `~/.config/memecoin-shadow/env`. The desk reads names, and
never logs a value. What is absent is reported by name rather than silently
skipped:

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('sources ready :', d['source_mesh']['registry']['ready'], '/', d['source_mesh']['registry']['declared'])
t=d['source_mesh']['transports']
print('transports    :', t['built'], 'built,', len(t['pending_endpoint']), 'need an endpoint,', len(t['unconfigured']), 'need a key')
print('entities      :', d['entity_registry']['status'], d['entity_registry']['entities'], 'entities')
"
```

Verify source endpoints on THIS node, not from a repository -- a sandbox with
an egress allowlist reports almost everything unreachable, and that verdict is
about the sandbox:

```bash
.venv/bin/python tools/verify_sources.py                            # what answers here
.venv/bin/python tools/verify_sources.py --out config/sources.verified.yaml
```

The entity registry stays empty until entries are verified from the entity's
own published pages (`tools/verify_entities.py`). Empty is not "nothing is a
copycat" -- it is "we cannot tell", and the desk sizes accordingly.

## 3. Accumulate the forward ledger

The ledger is the scarce thing. It counts decisions the desk made in real
time, on markets it had not seen, and it is the only evidence that
distinguishes a model from a backtest.

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
print(json.dumps(json.load(sys.stdin)['forward_evidence']['evidence'], indent=2))
"
```

Cohorts and regimes are counted as SETS. Five thousand decisions about one
launch is one launch, and a ledger that counted it as five thousand would
promote a model that has seen a single market.

Nothing needs doing during this phase except leaving it running. Restarts are
fine -- the ledger persists -- but a desk that is down is a desk that is not
counting.

## 4. Read the distance to the next stage

The ladder is DISCOVERED → CHRONOLOGICAL_OOS → FORWARD_SHADOW → CANARY → LIVE.
Each stage's criteria are frozen in `src/research/promotion_gate.py`; the
status page reports the distance to each as ratios, because a gate that only
says FAIL cannot distinguish a week away from a year away.

CANARY wants, at minimum: 5,000 decisions, 1,000 real fills, 1,000 launch
cohorts, 3 regimes, non-negative net log growth, and rug losses under 15% of
gross.

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
d = json.load(sys.stdin)['forward_evidence']['distance']
print('at', d['stage'], '-> next', d['next_stage'])
for name, row in d['progress'].items():
    print(f\"  {name:16s} {row['have']:>7} / {row['need']:<7} {row['fraction']:.1%}\")
print('slowest:', d['slowest'])
for failure in d['verdict']['failures']:
    print('  FAIL', failure)
"
```

`slowest` is the one to act on. `unmeasured` is different from `below
required`: a criterion nobody has measured is not a criterion the desk is
failing, it is one nothing is reporting, and the two need different fixes.

The gate does not enable capital and cannot. It answers whether the evidence
would justify the next stage; enabling it is a separate, deliberate act.

## 5. Weekly

The audit pack runs Mondays at 18:00 Irish time and needs no approval. Read
it rather than the live status page: a weekly artefact is comparable
week-to-week, and a status page is only ever now.

```bash
ls -t ~/.local/opt/memecoin-shadow/data/state/audit/ | head -3
```

The shadow trainer runs on its own timer. Its report carries `split_warrants`,
which says whether the data would support cutting any age band further. It
reports and never acts: adding a band is an edit to `AGE_BANDS` with that
report recorded next to it.

## 6. A tiny canary, when and only when the gate says so

The canary is small enough that being wrong about everything costs an amount
you would not think about twice. Its purpose is not profit -- it is to find
out what the shadow ledger could not: real fills, real landing rates, real
slippage against real competitors.

Before it: confirm the gate passes CANARY, confirm the wallet holds only what
you are prepared to lose, and read `native_route_report()` to confirm the
native route is actually being taken rather than merely built.

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
d = json.load(sys.stdin)['native_route']
print('prepared share      :', d['prepared_share'])
print('blockhash from cache:', d['blockhash']['cache_hit_rate'])
print('outcomes            :', d['outcomes'])
print('pumpswap wired      :', d['pool_state_wired'], d['pool_account_wired'])
"
```

A prepared share near zero means the entry path is still paying round trips it
was supposed to have stopped paying, and a canary would be measuring the
fallback rather than the system.

## Stopping

```bash
systemctl --user stop memecoin-shadow.service
systemctl --user disable --now memecoin-health.timer memecoin-audit-pack.timer memecoin-shadow-trainer.timer
```

The evidence in `data/state` is the one thing here that cannot be
regenerated. A reinstall excludes it deliberately; do not clear it to "start
fresh" unless the model that produced it is gone too.
