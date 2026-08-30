# Runbook

The shadow desk, from a fresh box to a canary. Nothing here enables live
capital: that requires a deliberate, separate acknowledgement, and the gate
below decides whether it would be justified.

## The one-line answer to "is shadow running?"

```bash
systemctl --user is-active memecoin-shadow.service
systemctl --user is-active memecoin-watchdog.timer
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

The watchdog persists its current state in
`data/state/watchdog_state.json` and its action history in
`data/state/watchdog_events.jsonl`. For intentional maintenance, create
`data/state/maintenance.lock` before stopping the desk; the installer does
this automatically. A five-minute cooldown and three-restarts-per-hour hard
budget prevent provider failures from becoming restart storms.

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

Landing redundancy is mechanisms, not regions. Jito's seven regions are one
auction; when that auction is the problem all seven fail together. Two more
mechanisms register automatically when their variables are set, and register
DISABLED with the reason when they are not:

```bash
export SOLANA_STAKED_RPC_URL=...     # a staked / SWQoS-prioritised endpoint
export SOLANA_SENDER_URL=...         # a multi-path forwarder
```

Every route receives the SAME signed base64 string. That is what makes racing
safe: one signature, executed at most once by the runtime, delivered by
whoever arrives first. Racing two differently-signed variants of one intent
would be two transactions and a double-size position, and the router refuses
it by construction -- it fans out a string, not a decision.

```bash
curl -s http://127.0.0.1:18080/status | python3 -c "
import json,sys; d=json.load(sys.stdin)['landing_router']
print(d['status'], '|', d['detail'])
print('mechanisms :', d['mechanisms'])
print('measured   :', d['measured_routes'])"
```

`mechanisms` under two is reported DEGRADED. A route's landing rate stays
DATA_BLOCKED until thirty resolved attempts, because three landings out of
three is three attempts and not a perfect route.

After promotion the Rust kernel decides ALONE and Python verifies afterwards.
`/status.t0_kernel.python_on_hot_path` is the line that says which shape is
running; `parity_dropped` counts promoted decisions nobody will ever check,
and unverified is never reported as agreement.

Measure the wire before arguing about geography. Run it ON THE NODE, never
from a laptop or a sandbox -- a proxy in the path makes Tokyo look 5ms away
and the tool refuses to draw a conclusion when it detects that:

```bash
.venv/bin/python tools/measure_wire.py --samples 12
.venv/bin/python tools/measure_wire.py --json data/state/wire.json
```

It times every Jito block engine the submitter already races, plus the RPC
set, and quotes each as a fraction of a 400ms slot. Under 15ms to the nearest
engine is colocated-tier and geography is not your problem; over 90ms and
moving the box is worth more than every code change combined.

Build the Rust hot path on the node. It is optional -- the desk falls back to
Python and says so -- but the fallback costs about 40 microseconds per entry
on build-and-sign, and more importantly the Rust build is what unblocks the
kernel parity ladder:

```bash
cd native/solana_fastpath && cargo build --release
cp target/release/libsolana_fastpath.so ../../.venv/lib/python3.11/site-packages/solana_fastpath.so
cd ../.. && .venv/bin/python -c "import solana_fastpath; print(solana_fastpath.IMPLEMENTATION)"
```

Then check `/status.latency`. This is the only page that can tell you whether
the next hour belongs to code or to money: `dominant_controllable_stage` names
the slowest stage we own, and if the detail line says the wire dominates, stop
writing code and move the box.

```bash
curl -s http://127.0.0.1:18080/status | python3 -c "
import json,sys; d=json.load(sys.stdin)['latency']
print(d['status'], '|', d['detail'])
print('slowest ours:', d['dominant_controllable_stage'])
print('unmeasured  :', d['unmeasured_stages'])"
```

The substitution catalogue -- sixty public endpoints across eleven regions,
each a rung the desk falls back to when the one above it refuses this address
-- is a claim until it is probed from the node that will use it:

```bash
.venv/bin/python tools/verify_substitution.py                 # all ten domains
.venv/bin/python tools/verify_substitution.py --domain regional_venues
.venv/bin/python tools/verify_substitution.py --json data/state/substitution_probe.json
```

It exits non-zero only when a DOMAIN has no working endpoint at all. A few
dead rungs is what the ladder is for; a domain with none is a question the
desk asks continuously and cannot answer, and `/status.substitution.dark`
names it. A completely dark domain is usually one shared cause rather than
every operator dying at once, so the fixer for it lifts every quarantine
rather than restarting anything:

```bash
curl -s -X POST http://127.0.0.1:18080/release-sources | python3 -m json.tool
```

Public Telegram needs no configuration at all. The desk reads
`t.me/s/<channel>` previews with no account, harvests handles from the t.me
links it already mines (a Pump token's own profile links its own channel),
and verifies each candidate by fetching its preview before reading it. To run
a verification pass immediately rather than waiting for its hourly slot:

```bash
curl -s -X POST http://127.0.0.1:18080/verify-channels | python3 -m json.tool
curl -s http://127.0.0.1:18080/status | python3 -c "
import json,sys; d=json.load(sys.stdin)['telegram_channels']
print('verified :', d['verified'], '| candidates:', d['candidates'],
      '| addresses seen:', d['mints_seen'])"
```

`config/figures.yaml` ships 68 public figures, projects and brands with names
and NO channels, deliberately. Name matching, serial-impersonator detection
and contradiction all work without a single channel; what does not work is
saying ANNOUNCED. Filling a figure's `channels` from their own verified
profile is what lets the desk distinguish a real celebrity launch from the
very many that only claim one, and `/status.identity_watch` reports the
registry as DEGRADED with exactly that reason until you do.

## 2a. Confirm the desk sees your keys, and authorise Telegram

Setting a key and the desk seeing it are different facts. An env file loaded
by the wrong unit, or a variable exported in a shell the service never
inherited, looks exactly like a missing key from outside.

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
d = json.load(sys.stdin)['credentials']
print('present:', ', '.join(d['present']) or '(none)')
for row in d['absent']:
    print(f\"  absent  {row['name']:24s} -> {row['unlocks']}\")
t = d['telegram']
print(f\"telegram: keys={t['keys_present']} channels={t['channels_listed']} \"
      f\"session={t['session_authorised']} ready={t['ready']}\")
if t['authorise_with']:
    print('  run once, interactively:', t['authorise_with'])
"
```

Only presence is reported. No value is ever read, logged or returned.

**Telegram needs one interactive step that the keys alone do not cover.**
Telethon asks for a phone number and a login code when it finds no session
file, and a systemd unit has no stdin to ask on -- so the transport refuses to
start rather than hanging. Authorise once, by hand:

```bash
cd ~/.local/opt/memecoin-shadow
.venv/bin/python -m src.research.telegram_authorize
systemctl --user restart memecoin-shadow.service
```

It reads `~/.config/memecoin-shadow/env` itself, so it works from a plain
shell -- systemd loads that file through `EnvironmentFile=` and an interactive
shell does not, and a tool that fails at the exact moment this page tells you
to run it is a tool that gets run wrong every time. If it still cannot find
the keys it prints every path it looked in.

That writes `data/telegram/collector.session`, which both the social collector
and the source mesh read. It survives restarts and reinstalls.

**Channels come from one list.** `TELEGRAM_CHANNELS` in your env file feeds
both the social collector and the source mesh -- listing them twice is asking
for two lists that disagree. Each channel is added to the mesh with chat-rate
polling. A channel whose language matters gets its own entry in
`config/sources.yaml`; an expanded one deliberately carries no language,
because assigning one by list position would invent an attribute nobody gave.

```bash
grep -c . <<< "$(grep TELEGRAM_CHANNELS ~/.config/memecoin-shadow/env)"
curl -s localhost:18080/status | python3 -c "
import json,sys
t = json.load(sys.stdin)['source_mesh']['transports']['by_kind']
print('telegram transports built:', t.get('telegram', 0))
"
```

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

## 5. Watch the Rust kernel promote itself

The Rust T0 core shadows the Python policy on every ordinary decision. It
takes over only after `t0_kernel_promote_after` consecutive agreements (500 by
default -- roughly an hour of an active desk), and a SINGLE disagreement while
it is deciding demotes it for the rest of the session.

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
d = json.load(sys.stdin)['t0_kernel']
print('mode              :', d['mode'], '| native:', d['native'])
print('authoritative     :', d['rust_authoritative'])
print('agreement run     :', d['consecutive_agreements'], '/', d['promote_after'])
print('compared/diverged :', d['compared'], '/', d['divergences'])
print('decided by rust   :', d['decisions_by_rust'], f\"({d['rust_share']})\")
if d['demoted_reason']:
    print('DEMOTED:', d['demoted_reason'])
for row in d['divergence_examples'][:3]:
    print('  ', row['reason'])
"
```

A non-empty `demoted_reason` needs looking at before anything else: the two
implementations disagreed about a decision that was moving capital. It does
not re-promote on its own, and it should not be re-promoted by restarting.

`not_expressible_in_kernel` counting up is normal, not a fault -- re-entry and
replacement decisions have no kernel representation and go to Python by
design. `without_survival_inputs` climbing means callers are not passing the
raw distribution, which is worth fixing; a high `rust_errors` means the
extension is built wrong and the desk is quietly running on Python.

## 6. Confirm marking is local

```bash
curl -s localhost:18080/status | python3 -c "
import json,sys
d = json.load(sys.stdin)['marking']
print('local share  :', d['local_share'])
print('via router   :', d['marks_via_router'])
print('cross-checks :', d['cross_checks'], 'diverged', d['cross_checks_diverged'])
print('mean drift   :', d['mean_drift'], '(tolerance', d['divergence_tolerance'], ')')
"
```

Two different questions here. `local_share` near 1.0 means position
redecisions are not waiting on a router. `mean_drift` says whether the local
mark is RIGHT -- a desk marking entirely locally and drifting 40% from the
router is fast and wrong, which is worse than slow. Persistent divergence on
one token usually means its curve state has gone stale or it has moved to a
venue the desk is not reading.

## 7. Weekly

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

## 8. A tiny canary, when and only when the gate says so

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

## The desk's own terminal

The health server serves the operator dashboard from the same loopback
binding as `/status`:

```
http://127.0.0.1:18080/
```

It polls `/status` every ten seconds, so it shows the live desk with nothing
to paste. From a laptop, tunnel it rather than exposing the port — this page
renders the desk's whole interior and carries exactly the exposure `/status`
does:

```
ssh -N -L 18080:127.0.0.1:18080 quant@<vps>
```

then open `http://127.0.0.1:18080/` locally.

## Isolated signing

By default the private key is held in the trading process, and the desk says
so in `/status` under `signer.mode = "local"`. To move it out, run the signer
as its own unit and point the desk at its socket:

```
MEMECOIN_SIGNER_SOCKET=/run/memecoin/signer.sock
```

There is no fallback. If the socket is configured and the signer refuses or is
unreachable, the transaction fails — it never quietly returns to signing with
a local key, because that is the state the isolation exists to leave.

`data/state/HALT_SIGNING` stops all signing immediately, for any caller:

```
touch ~/.local/opt/memecoin-shadow/data/state/HALT_SIGNING
```

## Memory

The desk reads its own RSS against the cgroup ceiling and sheds context in two
bands before the kernel intervenes: census detail spills to disk, miner
concurrency halves, price paths for untraded tokens are dropped. None of it
touches the decision path. Watch `memory.band` in `/status`; a desk that lives
in `shed` is on a host too small for it, and trimming harder will not fix
that.


## Moving the key out of the desk

Two units. The desk holds no key; the signer holds nothing else.

```bash
# 1. The signer's own env file -- ONLY the key. Not the desk's env: every
#    variable the signer cannot read is one it cannot leak.
install -m 700 -d ~/.config/memecoin-shadow
printf 'SOLANA_PRIVATE_KEY=%s\n' "$YOUR_KEY" > ~/.config/memecoin-shadow/signer-env
chmod 600 ~/.config/memecoin-shadow/signer-env

# 2. Install and start the signer
cp deploy/systemd/memecoin-signer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memecoin-signer.service
systemctl --user status memecoin-signer.service --no-pager | head -5

# 3. Point the desk at it and remove the key from the desk's own env
sed -i '/^SOLANA_PRIVATE_KEY=/d' ~/.config/memecoin-shadow/env
echo "MEMECOIN_SIGNER_SOCKET=%t/memecoin/signer.sock" >> ~/.config/memecoin-shadow/env
systemctl --user restart memecoin-shadow.service
```

Confirm with `/status`: `signer.mode` should read `isolated` and
`signer.isolated` `true`. While it reads `local`, the key is in the trading
process -- which is a supported configuration, not a broken one, but it is
not the one to run live capital on.

The signer has `PrivateNetwork=true`: it talks to one unix socket and cannot
reach the network at all, so a compromise cannot send the key anywhere.

## Reading the dashboard from a laptop or phone

The desk binds to loopback and should stay there. Reach it by tunnel.

**Laptop** — forward the port, then open `http://127.0.0.1:18080/`:

```
ssh -N -L 18080:127.0.0.1:18080 quant@<vps>
```

**Phone, same WiFi as the laptop** — bind the forward to the laptop's LAN
interface instead, then browse to the laptop's IP:

```
ssh -N -L 0.0.0.0:18080:127.0.0.1:18080 quant@<vps>
```

**Phone, anywhere** — put the VPS and the phone on the same private network
with Tailscale, which runs without root in userspace mode, and browse to the
VPS's Tailscale address. Do NOT set `HEALTH_HOST=0.0.0.0` to achieve this:
that publishes the desk's whole interior to anything that can route to the
box.


## Running itself

Three units, on timers, so the node needs no laptop.

```bash
cp deploy/systemd/memecoin-supervisor.* deploy/systemd/memecoin-liveness.* \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memecoin-liveness.timer memecoin-supervisor.timer
systemctl --user list-timers | grep memecoin
```

Two cadences, because they cost different amounts. The liveness probe is one
HTTP call and runs every **30 seconds**, so a wedged desk is restarted within
half a minute. The full pass reads the whole status document, evaluates thirty
checks and may run the test suite, so it runs every **60 seconds**. Splitting
them is what makes the fast cadence affordable.

Every two minutes the supervisor does three things in order:

**Deploys.** Fetches `main`, and if there are new commits, runs the full test
suite ON THIS NODE against the new code before restarting anything. If the
suite fails, the checkout returns to the exact commit that was running and the
service is untouched. It refuses to deploy over a dirty tree, and it refuses
anything that is not a fast-forward.

**Corrects.** A fixed repertoire of remedies -- restart the desk when it stops
writing readiness, when the stream is connected and silent, when the
denominator freezes, when the feed is dead. Each is capped at three attempts
an hour with a four-minute cooldown; past that it stops acting and escalates,
because a service restarting for ever while nobody is told is worse than one
that stays down.

**Escalates.** Anything critical, and any fault the fixer gave up on, goes to
your Telegram Saved Messages using the session the collector already holds.
One message per distinct fault per hour, one line when it recovers, and the
full trail written to `data/state/escalations.jsonl` whether or not delivery
worked.

Check what it has been doing:

```bash
journalctl --user -u memecoin-supervisor.service -n 50 --no-pager
```

```bash
tail -20 ~/.local/opt/memecoin-shadow/data/state/escalations.jsonl
```

To watch without acting -- useful the first day:

```bash
.venv/bin/python -m ops.supervisor --root ~/.local/opt/memecoin-shadow --no-fix --no-deploy
```

What it deliberately cannot do: trade, sign, or touch capital. The unit clears
`ALLOW_LIVE_TRADING` and `SOLANA_PRIVATE_KEY` outright rather than merely not
using them. It restarts processes and moves files, and that boundary is what
makes it safe to run unattended.
