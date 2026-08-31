# Two-box split: dedicated collector, dedicated trainer

## Why, and what it does not buy

Training skipped **six consecutive hourly runs** on 2026-08-28 with available
memory falling 556 -> 483 -> 401 -> 314 -> 205 MiB. The desk was not the cause:
it held steady at 513 MB against a 568 MB peak, memory band `calm`. The box was
simply oversubscribed -- three Claude Code sessions at 865 MB combined,
seventeen `claude` processes, plus the quant-platform desks.

So the trainer and the always-on collector are competing for a 2 vCPU / 4 GB
box, and the collector must win that fight, because a gap in its stream is
evidence that cannot be recreated later. Training is the part that can move.

**This split unblocks the training loop. It does not create edge.** The model
is rejected on its merits (Brier skill negative, return head losing to a median
baseline by ~2x even after the log-space fix), and no amount of hardware
changes that. What it buys is iteration speed on the feature research that
might: hourly training that actually runs instead of skipping.

## Location: stay in Helsinki

Measured from the current box (hel1) on 2026-08-28:

| endpoint | TCP connect |
|---|---|
| Yellowstone gRPC (publicnode) | 16 ms |
| Jupiter API | 5 ms |
| Helius RPC | 3 ms |

Yellowstone reports `STREAMING` with 0.1 s since last response. There is no
latency problem to solve, and the desk submits nothing today (`entered: 0`,
live submission locked), so validator proximity buys nothing yet either.

Put the new box in **hel1, the same location**, for two concrete reasons:

1. Hetzner private networks work within a network zone. `hel1`, `fsn1` and
   `nbg1` are all `eu-central`, so same-zone placement gives a private network
   with free internal traffic and no public exposure of either box.
2. Nothing has to move. Relocating the collector risks the one asset that
   cannot be rebuilt -- its continuous stream -- to fix a problem the
   measurements say does not exist.

Revisit this only when the desk starts submitting transactions. At that point
Jito block-engine proximity becomes a real question, and it is a different
question from where the collector lives.

## Sizing

The trainer peaks at **432 MB** (measured across nine runs: 339-482 MB). A 4 GB
box is not merely adequate, it is generous -- a 2 GB box would serve. Do not
buy CPU for it either: it is one hourly ~12-minute job.

The value is **isolation**, not capacity.

## Do this first -- it is free and may be enough

Before provisioning anything, reclaim what is already there:

```bash
ps aux --sort=-rss | grep '[c]laude' | head
```

Three sessions were holding 865 MB. Closing idle ones restored training once
before (2026-08-25, per the desk notes). Then confirm:

```bash
.venv/bin/python -m src.runtime.training_guard --min-available-mib 900
```

Exit 0 means the next hourly tick will train. If that holds for a day, you may
not need a second box at all -- and the honest version of this blueprint is
that you should check before you spend.

## Topology

    Box A (hel1, existing)              Box B (hel1, new)
    memecoin-shadow.service     <--->   memecoin-remote-trainer.timer
    collector, always on                hourly, isolated
    data/launch_episodes  --------->    pulled by rsync
    models/  <---------------------     pushed back after a PASSED run

Box A keeps the desk, the watchdog, the health checks and the feed doctor.
Box B runs only training. The trainer timer on Box A is disabled, not deleted,
so the split can be reverted by re-enabling one unit.

## Why pushing models back is safe

`MultiHeadPredictor.load()` refuses an artifact unless it carries the current
`ARTIFACT_VERSION`, a matching `feature_schema_hash`, a matching
`log_space_targets` stamp, and a validation report whose status is `PASSED`.
`save()` refuses to write one without a passed report in the first place. So a
model arriving over the network is held to exactly the same bar as one trained
locally -- a corrupted or stale bundle is rejected on load, not trusted because
it appeared in the directory.

Sync the reports alongside the models. Box A's watchdog measures training
staleness from `models/last_*_report.json` mtimes, and without them it would
conclude training had stopped and start trying to repair it.

## Setup

### 1. Provision

Hetzner Cloud, **hel1**, CX22 (2 vCPU / 4 GB) or smaller. Attach both boxes to
a private network in `eu-central`.

### 2. Key the sync, one direction, restricted

On **Box B**:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/memecoin_sync -N ""
```

On **Box A**, append Box B's public key to `~/.ssh/authorized_keys`, pinned to
the private IP and confined to rsync inside the desk tree. `rrsync` ships with
rsync and is already at `/usr/bin/rrsync`:

```
from="10.0.0.3",restrict,command="/usr/bin/rrsync /home/quant/.local/opt/memecoin-shadow" ssh-ed25519 AAAA...
```

`restrict` disables port forwarding, agent forwarding, PTY and X11; `rrsync`
confines the key to rsync under that one directory, so a key that leaks cannot
be used for a shell or to read anything else on the collector. Replace
`10.0.0.3` with Box B's private address.

Because `rrsync` re-roots paths at the directory it is given, the remote paths
in `sync-and-train.sh` become tree-relative. Set `REMOTE_ROOT=` (empty) in the
unit when using `rrsync`, or drop the `-o` confinement and keep absolute paths
if you would rather trust the key fully. The confined form is the default and
the one worth keeping: this key exists to move two directories, and nothing
about the collector's private keys or env file needs to be reachable by it.

### 3. Install the desk on Box B

Same tree, same venv, no credentials beyond what training needs. Box B must
**never** carry `SOLANA_PRIVATE_KEY` or `ALLOW_LIVE_TRADING`: it does not
trade, and a box that cannot sign cannot be made to sign by a mistake.

### 4. Install the units on Box B

```bash
cp deploy/two-box/memecoin-remote-trainer.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memecoin-remote-trainer.timer
loginctl enable-linger quant     # so user units survive logout
```

### 5. Disable the local trainer on Box A

```bash
systemctl --user disable --now memecoin-shadow-trainer.timer
```

Leave the unit file in place.

### 6. Verify before trusting it

```bash
systemctl --user start memecoin-remote-trainer.service   # on Box B
journalctl --user -u memecoin-remote-trainer -f
```

Then on Box A confirm a fresh model actually landed and loaded:

```bash
ls -la models/ | tail
journalctl --user -u memecoin-shadow --since '-10 min' | grep -i 'Loaded models\|rejected'
```

A bundle that syncs but is rejected on load is the failure mode to watch for --
it looks like success in the file listing and changes nothing in the desk.

## What is still blocked afterwards

This split fixes training capacity and nothing else. Still open, in the order
they gate promotion:

1. `shadow_policy_trades: 0` -- the policy never wants to trade out-of-sample;
   promotion needs 10. This sits ahead of the return head now.
2. `flash` band `DATA_BLOCKED` -- the 0-0.5 s window, which is the one that
   matters for sniping, has no model at all.
3. `pumpswap_execution` blocked on `fee_schedule_unobserved`,
   `pools_priceable: 0` -- this is what starves local marking to 0.18/s.
4. `rug_30s` rejected on five out-of-sample positives.
5. Zero fills, so landing and exit-latency models cannot populate in dry run.

And two things no box can supply: a **Jupiter API key** (halves the 2.05 s
quote interval, free tier) and **RPC quota** (all three Solana endpoints are at
429; Helius reports "max usage reached").
