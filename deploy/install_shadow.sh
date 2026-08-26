#!/usr/bin/env bash
# Install and start the shadow desk as a user service.
#
# Everything the health checks, the audit pack and the shadow trainer already
# assumed was running. Those units all declare `After=memecoin-shadow.service`
# and that unit did not exist, so the monitoring layer was watching a desk
# nobody had started -- which is why the forward-evidence ledger reads zero
# decisions no matter how long it has been since the code was written.
#
# Deliberately a user service, not a system one. It needs no privileges, and
# a trading process that runs as root is a trading process whose blast radius
# is the machine.
#
#   bash deploy/install_shadow.sh
#
# It never enables live capital. ALLOW_LIVE_TRADING is cleared in the unit
# after the environment file is read, so a stale env cannot promote a shadow
# run into a live one by accident.

set -euo pipefail

ROOT="${MEMECOIN_ROOT:-$HOME/.local/opt/memecoin-shadow}"
UNITS="$HOME/.config/systemd/user"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "installing from $SOURCE to $ROOT"
mkdir -p "$ROOT" "$UNITS" "$HOME/.config/memecoin-shadow"

# rsync rather than a copy so a reinstall over a running node does not
# clobber the accumulated evidence, which is the one thing here that cannot
# be regenerated.
# The *.verified.yaml files are excluded for the same reason data/ is: they
# are produced ON THIS HOST by probing endpoints and reading published pages,
# they are not in the repository, and --delete would remove them on every
# reinstall. Losing them silently un-configures the source mesh and empties
# the entity registry, and the desk would keep running and report less
# coverage without anything saying why.
rsync -a --delete \
  --exclude 'data/' --exclude '.git/' --exclude '.venv/' \
  --exclude 'config/*.verified.yaml' \
  --exclude 'native/solana_fastpath/target/' \
  "$SOURCE/" "$ROOT/"
mkdir -p "$ROOT/data/state" "$ROOT/models"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "creating the virtualenv"
  python3 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" install --quiet --upgrade pip
"$ROOT/.venv/bin/pip" install --quiet -r "$ROOT/requirements.txt"

# The native extension is optional: the Python path is the reference
# implementation and runs without it. Built when cargo is present, skipped
# with a note when it is not, because a missing toolchain should not stop a
# shadow run from accumulating evidence.
if command -v cargo >/dev/null 2>&1; then
  echo "building the native extension"
  cargo build --release --manifest-path "$ROOT/native/solana_fastpath/Cargo.toml"
  cp "$ROOT/native/solana_fastpath/target/release/libsolana_fastpath.so" \
     "$ROOT/.venv/lib/python3."*"/site-packages/solana_fastpath.so"
else
  echo "cargo not found; running on the Python reference path"
fi

cp "$SOURCE/deploy/systemd/"*.service "$SOURCE/deploy/systemd/"*.timer "$UNITS/"
systemctl --user daemon-reload

# Lets the units keep running when nobody is logged in. Without it a shadow
# run ends at logout, which is the other way an evidence ledger silently
# stops counting.
loginctl enable-linger "$USER" 2>/dev/null || \
  echo "could not enable linger; the desk will stop at logout"

systemctl --user enable --now memecoin-shadow.service
systemctl --user enable --now memecoin-health.timer
systemctl --user enable --now memecoin-audit-pack.timer
systemctl --user enable --now memecoin-shadow-trainer.timer

sleep 3
systemctl --user --no-pager status memecoin-shadow.service | head -15

cat <<'NOTE'

Started in DRY RUN. To confirm:

  curl -s localhost:18080/status | python3 -m json.tool | head -20

Watch the forward-evidence ledger fill:

  python3 -c "import json;d=json.load(open('$HOME/.local/opt/memecoin-shadow/data/state/forward_evidence.json'));print(d['decisions'],'decisions')"

The gate wants 5,000 decisions, 1,000 launch cohorts and 3 regimes before the
canary stage. Readiness reports the distance to each under "forward_evidence".
NOTE
