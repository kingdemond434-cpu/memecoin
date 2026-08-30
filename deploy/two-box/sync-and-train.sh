#!/usr/bin/env bash
#
# Pull episodes from the collector, train, push passing models back.
#
# Ordering matters and is not arbitrary:
#
#   1. Pull episodes BEFORE training, so the run sees the newest evidence.
#   2. Train against the local copy, never against a live remote directory --
#      the collector writes continuously, and training off a moving tree
#      produces a model fitted on a set that never existed at any instant.
#   3. Push models back ONLY if the trainer wrote a fresh report. A run that
#      skipped or failed must not overwrite the collector's working model with
#      whatever happens to be in this box's models directory.
#
# The collector is the source of truth for episodes; this box is the source of
# truth for models. Neither direction is a two-way sync, because a two-way sync
# between a continuously-appending dataset and a periodically-rewritten model
# directory resolves conflicts by coin flip.

set -euo pipefail

COLLECTOR="${COLLECTOR:-10.0.0.2}"
REMOTE_USER="${REMOTE_USER:-quant}"
# Empty when the collector confines this key with rrsync, which re-roots every
# path at the desk tree. Set to an absolute path only if the key is unconfined.
REMOTE_ROOT="${REMOTE_ROOT-}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/quant/.local/opt/memecoin-shadow}"
SSH_KEY="${SSH_KEY:-/home/quant/.ssh/memecoin_sync}"
MIN_SAMPLES="${MIN_SAMPLES:-250}"

SSH="ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
REMOTE="${REMOTE_USER}@${COLLECTOR}"

log() { printf '%s remote-trainer: %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cd "${LOCAL_ROOT}"

# --- 1. pull -----------------------------------------------------------------
# --partial so a dropped transfer resumes rather than restarting a 250 MB pull
# on a box whose whole job is to run for twelve minutes every hour.
log "pulling episodes from ${COLLECTOR}"
rsync -az --partial --timeout=120 -e "${SSH}" \
    "${REMOTE}:${REMOTE_ROOT}/data/launch_episodes/" \
    "${LOCAL_ROOT}/data/launch_episodes/"

# The trainer reads forward evidence and outcome indices from data/state.
# Pulled read-only: this box must never write back into the collector's state,
# which is hash-chained and belongs to the process that appends to it.
log "pulling research state"
rsync -az --partial --timeout=120 -e "${SSH}" \
    --include='forward_evidence.json' \
    --include='trade_outcomes.jsonl' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_ROOT}/data/state/" \
    "${LOCAL_ROOT}/data/state/" || log "state pull incomplete; continuing"

# --- 2. train ----------------------------------------------------------------
# Stamped before the run so "did this run produce anything" is a file-age
# question rather than a parse of the trainer's own report.
STAMP="${LOCAL_ROOT}/data/state/.train-started"
: > "${STAMP}"

log "training"
FAILED=0
.venv/bin/python -m src.research.shadow_trainer \
    --storage data/launch_episodes --model-dir models \
    --min-samples "${MIN_SAMPLES}" || FAILED=1
.venv/bin/python -m src.research.hazard_trainer \
    --storage data/launch_episodes --model-dir models \
    --min-rows "${MIN_SAMPLES}" || FAILED=1
.venv/bin/python -m src.research.exit_policy_trainer || FAILED=1

if [ "${FAILED}" -ne 0 ]; then
    log "a trainer exited non-zero; NOT pushing models"
    exit 1
fi

# --- 3. push -----------------------------------------------------------------
# Only artifacts newer than the stamp. A trainer that legitimately declined to
# write a model (too few samples, validation not passed) leaves the previous
# bundle in place here, and pushing it would silently re-install a model the
# collector already has -- or worse, an older one.
if [ -z "$(find models -newer "${STAMP}" -type f -print -quit 2>/dev/null)" ]; then
    log "no fresh artifacts; nothing to push"
    exit 0
fi

# Reports go with the models. The collector's watchdog measures training
# staleness from models/last_*_report.json mtimes; without them it concludes
# training has stopped and starts trying to repair a desk that is fine.
log "pushing models and reports"
rsync -az --partial --timeout=120 -e "${SSH}" \
    --include='*.joblib' --include='*.json' --exclude='*' \
    "${LOCAL_ROOT}/models/" \
    "${REMOTE}:${REMOTE_ROOT}/models/"

log "done"
