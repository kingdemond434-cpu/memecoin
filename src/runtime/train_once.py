"""One training pass, for systemd. The other half of the mutual exclusion.

The trainer unit used to be three `ExecStart=` lines invoking three trainers
directly. That had three problems, and only one of them was memory.

**It could not be excluded.** Each line was its own process taking no lock, so
it ran happily alongside the desk's in-process training round over the same
corpus. Two shadow passes at once is roughly twice the resident corpus on a
4 GB box, and the kernel picks the victim.

**It was missing a trainer.** `action_value_trainer` was never in the list, so
on this box the exit policy's incumbent artifact could only ever be produced by
the in-process path -- the one that competes with this unit for memory.

**It could not report a skip.** Three independent ExecStarts have three exit
codes and no shared notion of "the corpus has not grown, do not bother".

So the unit runs this instead: one process, one lock, one round through the
same `TrainingSupervisor` the desk uses, and one report. The supervisor decides
which jobs are due; this only decides whether to run at all.

Exit codes are chosen for systemd's benefit:

    0   a round ran (whatever the individual verdicts were -- DATA_BLOCKED and
        REJECTED are trainers working correctly, not unit failures)
    0   nothing was due, or the corpus has not grown enough
    75  another process holds the training lock (EX_TEMPFAIL); the unit
        declares this a success so a skip does not show up in
        `list-units --failed` as noise a health sweep would chase
    1   the round raised
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from src.runtime.training import TrainingSupervisor
from src.runtime.training_lock import TrainingBusy, holder

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BUSY = 75
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default="data/launch_episodes",
                        help="episode corpus directory")
    parser.add_argument("--model-dir",
                        default=os.getenv("MODEL_DIR", "models"))
    parser.add_argument("--new-episodes", type=int, default=250,
                        help="episodes that must have resolved since the last "
                             "round before another one is worth running")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--force", action="store_true",
                        help="run even if the corpus has not grown; for a "
                             "first pass on an existing corpus")
    return parser


def main(argv: Any = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    storage = Path(args.storage)
    supervisor = TrainingSupervisor(
        storage, Path(args.model_dir),
        new_episodes_required=int(args.new_episodes),
        timeout_s=float(args.timeout))

    busy, who = holder(storage.parent if storage.name == "launch_episodes"
                       else storage)
    if busy:
        # Reported before the round rather than discovered inside it, so the
        # journal line names the holder instead of an opaque skip.
        print(json.dumps({"status": "SKIPPED_LOCKED", "holder": who}))
        return EXIT_BUSY

    if not args.force and not supervisor.should_train():
        print(json.dumps({"status": "NOT_DUE",
                          "detail": supervisor.pending_reason(),
                          "episodes": supervisor.resolved_episodes()}))
        return EXIT_OK

    try:
        results: Dict[str, Any] = asyncio.run(supervisor.run_round())
    except TrainingBusy as exc:
        print(json.dumps({"status": "SKIPPED_LOCKED", "detail": str(exc)}))
        return EXIT_BUSY
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("training round failed")
        print(json.dumps({"status": "ERROR",
                          "error": f"{type(exc).__name__}: {exc}"}))
        return EXIT_ERROR

    print(json.dumps({"status": "OK", "episodes":
                      supervisor.resolved_episodes(), "results": results},
                     default=str))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
