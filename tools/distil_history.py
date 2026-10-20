"""Compress reconstructed launch history into priors a running desk can hold.

`tools/backfill_history.py` writes one JSON episode per launch. That is the
right shape for training and the wrong shape for a decision: a 4GB box
running a live desk cannot hold a million episodes, and a prior that has to
be recomputed by scanning a directory is a prior no T0 decision will ever
consult.

This reads that directory once, offline, and writes a single artifact the
desk loads at startup and reads in a dictionary lookup -- roughly four
megabytes for two hundred thousand deployers. Run it on whatever machine has
the history, never on the trading node.

    python -m tools.distil_history \\
        --episodes data/launch_episodes/reconstructed \\
        --out data/state/cold_distillate.json

Everything it emits is stamped RECONSTRUCTED. A reconstruction flatters
itself in known ways -- survivorship, latency, depth -- and a prior that
arrives without its provenance cannot be discounted for any of them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

from src.research.cold_distillation import ColdDistillate, distil

logger = logging.getLogger("distil_history")


def read_episodes(directory: Path) -> Iterator[Dict[str, Any]]:
    """Every episode file, one at a time.

    A generator rather than a list: the whole point is that the corpus does
    not fit in memory, and loading it to distil it would defeat the exercise.
    """
    for path in sorted(directory.glob("*.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            continue
        if not isinstance(episode, dict):
            continue
        yield {
            "creator": episode.get("deployer", ""),
            "created_at": episode.get("created_at", 0),
            "venue": episode.get("venue") or episode.get("factory") or "pump",
            "funding_transfers": episode.get("funding_transfers", []),
            "final_outcome": episode.get("final_outcome", {}),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default="data/launch_episodes/reconstructed",
                        help="directory of reconstructed episode JSON files")
    parser.add_argument("--out", default="data/state/cold_distillate.json")
    parser.add_argument("--max-deployers", type=int, default=200_000)
    parser.add_argument("--source", default="",
                        help="what this history came from, recorded in the artifact")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory = Path(args.episodes)
    if not directory.is_dir():
        print(f"no such directory: {directory}", file=sys.stderr)
        return 2

    distillate = distil(read_episodes(directory),
                        source=args.source or str(directory),
                        max_deployers=args.max_deployers)
    if not distillate.launches_distilled:
        # A silent zero here would put an empty artifact in front of the desk
        # and make "no history" indistinguishable from "history says nothing".
        print("WARNING: no resolved launches were distilled; the episode "
              "directory may be empty or every episode may be unresolved",
              file=sys.stderr)

    out = Path(args.out)
    if not distillate.save(out):
        return 1
    report = distillate.report()
    print(json.dumps({**report, "written_to": str(out),
                      "bytes": out.stat().st_size if out.exists() else 0},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
