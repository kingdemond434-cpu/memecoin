"""Audit retained episodes or seal a fresh, declared chronological corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.replay_corpus import audit_corpus, freeze_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--windows", type=Path, help="JSON with explicit train, validation and test [start,end] clocks")
    args = parser.parse_args(argv)
    if bool(args.output) != bool(args.windows):
        parser.error("--output and --windows must be supplied together")
    if args.output:
        report = freeze_corpus(args.episodes, args.output,
            windows=json.loads(args.windows.read_text(encoding="utf-8-sig")), max_episodes=args.max_episodes)
        result = {key: report[key] for key in ("status", "partition_counts", "frozen_at", "source_inventory_hash")}
        result.update(manifest=str(args.output / "split.json"), blocked_objects=len(report["blocked"]))
    else:
        result = audit_corpus(args.episodes, max_episodes=args.max_episodes)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"INPUTS_PRESENT_NOT_VALIDATED", "SEALED_NOT_EVALUATED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
