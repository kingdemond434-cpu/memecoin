"""Run the statistical gauntlet over the stored launch corpus.

Prints the mechanism scoreboard -- one row per candidate edge, one verdict
column -- and, with `--record`, writes the survivor count into the forward
evidence ledger, which is where the promotion gate reads it from. CANARY
requires one survivor and LIVE requires two, and until something runs this
that field is unmeasured and both rungs fail closed.

    python -m tools.run_gauntlet --storage data/launch_episodes
    python -m tools.run_gauntlet --record data/state/forward_evidence.json

Reads the corpus and writes at most that one ledger file. It does not train,
promote, or touch capital; promotion still happens through the gate on the
desk's own schedule, against a count this only supplies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from src.research.forward_evidence import ForwardEvidence
from src.research.gauntlet import Gauntlet
from src.research.gauntlet_feed import GauntletFeed, iter_episodes

EXIT_OK = 0
EXIT_NO_EDGE = 3
EXIT_DATA_BLOCKED = 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default="data/launch_episodes",
                        help="directory of stored launch episodes")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N resolved episodes")
    parser.add_argument("--record", nargs="?", default=None,
                        const="data/state/gauntlet.json",
                        help="write the survivor count to this file, which "
                             "the desk reads into its forward evidence "
                             "ledger (default data/state/gauntlet.json)")
    parser.add_argument("--min-observations", type=int, default=None,
                        help="override the gauntlet's minimum sample size")
    parser.add_argument("--allow-unmeasured-regime", action="store_true",
                        help="score launches whose market was never measured, "
                             "in one 'unknown' bucket. Off by default: the "
                             "regime requirement exists to show a mechanism "
                             "generalises, and a bucket named for missing "
                             "measurement cannot show that")
    parser.add_argument("--json", action="store_true",
                        help="emit the full report as JSON instead of a table")
    args = parser.parse_args(argv)

    storage = Path(args.storage)
    if not storage.exists():
        print(json.dumps({"status": "DATA_BLOCKED",
                          "detail": f"no corpus at {storage}"}, indent=2))
        return EXIT_DATA_BLOCKED

    gauntlet = (Gauntlet(min_observations=args.min_observations)
            if args.min_observations else Gauntlet())
    feed = GauntletFeed(
        require_measured_regime=not args.allow_unmeasured_regime)
    report: dict[str, Any] = feed.run(
        iter_episodes(storage, limit=args.limit), gauntlet=gauntlet)

    if args.json:
        printable = {key: value for key, value in report.items()
                     if key != "table"}
        print(json.dumps(printable, indent=2, default=str))
    else:
        print(report["table"])
        print()
        coverage = report["coverage"]
        print(f"episodes scored      {coverage['observations']:>8d} "
              f"observations from {coverage['episodes']} episodes")
        print(f"dropped, no prices   {coverage['dropped_no_lifecycle']:>8d}")
        print(f"dropped, no regime   "
              f"{coverage['dropped_no_measured_regime']:>8d}")
        print(f"regimes observed     {len(coverage['regimes']):>8d}  "
              f"{', '.join(coverage['regimes']) or '-'}")
        print(f"survivors            {report['survivors']:>8d}")
        if report["detail"]:
            print(f"\n{report['detail']}")

    if args.record:
        # Written to its own file, never into forward_evidence.json. The desk
        # owns that ledger and rewrites it whole on every save, so a second
        # writer there either loses this verdict or destroys the desk's
        # counters depending on who saved last.
        if not report["mechanisms"]:
            print("\nnot recorded: the run scored no mechanisms, which is no "
                  "measurement rather than zero survivors", file=sys.stderr)
        elif ForwardEvidence.write_gauntlet(Path(args.record), report):
            print(f"\nrecorded {report['survivors']} survivor(s) "
                  f"into {args.record}")
        else:
            print(f"\ncould not write {args.record}", file=sys.stderr)
            return EXIT_DATA_BLOCKED

    if not report["mechanisms"]:
        return EXIT_DATA_BLOCKED
    return EXIT_OK if report["survivors"] else EXIT_NO_EDGE


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
