"""How long until the next rung, in plain text.

Answers "how many days, or is it already eligible" from the desk's own
ledgers, and keeps the two kinds of answer apart. The counting requirements
have an observed rate and therefore a date. Three regimes, positive growth,
2.0 monster enrichment and a gauntlet survivor do not: a desk short of those
is waiting on the market or on itself, and a single number covering both
would put a confident date on the requirement least likely to arrive.

    python -m tools.canary_eta

Reads two state files and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.research.forward_evidence import ForwardEvidence
from src.research.promotion_gate import DEFAULT_CRITERIA, PromotionLedger

EXIT_ELIGIBLE = 0
EXIT_WAITING = 3
EXIT_DATA_BLOCKED = 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="data/state")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    state = Path(args.state_dir)
    evidence_path = state / "forward_evidence.json"
    if not evidence_path.exists():
        print(f"no forward evidence ledger at {evidence_path}; the desk has "
              "not recorded a decision yet")
        return EXIT_DATA_BLOCKED

    # `promotion.jsonl`, matching src/runtime/wiring.py. A wrong name here
    # does not fail: `current_stage` reads a missing stage file as HISTORICAL,
    # so this would confidently report the bottom rung and a date to match.
    ledger_path = state / "promotion.jsonl"
    if not ledger_path.with_name(ledger_path.stem + "_stage.json").exists():
        print(f"no stage record beside {ledger_path}; the ladder has not "
              "recorded an arrival yet, so this would report the bottom rung "
              "whatever the desk has actually earned")
        return EXIT_DATA_BLOCKED
    ledger = PromotionLedger(ledger_path)
    forward = ForwardEvidence(evidence_path, stage=ledger.current_stage())
    forward.load_gauntlet(state / "gauntlet.json")
    eta = ledger.eta(forward.evidence(), forward.observed_days())

    if args.json:
        print(json.dumps(eta, indent=2))
        return EXIT_ELIGIBLE if eta.get("eligible_now") else EXIT_WAITING
    if eta.get("status") != "OK":
        print(eta.get("detail", "no answer"))
        return EXIT_DATA_BLOCKED

    print(f"at {eta['stage']}, next rung {eta['next_stage']}")
    print(f"observed {eta['observed_days']:g} days, "
          f"{eta['days_at_stage']:g} of them at this stage")
    print()
    if eta["eligible_now"]:
        print("ELIGIBLE NOW -- the gate passes on the next sweep")
        return EXIT_ELIGIBLE

    print(f"{'requirement':22s} {'have':>9s} {'need':>9s} {'per day':>9s} "
          f"{'days left':>10s}")
    for name, row in eta["counting"].items():
        days = row["days_remaining"]
        print(f"{name:22s} {row['have']:>9d} {row['need']:>9d} "
              f"{row['per_day']:>9.2f} "
              f"{'no rate' if days is None else format(days, '>10.1f')}")
    if eta["dwell_days_remaining"] > 0:
        print(f"{'days_at_stage':22s} {eta['days_at_stage']:>9.1f} "
              f"{DEFAULT_CRITERIA[ledger.current_stage()].min_days_at_stage:>9.0f} "
              f"{'':>9s} {eta['dwell_days_remaining']:>10.1f}")
    print()

    counts = eta["days_until_counts_are_met"]
    if counts is None:
        print("counts with no observed rate yet: "
              + ", ".join(eta["counting_without_a_rate"]))
    else:
        print(f"counts and elapsed time: about {counts:.0f} more day(s)"
              + (f", slowest is {eta['slowest_count']}"
                 if eta["slowest_count"] else ""))

    blocking = eta["blocking_regardless_of_time"]
    if blocking:
        print()
        print("NOT fixed by waiting:")
        for item in blocking:
            print(f"  - {item}")
    return EXIT_WAITING


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
