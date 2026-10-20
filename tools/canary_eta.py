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
from src.research.promotion_gate import (
    DEFAULT_CRITERIA, PromotionLedger, Stage)

EXIT_ELIGIBLE = 0
EXIT_WAITING = 3
EXIT_DATA_BLOCKED = 4


def _print_canary_path(ledger: PromotionLedger,
                       forward: ForwardEvidence) -> None:
    """The whole ladder to real money, not just the next rung.

    The question is almost never about the next promotion. CANARY is the first
    stage that may spend anything, and from the bottom it is three of them --
    which do not overlap, because one rung is earned per passing verdict and
    FORWARD_SHADOW cannot be left in under fourteen days however fast the
    counts arrive.
    """
    if ledger.current_stage() is Stage.CANARY:
        return
    path = ledger.eta_to(Stage.CANARY, forward.evidence(),
                         forward.observed_days())
    if not path.get("rungs"):
        return
    print()
    print(f"all the way to canary ({len(path['rungs'])} promotion(s)):")
    for rung in path["rungs"]:
        dwell = (f", {rung['dwell_required']:g}d dwell"
                 if rung["dwell_required"] else "")
        print(f"  {rung['stage']:20s} -> {rung['leaves_for']:20s} "
              f"{rung['days']:>7.1f}d{dwell}")
    if path["days"] is None:
        print("  total: cannot be projected -- "
              + ", ".join(path["counting_without_a_rate"]))
    else:
        print(f"  total: about {path['days']:.0f} days, and only if nothing "
              "above is blocked on evidence rather than time")
    for item in path["blocking_regardless_of_time"]:
        print(f"  blocked now: {item}")


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

    # `promotion.jsonl`, matching src/runtime/wiring.py.
    #
    # A missing stage file is NOT an error and must not refuse: `_write_stage`
    # runs only on a promotion (and on arrival at the two rungs with a dwell
    # requirement), so a desk that has never been promoted legitimately has
    # none, and `current_stage` reading that as HISTORICAL is the truth --
    # HISTORICAL is the bottom rung and carries no authorisation at all.
    # Refusing here withheld the most useful answer there is: you are at the
    # bottom, and here is what leaving it costs.
    #
    # The failure genuinely worth catching is a WRONG FILENAME, which looks
    # identical from the stage file's absence alone. A stage record under some
    # other basename in the same directory is that, and only that.
    ledger_path = state / "promotion.jsonl"
    stage_path = ledger_path.with_name(ledger_path.stem + "_stage.json")
    if not stage_path.exists():
        stray = sorted(p.name for p in state.glob("*_stage.json"))
        if stray:
            print(f"expected {stage_path.name} but found {', '.join(stray)}; "
                  "this tool and the desk disagree about where the ladder "
                  "lives, and answering would describe the wrong rung")
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
    if not stage_path.exists():
        print("  (no promotion has ever been recorded, so this is the bottom "
              "rung by construction, not by default)")
    print(f"observed {eta['observed_days']:g} days, "
          f"{eta['days_at_stage']:g} of them at this stage")
    print()
    if eta["eligible_now"]:
        print(f"ELIGIBLE NOW for {eta['next_stage']} -- the gate passes on "
              "the next sweep")
        _print_canary_path(ledger, forward)
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

    _print_canary_path(ledger, forward)
    return EXIT_WAITING


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
