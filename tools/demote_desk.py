"""Lower the desk's trading authority by one stage, by hand.

`PromotionLedger.demote` is the only thing in this system that reduces what
the desk may spend. It was written and tested on 2026-08-something and never
given a caller -- not an automatic one, and not an operator one. An operator
watching the desk misbehave had no way to take authority away short of
editing the stage file, which leaves no record of who did it or why.

Automatic demotion on a catastrophic loss now exists in
`src/runtime/regime.py`. This is the other half: the judgement call.

    python -m tools.demote_desk --reason "exit latency doubled since Tuesday"

One stage, never more. Recovering it means passing the gate again on measured
evidence rather than being handed back what was taken away -- so this is
cheap to do and expensive to undo, which is the correct asymmetry for a
control that exists to stop losses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.research.promotion_gate import PromotionLedger, Stage

EXIT_OK = 0
EXIT_REFUSED = 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="data/state")
    parser.add_argument("--reason", required=True,
                        help="why, in the operator's own words; it is written "
                             "into the stage record and cannot be edited later")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    reason = args.reason.strip()
    if len(reason) < 10:
        print("refusing: give a real reason. This is the permanent record of "
              "why the desk's authority was lowered.")
        return EXIT_REFUSED

    ledger = PromotionLedger(Path(args.state_dir) / "promotion.jsonl")
    stage = ledger.current_stage()
    if stage is Stage.HISTORICAL:
        print(f"already at {stage.value}, the bottom rung; nothing to lower")
        return EXIT_REFUSED

    authorised, detail = ledger.authorises_live_capital()
    print(f"current stage:  {stage.value}")
    print(f"live capital:   {'YES' if authorised else 'no'} -- {detail}")
    print(f"reason:         {reason}")
    print()
    if not args.yes:
        answer = input("lower one stage? this is not undone by waiting [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("no change")
            return EXIT_REFUSED

    earned = ledger.demote(reason)
    print(f"{stage.value} -> {earned.value}")
    print("recovering it means passing the gate again on measured evidence.")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
