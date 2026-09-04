"""Why has the desk decided so much and entered nothing?

Measured on the live desk 2026-09-04: 150,278 decisions over 8.76 days across
14,097 distinct launches, and `net_log_growth`, `rug_loss_share` and
`monster_enrichment` all reported NOT MEASURED rather than below threshold.
Those three are None only when nothing has been entered -- `net_log_growth`
is None exactly when `entered == 0` -- so the desk was screening every launch
and had never opened and closed a position. The promotion ladder cannot leave
its bottom rung in that state, however many decisions accumulate: three of
HISTORICAL's requirements are ratios over entered positions, and a ratio with
no denominator is not a small number, it is no number.

The census already counts why each launch was refused. This reads that count
off disk and puts the funnel next to it, so the answer is one command rather
than an HTTP endpoint and a JSON path.

    python -m tools.why_no_entries

Reads two state files and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

EXIT_OK = 0
EXIT_NO_ENTRIES = 3
EXIT_DATA_BLOCKED = 4

#: Screens that mean "the desk could not answer", as opposed to "the desk
#: looked and said no". The difference decides what to fix: the first is a
#: pipeline or a model that is not running, the second is a policy.
_UNANSWERED = ("data_blocked", "no_prediction", "not_trained", "untrained",
               "no_model", "not_promoted", "unavailable", "missing", "timeout",
               "no_liquidity", "unpriced", "stale")

#: What to do about the screens the desk actually emits. Keyed by substring
#: because the reason strings carry detail; first match wins, so the more
#: specific entries come first. A histogram tells you what is happening; this
#: is the difference between that and an answer.
_REMEDY = (
    ("data_blocked_prediction_model",
     "the multi-head predictor is not trained, so EVERY launch is refused "
     "before liquidity is even resolved. Nothing downstream can run.\n"
     "    .venv/bin/python -m src.runtime.train_once "
     "--storage data/launch_episodes --model-dir models"),
    ("champion_not_promoted",
     "a model exists but the champion/challenger framework has not promoted "
     "it to live authority. Shadow mode does not need this; a live desk "
     "does. Check `champions` on /status."),
    ("data_blocked_liquidity",
     "depth could not be resolved for these launches. The RPC ladder or the "
     "pool reader is the thing to look at, not the policy."),
    ("reentry_", "these are re-entry attempts, not first entries; they are "
     "not what is stopping the ladder."),
)


def _remedy(reason: str) -> str:
    lowered = reason.lower()
    for marker, text in _REMEDY:
        if marker in lowered:
            return text
    return ""


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="data/state")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)

    state = Path(args.state_dir)
    census = _load(state / "launch_census.json")
    if not census:
        print(f"no launch census at {state / 'launch_census.json'}")
        return EXIT_DATA_BLOCKED
    totals = census.get("totals") or {}
    seen = int(totals.get("seen", 0) or 0)
    if not seen:
        print("the census has seen no launches; the denominator is empty")
        return EXIT_DATA_BLOCKED

    entered = int(totals.get("entered", 0) or 0)
    screened = int(totals.get("screened", 0) or 0)
    decided = int(totals.get("decided", 0) or 0)

    print(f"{'seen':22s} {seen:>10d}")
    print(f"{'screened out':22s} {screened:>10d}  "
          f"{screened / seen:>6.1%}")
    print(f"{'reached a decision':22s} {decided:>10d}  {decided / seen:>6.1%}")
    print(f"{'ENTERED':22s} {entered:>10d}  {entered / seen:>6.1%}")
    print()

    reasons = dict(totals.get("screened_by_reason") or {})
    rejected = dict(totals.get("rejected_by_reason") or {})
    if reasons:
        print("screened before a decision, by reason:")
        for name, count in sorted(reasons.items(), key=lambda item: -item[1])[:args.top]:
            print(f"  {name:44s} {count:>10d}  {count / seen:>6.1%}")
    if rejected:
        print()
        print("reached a decision and was refused, by reason:")
        for name, count in sorted(rejected.items(), key=lambda item: -item[1])[:args.top]:
            print(f"  {name:44s} {count:>10d}  {count / seen:>6.1%}")

    # What the desk could not answer, versus what it answered no to. Only the
    # first is fixed by making something run.
    combined = {**reasons, **rejected}
    unanswered = sum(count for name, count in combined.items()
                     if any(mark in name.lower() for mark in _UNANSWERED))
    if combined:
        print()
        print(f"{'could not answer':22s} {unanswered:>10d}  "
              f"{unanswered / max(1, sum(combined.values())):>6.1%} of refusals")
        print(f"{'answered no':22s} "
              f"{sum(combined.values()) - unanswered:>10d}")

    monsters_by_screen = dict(totals.get("monsters_by_screen") or {})
    if monsters_by_screen:
        print()
        print("10x+ launches this desk refused, by the screen that refused them:")
        for name, count in sorted(monsters_by_screen.items(),
                                  key=lambda item: -item[1])[:args.top]:
            print(f"  {name:44s} {count:>10d}")

    if entered:
        return EXIT_OK
    print()
    print("NOTHING HAS BEEN ENTERED.")
    top = max(combined, key=lambda name: combined[name]) if combined else ""
    if top:
        print(f"top refusal: {top} ({combined[top]:,}, "
              f"{combined[top] / seen:.1%} of launches seen)")
        remedy = _remedy(top)
        if remedy:
            print(f"  -> {remedy}")
        print()
    print("Three of the bottom rung's requirements -- net_log_growth,")
    print("rug_loss_share, monster_enrichment -- are ratios over entered")
    print("positions. With no denominator they report NOT MEASURED, the gate")
    print("fails closed, and no number of further decisions changes that.")
    print("The top screen above is what to fix; nothing else on the ladder")
    print("moves until it does.")
    return EXIT_NO_ENTRIES


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
