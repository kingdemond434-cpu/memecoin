"""What each safety veto actually cost, measured against what happened next.

A veto is only defensible if the launches it discarded went on to do badly.
Nothing in the desk measured that: the hard RiskVeto rejected 100% of decided
launches on 2026-08-29 and no report asked whether those launches then rugged
or ran. A filter that rejects everything has no losses on any win-rate
measure, which is precisely why the census exists and why this replays
against it.

The question this answers, per veto cause:

    of the launches THIS cause rejected, what did they resolve to, and what
    would a probe-sized position in them have returned?

Measured in log space, because the desk's objective is E[log W] and an
arithmetic mean of multiples is dominated by the one outcome that ran.

The verdict is taken on the REALISABLE return, not the peak. Nearly every
launch trades above its open at some instant, so a peak-based judgement
declares every veto guilty and argues for removing safety from launches
that in fact lose money. The peak is discounted by this desk's own measured
peak-to-realised gap before any veto is called costly.

Deliberately conservative about what it will claim:

* Rejected launches that never resolved are excluded and counted, never
  treated as flat. Treating unobserved as zero-return makes every veto look
  free -- the error that runs in the expensive direction.
* A cause with fewer than MIN_SAMPLES resolved rejections reports
  DATA_BLOCKED rather than a number. Ten launches cannot price a tail.
* The replay prices the PEAK, which is an upper bound no real exit achieves.
  It is stated as such, and a veto that is not costing anything even at the
  peak is definitively not costing anything.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

#: Below this many resolved rejections a cause is unpriceable, not free.
MIN_SAMPLES = 30

#: What the census calls a monster. Mirrored rather than imported so this
#: tool keeps running if the census module moves.
MONSTER_MULTIPLE = 10.0

#: Log units between a launch's PEAK and what an exit policy actually
#: realises on it. Nobody sells at the peak: the gap is fees, slippage,
#: landing latency and the fact that the peak is only knowable afterwards.
#:
#: Calibrated from this desk's own chronological exit-policy replay rather
#: than assumed. That run measured a realised hold-baseline E[logW] of
#: -0.2776 over 1,542 out-of-sample episodes. Against a population mean log
#: PEAK of about +0.35, the peak-to-realised gap is ~0.63 log units.
#:
#: This is the difference between a defensible verdict and a dangerous one:
#: almost every launch peaks above its open at some instant, so a
#: peak-based replay declares every veto "costing growth" and would argue
#: for removing safety on launches that in fact lose money.
DEFAULT_REALISED_HAIRCUT = 0.63


def iter_records(state_path: Path, spill_path: Optional[Path]) -> Iterator[Dict[str, Any]]:
    """Every launch record, from live state and the spilled lake.

    The lake matters more than the live file here: detail is evicted as it
    resolves, so the launches with outcomes are disproportionately the ones
    that have already left memory.
    """
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        yield from (state.get("records") or [])
    except (OSError, ValueError):
        pass
    if spill_path is None or not spill_path.exists():
        return
    try:
        with spill_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def veto_causes(reason: str) -> List[str]:
    """Split a compound reject reason into the causes that fired.

    ``safety_veto:native_risk_level:critical,sell_route_unavailable`` is two
    causes, and attributing it wholly to the first is how a veto hides
    behind a louder one.
    """
    if not reason.startswith("safety_veto:"):
        return []
    return [part.strip() for part in reason.split(":", 1)[1].split(",") if part.strip()]


def replay(records: Iterable[Dict[str, Any]],
           min_samples: int = MIN_SAMPLES,
           realised_haircut: float = DEFAULT_REALISED_HAIRCUT) -> Dict[str, Any]:
    """Per-cause outcome distribution over the launches it rejected."""
    resolved: Dict[str, List[float]] = defaultdict(list)
    unresolved: Dict[str, int] = defaultdict(int)
    seen_mints: set = set()
    for record in records:
        mint = record.get("mint")
        if not mint or mint in seen_mints:
            continue
        seen_mints.add(mint)
        if record.get("disposition") != "DECIDED_REJECT":
            continue
        causes = veto_causes(str(record.get("disposition_reason", "") or ""))
        if not causes:
            continue
        peak = record.get("peak_multiple")
        for cause in causes:
            if peak is None:
                unresolved[cause] += 1
            else:
                resolved[cause].append(float(peak))

    rows = []
    for cause in sorted(set(resolved) | set(unresolved)):
        peaks = resolved.get(cause, [])
        blocked = unresolved.get(cause, 0)
        if len(peaks) < min_samples:
            rows.append({
                "cause": cause, "status": "DATA_BLOCKED",
                "resolved": len(peaks), "unresolved": blocked,
                "detail": (f"{len(peaks)} resolved rejections; {min_samples} "
                           "needed before a cost can be priced"),
            })
            continue
        # Log space: the objective is geometric growth, and an arithmetic
        # mean of multiples is decided by whichever one ran furthest.
        logs = [math.log(max(peak, 1e-9)) for peak in peaks]
        mean_log = sum(logs) / len(logs)
        monsters = sum(1 for peak in peaks if peak >= MONSTER_MULTIPLE)
        rows.append({
            "cause": cause,
            "status": "OK",
            "resolved": len(peaks),
            "unresolved": blocked,
            "mean_log_peak": mean_log,
            "median_peak": sorted(peaks)[len(peaks) // 2],
            "max_peak": max(peaks),
            "monsters_discarded": monsters,
            "monster_share": monsters / len(peaks),
            # The headline, and it is deliberately NOT the peak. A veto is
            # only costing growth if the launches it discarded would have
            # been profitable to actually trade -- peak minus what an exit
            # policy really achieves. Judging on the raw peak would declare
            # every veto guilty, because nearly every launch trades above
            # its open at some instant.
            "mean_log_realisable": mean_log - realised_haircut,
            "verdict": ("COSTING_GROWTH"
                        if mean_log - realised_haircut > 0
                        else "EARNING_ITS_PLACE"),
        })
    rows.sort(key=lambda row: (row["status"] != "OK",
                               -(row.get("mean_log_peak") or 0.0)))
    return {
        "schema": "v1",
        "min_samples": min_samples,
        "measured": "peak multiple, an UPPER BOUND no real exit achieves",
        "realised_haircut": realised_haircut,
        "haircut_source": ("measured peak-to-realised gap from this desk's "
                           "chronological exit-policy replay"),
        "causes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path,
                        default=Path("data/state/launch_census.json"))
    parser.add_argument("--spill", type=Path,
                        default=Path("data/state/launch_census.jsonl"))
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--realised-haircut", type=float,
                        default=DEFAULT_REALISED_HAIRCUT)
    args = parser.parse_args()
    report = replay(iter_records(args.state, args.spill), args.min_samples,
                    args.realised_haircut)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
