"""Distil the desk's own observations into compact per-wallet DNA.

The moat was already being accumulated and nothing read it. Measured
2026-08-29, the hazard spill held 83,953 tokens, 207,885 wallet-trades and
56,636 distinct wallets in 221 MB -- every one of them a point-in-time
observation this desk made itself, which is a stronger asset than a public
archive because nobody else has it in this form.

That is why this reads local spill rather than SolArchive or Old Faithful.
SolArchive's advertised data paths return 404 from this host and Old
Faithful is CAR-format requiring a parser; both are worth having later, and
neither is needed to start, because the executor's own history already
answers the question a wallet-DNA layer exists to answer.

The shape is the one the hot state was designed for:

    millions of cold observations -> distillation -> compact DNA
    -> only the currently relevant entities resident

so this writes a small artifact, not a database. 56,636 wallets at roughly
200 bytes is ~11 MB, which a 4 GB box can hold and an executor can index.

Two disciplines carry through from the census.

**Never turn unknown into zero.** A wallet whose tokens never resolved has
no measurable enrichment, and is emitted with a null rather than a 0.0 that
would read as "measured, and bad".

**Entry lag is measured against the token, not the clock.** How early a
wallet was is meaningful only relative to that token's own first observed
trade; wall-clock time says more about when the desk was running.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

#: A wallet seen on fewer tokens than this cannot be characterised: one
#: lucky launch is not a pattern, and a DNA record implying otherwise is
#: worse than no record.
MIN_TOKENS = 3

#: What the census calls a monster. Mirrored rather than imported so this
#: tool survives the census module moving.
MONSTER_MULTIPLE = 10.0

#: Pseudo-observations of the universe rate mixed into every wallet's
#: monster rate. Without it a wallet with ONE resolved token that happened
#: to moon scores 1.00 and outranks a wallet that hit three monsters in
#: ten -- which is exactly what the first run of this tool produced, and
#: is noise wearing the costume of a ranking.
#:
#: 20 is deliberately heavy against this data: the universe monster rate is
#: ~0.55%, so a wallet needs many resolved tokens before its own rate moves
#: the estimate far. A ranking that is hard to climb by luck is the point.
SHRINKAGE_PSEUDO_OBSERVATIONS = 20.0


def load_outcomes(census_spill: Path, census_state: Path) -> Dict[str, float]:
    """mint -> peak multiple, from every census record we can find.

    The spilled lake matters more than live state: detail is evicted as it
    resolves, so resolved launches are disproportionately the ones already
    written out.
    """
    outcomes: Dict[str, float] = {}

    def absorb(record: Dict[str, Any]) -> None:
        mint = record.get("mint")
        peak = record.get("peak_multiple")
        if mint and peak is not None:
            try:
                outcomes[mint] = float(peak)
            except (TypeError, ValueError):
                pass

    try:
        state = json.loads(census_state.read_text(encoding="utf-8"))
        for record in state.get("records") or []:
            absorb(record)
    except (OSError, ValueError):
        pass
    try:
        with census_spill.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    absorb(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return outcomes


def iter_tokens(spill: Path) -> Iterator[Tuple[str, list]]:
    """(mint, observations) per spilled token, streamed.

    Streamed deliberately: the spill is 221 MB and growing on a box that
    has been OOM-killed for less. Nothing here holds more than one token's
    observations at a time.
    """
    try:
        with spill.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                token = record.get("token")
                if token:
                    yield token, (record.get("observations") or [])
    except OSError:
        return


def distil(spill: Path, outcomes: Dict[str, float],
           min_tokens: int = MIN_TOKENS,
           shrinkage: float = SHRINKAGE_PSEUDO_OBSERVATIONS) -> Dict[str, Any]:
    """Per-wallet DNA over every token the desk observed it touching."""
    tokens_touched: Dict[str, set] = defaultdict(set)
    buys: Dict[str, int] = defaultdict(int)
    sells: Dict[str, int] = defaultdict(int)
    lags: Dict[str, list] = defaultdict(list)
    resolved_peaks: Dict[str, list] = defaultdict(list)

    for token, observations in iter_tokens(spill):
        trades = [row for row in observations
                  if row.get("type") == "trade" and row.get("wallet")]
        if not trades:
            continue
        stamps = [float(row.get("timestamp", 0) or 0) for row in trades]
        opened = min(stamps) if stamps else 0.0
        first_seen: Dict[str, float] = {}
        for row in trades:
            wallet = str(row["wallet"])
            tokens_touched[wallet].add(token)
            side = str(row.get("side", "")).lower()
            if side == "sell":
                sells[wallet] += 1
            else:
                buys[wallet] += 1
            stamp = float(row.get("timestamp", 0) or 0)
            if wallet not in first_seen or stamp < first_seen[wallet]:
                first_seen[wallet] = stamp
        peak = outcomes.get(token)
        for wallet, stamp in first_seen.items():
            # Relative to the token's own opening trade: how early they
            # were, not what time it happened to be.
            lags[wallet].append(max(0.0, stamp - opened))
            if peak is not None:
                resolved_peaks[wallet].append(peak)

    universe_peaks = list(outcomes.values())
    universe_monster_rate = (
        sum(1 for p in universe_peaks if p >= MONSTER_MULTIPLE) / len(universe_peaks)
        if universe_peaks else 0.0)

    records = []
    for wallet, tokens in tokens_touched.items():
        if len(tokens) < min_tokens:
            continue
        peaks = resolved_peaks.get(wallet, [])
        wallet_lags = sorted(lags.get(wallet, []))
        median_lag = (wallet_lags[len(wallet_lags) // 2]
                      if wallet_lags else None)
        if peaks:
            monsters = sum(1 for peak in peaks if peak >= MONSTER_MULTIPLE)
            # Log space, because the objective is geometric growth and an
            # arithmetic mean of multiples is decided by the largest one.
            mean_log_peak = sum(math.log(max(p, 1e-9)) for p in peaks) / len(peaks)
        else:
            monsters = 0
            mean_log_peak = None
        records.append({
            "wallet": wallet,
            "tokens_touched": len(tokens),
            "buys": buys.get(wallet, 0),
            "sells": sells.get(wallet, 0),
            # How early this wallet typically arrives, in seconds after the
            # token's first observed trade. The core early-entry signal.
            "median_entry_lag_s": median_lag,
            # Unknown stays null. A wallet whose tokens never resolved has
            # no measured enrichment, and 0.0 would read as "measured, bad".
            "resolved_tokens": len(peaks),
            "monsters_touched": monsters if peaks else None,
            "monster_rate": (monsters / len(peaks)) if peaks else None,
            # What the rate is worth once the sample size is accounted for.
            # This, not the raw rate, is what anything downstream should
            # rank on: it cannot be climbed by getting lucky once.
            "shrunk_monster_rate": (
                (monsters + shrinkage * universe_monster_rate)
                / (len(peaks) + shrinkage)) if peaks else None,
            # How far above the universe this wallet actually sits. 1.0 is
            # no skill; a wallet cannot look good merely by existing in a
            # market where monsters happen.
            "monster_enrichment": (
                ((monsters + shrinkage * universe_monster_rate)
                 / (len(peaks) + shrinkage)) / universe_monster_rate
                if peaks and universe_monster_rate > 0 else None),
            "mean_log_peak": mean_log_peak,
        })
    records.sort(key=lambda row: (-(row["shrunk_monster_rate"] or 0.0),
                                  -row["tokens_touched"]))
    return {
        "schema": "v1",
        "wallets": len(records),
        "min_tokens": min_tokens,
        # The denominator every wallet claim is judged against. Without it
        # a 2% monster rate looks like skill when the universe is 2%.
        "universe_monster_rate": universe_monster_rate,
        "universe_resolved": len(universe_peaks),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spill", type=Path,
                        default=Path("data/launch_episodes/hazard_observations.jsonl"))
    parser.add_argument("--census-spill", type=Path,
                        default=Path("data/state/launch_census.jsonl"))
    parser.add_argument("--census-state", type=Path,
                        default=Path("data/state/launch_census.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/state/wallet_dna.json"))
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--top", type=int, default=20,
                        help="how many rows to print")
    args = parser.parse_args()

    outcomes = load_outcomes(args.census_spill, args.census_state)
    report = distil(args.spill, outcomes, args.min_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report), encoding="utf-8")

    summary = dict(report)
    summary["records"] = report["records"][:args.top]
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
