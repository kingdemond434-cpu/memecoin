"""Many tiny lottery tickets, or fewer large ones? Simulated, not chosen.

    PYTHONPATH=. python3 tools/ticket_size.py --bankroll 200 --target 100000

Sweeps ticket size, concurrency and reserve over thousands of drawn launch
sequences and reports which policy most often reaches the target WITHOUT
ruining first. The ruin budget is a filter, not a penalty: without it the
optimiser bets everything on the most explosive coin, which maximises the
lottery payoff and produces a near-certain dead account.

Read the first line of output before anything else. The whole sweep is a
consequence of the survival curve it was given, and a curve implying a
generous edge will conclude that nearly any policy succeeds.

Defaults to the desk's own historical corpus rates, which imply a heavily
NEGATIVE unconditional edge -- meaning no ticket size reaches any target by
playing the base rate, and every cent of the desk's value has to come from
selection lifting the conditional distribution above it. That is a finding,
not a bug in the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chains.pumpswap_curve import PumpSwapPoolState
from src.research.migration_lineage import MigrationLineage
from src.strategies.wealth_target import (
    LaunchOutcomeModel, WealthTargetSimulator, default_policy_grid)

EXIT_OK = 0
EXIT_NO_ADMISSIBLE_POLICY = 3

#: Of 32,542 episodes in the desk's corpus: 683 reached 2x, 221 5x, 94 10x,
#: 40 20x, 13 50x, 5 100x. The UNCONDITIONAL rates, which is the point.
CORPUS_EPISODES = 32_542
CORPUS_COUNTS = ((2.0, 683), (5.0, 221), (10.0, 94), (20.0, 40),
                 (50.0, 13), (100.0, 5))

#: A representative migrated pool, used to price what a ticket of a given size
#: could actually take out of a printed move.
POOL_BASE = 206_900_000_000_000
POOL_QUOTE = 85_000_000_000


def _pool(multiple: float = 1.0) -> PumpSwapPoolState:
    product = POOL_BASE * POOL_QUOTE
    quote = int(POOL_QUOTE * math.sqrt(multiple))
    return PumpSwapPoolState(
        pool="reference", base_mint="b", quote_mint="q",
        base_reserves=product // quote, quote_reserves=quote,
        virtual_quote_reserves=0, total_fee_bps=100, updated_at=time.time())


def _capture_fn(sol_price_usd: float):
    """What a ticket of this size could really take out of a printed move."""
    price = POOL_QUOTE / POOL_BASE
    cache: Dict[Any, float] = {}

    def capture(printed: float, ticket_usd: float) -> float:
        key = (round(printed, 2), round(ticket_usd, 1))
        if key in cache:
            return cache[key]
        cost = int(ticket_usd / sol_price_usd * 1e9)
        if cost <= 0:
            return printed
        lineage = MigrationLineage()
        lineage.set_reference_position("ref", int(cost / price), cost)
        lineage.observe_migration("ref", _pool(1.0))
        lineage.observe_pool("ref", _pool(printed))
        taken = lineage.get("ref").max_executable_multiple
        cache[key] = float(taken) if taken is not None else printed
        return cache[key]

    return capture


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bankroll", type=float, default=200.0)
    parser.add_argument("--target", type=float, default=100_000.0)
    parser.add_argument("--floor", type=float, default=20.0)
    parser.add_argument("--launches", type=int, default=300)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--sol-price", type=float, default=200.0)
    parser.add_argument("--max-ruin", type=float, default=0.20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    curve = [(level, count / CORPUS_EPISODES) for level, count in CORPUS_COUNTS]
    model = LaunchOutcomeModel(curve, capture=_capture_fn(args.sol_price))
    simulator = WealthTargetSimulator(model, trials=args.trials,
                                      max_ruin=args.max_ruin)
    report = simulator.sweep(
        default_policy_grid(concurrency=(1, 10, 50), reserves=(0.0, 0.3)),
        bankroll=args.bankroll, target=args.target, floor=args.floor,
        launches=args.launches)

    if args.json:
        print(json.dumps(report, indent=2))
        return EXIT_OK if report.get("best") else EXIT_NO_ADMISSIBLE_POLICY

    edge = report.get("implied_edge_per_launch")
    print(f"survival curve implies E[multiple] = "
          f"{report['expected_multiple_per_launch']:.4f} per launch "
          f"({edge:+.1%} edge)")
    if report["input_warning"]:
        print(f"  ! {report['input_warning']}")
    print(f"${args.bankroll:,.0f} -> ${args.target:,.0f}, floor ${args.floor:,.0f}, "
          f"{args.launches} launches, {args.trials} sequences each")
    print()
    print(f"  {'policy':>18} {'P(target)':>10} {'P(ruin)':>9} "
          f"{'median':>12} {'p90':>12}")
    for row in report["ranked"][:12]:
        mark = " " if row["admissible"] else "x"
        print(f"{mark} {row['policy']['name']:>18} {row['hit_target']:>9.1%} "
              f"{row['ruined']:>8.1%} ${row['median_terminal']:>11,.2f} "
              f"${row['p90_terminal']:>11,.2f}")
    print()
    if report["best"]:
        print(f"  best admissible policy: {report['best']['policy']['name']}")
    else:
        print(f"  {report['detail']}")
    return EXIT_OK if report.get("best") else EXIT_NO_ADMISSIBLE_POLICY


if __name__ == "__main__":
    raise SystemExit(main())
