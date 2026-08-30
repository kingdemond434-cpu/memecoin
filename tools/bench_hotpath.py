"""Microbenchmarks for the hot path, with baselines CI can fail on.

A correctness regression fails the build. A latency regression currently does
not, which means the one property this system is supposed to have is the one
nothing defends. This closes that: every stage of the local hot path is timed
against a committed baseline, and a significant slowdown is a build failure.

What is measured, and only this:

  native_decode       decoding a bonding-curve account
  native_quote        pricing a buy off decoded reserves
  t0_decide           the whole Rust decision, survival bins included
  compile_message     assembling a 27-account v0 message
  assemble_tx         signatures plus message into wire bytes
  python_policy       the Python ActionValuePolicy, as the thing Rust is
                      replacing and therefore the number worth watching

What is deliberately NOT measured: anything with a network in it. A CI runner
sharing a host with forty other jobs cannot time a socket to within a factor
of three, and a threshold that noisy either fails constantly and gets disabled
or is set so loose it catches nothing. Network latency is measured on the node
by `tools/measure_wire.py` and lives in the latency ledger, where it belongs.

The tolerance is deliberately wide. Shared CI runners vary by a factor of two
between runs on identical code, so a 20% threshold would be a coin flip. This
catches an order-of-magnitude regression -- an accidental O(n^2), a lost
cache, a Rust path silently falling back to Python -- which is the class of
mistake worth failing a build over. Anything subtler than that belongs in a
measurement on the node, not in CI.

    .venv/bin/python tools/bench_hotpath.py                 # measure and compare
    .venv/bin/python tools/bench_hotpath.py --update        # rewrite baselines
    .venv/bin/python tools/bench_hotpath.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "hotpath_baseline.json")

#: How much slower than baseline is a failure. Wide on purpose -- see above.
DEFAULT_TOLERANCE = 4.0

#: Below this a measurement is dominated by timer resolution and scheduler
#: noise, and comparing it to anything is superstition.
NOISE_FLOOR_US = 0.5


def bench(fn: Callable[[], Any], *, iterations: int, repeats: int = 5) -> float:
    """Microseconds per call, taking the MINIMUM of several runs.

    The minimum rather than the mean: every source of noise on a shared runner
    makes a measurement slower and none makes it faster, so the fastest run is
    the closest thing to the code's own cost. A mean measures the runner.
    """
    fn()  # warm caches, imports, and any lazy initialisation
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        elapsed = (time.perf_counter_ns() - started) / iterations / 1_000.0
        best = min(best, elapsed)
    return best


def build_cases() -> Dict[str, Dict[str, Any]]:
    """Every case, or a clear reason it could not be built."""
    cases: Dict[str, Dict[str, Any]] = {}
    try:
        import solana_fastpath as fp
    except ImportError as exc:
        return {"_error": {"detail": f"native extension unavailable: {exc}"}}

    # A real bonding-curve account layout: discriminator plus five u64 and a
    # flag. Built here rather than fetched so the benchmark has no network.
    account = (bytes.fromhex("17b7f83760d8ac60")
               + (1_073_000_000_000_000).to_bytes(8, "little")
               + (30_000_000_000).to_bytes(8, "little")
               + (793_100_000_000_000).to_bytes(8, "little")
               + (5_000_000_000).to_bytes(8, "little")
               + (1_000_000_000_000_000).to_bytes(8, "little")
               + b"\x00")
    cases["native_decode"] = {
        "fn": lambda: fp.decode_bonding_curve(account), "iterations": 20_000}
    cases["native_quote"] = {
        "fn": lambda: fp.quote_buy_from_account(account, 1_000_000_000),
        "iterations": 20_000}

    levels = [0.62, 0.41, 0.24, 0.13, 0.06, 0.03, 0.012, 0.004]
    cases["t0_decide"] = {
        "fn": lambda: fp.t0_decide(
            2.5, 30_000_000_000, 1_073_000_000_000_000, levels, 0.09, 0.31, 2.4,
            0.5, 1.8, 0.01, 0.01, 0.9, 0.8, 0.0, 90.0, 0.1, 0.05, 0.02,
            1e-4, 0.5, False, 1.0, 1.0, 0.0, 0.0, True),
        "iterations": 20_000}

    try:
        from solders.hash import Hash
        from solders.instruction import AccountMeta, Instruction
        from solders.keypair import Keypair
        from solders.message import MessageV0, to_bytes_versioned
        from solders.pubkey import Pubkey
    except ImportError as exc:
        cases["_solders"] = {"detail": f"solders unavailable: {exc}"}
        return cases

    keypair = Keypair.from_seed(bytes([3] * 32))
    payer = keypair.pubkey()
    blockhash = Hash(bytes([7] * 32))
    metas = [AccountMeta(Pubkey(bytes([index % 251 + 1] * 32)), False, index % 3 == 0)
             for index in range(26)]
    metas.append(AccountMeta(payer, True, True))
    program = Pubkey(bytes([200] * 32))
    data = bytes(range(24))
    raw = [(bytes(program),
            [(bytes(meta.pubkey), meta.is_signer, meta.is_writable) for meta in metas],
            data)]
    payer_bytes, blockhash_bytes = bytes(payer), bytes(blockhash)
    cases["compile_message"] = {
        "fn": lambda: fp.compile_v0_message(payer_bytes, raw, blockhash_bytes),
        "iterations": 5_000}

    message = MessageV0.try_compile(payer, [Instruction(program, data, metas)],
                                    [], blockhash)
    message_bytes = bytes(to_bytes_versioned(message))
    signature = bytes(keypair.sign_message(message_bytes))
    cases["assemble_tx"] = {
        "fn": lambda: fp.assemble_transaction(message_bytes, [signature]),
        "iterations": 20_000}

    try:
        from src.strategies.action_value import ActionValuePolicy, PositionState
    except ImportError as exc:  # pragma: no cover - defensive
        cases["_policy"] = {"detail": f"policy unavailable: {exc}"}
        return cases
    policy = ActionValuePolicy(min_edge=1e-4, max_add_fraction=0.5)
    forward = tuple(fp.survival_bins(levels, 0.09, 0.31, 2.4))
    state = PositionState(
        held_fraction=0.5, current_multiple=1.8, forward_bins=forward,
        exit_cost=0.01, entry_cost=0.01, exit_capacity_ratio=0.9,
        escape_probability=0.8, expected_remaining_seconds=90.0,
        alternative_growth_per_second=0.0, add_fraction=0.1, probe_fraction=0.05)
    cases["python_policy"] = {"fn": lambda: policy.score(state), "iterations": 2_000}
    return cases


def load_baseline() -> Dict[str, float]:
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
            return dict(json.load(handle).get("microseconds") or {})
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="rewrite the baselines from this run")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="multiple of baseline that counts as a regression")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    cases = build_cases()
    problems = {name: case["detail"] for name, case in cases.items()
                if name.startswith("_")}
    for name in list(cases):
        if name.startswith("_"):
            cases.pop(name)
    if problems:
        for name, detail in problems.items():
            print(f"skipped: {detail}")
    if not cases:
        # Not a pass. A benchmark suite that measured nothing has not shown
        # that anything is fast, and reporting success would be a lie CI then
        # repeats on every build.
        print("no benchmark could be built; nothing was measured", file=sys.stderr)
        return 2

    baseline = load_baseline()
    results: Dict[str, float] = {}
    print(f"{'case':<20} {'us/call':>10} {'baseline':>10} {'ratio':>8}  verdict")
    regressions: List[str] = []
    for name, case in cases.items():
        micros = bench(case["fn"], iterations=case["iterations"])
        results[name] = round(micros, 4)
        previous = baseline.get(name)
        if previous is None:
            verdict = "new (no baseline)"
            ratio = ""
        elif previous < NOISE_FLOOR_US and micros < NOISE_FLOOR_US:
            verdict = "below the noise floor"
            ratio = ""
        else:
            factor = micros / max(previous, 1e-9)
            ratio = f"{factor:.2f}x"
            if factor > args.tolerance:
                verdict = "REGRESSION"
                regressions.append(
                    f"{name}: {micros:.2f}us against a {previous:.2f}us baseline "
                    f"({factor:.1f}x slower)")
            else:
                verdict = "ok"
        shown = f"{previous:.2f}" if previous is not None else "-"
        print(f"{name:<20} {micros:>10.2f} {shown:>10} {ratio:>8}  {verdict}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"measured_at": time.time(), "microseconds": results},
                      handle, indent=2)

    if args.update:
        with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
            json.dump({"note": "minimum microseconds per call; see "
                               "tools/bench_hotpath.py for why the minimum",
                       "tolerance": args.tolerance,
                       "microseconds": results}, handle, indent=2, sort_keys=True)
        print(f"\nbaselines written to {BASELINE_PATH}")
        return 0

    if regressions:
        print("\nLATENCY REGRESSION")
        for line in regressions:
            print(f"  {line}")
        print("\nIf this is a deliberate change, re-run with --update and commit "
              "the new baseline in the same change that caused it.")
        return 1
    missing = [name for name in cases if name not in baseline]
    if missing:
        print(f"\nno baseline yet for: {', '.join(sorted(missing))}")
        print("run with --update and commit the result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
