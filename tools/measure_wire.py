"""How far this box actually is from the places a transaction has to reach.

The largest latency term in this desk is not code. It is the distance between
this host and (a) whoever streams us the chain and (b) the block engine that
relays a bundle to a leader. Both are physics plus peering, both are fixed by
moving the box or changing the provider, and neither is arguable once
measured. This measures them.

Run it ON THE NODE. A number from a laptop is a number about the laptop.

    .venv/bin/python tools/measure_wire.py
    .venv/bin/python tools/measure_wire.py --samples 20 --json data/state/wire.json

What it reports, and why each matters:

  BLOCK ENGINES   round trip to every Jito region the submitter races. The
                  desk already sends to all of them and takes first receipt,
                  so what matters is the BEST one -- that is your real
                  submission latency, and the gap between best and worst is
                  what moving the box would change.

  RPC ENDPOINTS   round trip to each configured Solana RPC, which bounds how
                  fast a blockhash refresh or a confirmation poll can be.

  SLOT REFERENCE  a slot is about 400ms. Every round trip is quoted as a
                  fraction of one, because "38ms" means nothing on its own
                  and "one tenth of a slot" is a decision.

Method: an HTTPS GET per sample, timed with `perf_counter_ns`, taking the
MINIMUM rather than the mean. The minimum is the closest thing to the true
path latency -- means are dominated by scheduler noise and the occasional
retransmit, and a sniper cares about the floor because that is what a
well-timed submission gets. TCP and TLS setup are excluded after the first
sample by reusing the connection, which is what the desk itself does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.jupiter_jito import JitoClient  # noqa: E402

SLOT_MS = 400.0

#: Cheap, documented, unauthenticated endpoints on each host. A health path
#: rather than a bundle submission: this measures the WIRE, and sending real
#: bundles to five regions to time them would be both rude and misleading.
BLOCK_ENGINE_PROBE = "/api/v1/bundles/tip_floor"

DEFAULT_RPCS = (
    ("solana_foundation", "https://api.mainnet-beta.solana.com"),
    ("publicnode", "https://solana-rpc.publicnode.com"),
    ("ankr", "https://rpc.ankr.com/solana"),
    ("drpc", "https://solana.drpc.org"),
)


async def _time_get(session: Any, url: str, samples: int) -> Dict[str, Any]:
    timings: List[float] = []
    error = ""
    for index in range(samples):
        started = time.perf_counter_ns()
        try:
            async with session.get(url) as response:
                await response.read()
                status = response.status
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        # The first sample pays TCP and TLS setup, which the desk pays once at
        # startup and never again. Including it would report a latency nobody
        # experiences during a launch.
        if index > 0:
            timings.append(elapsed_ms)
        if status >= 500:
            error = f"HTTP {status}"
    if not timings:
        return {"reachable": False, "detail": error or "no samples"}
    return {
        "reachable": True,
        "best_ms": round(min(timings), 2),
        "median_ms": round(statistics.median(timings), 2),
        "worst_ms": round(max(timings), 2),
        "samples": len(timings),
        "slots": round(min(timings) / SLOT_MS, 3),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8,
                        help="requests per endpoint; the first is discarded")
    parser.add_argument("--json", default="", help="write the full result here")
    parser.add_argument("--rpc", action="append", default=[],
                        help="extra RPC url to time; repeatable")
    args = parser.parse_args()

    import aiohttp

    targets: List[tuple] = []
    for base in JitoClient.DEFAULT_REGIONS:
        name = base.split("//")[-1].split(".")[0]
        targets.append(("block_engine", name, base + BLOCK_ENGINE_PROBE))
    rpcs = list(DEFAULT_RPCS)
    for url in args.rpc:
        rpcs.append((url.split("//")[-1].split("/")[0], url))
    configured = os.getenv("SOLANA_RPC_URL", "").strip()
    if configured:
        rpcs.insert(0, ("configured", configured))
    for name, url in rpcs:
        targets.append(("rpc", name, url))

    timeout = aiohttp.ClientTimeout(total=15)
    results: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for kind, name, url in targets:
            row = await _time_get(session, url, max(2, args.samples))
            row.update({"kind": kind, "name": name, "url": url.split("?")[0]})
            results.append(row)

    def show(kind: str, title: str) -> Optional[Dict[str, Any]]:
        rows = [row for row in results if row["kind"] == kind]
        live = sorted((row for row in rows if row["reachable"]),
                      key=lambda row: row["best_ms"])
        print(f"\n{title}")
        for row in live:
            print(f"  {row['name']:<22} {row['best_ms']:7.1f} ms best   "
                  f"{row['median_ms']:7.1f} ms median   "
                  f"{row['slots']:.3f} slots")
        for row in rows:
            if not row["reachable"]:
                print(f"  {row['name']:<22}      --            {row['detail'][:60]}")
        return live[0] if live else None

    best_engine = show("block_engine", "Jito block engines (the submitter races all of these)")
    best_rpc = show("rpc", "Solana RPC endpoints")

    # An interception check, before any conclusion is drawn from the numbers.
    # Tokyo and Amsterdam are ~9,000 km apart; light in fibre needs ~90ms one
    # way and the real round trip is 200ms or more. If they measure within a
    # few milliseconds of each other, something local is terminating the
    # connection -- a corporate proxy, a CDN edge, a sandbox egress gateway --
    # and every number above is about that hop rather than about the internet.
    # Reporting a colocated-tier verdict off proxy numbers would be worse than
    # reporting nothing.
    engines = {row["name"]: row for row in results
               if row["kind"] == "block_engine" and row["reachable"]}
    far_pair = ("tokyo", "amsterdam")
    intercepted = (all(name in engines for name in far_pair)
                   and abs(engines["tokyo"]["best_ms"]
                           - engines["amsterdam"]["best_ms"]) < 50.0)
    if intercepted:
        print("\nTHESE NUMBERS ARE NOT THE INTERNET")
        print(f"  tokyo {engines['tokyo']['best_ms']:.1f} ms and amsterdam "
              f"{engines['amsterdam']['best_ms']:.1f} ms are ~9,000 km apart and "
              "cannot both be that close.")
        print("  Something local is terminating these connections -- a proxy, a "
              "CDN edge, or a sandbox egress gateway.")
        print("  Run this on the trading node itself, outside any proxy. Nothing "
              "below is a measurement of your real submission latency.")
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump({"measured_at": time.time(), "slot_ms": SLOT_MS,
                           "intercepted": True, "results": results}, handle, indent=2)
        return 2

    print("\nwhat this means")
    if best_engine is None:
        print("  no block engine answered from this box. Submission would have "
              "no route at all; this is the finding, not a measurement error.")
    else:
        best = best_engine["best_ms"]
        print(f"  nearest block engine: {best_engine['name']} at {best:.1f} ms "
              f"({best / SLOT_MS:.2f} of a 400ms slot)")
        if best <= 15:
            print("  that is colocated-tier. Geography is NOT your bottleneck; "
                  "spend the next effort on code and on the stream instead.")
        elif best <= 40:
            print("  that is a normal European or US datacentre next to a region. "
                  "Moving the box would buy you tens of milliseconds at most.")
        elif best <= 90:
            print("  you are one region away from the nearest engine. Moving the "
                  "box to that region is worth roughly "
                  f"{best - 15:.0f} ms per submission, which is "
                  f"{(best - 15) / SLOT_MS:.2f} of a slot on every trade.")
        else:
            print("  you are FAR from every engine. This single term is larger "
                  "than everything the code does put together, and moving the "
                  "box is the highest-value change available.")
        spread = max(row["best_ms"] for row in results
                     if row["kind"] == "block_engine" and row["reachable"]) - best
        print(f"  spread across regions: {spread:.0f} ms "
              "-- that is what racing all of them already saves you")
    if best_rpc is not None:
        print(f"  nearest RPC: {best_rpc['name']} at {best_rpc['best_ms']:.1f} ms "
              "(bounds blockhash refresh and confirmation polling)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"measured_at": time.time(), "slot_ms": SLOT_MS,
                       "results": results}, handle, indent=2)
        print(f"\nwritten to {args.json}")
    return 0 if best_engine is not None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
