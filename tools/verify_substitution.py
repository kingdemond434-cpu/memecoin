"""Probe every declared endpoint from THIS node and report which answer.

The catalogue in ``src/research/source_catalogue.py`` is a claim: sixty public
interfaces that should work without an account. A claim is not a measurement,
and the gap between the two is exactly where a coverage number goes wrong in
the flattering direction -- an endpoint that quietly refuses this address is
reported as declared breadth for ever unless somebody actually asks it.

So run this on the node that will use them, not on a laptop. Public endpoints
refuse datacentre ranges routinely and a rung that answers from your desk may
403 from the VPS, which is a real difference and the only one that matters.

    .venv/bin/python tools/verify_substitution.py
    .venv/bin/python tools/verify_substitution.py --domain regional_venues
    .venv/bin/python tools/verify_substitution.py --json out.json

Endpoints with a placeholder in the path are probed with a well-known value so
the shape of the request is real. Nothing here writes to the catalogue: what
to do about a dead rung is a decision, and a tool that edited the source
because a network blip happened during a probe would be worse than the blip.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collectors.transports import HttpClient  # noqa: E402
from src.research.source_catalogue import DOMAINS  # noqa: E402

#: Stand-ins so a templated URL is probed as a real request rather than with a
#: literal brace in the path. Wrapped SOL and a widely-listed article: both are
#: certain to exist, so a failure is the endpoint rather than the argument.
PROBE_VALUES = {
    "mint": "So11111111111111111111111111111111111111112",
    "mints": "So11111111111111111111111111111111111111112",
    "llama_ids": "solana:So11111111111111111111111111111111111111112",
    "channel": "telegram",
    "sub": "solana",
    "query": "solana",
    "article": "Solana_(blockchain_platform)",
    "start": "20260101",
    "end": "20260102",
    "helius_key": os.getenv("HELIUS_API_KEY", ""),
    "alchemy_key": os.getenv("ALCHEMY_API_KEY", ""),
}


async def probe(client: HttpClient, domain: str, endpoint) -> Dict[str, Any]:
    missing = endpoint.missing_credentials()
    if missing:
        return {"domain": domain, "endpoint": endpoint.name,
                "region": endpoint.region, "state": "UNCONFIGURED",
                "detail": "missing: " + ", ".join(missing)}
    url = endpoint.format(**PROBE_VALUES)
    started = time.time()
    try:
        status, body, _headers = await client.get(url)
    except Exception as exc:
        return {"domain": domain, "endpoint": endpoint.name,
                "region": endpoint.region, "state": "UNREACHABLE",
                "detail": f"{type(exc).__name__}: {exc}"}
    latency_ms = int((time.time() - started) * 1000)
    row = {"domain": domain, "endpoint": endpoint.name,
           "region": endpoint.region, "http": status,
           "latency_ms": latency_ms, "bytes": len(body or "")}
    if status == 429:
        # The endpoint works and wants us to wait. Reporting this as dead is
        # how a healthy source gets removed from a catalogue.
        row["state"] = "RATE_LIMITED"
    elif status == 403:
        row["state"] = "REFUSED"
        row["detail"] = "public endpoint refusing this address"
    elif status >= 400:
        row["state"] = "ERROR"
    elif not (body or "").strip():
        row["state"] = "EMPTY"
        row["detail"] = "200 with an empty body"
    else:
        row["state"] = "LIVE"
    return row


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", default=[],
                        help="probe only these domains; repeatable")
    parser.add_argument("--json", default="", help="write the full result here")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    wanted = set(args.domain) or set(DOMAINS)
    unknown = wanted - set(DOMAINS)
    if unknown:
        print(f"unknown domain(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    client = HttpClient()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def bounded(domain: str, endpoint) -> Dict[str, Any]:
        async with semaphore:
            return await probe(client, domain, endpoint)

    tasks = [bounded(domain, endpoint)
             for domain in sorted(wanted)
             for endpoint in DOMAINS[domain]]
    rows = await asyncio.gather(*tasks)
    await client.close()

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)

    live_regions = set()
    dark_domains = []
    for domain in sorted(by_domain):
        entries = by_domain[domain]
        live = [row for row in entries if row["state"] in ("LIVE", "RATE_LIMITED")]
        if not live:
            dark_domains.append(domain)
        print(f"\n{domain}  {len(live)}/{len(entries)} answering")
        for row in entries:
            if row["state"] in ("LIVE", "RATE_LIMITED"):
                live_regions.add(row["region"])
            mark = {"LIVE": "ok  ", "RATE_LIMITED": "limit",
                    "REFUSED": "403 ", "UNCONFIGURED": "key ",
                    "EMPTY": "empty", "ERROR": "err ",
                    "UNREACHABLE": "down"}.get(row["state"], "?   ")
            extra = row.get("detail", "")
            latency = f"{row.get('latency_ms', 0)}ms" if "latency_ms" in row else ""
            print(f"  [{mark}] {row['endpoint']:<24} {row['region']:<7} "
                  f"{latency:<8} {extra}")

    total = len(rows)
    answering = sum(1 for row in rows if row["state"] in ("LIVE", "RATE_LIMITED"))
    print(f"\n{answering}/{total} endpoints answering from this node")
    print(f"regions proven from here: {', '.join(sorted(live_regions)) or 'none'}")
    if dark_domains:
        # The only failure worth an exit code. A few dead rungs is what the
        # ladder is for; a domain with none is a question with no answer.
        print(f"DOMAINS WITH NO WORKING ENDPOINT: {', '.join(dark_domains)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"probed_at": time.time(), "rows": rows,
                       "dark_domains": dark_domains}, handle, indent=2)
        print(f"written to {args.json}")
    return 1 if dark_domains else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
