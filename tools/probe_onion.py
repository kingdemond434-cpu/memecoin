"""Read one explicitly selected public onion source through the real Tor path.

This is a connectivity/evidence probe, never an alpha or production benchmark.
"""
import argparse
import asyncio
import json
from pathlib import Path

from src.collectors.adapters import onion_source
from src.collectors.onion import OnionResearchTransport


async def probe(args):
    transport = OnionResearchTransport("onion:probe", args.url,
        source_ref=args.source_ref, max_pages=args.max_pages, max_depth=args.max_depth)
    try:
        source = onion_source(transport.source_id, transport)
        events = await source.collect()
        return {"scope": "PUBLIC_ONION_CONNECTIVITY_ONLY", "execution_authority": False,
                "transport": transport.report(), "events": [event.to_dict() for event in events]}
    finally:
        await transport.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(probe(args))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"observed": report["transport"]["status"],
                      "events": len(report["events"]), "report": str(args.output)}))
    return 0 if report["transport"]["status"] == "OBSERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
