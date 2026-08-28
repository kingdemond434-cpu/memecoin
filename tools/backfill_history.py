"""Backfill reconstructed launch episodes from public RPC, at zero cost.

The extraction layer (src/research/history_warehouse.py) has existed for a
while; what never existed was a command that runs it. This is that command,
for the one backend that needs no account: walk the Pump program's signatures
backwards over a bounded window, fetch the transactions, decode them through
the SAME decoders the live feed uses (PumpFunMonitor accepts canonical
getTransaction JSON -- the historical-fixture tests already rely on this),
group the events into RawLaunch records, and hand them to the existing
reconstruction path, which stamps every episode as reconstructed so nothing
downstream can mistake it for an observed one.

Why this exists: the models' scarcest inputs are LABELS for rare outcomes.
After 2.7 days of forward observation the desk had 10 resolved monsters and
ZERO resolved rugs out of 2,618 outcomes -- not because rugs stopped, but
because a rug must be OBSERVED collapsing, and sparse marks miss the window.
The chain's history already contains thousands of complete lifecycles;
extraction is the only thing between the trainers and them.

Honesty notes, enforced by the reconstruction path rather than promised here:
reconstructed episodes carry survivorship, no-observation-latency, no-social,
no-route-feasibility limitations in their provenance stamp; collapses are
recorded as collapses, never as rugs, because price alone cannot attribute
intent; and thin launches (fewer than --min-trades trades) are refused rather
than reconstructed into noise.

This deliberately does NOT go through HistoryWarehouse.run(), which calls
backend.launches() synchronously and cannot drive the async RpcBackend; the
paging loop here is self-contained and checkpointed by newest-covered
signature instead.

Budget: one signature page is one request; each transaction fetch is batched
25 at a time through the method-aware router, which learns which free
endpoints serve getTransaction. A full day of Pump history is roughly 200k
signatures -- run bounded (--max-signatures) and let the checkpoint resume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chains.rpc_manager import ChainRegistry
from src.chains.yellowstone_grpc import (
    PumpFunMonitor, apply_event_timing, transaction_parts,
)
from src.research.backfill import RawLaunch, run_backfill

logger = logging.getLogger("backfill_history")

PUMP_PROGRAM = PumpFunMonitor.PUMP_FUN_PROGRAM
TRANSACTION_OPTIONS = {
    # "json", not "jsonParsed": the decoders read raw instruction data, and
    # the historical fixtures that pin their behaviour are in this encoding.
    "encoding": "json", "commitment": "confirmed",
    "maxSupportedTransactionVersion": 0,
}


class _StubYellowstone:
    """PumpFunMonitor subscribes in its constructor; here there is no stream."""

    def on(self, *_args, **_kwargs) -> None:
        return None


class EventCollector:
    """Feed getTransaction JSON through the live decoder, keep the events."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.monitor = PumpFunMonitor(_StubYellowstone(), self.events.append)

    async def ingest(self, tx_json: Dict[str, Any]) -> None:
        # The monitor stamps wall-clock receipt times; apply_event_timing
        # inside the decode path replaces them with block time when present,
        # which for a backfill is the only honest timestamp.
        await self.monitor._on_transaction(tx_json)


def _group_into_launches(events: List[Dict[str, Any]]) -> List[RawLaunch]:
    """Events -> RawLaunch, conservatively.

    A launch is only built when its CREATION was decoded inside the window:
    trades without an observed creation belong to a token born before the
    window started, and reconstructing a lifecycle from its middle would put
    a truncated price path into training wearing the shape of a full one.
    """
    created: Dict[str, Dict[str, Any]] = {}
    trades: Dict[str, List[Dict[str, Any]]] = {}
    migrated: Dict[str, float] = {}
    for event in events:
        token = str(event.get("token", "") or "")
        if not token or event.get("data_status") != "OK":
            continue
        kind = event.get("type")
        if kind == "token_created":
            created.setdefault(token, event)
        elif kind == "token_trade":
            trades.setdefault(token, []).append({
                "timestamp": float(event.get("timestamp", 0) or 0),
                "side": event.get("side"),
                "wallet": event.get("wallet"),
                "notional_sol": event.get("notional_sol"),
                # _price_path reads price_sol_per_token or curve_price_raw.
                "curve_price_raw": event.get("curve_price_raw"),
            })
        elif kind == "token_migrated":
            migrated[token] = float(event.get("timestamp", 0) or 0)

    launches: List[RawLaunch] = []
    for token, creation in created.items():
        rows = sorted(trades.get(token, ()), key=lambda item: item["timestamp"])
        launches.append(RawLaunch(
            token=token,
            created_at=float(creation.get("timestamp", 0) or 0),
            creator=str(creation.get("creator", "") or ""),
            bonding_curve=str(creation.get("bonding_curve", "") or ""),
            trades=rows,
            migrated_at=migrated.get(token),
            last_seen_at=rows[-1]["timestamp"] if rows else None,
        ))
    return launches


async def _fetch_transactions(rpc: Any, signatures: List[str],
                              batch_size: int) -> List[Optional[Dict[str, Any]]]:
    """Batched with sequential fallback, mirroring the wallet-history path."""
    out: List[Optional[Dict[str, Any]]] = []
    for start in range(0, len(signatures), batch_size):
        chunk = signatures[start:start + batch_size]
        payload = [{"jsonrpc": "2.0", "id": index, "method": "getTransaction",
                    "params": [signature, TRANSACTION_OPTIONS]}
                   for index, signature in enumerate(chunk)]
        try:
            out.extend(await rpc.batch_request(payload))
            continue
        except Exception as exc:
            logger.debug("batch refused (%s); sequential fallback", exc)
        semaphore = asyncio.Semaphore(3)

        async def fetch_one(signature: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    return await rpc.request(
                        "getTransaction", [signature, TRANSACTION_OPTIONS])
                except Exception:
                    return None

        out.extend(await asyncio.gather(*(fetch_one(s) for s in chunk)))
    return out


async def extract(args: argparse.Namespace) -> int:
    registry = ChainRegistry(args.config)
    await registry.start_all(["solana"])
    rpc = registry.get_rpc("solana")
    checkpoint_path = Path(args.checkpoint)
    checkpoint: Dict[str, Any] = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
        except (OSError, ValueError):
            checkpoint = {}

    cutoff = time.time() - args.hours * 3600.0
    collector = EventCollector()
    fetched = 0
    pages = 0
    before: Optional[str] = checkpoint.get("resume_before") or None
    oldest_seen: Optional[float] = None

    try:
        while fetched < args.max_signatures:
            params: Dict[str, Any] = {"limit": min(1000, args.max_signatures - fetched),
                                      "commitment": "confirmed"}
            if before:
                params["before"] = before
            page = await rpc.request("getSignaturesForAddress",
                                     [PUMP_PROGRAM, params])
            page = [row for row in (page or []) if isinstance(row, dict)]
            if not page:
                break
            pages += 1
            usable: List[str] = []
            done = False
            for row in page:
                block_time = row.get("blockTime")
                if block_time is not None:
                    oldest_seen = float(block_time)
                    if float(block_time) < cutoff:
                        done = True
                        break
                if row.get("err") is None and row.get("signature"):
                    usable.append(str(row["signature"]))
            transactions = await _fetch_transactions(rpc, usable, args.batch)
            for tx_json in transactions:
                if isinstance(tx_json, dict):
                    await collector.ingest(tx_json)
            fetched += len(usable)
            before = page[-1].get("signature") or before
            logger.info("page %d: %d signatures (%d total), oldest %s, "
                        "events so far %d", pages, len(usable), fetched,
                        time.strftime("%H:%M:%S", time.gmtime(oldest_seen))
                        if oldest_seen else "?", len(collector.events))
            if done:
                break
    finally:
        await registry.stop_all()

    launches = _group_into_launches(collector.events)
    report = run_backfill(launches, Path(args.out), min_trades=args.min_trades)

    checkpoint.update({
        "resume_before": before,
        "oldest_seen": oldest_seen,
        "updated_at": time.time(),
        "fetched_total": int(checkpoint.get("fetched_total", 0)) + fetched,
    })
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=1))

    decode = collector.monitor.matched
    print(json.dumps({
        "signatures_fetched": fetched,
        "events_decoded": len(collector.events),
        "decoder_matched": decode,
        "launches_with_creation": len(launches),
        "episodes_written": report.reconstructed,
        "episodes_refused_thin": report.blocked,
        "refusal_reasons": report.reasons,
        "limitations": report.limitation_counts,
        "output_dir": args.out,
        "resume_before": before,
    }, indent=1))
    # Zero episodes from nonzero signatures is the silent-zero shape this
    # desk keeps finding: make it loud.
    if fetched > 0 and report.reconstructed == 0:
        print("WARNING: fetched history produced no episodes; check "
              "refusal_reasons and decoder_matched above", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="reconstruct historical Pump launches from public RPC")
    parser.add_argument("--config", default="config/chains.yaml")
    parser.add_argument("--hours", type=float, default=6.0,
                        help="window backwards from now")
    parser.add_argument("--max-signatures", type=int, default=20_000,
                        help="hard request budget for this run; the checkpoint "
                             "resumes where it stopped")
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--out", default="data/launch_episodes/reconstructed")
    parser.add_argument("--checkpoint",
                        default="data/state/backfill_checkpoint.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(extract(args))


if __name__ == "__main__":
    raise SystemExit(main())
