"""One explicitly invoked public-source proof, isolated from trading state.

Uses the production scout, shared HttpClient and existing TradeEvidenceLedger.
This does not start the desk, use credentials or contact a blockchain account.
It records the source response and registry state actually observed by this
local process. It is not forward-return, execution or live-trading evidence.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collectors.transports import HttpClient
from src.research.chain_opportunity import ChainOpportunityScout
from src.research.frontier import FrontierIndex
from src.research.trade_evidence import TradeEvidenceLedger


async def prove():
    destination = ROOT.parent / "memecoin-chain-discovery-proof"
    destination.mkdir(parents=True, exist_ok=True)
    evidence_path = destination / "canonical_evidence.jsonl"
    index_path = destination / "frontier.sqlite"
    ledger = TradeEvidenceLedger(evidence_path)
    frontier = FrontierIndex(index_path)
    newly_registered = []

    def sink(record):
        if record["kind"] == "chain":
            was_absent = frontier.chain(record["identity"]) is None
            frontier.observe_discovery("chain", record["identity"], family="UNKNOWN",
                                        metrics=record["metrics"], evidence_refs=[record["source_response_ref"]])
            if was_absent:
                newly_registered.append(record["identity"])
            return record["source_response_ref"]
        if record["kind"] == "protocol":
            frontier.observe_discovery("protocol", record["identity"], axes={"chain": record["chain_id"]},
                                        version=record["version"], contract=record["contract"],
                                        evidence_refs=[record["source_response_ref"]])
            return record["source_response_ref"]
        reference = ledger.record_reference("unknown_opportunity_public_source_response", record,
                                            timestamp=record["observed_at"])
        if reference is None:
            raise RuntimeError("canonical evidence append failed")
        return reference

    http = HttpClient(timeout_s=20)
    try:
        result = await ChainOpportunityScout(http, sink, max_pair_requests=4).run_once()
    finally:
        await http.close()
    intact, detail, records = TradeEvidenceLedger.verify(evidence_path)
    restored = FrontierIndex(index_path)
    result.update(
        proof_scope="ONE_LOCAL_PUBLIC_SOURCE_DISCOVERY_PASS_NO_TRADING_OR_FORWARD_PERFORMANCE_CLAIM",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        canonical_evidence_path=str(evidence_path), frontier_path=str(index_path),
        evidence_integrity={"intact": intact, "detail": detail, "records": records},
        newly_registered_chain_ids=sorted(newly_registered),
        restart_restored_all_chains=all(restored.chain(chain) is not None for chain in result["chains"]),
        unknown_metrics_preserved=all(all(value is None for value in restored.chain(chain)["metrics"].values())
                                      for chain in result["chains"]),
        registry_report=restored.report(),
    )
    manifest_path = ROOT / "docs" / "audits" / "chain_discovery_proof" / "result.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    asyncio.run(prove())
