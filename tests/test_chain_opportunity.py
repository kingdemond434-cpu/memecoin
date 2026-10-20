"""Cross-chain identities survive public-source and per-pass boundaries."""

import asyncio
import json

from src.research.chain_opportunity import ChainOpportunityScout
from src.research.frontier import CHAIN_METRICS, FrontierIndex


class HTTP:
    def __init__(self, profiles, pairs=None, status=200):
        self.profiles, self.pairs, self.status = profiles, pairs or [], status
        self.calls = []

    async def get(self, url):
        self.calls.append(url)
        body = self.profiles if "/token-profiles/" in url or "/token-boosts/" in url else self.pairs
        return self.status, json.dumps(body), {"Retry-After": "120"}


class Sink:
    def __init__(self, index=None):
        self.records = []
        self.index = index

    def __call__(self, record):
        self.records.append(record)
        if self.index and record["kind"] == "chain":
            self.index.observe_discovery("chain", record["identity"], family="UNKNOWN",
                                         metrics=record["metrics"], evidence_refs=[record["source_response_ref"]])
        if self.index and record["kind"] == "protocol":
            self.index.observe_discovery("protocol", record["identity"], axes={"chain": record["chain_id"]},
                                         version=record["version"], contract=record["contract"],
                                         evidence_refs=[record["source_response_ref"]])
        return f"canonical-evidence:{len(self.records)}"


def test_non_solana_chain_and_protocol_enter_frontier_without_handwritten_chain_list(tmp_path):
    async def scenario():
        index = FrontierIndex(tmp_path / "frontier.sqlite")
        sink = Sink(index)
        http = HTTP([{"chainId": "previously-unknown-ecosystem", "tokenAddress": "native-token"},
                     {"chainId": "solana", "tokenAddress": "sol-token"}],
                    [{"chainId": "previously-unknown-ecosystem", "dexId": "newly-observed-protocol", "pairAddress": "pool-not-a-factory"}])
        report = await ChainOpportunityScout(http, sink).run_once()
        assert report["chains"] == ["previously-unknown-ecosystem", "solana"]
        assert report["protocols"] == [{"chain_id": "previously-unknown-ecosystem", "protocol_id": "newly-observed-protocol"}]
        restarted = FrontierIndex(index.path)
        row = restarted.chain("previously-unknown-ecosystem")
        assert row["family"] == "UNKNOWN"
        assert set(row["metrics"]) == set(CHAIN_METRICS)
        assert all(value is None for value in row["metrics"].values())
        protocol = restarted.protocol("newly-observed-protocol", "previously-unknown-ecosystem", "UNKNOWN")
        assert protocol["contract"]["addresses"] is None
        assert protocol["contract"]["execution_support"] is None
        assert sink.records[0]["kind"] == "source_response"
        assert all(row["source_at"] is None for row in sink.records)
    asyncio.run(scenario())


def test_every_chain_is_recorded_before_pair_budget_and_raw_debt_is_retained():
    async def scenario():
        http, sink = HTTP([{"chainId": f"new-{i}", "tokenAddress": f"t-{i}"} for i in range(5)]), Sink()
        scout = ChainOpportunityScout(http, sink, max_pair_requests=1)
        report = await scout.run_once()
        assert len(report["chains"]) == 5
        assert len(http.calls) == 2
        assert report["deferred"][0]["remaining_targets"] == 4
        reference = report["deferred"][0]["evidence_ref"]
        assert reference == "canonical-evidence:1"
        assert len(json.loads(sink.records[0]["raw_body"])) == 5
        assert (await scout.run_once())["pass_status"] == "NOT_THIS_PASS_CADENCE"
        assert len(http.calls) == 2
    asyncio.run(scenario())


def test_row_budget_defers_full_canonical_response_without_truncation():
    async def scenario():
        http, sink = HTTP([{"chainId": f"x-{i}", "tokenAddress": "t"} for i in range(3)]), Sink()
        report = await ChainOpportunityScout(http, sink, max_rows=2).run_once()
        assert report["status"] == "OBSERVED_WITH_PERSISTED_DEBT"
        assert report["deferred"][0]["reason"] == "NOT_THIS_PASS_ROW_BUDGET"
        assert len(sink.records) == 1
        assert len(json.loads(sink.records[0]["raw_body"])) == 3
    asyncio.run(scenario())


def test_large_body_is_persisted_before_json_parse_budget():
    async def scenario():
        http, sink = HTTP([{"chainId": "x", "description": "z" * 2000}]), Sink()
        report = await ChainOpportunityScout(http, sink, max_body_bytes=1024).run_once()
        assert report["deferred"][0]["reason"] == "NOT_THIS_PASS_BODY_BUDGET"
        assert len(sink.records[0]["raw_body"]) > 1024
    asyncio.run(scenario())


def test_rate_limit_is_visible_and_does_not_send_further_requests():
    async def scenario():
        http, sink = HTTP({"error": "slow down"}, status=429), Sink()
        scout = ChainOpportunityScout(http, sink)
        report = await scout.run_once()
        assert report["status"] == "DATA_BLOCKED"
        assert report["blocked"][0]["reason"] == "RATE_LIMITED"
        assert sink.records[0]["http_status"] == 429
        await scout.run_once()
        assert len(http.calls) == 1
    asyncio.run(scenario())


def test_no_sink_acknowledgement_creates_no_discovered_identity():
    async def scenario():
        http = HTTP([{"chainId": "x", "tokenAddress": "t"}])
        report = await ChainOpportunityScout(http, lambda record: None).run_once()
        assert report["chains"] == []
        assert report["status"] == "DATA_BLOCKED"
        assert report["blocked"][0]["reason"] == "CANONICAL_SINK_OR_LOCAL_FAILURE"
    asyncio.run(scenario())


def test_pair_budget_reaches_new_ecosystems_instead_of_only_dominant_profile_chain():
    async def scenario():
        rows = [{"chainId": "dominant", "tokenAddress": f"t-{i}"} for i in range(10)]
        rows.append({"chainId": "new-ecosystem", "tokenAddress": "new-token"})
        http, sink = HTTP(rows), Sink()
        await ChainOpportunityScout(http, sink, max_pair_requests=2).run_once()
        assert "/dominant/t-0" in http.calls[1]
        assert "/new-ecosystem/new-token" in http.calls[2]
    asyncio.run(scenario())
