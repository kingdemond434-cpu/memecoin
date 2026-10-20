from concurrent.futures import ThreadPoolExecutor
import json

from src.research.trade_evidence import TradeEvidenceLedger


def test_concurrent_research_refs_resolve_to_their_own_hash_chained_rows(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "canonical.jsonl")
    def append(index):
        return index, ledger.record_reference("research", {"index": index})
    with ThreadPoolExecutor(max_workers=8) as workers:
        refs = list(workers.map(append, range(80)))
    assert TradeEvidenceLedger.verify(ledger.path) == (True, "OK", 80)
    with ledger.path.open("rb") as handle:
        for index, ref in refs:
            _, digest, offset = ref.split(":")
            handle.seek(int(offset))
            row = json.loads(handle.readline())
            assert row["hash"] == digest
            assert row["payload"]["index"] == index
