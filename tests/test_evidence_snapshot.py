import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from src.research import trade_evidence
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.proposal_rounds import ProposalRoundRunner
from test_proposal_rounds import FakeProvider, fact


def test_verified_snapshot_excludes_later_appends_without_blocking_writer(tmp_path, monkeypatch):
    ledger=TradeEvidenceLedger(tmp_path/'evidence.jsonl')
    ledger.record('fixture',{'round_id':'target','value':1})
    entered,release=threading.Event(),threading.Event()
    original=trade_evidence._verified_row
    def paused(*args):
        entered.set()
        assert release.wait(3), 'reader could not continue'
        return original(*args)
    monkeypatch.setattr(trade_evidence,'_verified_row',paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(ledger.matching_snapshot,'round_id','target')
        assert entered.wait(3)
        # The read must not own the append lock during verification.
        assert ledger._append_lock.acquire(blocking=False)
        try:
            ledger.record('fixture',{'round_id':'target','value':2})
        finally:
            ledger._append_lock.release();release.set()
        rows=future.result(timeout=3)
    assert [row['payload']['value'] for row in rows]==[1]
    assert TradeEvidenceLedger.verify(ledger.path)[0]


def test_nonmatching_tampered_rows_still_fail_whole_prefix_verification(tmp_path):
    ledger=TradeEvidenceLedger(tmp_path/'evidence.jsonl')
    ledger.record('fixture',{'round_id':'other','value':1})
    ledger.record('fixture',{'round_id':'target','value':2})
    rows=[json.loads(line) for line in ledger.path.read_text().splitlines()]
    rows[0]['payload']['value']=999
    ledger.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='row hash mismatch'):
        ledger.matching_snapshot('round_id','target')
    assert not TradeEvidenceLedger.verify(ledger.path)[0]


def test_incomplete_prefix_is_not_accepted_as_a_finished_snapshot(tmp_path):
    ledger=TradeEvidenceLedger(tmp_path/'evidence.jsonl')
    ledger.record('fixture',{'round_id':'target'})
    raw=ledger.path.read_bytes()
    ledger.path.write_bytes(raw[:-1])
    with pytest.raises(ValueError,match='incomplete committed'):
        ledger.matching_snapshot('round_id','target')
    ledger.path.write_bytes(b'\xef\xbb\xbf'+raw)
    with pytest.raises(ValueError): ledger.matching_snapshot('round_id','target')
    assert not TradeEvidenceLedger.verify(ledger.path)[0]


def test_proposal_history_read_allows_event_loop_callbacks(tmp_path, monkeypatch):
    async def scenario():
        runner=ProposalRoundRunner(TradeEvidenceLedger(tmp_path/'evidence.jsonl'),[FakeProvider('codex')],required_providers=['codex'])
        loop=asyncio.get_running_loop()
        def history(round_id):
            release=threading.Event()
            loop.call_soon_threadsafe(release.set)
            if not release.wait(3): raise RuntimeError('event loop blocked by evidence scan')
            return []
        monkeypatch.setattr(runner,'_previous',history)
        result=await runner.run_round(round_id='target',facts=[fact()],schemas={'schema':1})
        assert result['status']=='OK'
    asyncio.run(scenario())
