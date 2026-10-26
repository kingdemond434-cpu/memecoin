import hashlib
from types import SimpleNamespace

import pytest

from src.research.neural_sentiment import MINER, SentimentResearch


class Worker:
    async def infer(self, text, offset):
        total = len(text.split())
        return {'token_start': offset, 'token_stop': min(total, offset+510),
                'token_count': total, 'raw_model_scores': {'positive': .6, 'neutral': .3, 'negative': .1}}

    async def stop(self):
        pass


def enqueue(queue, text='public fixture', source='source:one', job='job:one'):
    return queue.enqueue({'text': text, 'source_id': source, 'observed_at': 100,
        'event_at': 90, 'language': 'en', 'platform': 'fixture'},
        source_job=job, source_ref='canonical:'+job, tokens=['token:one'])


class CountingWorker(Worker):
    def __init__(self):
        self.calls=0

    async def infer(self,text,offset):
        self.calls+=1
        return await super().infer(text,offset)


@pytest.mark.asyncio
@pytest.mark.parametrize('acknowledged',[False,True])
async def test_owner_loop_observation_uses_committed_snapshot_without_waiting_for_writer(tmp_path,acknowledged):
    import asyncio,threading
    from aiohttp import web
    from aiohttp.test_utils import TestClient,TestServer
    queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=CountingWorker());enqueue(queue)
    rows=await queue.fetch();identity=rows[0]['annotation_id']
    if acknowledged:queue.acknowledge(rows,'canonical:before-write')
    entered,released=threading.Event(),threading.Event()
    def slow_write():
        with queue.lock,queue.db:
            if not acknowledged:
                queue.db.execute('UPDATE results SET ref=?,available=? WHERE id=?',
                    ('canonical:uncommitted-fixture',200.,identity))
            else:
                queue.db.execute('UPDATE inputs SET served=served+1')
            entered.set();assert released.wait(5)
    writer=asyncio.create_task(asyncio.to_thread(slow_write))
    assert await asyncio.to_thread(entered.wait,3)
    async def health(request):
        proof=queue.observation(identity,as_of=10**12)
        return web.json_response({'observed':proof is not None})
    app=web.Application();app.router.add_get('/health',health)
    try:
        async with TestClient(TestServer(app)) as client:
            response=await asyncio.wait_for(client.get('/health'),1)
            assert (await response.json())['observed']==acknowledged
        assert not released.is_set() and not writer.done()
    finally:
        released.set();await writer
    assert queue.observation(identity,as_of=10**12) is not None
    if not acknowledged:assert queue.observation(identity,as_of=199.) is None
    queue.close()


def test_observation_connection_is_read_only_and_closed_with_its_owner(tmp_path):
    import sqlite3
    queue=SentimentResearch(tmp_path/'nlp.sqlite');reader=queue.observation_db
    with pytest.raises(sqlite3.OperationalError):reader.execute('DELETE FROM results')
    queue.close()
    with pytest.raises(sqlite3.ProgrammingError):reader.execute('SELECT count(*) FROM results')


@pytest.mark.asyncio
async def test_slow_report_does_not_lock_intake_or_canonical_delivery(tmp_path):
    import asyncio,threading
    queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=CountingWorker());enqueue(queue)
    rows=await queue.fetch();queue.acknowledge(rows,'canonical:before-report')
    entered,released=threading.Event(),threading.Event()
    def gate():
        if not entered.is_set():entered.set();assert released.wait(5)
        return 0
    queue.report_db.set_progress_handler(gate,1)
    pending=asyncio.create_task(asyncio.to_thread(queue.report))
    assert await asyncio.to_thread(entered.wait,3)
    try:
        # These are the synchronous methods called by actual collection and
        # owner-loop delivery. Neither shares the long-report reader's lock.
        assert enqueue(queue,'second public post',job='second')
        assert queue.observation(rows[0]['annotation_id'],as_of=10**12)
        assert not pending.done()
    finally:released.set();await pending
    queue.report_db.set_progress_handler(None,0)
    assert queue.report()['inputs']==2
    queue.close()


@pytest.mark.asyncio
async def test_report_counter_groups_share_one_committed_database_snapshot(tmp_path):
    queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=CountingWorker());enqueue(queue)
    rows=await queue.fetch();committed=[]
    def race(sql):
        if sql.startswith('SELECT count(*),count(ref)') and not committed:
            queue.acknowledge(rows,'canonical:during-report');committed.append(True)
    queue.report_db.set_trace_callback(race)
    first=queue.report();queue.report_db.set_trace_callback(None)
    assert committed and first['completed_inputs']==first['canonical_windows']==0
    second=queue.report()
    assert second['completed_inputs']==second['canonical_windows']==1
    queue.close()


@pytest.mark.asyncio
async def test_pending_queue_lock_cannot_hold_event_loop_and_cancel_drains_query(tmp_path):
    import asyncio,threading
    queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=CountingWorker());enqueue(queue)
    entered,released=threading.Event(),threading.Event()
    original=queue._pending_results
    def blocked():
        entered.set();assert released.wait(5);return original()
    queue._pending_results=blocked
    task=asyncio.create_task(queue.fetch())
    assert await asyncio.to_thread(entered.wait,3)
    # The owner loop still handles real network requests during the gated SQL operation.
    from aiohttp import web
    from aiohttp.test_utils import TestClient,TestServer
    async def health(request):return web.json_response({'responsive':True})
    app=web.Application();app.router.add_get('/health',health)
    try:
        async with TestClient(TestServer(app)) as client:
            response=await asyncio.wait_for(client.get('/health'),1)
            assert (await response.json())['responsive']
        task.cancel();await asyncio.sleep(0);assert not task.done()
    finally:released.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert queue.report()['inputs']==1 and queue.report()['canonical_windows']==0
    assert queue.last_fetch_profile['disk_wait_seconds']>=0
    queue.close()


@pytest.mark.asyncio
async def test_cancelled_result_commit_is_drained_and_replayed_without_duplicate_inference(tmp_path):
    import asyncio,threading
    worker=CountingWorker();queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=worker);enqueue(queue)
    entered,released=threading.Event(),threading.Event();owner=threading.get_ident();original=queue._commit_window
    def blocked(*args):
        assert threading.get_ident()!=owner
        entered.set();assert released.wait(5);return original(*args)
    queue._commit_window=blocked
    task=asyncio.create_task(queue.fetch());assert await asyncio.to_thread(entered.wait,3)
    task.cancel();await asyncio.sleep(0);assert not task.done();released.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert queue.report()['awaiting_canonical_ack']==1 and queue.report()['canonical_windows']==0
    rows=await queue.fetch()
    assert len(rows)==worker.calls==1 and queue.observation(rows[0]['annotation_id'],as_of=10**12) is None
    queue.acknowledge(rows,'canonical:after-cancellation')
    assert queue.report()['completed_inputs']==1 and queue.report()['canonical_windows']==1
    queue.close()


@pytest.mark.asyncio
async def test_result_transaction_failure_does_not_publish_cache_annotation_or_cursor(tmp_path):
    worker=CountingWorker();queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=worker);identity=enqueue(queue)
    before=queue.db.execute('SELECT cursor FROM dispatch').fetchone()[0]
    # Abort after the transaction has written its cache and annotation. All must roll back.
    queue.db.execute("CREATE TRIGGER reject_result BEFORE UPDATE ON inputs WHEN NEW.served<>OLD.served BEGIN SELECT RAISE(ABORT,'synthetic commit refusal'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):await queue.fetch()
    assert queue.db.execute('SELECT count(*) FROM inference_cache').fetchone()[0]==0
    assert queue.db.execute('SELECT count(*) FROM results').fetchone()[0]==0
    assert queue.db.execute('SELECT offset,complete FROM inputs WHERE id=?',(identity,)).fetchone()==(0,0)
    assert queue.db.execute('SELECT cursor FROM dispatch').fetchone()[0]==before
    queue.db.execute('DROP TRIGGER reject_result')
    queue.close()


def paced_worker(monkeypatch):
    """Deterministic four-window time quantum, independent of host disk speed."""
    tick = [0.]
    monkeypatch.setattr('src.research.neural_sentiment.time.perf_counter', lambda: tick[0])
    class PacedWorker(Worker):
        async def infer(self, text, offset):
            tick[0] += .5
            return await super().infer(text, offset)
    return PacedWorker()


@pytest.mark.asyncio
async def test_fast_inference_uses_available_time_and_commits_once_per_result(tmp_path,monkeypatch):
    monkeypatch.setattr('src.research.neural_sentiment.time.perf_counter', lambda: 0.)
    worker = CountingWorker()
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=worker)
    for index in range(12):
        enqueue(queue, f'distinct public fixture {index}', job=f'job:{index}')
    commits = []
    queue.db.set_trace_callback(lambda sql: commits.append(sql) if sql == 'COMMIT' else None)
    rows = await queue.fetch()
    assert len(rows) == worker.calls == 12
    assert len(commits) == 12
    assert queue.report()['awaiting_canonical_ack'] == 12
    assert queue.report()['canonical_windows'] == 0
    queue.close()


@pytest.mark.asyncio
async def test_failed_inference_consumes_dispatch_time_without_dropping_inputs(tmp_path,monkeypatch):
    tick = [0.]
    monkeypatch.setattr('src.research.neural_sentiment.time.perf_counter', lambda: tick[0])
    class FailingWorker(Worker):
        calls = 0
        async def infer(self, text, offset):
            self.calls += 1
            tick[0] += 1.
            raise TimeoutError('fixture worker unavailable')
    worker = FailingWorker()
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=worker)
    for index in range(12):
        enqueue(queue, f'public fixture {index}', job=f'job:{index}')
    assert await queue.fetch() == []
    assert worker.calls == 2
    assert queue.report()['pending_inputs'] == 12
    queue.close()


@pytest.mark.asyncio
async def test_result_write_failure_rolls_back_cache_and_annotation_but_counts_real_work(tmp_path):
    worker = CountingWorker()
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=worker)
    enqueue(queue)
    queue.db.execute("CREATE TRIGGER fixture_fail BEFORE INSERT ON results BEGIN SELECT RAISE(ABORT,'fixture'); END")
    assert await queue.fetch() == []
    assert worker.calls == 1
    assert queue.db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == 0
    assert queue.db.execute('SELECT COUNT(*) FROM inference_cache').fetchone()[0] == 0
    assert queue.report()['inference_metrics']['worker_inferences'] == 1
    assert queue.report()['pending_inputs'] == 1
    queue.db.execute('DROP TRIGGER fixture_fail')
    rows = await queue.fetch()
    assert worker.calls == 2 and len(rows) == 1
    assert queue.report()['inference_metrics']['worker_inferences'] == 2
    assert queue.report()['canonical_windows'] == 0
    queue.close()


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '4', None, 9])
def test_cpu_profile_rejects_invalid_or_oversubscribed_thread_counts(value,monkeypatch):
    from src.research.neural_sentiment import OfflineWorker
    monkeypatch.setattr('src.research.neural_sentiment.os.cpu_count', lambda: 8)
    with pytest.raises(ValueError, match='thread count'):
        OfflineWorker('python', 'public-model', intra_op_threads=value)


@pytest.mark.asyncio
async def test_cpu_profile_is_reported_and_separates_cache_identity(tmp_path,monkeypatch):
    from src.research.neural_sentiment import inference_profile
    monkeypatch.setattr('src.research.neural_sentiment.os.cpu_count', lambda: 8)
    worker = CountingWorker()
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=worker)
    enqueue(queue, job='first')
    first = await queue.fetch()
    queue.acknowledge(first, 'canonical:first-profile')
    queue.inference_profile = inference_profile(4)
    enqueue(queue, job='second')
    second = await queue.fetch()
    assert worker.calls == 2
    assert first[0]['inference_cache_key'] != second[0]['inference_cache_key']
    assert second[0]['inference_profile']['intra_op_threads'] == 4
    assert queue.report()['inference_profile'] == second[0]['inference_profile']
    queue.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch', [False, True])
async def test_worker_passes_profile_without_credentials_and_verifies_child_receipt(monkeypatch,mismatch):
    import json
    from unittest.mock import AsyncMock, Mock
    from src.research.neural_sentiment import OfflineWorker, inference_profile
    monkeypatch.setattr('src.research.neural_sentiment.os.cpu_count', lambda: 8)
    response = {'ok': True, 'result': {'fixture': True},
                'inference_profile': inference_profile(1 if mismatch else 4)}
    process = SimpleNamespace(returncode=None,
        stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
        stdout=SimpleNamespace(readline=AsyncMock(return_value=(json.dumps(response)+'\n').encode())),
        kill=Mock(), wait=AsyncMock())
    async def spawn(*args, **kwargs):
        assert args[-2:] == ('--threads', '4')
        assert set(kwargs['env']) <= {'SYSTEMROOT','WINDIR','TEMP','TMP','PYTHONPATH',
                                     'PYTHONIOENCODING','HF_HUB_OFFLINE','TOKENIZERS_PARALLELISM'}
        return process
    monkeypatch.setattr('asyncio.create_subprocess_exec', spawn)
    worker = OfflineWorker('python', 'public-model', intra_op_threads=4)
    if mismatch:
        with pytest.raises(ValueError):
            await worker.infer('public fixture', 0)
        process.kill.assert_called_once()
        assert worker.process is None
    else:
        assert await worker.infer('public fixture', 0) == {'fixture': True}
        await worker.stop()


@pytest.mark.asyncio
async def test_newest_post_bypasses_live_backlog_without_starving_old_live_or_replay(tmp_path,monkeypatch):
    path=tmp_path/'nlp.sqlite'
    worker = paced_worker(monkeypatch)
    queue=SentimentResearch(path,worker=worker)
    for index in range(100):
        enqueue(queue,f'old live text {index}',job=f'old:{index}')
    queue.enqueue({'text':'historical text','source_id':'historical','observed_at':1},
        source_job=None,source_ref='canonical:history')
    # Use the persistent arrival ticket, not the corrected host wall clock.
    monkeypatch.setattr('src.research.neural_sentiment.time.time',lambda:50.)
    enqueue(queue,'newest live text',job='fresh:arrival')
    queue.close()
    queue=SentimentResearch(path,worker=worker)
    rows=await queue.fetch()
    assert rows[0]['source_job']=='fresh:arrival'
    assert any(row['source_job']=='old:0' for row in rows)
    assert any(row['input_origin']=='REPLAY' for row in rows)
    queue.acknowledge(rows,'canonical:all-lanes')
    for turn in range(3):
        enqueue(queue,f'fresh turn {turn}',job=f'fresh:{turn}')
        rows=await queue.fetch()
        assert any(row['source_job']==f'fresh:{turn}' for row in rows)
        assert any(str(row['source_job']).startswith('old:') for row in rows)
        queue.acknowledge(rows,f'canonical:turn:{turn}')
    queue.close()


@pytest.mark.asyncio
async def test_live_intake_progresses_ahead_of_large_history_without_starving_history(tmp_path,monkeypatch):
    path=tmp_path/'nlp.sqlite'
    worker = paced_worker(monkeypatch)
    queue=SentimentResearch(path,worker=worker)
    for index in range(100):
        queue.enqueue({'text':f'old document {index}', 'source_id':'historical', 'observed_at':10},
            source_job=None,source_ref=f'canonical:replay:{index}')
    enqueue(queue,'current live observation',job='live:first')
    queue.close()
    queue=SentimentResearch(path,worker=worker)
    rows=await queue.fetch()
    assert any(row['source_job']=='live:first' and row['input_origin']=='LIVE' for row in rows)
    assert any(row['input_origin']=='REPLAY' for row in rows)
    queue.acknowledge(rows,'canonical:mixed')
    enqueue(queue,'another current live observation',job='live:second')
    rows=await queue.fetch()
    assert any(row['source_job']=='live:second' for row in rows)
    assert any(row['input_origin']=='REPLAY' for row in rows)
    assert queue.report()['queue_lanes']['REPLAY']['pending'] > 90
    queue.close()


@pytest.mark.asyncio
async def test_identical_text_reuses_inference_but_preserves_each_source_and_ack(tmp_path):
    worker=CountingWorker()
    queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=worker)
    for index in range(3):
        enqueue(queue, 'same exact text',source=f'source:{index}',job=f'job:{index}')
    rows=await queue.fetch()
    assert len(rows)==3 and worker.calls==1
    assert sum(row['inference_cache_hit'] for row in rows)==2
    assert len({row['source_ref'] for row in rows})==3
    assert len({row['annotation_id'] for row in rows})==3
    assert len({row['inference_cache_key'] for row in rows})==1
    for row in rows:
        assert queue.observation(row['annotation_id'],as_of=10**12) is None
    queue.acknowledge(rows,'canonical:independent-observations')
    assert queue.report()['inference_metrics']=={'worker_inferences':1,'cache_hits':2}
    assert queue.report()['canonical_windows']==3
    queue.close()


@pytest.mark.asyncio
async def test_corrupt_cache_is_recomputed_and_model_code_changes_invalidate_cache(tmp_path):
    worker=CountingWorker()
    queue=SentimentResearch(tmp_path/'nlp.sqlite',worker=worker)
    enqueue(queue,'shared text',job='first')
    first=await queue.fetch()
    queue.acknowledge(first,'canonical:first')
    with queue.db:
        queue.db.execute('UPDATE inference_cache SET body=?', ('{"broken":true}',))
    enqueue(queue,'shared text',job='second')
    second=await queue.fetch()
    assert worker.calls==2 and not second[0]['inference_cache_hit']
    assert queue.report()['inference_metrics']['cache_corruptions']==1
    queue.acknowledge(second,'canonical:second')
    queue.inference_code_sha256='fixture-new-inference-contract'
    enqueue(queue,'shared text',job='third')
    third=await queue.fetch()
    assert worker.calls==3 and third[0]['inference_cache_key']!=first[0]['inference_cache_key']
    queue.close()


@pytest.mark.asyncio
async def test_full_document_survives_restart_windows_and_canonical_boundary(tmp_path):
    path = tmp_path/'nlp.sqlite'
    text = ' '.join('word' for _ in range(2300))
    queue = SentimentResearch(path, worker=Worker())
    identity = enqueue(queue, text)
    assert enqueue(queue, text) == identity
    windows = []
    for index in range(5):
        rows = await queue.fetch()
        assert len(rows) == 1
        row = rows[0]
        assert row['text_sha256'] == hashlib.sha256(text.encode()).hexdigest()
        assert row['execution_authority'] is False and row['calibrated'] is False
        assert queue.observation(row['annotation_id'], as_of=10**12) is None
        assert await queue.fetch() == rows  # no cursor advance before canonical ACK
        queue.close()
        queue = SentimentResearch(path, worker=Worker())
        assert await queue.fetch() == rows
        queue.acknowledge(rows, 'canonical:output:'+str(index))
        observation = queue.observation(row['annotation_id'], as_of=10**12)
        assert queue.observation(row['annotation_id'], as_of=observation['available_at']-.001) is None
        queue.acknowledge(rows, 'canonical:duplicate')  # keeps first availability
        assert queue.observation(row['annotation_id'], as_of=10**12) == observation
        windows.append((row['token_start'], row['token_stop']))
    assert windows == [(0,510), (510,1020), (1020,1530), (1530,2040), (2040,2300)]
    assert await queue.fetch() == []
    assert queue.report()['completed_inputs'] == 1 and queue.report()['canonical_windows'] == 5
    queue.close()


@pytest.mark.asyncio
async def test_real_miner_retains_then_acks_and_rejects_forged_records(tmp_path):
    from src.research.data_miners import DataMinerPool
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=Worker())
    enqueue(queue)
    delivered = []
    def retain(miner, rows, now):
        assert miner == MINER and rows[0]['_miner'] == MINER
        queue.acknowledge(rows, 'canonical:stamped')
        return 'canonical:stamped'
    pool = DataMinerPool(retain_batch=retain, on_records=lambda miner, rows: delivered.extend(rows))
    queue.register(pool)
    result = await pool._run_one(MINER, 1000)
    assert result.status == 'OK' and len(delivered) == 1
    assert queue.report()['awaiting_canonical_ack'] == 0
    for forged in ({**delivered[0], 'extra': True}, {**delivered[0], '_miner': 'foreign'},
                   {**delivered[0], 'text_sha256': 'invented'}):
        with pytest.raises(ValueError):
            queue.acknowledge([forged], 'canonical:forged')
    queue.close()


@pytest.mark.asyncio
async def test_failure_retains_raw_input_and_old_sources_are_not_starved(tmp_path):
    class Failing(Worker):
        async def infer(self, text, offset):
            if text == 'broken':
                raise RuntimeError('do-not-print-text-or-a-url')
            return await super().infer(text, offset)
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=Failing())
    enqueue(queue, 'broken')
    enqueue(queue, 'old public post', job='job:old')
    for index in range(8):
        enqueue(queue, job='job:new:'+str(index))
    rows = await queue.fetch()
    assert any(r['source_job'] == 'job:old' for r in rows)
    queue.acknowledge(rows, 'canonical:fair')
    assert queue.report()['pending_inputs'] >= 1
    assert queue.db.execute('SELECT count(*) FROM inputs WHERE json_extract(body,?)=?', ('$.text','broken')).fetchone()[0] == 1
    queue.close()


@pytest.mark.asyncio
async def test_new_inputs_cannot_starve_remaining_windows_of_a_long_post(tmp_path, monkeypatch):
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=Worker())
    enqueue(queue, ' '.join(['word']*1200), job='long:original')
    first = await queue.fetch()
    queue.acknowledge(first, 'canonical:first')
    assert first[0]['token_stop'] == 510
    # A backwards wall-clock correction must not give new work priority.
    monkeypatch.setattr('src.research.neural_sentiment.time.time', lambda: 50.)
    for turn in range(3):
        for index in range(4):
            enqueue(queue, job=f'new:{turn}:{index}')
        rows = await queue.fetch()
        queue.acknowledge(rows, f'canonical:turn:{turn}')
    original = queue.db.execute('SELECT complete FROM inputs WHERE json_extract(body,?)=?',
                               ('$.source_job', 'long:original')).fetchone()
    assert original == (1,)
    assert all(r['worker_request_seconds'] >= 0 for r in rows)
    queue.close()


def test_runtime_only_queues_after_canonical_source_and_retains_output_before_ack(tmp_path):
    from src.runtime.source_intelligence import SourceIntelligence
    from src.runtime.research_runtime import ResearchRuntime
    queue = SentimentResearch(tmp_path/'nlp.sqlite', worker=Worker())
    event = SimpleNamespace(source_id='public:fixture', token_addresses=['mint'], observed_at=100,
                            metadata={})
    payload = {'text': 'public post', 'source_id': event.source_id, 'observed_at': 100}
    bot = SimpleNamespace(neural_sentiment=queue, unknown_engine=SimpleNamespace(
        frontier=SimpleNamespace(job_payload_ref=lambda job: 'canonical:raw')),
        public_claim_links=SimpleNamespace(record=lambda **kwargs: None))
    assert SourceIntelligence._record_population_narrative(bot, event,
        prepared_payload=payload, canonical_job_result='job:raw') == 'job:raw'
    assert queue.report()['inputs'] == 1
    called = []
    bot = SimpleNamespace(unknown_engine=SimpleNamespace(_record=lambda *args: called.append('retained') or 'canonical:output'),
        neural_sentiment=SimpleNamespace(acknowledge=lambda rows, ref: called.append('acked')))
    ResearchRuntime._retain_mined_batch(bot, MINER, [], 100)
    assert called == ['retained', 'acked']
    queue.close()


def test_unconfigured_pipeline_keeps_pending_work_and_invalid_scores_are_refused(tmp_path):
    queue = SentimentResearch(tmp_path/'nlp.sqlite')
    enqueue(queue)
    assert queue.report()['status'] == 'NOT_CONFIGURED' and queue.report()['pending_inputs'] == 1
    with pytest.raises(ValueError):
        queue._validate_window({'token_start': 0,'token_stop': 10,'token_count': 10,
            'raw_model_scores': {'positive': float('nan'), 'negative': .1, 'neutral': .9}}, 0)
    queue.close()


def test_hash_verified_replay_repairs_intake_and_resumes_partial_lines(tmp_path):
    from src.research.trade_evidence import TradeEvidenceLedger
    path = tmp_path/'canonical.jsonl'
    ledger = TradeEvidenceLedger(path)
    payload = {'text': 'old public observation', 'source_id': 'public:old', 'observed_at': 100,
               'token_id': 'mint'}
    reference = ledger.record_reference('research_typed_work',
        {'work_kind': 'NARRATIVE_OBSERVATION', 'point_payload': payload}, timestamp=100)
    queue = SentimentResearch(tmp_path/'nlp.sqlite', ledger_path=path)
    queue.reconcile(byte_quantum=1)  # whole first row, even beyond quantum
    assert queue.report()['inputs'] == 1 and queue.report()['source_ledger_backlog_bytes'] == 0
    assert queue.db.execute('SELECT json_extract(body,?) FROM inputs', ('$.source_ref',)).fetchone()[0] == reference
    enqueue_id = queue.enqueue(payload, source_job='actual:job', source_ref=reference, tokens=['mint'])
    assert enqueue_id and queue.report()['inputs'] == 1
    checkpoint = queue.report()['replayed_canonical_bytes']
    with path.open('ab') as handle:
        handle.write(b'{"incomplete":')
    queue.reconcile()
    assert queue.report()['replayed_canonical_bytes'] == checkpoint
    # A complete corrupt row is visible; it is never skipped to later posts.
    with path.open('ab') as handle:
        handle.write(b'1}\n')
    with pytest.raises((ValueError, KeyError)):
        queue.reconcile()
    assert queue.report()['replayed_canonical_bytes'] == checkpoint
    queue.close()


def replay_fixture(tmp_path,count=40):
    from src.research.trade_evidence import TradeEvidenceLedger
    path=tmp_path/'canonical.jsonl'
    ledger=TradeEvidenceLedger(path)
    for index in range(count):
        ledger.record('research_typed_work',{'work_kind':'NARRATIVE_OBSERVATION',
            'point_payload':{'text':f'public post {index}','source_id':'fixture','observed_at':100}},timestamp=100)
    return path,SentimentResearch(tmp_path/'nlp.sqlite',worker=Worker(),ledger_path=path)


def test_replay_inputs_and_checkpoint_use_one_durable_commit(tmp_path):
    path,queue=replay_fixture(tmp_path)
    commits=[]
    queue.db.set_trace_callback(lambda statement: commits.append(1) if statement=='COMMIT' else None)
    queue.reconcile(byte_quantum=10**7)
    assert queue.report()['inputs']==40
    assert queue.report()['replayed_canonical_bytes']==path.stat().st_size
    assert len(commits)==1 and queue.report()['inference_metrics']['replay_commits']==1
    queue.reconcile()
    assert len(commits)==1  # EOF does not fsync another identical checkpoint.
    queue.close()


def test_replay_intake_failure_rolls_back_whole_batch_and_source_cursor(tmp_path,monkeypatch):
    path,queue=replay_fixture(tmp_path)
    original=queue._enqueue_locked
    calls=[]
    def fail_third(*args,**kwargs):
        calls.append(1)
        if len(calls)==3: raise OSError('fixture write failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(queue,'_enqueue_locked',fail_third)
    with pytest.raises(OSError): queue.reconcile(byte_quantum=10**7)
    assert queue.report()['inputs']==0 and queue.report()['replayed_canonical_bytes']==0
    monkeypatch.setattr(queue,'_enqueue_locked',original)
    queue.reconcile(byte_quantum=10**7)
    assert queue.report()['inputs']==40 and queue.report()['source_ledger_backlog_bytes']==0
    queue.close()


def test_concurrent_replay_cursor_is_never_rewound(tmp_path,monkeypatch):
    path,queue=replay_fixture(tmp_path,count=2)
    import src.research.trade_evidence as evidence
    original=evidence._verified_row
    moved=[]
    def concurrent(*args):
        row,previous=original(*args)
        if not moved:
            moved.append(1)
            # Simulate a separately committed newer replay pass while this
            # pass is reading its prefix outside the queue's writer lock.
            with queue.db:
                queue.db.execute('UPDATE replay SET offset=12345 WHERE id=1')
        return row,previous
    monkeypatch.setattr(evidence,'_verified_row',concurrent)
    queue.reconcile(byte_quantum=10**7)
    assert queue.report()['replayed_canonical_bytes']==12345 and queue.report()['inputs']==0
    queue.close()


@pytest.mark.asyncio
async def test_worker_environment_excludes_credentials_and_failed_child_is_reaped(monkeypatch):
    from src.research.neural_sentiment import OfflineWorker
    monkeypatch.setenv('HELIUS_API_KEY', 'fixture-private-do-not-inherit')
    monkeypatch.setenv('SOLANA_PRIVATE_KEY', 'fixture-private-do-not-inherit')
    called = []
    class Pipe:
        def write(self, value):
            pass
        async def drain(self):
            pass
        async def readline(self):
            return b'not-json\n'
    class Process:
        returncode = None
        stdin = stdout = Pipe()
        def kill(self):
            called.append('killed')
            self.returncode = 1
        async def wait(self):
            called.append('reaped')
    async def spawn(*args, **kwargs):
        assert not any('KEY' in key or 'TOKEN' in key and key != 'TOKENIZERS_PARALLELISM' for key in kwargs['env'])
        return Process()
    monkeypatch.setattr('asyncio.create_subprocess_exec', spawn)
    worker = OfflineWorker('python', 'public-model')
    with pytest.raises(ValueError):
        await worker.infer('public fixture', 0)
    assert called == ['killed', 'reaped'] and worker.process is None
