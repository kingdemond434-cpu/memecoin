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
async def test_newest_post_bypasses_live_backlog_without_starving_old_live_or_replay(tmp_path,monkeypatch):
    path=tmp_path/'nlp.sqlite'
    queue=SentimentResearch(path,worker=Worker())
    for index in range(100):
        enqueue(queue,f'old live text {index}',job=f'old:{index}')
    queue.enqueue({'text':'historical text','source_id':'historical','observed_at':1},
        source_job=None,source_ref='canonical:history')
    # Use the persistent arrival ticket, not the corrected host wall clock.
    monkeypatch.setattr('src.research.neural_sentiment.time.time',lambda:50.)
    enqueue(queue,'newest live text',job='fresh:arrival')
    queue.close()
    queue=SentimentResearch(path,worker=Worker())
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
async def test_live_intake_progresses_ahead_of_large_history_without_starving_history(tmp_path):
    path=tmp_path/'nlp.sqlite'
    queue=SentimentResearch(path,worker=Worker())
    for index in range(100):
        queue.enqueue({'text':f'old document {index}', 'source_id':'historical', 'observed_at':10},
            source_job=None,source_ref=f'canonical:replay:{index}')
    enqueue(queue,'current live observation',job='live:first')
    queue.close()
    queue=SentimentResearch(path,worker=Worker())
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
        frontier=SimpleNamespace(job=lambda job: {'payload_ref': 'canonical:raw'})),
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
