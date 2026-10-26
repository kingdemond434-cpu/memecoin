import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient,TestServer

from src.research.program_tape import ProgramEventTape
from src.runtime.program_capture import ProgramCapture,create_app


@pytest.mark.asyncio
async def test_interleaved_subscription_notifications_retain_full_logs_and_raw_bytes(tmp_path):
    program='11111111111111111111111111111111'
    raw=b'unknown event payload'
    async def socket(request):
        ws=web.WebSocketResponse();await ws.prepare(request)
        first=await ws.receive_json();second=await ws.receive_json()
        await ws.send_json({'id':first['id'],'result':17})
        # An event may arrive before the second program is refused.
        await ws.send_json({'method':'logsNotification','params':{'subscription':17,'result':{
            'context':{'slot':42},'value':{'signature':'public-signature',
                'err':None,'logs':[
                    f'Program {program} invoke [1]',
                    'Program data: '+base64.b64encode(raw).decode(),
                    f'Program {program} success']}}}})
        await ws.send_json({'method':'logsNotification','params':{'subscription':17,'result':{
            'context':{'slot':43},'value':{'signature':'failed-public-signature',
                'err':{'InstructionError':[0,'Custom']},'logs':[
                    f'Program {program} invoke [1]',
                    'Program data: '+base64.b64encode(b'failed payload').decode(),
                    f'Program {program} failed: custom program error']}}}})
        await ws.send_json({'id':second['id'],'error':{'code':-32602,'message':'not authorized'}})
        async for _ in ws:pass
        return ws
    app=web.Application();app.router.add_get('/ws',socket)
    async with TestServer(app) as server:
        class RPC:
            def get_ws_urls(self):return [str(server.make_url('/ws')).replace('http://','ws://')]
            async def stop(self):pass
        tape=ProgramEventTape(tmp_path/'tape.sqlite',batch_delay=0)
        capture=ProgramCapture(RPC(),tape,[program,'refused-program'])
        await capture.start()
        try:
            async with asyncio.timeout(5):
                while capture.status!='PARTIAL_RPC_WS':await asyncio.sleep(.001)
            await tape.flush()
            rows=[json.loads(row[0]) for row in tape.db.execute('SELECT body FROM records ORDER BY sequence')]
            transaction=next(row for row in rows if row['kind']=='transaction')['payload']
            event=next(row for row in rows if row['kind']=='program_data')['payload']
            assert transaction['notification']['result']['value']['logs'][1].startswith('Program data:')
            assert base64.b64decode(event['data']['data'])==raw
            assert event['slot']==42 and event['source_commitment']=='processed'
            assert event['transaction_error'] is None
            transactions=[row['payload'] for row in rows if row['kind']=='transaction']
            assert transactions[1]['notification']['result']['value']['err']=={'InstructionError':[0,'Custom']}
            assert len([row for row in rows if row['kind']=='program_data'])==1
            assert capture.report()['subscribed_programs']==[program]
            assert capture.report()['refused_program_codes']=={'refused-program':-32602}
            assert capture.report()['real_transactions_submitted']==0
        finally:await capture.stop()
        assert ProgramEventTape.verify(tape.path)['records']>=4


@pytest.mark.asyncio
async def test_raw_bytes_receipt_clocks_and_hashes_survive_restart(tmp_path):
    path=tmp_path/'tape.sqlite';tape=ProgramEventTape(path,batch_delay=0)
    raw={'program':'public-program','signature':'observed-signature','slot':30,
         'data':b'\x00\xffwire','received_ns':123,'source_commitment':'processed'}
    await tape.enqueue('program_data',raw);raw['data']=b'changed after admission'
    await tape.flush()
    assert tape.report()['committed_records']==1
    body=json.loads(tape.db.execute('SELECT body FROM records').fetchone()[0])
    assert base64.b64decode(body['payload']['data']['data'])==b'\x00\xffwire'
    assert body['payload']['received_ns']==123 and body['capture_received_at_ns']>123
    assert body['payload']['source_commitment']=='processed'
    assert ProgramEventTape.verify(path)['records']==1
    await tape.stop();restored=ProgramEventTape(path,batch_delay=0)
    assert restored.report()['previous_shutdown_clean'] is True
    await restored.enqueue('source_transition',{'status':'restarted'})
    await restored.stop()
    assert ProgramEventTape.verify(path)['records']==2


@pytest.mark.asyncio
async def test_committed_prefix_does_not_include_queued_or_failed_batches(tmp_path,monkeypatch):
    tape=ProgramEventTape(tmp_path/'tape.sqlite',batch_delay=0)
    entered,released=threading.Event(),threading.Event();original=tape._append
    def delayed(batch):
        entered.set();assert released.wait(5)
        original(batch)
    monkeypatch.setattr(tape,'_append',delayed)
    await tape.enqueue('program_data',{'data':b'first'})
    async with asyncio.timeout(3):
        while not entered.is_set():await asyncio.sleep(.001)
    await tape.enqueue('program_data',{'data':b'second'})
    assert tape.report()['committed_records']==0
    assert ProgramEventTape.verify(tape.path)['records']==0
    released.set();await tape.stop()
    assert ProgramEventTape.verify(tape.path)['records']==2


@pytest.mark.asyncio
async def test_transient_write_failure_retries_same_batch_without_false_ack(tmp_path,monkeypatch):
    tape=ProgramEventTape(tmp_path/'tape.sqlite',batch_delay=0)
    original=tape._append;attempts=[]
    def transient(batch):
        attempts.append(batch[0]['observation_id'])
        if len(attempts)==1:raise OSError('synthetic unavailable disk')
        original(batch)
    monkeypatch.setattr(tape,'_append',transient)
    await tape.enqueue('program_data',{'data':b'wire'})
    await asyncio.wait_for(tape.flush(),4)
    assert attempts[0]==attempts[1] and tape.report()['committed_records']==1
    assert tape.report()['write_errors_this_run']==1
    await tape.stop()


@pytest.mark.asyncio
async def test_tampered_tape_is_detected_and_unclean_shutdown_is_explicit(tmp_path):
    path=tmp_path/'tape.sqlite';tape=ProgramEventTape(path,batch_delay=0)
    await tape.enqueue('program_data',{'data':b'wire'});await tape.flush()
    # Simulate loss of the owning process after its last committed batch.
    tape.worker.cancel();await asyncio.gather(tape.worker,return_exceptions=True);tape.db.close()
    restored=ProgramEventTape(path)
    assert restored.report()['previous_shutdown_clean'] is False
    assert restored.report()['uncommitted_loss_after_unclean_shutdown']=='UNKNOWN'
    with restored.db:restored.db.execute("UPDATE records SET body='{}' WHERE sequence=1")
    with pytest.raises((ValueError,KeyError)):ProgramEventTape.verify(path)
    await restored.stop()


@pytest.mark.asyncio
async def test_unavailable_feed_is_distinct_from_a_running_readonly_recorder(tmp_path):
    class RPC:
        def get_ws_urls(self):return []
        async def stop(self):pass
    tape=ProgramEventTape(tmp_path/'tape.sqlite',batch_delay=0)
    capture=ProgramCapture(RPC(),tape,['public-program'])
    async with TestClient(TestServer(create_app(capture))) as client:
        async with asyncio.timeout(3):
            while capture.status!='DATA_BLOCKED_NO_ELIGIBLE_WS':await asyncio.sleep(.001)
        health=await (await client.get('/health')).json()
        assert health['source_status']=='DATA_BLOCKED_NO_ELIGIBLE_WS'
        assert health['committed_program_records']==0 and health['live_submission_locked'] is True
        report=await (await client.get('/status')).json()
        assert report['real_transactions_submitted']==0 and report['all_programs_coverage'] is False
        assert (await client.get('/status',headers={'Host':'remote.invalid'})).status==403
    assert ProgramEventTape.verify(tape.path)['records']>=1


def test_refusal_honors_retry_after_without_disclosing_credential_url(tmp_path,monkeypatch):
    tape=ProgramEventTape(tmp_path/'tape.sqlite')
    capture=ProgramCapture(SimpleNamespace(),tape,['public-program'])
    monkeypatch.setattr('src.runtime.program_capture.time.time',lambda:100.)
    private='wss://example.invalid/?key=NEVER_EXPORT'
    assert capture._defer(private,429,{'Retry-After':'7200'})==7200
    assert capture._until(private)==7300
    assert 'NEVER_EXPORT' not in json.dumps(capture.report())
    asyncio.run(tape.stop())
