import asyncio
import json
import sqlite3
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.research.fork_probe import ForkProbe


@pytest.mark.parametrize('already_acknowledged', [False, True])
def test_owner_http_observation_does_not_wait_for_archive_writer(tmp_path, already_acknowledged):
    async def scenario():
        probe = ForkProbe(tmp_path/'fork.sqlite')
        body = json.dumps({'token': 'token-one', 'native_source_refs': [
            {'available_at': 5, 'ingested_at': 6}]})
        with probe.lock, probe.db:
            probe.db.execute('INSERT INTO results VALUES(?,?,?,?,?,?)',
                ('result-one', 'job-one', body, 7, 'canonical:old' if already_acknowledged else None,
                 8 if already_acknowledged else None))
        entered, released = threading.Event(), threading.Event()
        def writer():
            with probe.lock, probe.db:
                probe.db.execute('UPDATE results SET ref=?,available=? WHERE id=?',
                    ('canonical:new', 9, 'result-one'))
                entered.set()
                assert released.wait(10)
        async def handler(request):
            result = probe.latest('token-one', as_of=10)
            return web.json_response({'ref': result['source_ref'] if result else None})
        app = web.Application(); app.router.add_get('/observation', handler)
        client = TestClient(TestServer(app)); await client.start_server()
        pending = asyncio.create_task(asyncio.to_thread(writer))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            response = await asyncio.wait_for(client.get('/observation'), 1)
            assert await response.json() == {'ref': 'canonical:old' if already_acknowledged else None}
            assert not released.is_set() and not pending.done()
            released.set(); await pending
            assert probe.latest('token-one', as_of=10)['source_ref'] == 'canonical:new'
            assert probe.latest('token-one', as_of=8) is None
            assert probe.latest('unknown', as_of=10) is None
            assert probe.db.execute('PRAGMA synchronous').fetchone()[0] == 2
            with pytest.raises(sqlite3.OperationalError):
                probe.observation_db.execute('UPDATE results SET available=0')
        finally:
            released.set(); await pending
            await client.close(); probe.close()
        with pytest.raises(sqlite3.ProgrammingError):
            probe.observation_db.execute('SELECT 1')
    asyncio.run(scenario())


def test_committed_native_source_point_in_time_guard_and_reopen(tmp_path):
    path = tmp_path/'fork.sqlite'; probe = ForkProbe(path)
    body = json.dumps({'token': 'token-one', 'native_source_refs': [
        {'available_at': 20, 'ingested_at': 30}]})
    with probe.lock, probe.db:
        probe.db.execute('INSERT INTO results VALUES(?,?,?,?,?,?)',
            ('result-one', 'job-one', body, 5, 'canonical:one', 10))
    assert probe.latest('token-one', as_of=25) is None
    assert probe.latest('token-one', as_of=30)['source_ref'] == 'canonical:one'
    probe.close(); probe = ForkProbe(path)
    try:
        assert probe.latest('token-one', as_of=29) is None
        assert probe.latest('token-one', as_of=30)['available_at'] == 10
    finally:
        probe.close()
