"""Integration invariants: canonical-first indexing, fairness and partial faults."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from src.research.data_miners import DataMinerPool, RateLimited
from src.research.phantom_perps import PhantomPerpsResearch
from src.runtime.research_runtime import ResearchRuntime
from src.runtime.ingestion import MinedRecordIngestion
from tests.test_phantom_perps import catalog, book


def setup_markets(state, count):
    row = catalog()
    asset = row['meta']['universe'][0]
    context = row['contexts'][0]
    row['meta']['universe'] = [{**asset, 'name':f'COIN{i:04d}'} for i in range(count)]
    row['contexts'] = [dict(context) for _ in range(count)]
    state.ingest(row)


def observation(role, **fields):
    return {'provider':'hyperliquid', 'product':'perpetual', 'perps_role':role,
            'observed_at':time.time(), **fields}


@pytest.mark.asyncio
async def test_runtime_indexes_before_handoff_and_never_writes_on_hot_dispatch(tmp_path, monkeypatch):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    sequence = []
    def retain(kind, payload, now):
        assert state.report()['active_markets'] == 0
        sequence.append('canonical')
        return 'canonical:perps:catalog'
    desk = SimpleNamespace(unknown_engine=SimpleNamespace(_record=retain), phantom_perps=state)
    def deliver(name, rows):
        sequence.append('handoff')
        assert state.report()['active_markets'] == 1
        # No perps object required on the receiving spot/World seam.
        MinedRecordIngestion._ingest_mined_records(SimpleNamespace(), name, rows)
    pool = DataMinerPool(retain_batch=lambda *args:ResearchRuntime._retain_mined_batch(desk,*args), on_records=deliver)
    state.register(pool, None)
    monkeypatch.setattr(state.acquisition, 'request', lambda *_:None)
    async def fetch(): return [catalog()]
    pool._callables['perps:catalog'] = fetch
    result = await pool._run_one('perps:catalog', time.time())
    assert result.ok and sequence == ['canonical','handoff']
    assert state.report()['canonical_batches_indexed'] == 1
    # Canonical retries are exactly-once in this derived index.
    state.ingest_batch(result.records, 'canonical:perps:catalog')
    assert state.report()['canonical_batches_indexed'] == 1
    state.close()


@pytest.mark.asyncio
async def test_failed_canonical_write_cannot_index_or_deliver_perps(tmp_path):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    def fail(*_): raise OSError('archive unavailable')
    desk = SimpleNamespace(unknown_engine=SimpleNamespace(_record=fail), phantom_perps=state)
    pool = DataMinerPool(retain_batch=lambda *args:ResearchRuntime._retain_mined_batch(desk,*args),
                         on_records=lambda *_:pytest.fail('must not deliver'))
    state.register(pool,None)
    async def fetch(): return [catalog()]
    pool._callables['perps:catalog'] = fetch
    assert not (await pool._run_one('perps:catalog', time.time())).ok
    assert state.report()['active_markets'] == 0
    state.close()


@pytest.mark.asyncio
async def test_partial_http_failure_keeps_other_books_and_continues_universe_after_restart(tmp_path):
    path = tmp_path/'perps.sqlite'
    state = PhantomPerpsResearch(path)
    setup_markets(state, 36)  # three per pass for a minute-long objective
    seen = []
    active, peak = 0, 0
    async def request(http, payload):
        nonlocal active, peak
        coin = payload['coin']
        seen.append(coin)
        active += 1
        peak = max(peak,active)
        try:
            await asyncio.sleep(0)
            if coin == 'COIN0001': raise TimeoutError('single public request')
            return {**book(time.time()), 'coin':coin}
        finally: active -= 1
    state.acquisition.request = request
    rows = await state.acquisition.books(None, observation)
    assert len(rows) == 3 and sum(r['perps_role']=='book' for r in rows) == 2
    def remaining_targets(stage):
        targets=state.acquisition.targets('book',3)
        assert targets==['COIN0003','COIN0004','COIN0005'], stage
    remaining_targets('after_fetch')
    state.ingest_batch(rows, 'canonical:first')
    remaining_targets('after_ingest')
    assert state.report()['books_recorded'] == 2
    assert state.report()['acquisition']['recent_request_errors'][0]['error'] == 'TimeoutError'
    assert peak <= 4
    state.close()
    state = PhantomPerpsResearch(path)
    state.acquisition.request = request
    remaining_targets('after_reopen')
    await state.acquisition.books(None, observation)
    assert seen == [f'COIN{i:04d}' for i in range(6)]
    state.close()


def test_unserved_market_ties_across_catalogs_are_not_skipped_or_duplicated(tmp_path):
    path=tmp_path/'selection.sqlite'
    state=PhantomPerpsResearch(path)
    setup_markets(state,12)
    state.ingest(catalog(dex='second',coin='COIN0003'))
    with state.db:
        state.db.executemany('INSERT INTO acquisition_service VALUES(?,?,?,?)',
            [('book',f'COIN{i:04d}',1000.,'') for i in range(3)])
    assert state.acquisition.targets('book',3)==['COIN0003','COIN0004','COIN0005']
    assert state.acquisition.targets('funding',3)==['COIN0000','COIN0001','COIN0002']
    state.close();state=PhantomPerpsResearch(path)
    assert state.acquisition.targets('book',3)==['COIN0003','COIN0004','COIN0005']
    assert len(state.acquisition.targets('book',100))==12
    state.close()


@pytest.mark.asyncio
async def test_one_429_cools_all_public_methods_and_survives_restart(tmp_path):
    path = tmp_path/'perps.sqlite'
    state = PhantomPerpsResearch(path)
    setup_markets(state, 2)
    calls = []
    async def request(http,payload):
        calls.append(payload)
        raise RateLimited('provider refused')
    state.acquisition.request = request
    with pytest.raises(RateLimited):
        await state.acquisition.fetch(None, {'type':'l2Book','coin':'COIN0000'}, role='book',target='COIN0000')
    state.close()
    state = PhantomPerpsResearch(path)
    state.acquisition.request = request
    for payload in ({'type':'spotMeta'}, {'type':'fundingHistory','coin':'COIN0001','startTime':0}):
        assert await state.acquisition.fetch(None,payload,role='other',target='untouched') is None
    assert len(calls) == 1 and state.report()['acquisition']['provider_cooldown_until'] > time.time()
    assert state.acquisition.targets('book',1) == ['COIN0001']
    state.close()


@pytest.mark.asyncio
async def test_longer_retry_after_is_shared_and_books_leave_metadata_service_headroom(tmp_path):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    setup_markets(state,2)
    calls = []
    async def request(http,payload):
        calls.append(payload)
        if payload['type'] == 'perpDexs':
            error = RateLimited('provider cooldown')
            error.retry_after_seconds = 180
            raise error
        return {}
    state.acquisition.request = request
    with state.db:
        state.db.execute('INSERT INTO acquisition_weight(at,weight) VALUES(?,?)',(time.time(),800))
    assert await state.acquisition.fetch(None,{'type':'l2Book','coin':'COIN0000'},role='book',target='COIN0000') is None
    with pytest.raises(RateLimited):
        await state.acquisition.fetch(None,{'type':'perpDexs'},role='catalog',target='dex_inventory')
    assert len(calls) == 1
    assert state.report()['acquisition']['provider_cooldown_until'] > time.time()+175
    state.close()


@pytest.mark.asyncio
async def test_weight_exhaustion_does_not_send_http_or_advance_unserved_target(tmp_path):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    setup_markets(state,2)
    with state.db:
        state.db.execute('INSERT INTO acquisition_weight(at,weight) VALUES(?,?)',(time.time(),899))
    async def forbidden(*_): pytest.fail('budget exhausted')
    state.acquisition.request = forbidden
    assert await state.acquisition.books(None,observation) == []
    assert state.acquisition.targets('book',2) == ['COIN0000','COIN0001']
    state.close()


def test_old_catalog_collateral_and_books_never_rewind_current_view(tmp_path):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    now = time.time()
    state.ingest(catalog(at=now))
    state.ingest(catalog(coin='old-contract',at=now-1))
    assert state.market_names() == ['BTC']
    state.ingest(observation('collateral_inventory',tokens=[{'index':0,'name':'USDC','isCanonical':True}]))
    state.ingest({**observation('collateral_inventory',tokens=[{'index':0,'name':'bad'}]),'observed_at':now-100})
    assert json.loads(state.db.execute('SELECT metadata FROM collateral_tokens').fetchone()[0])['name'] == 'USDC'
    state.ingest(observation('book',coin='BTC',book=book(now)))
    state.ingest(observation('book',coin='BTC',book=book(now-1)))
    assert json.loads(state.db.execute('SELECT body FROM books').fetchone()[0])['time'] == int(now*1000)
    with pytest.raises(ValueError):
        state.ingest(observation('book',coin='BTC',book=book(now+86400)))
    assert json.loads(state.db.execute('SELECT body FROM books').fetchone()[0])['time'] == int(now*1000)
    assert state.report()['fresh_active_books'] == 1
    state.close()


def test_malformed_book_is_audited_without_discarding_valid_rows(tmp_path):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    setup_markets(state,2)
    valid = observation('book',coin='COIN0000',book={**book(time.time()),'coin':'COIN0000'})
    bad = observation('book',coin='COIN0001',book={'coin':'COIN0001','time':'not-a-clock'})
    state.ingest_batch([bad,valid], 'canonical:partial')
    report = state.report()
    assert report['derived_record_errors'] == 1 and report['books_recorded'] == 1
    assert state.db.execute('SELECT ref,record_index FROM ingestion_errors').fetchone() == ('canonical:partial',0)
    state.close()


def test_funding_cursor_cannot_advance_over_conflict_or_future_data(tmp_path):
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    row = observation('funding',coin='BTC',start_time=0,rows=[{'coin':'BTC','time':1000,'fundingRate':'0.01'}])
    state.ingest(row)
    with pytest.raises(ValueError):
        state.ingest({**row,'rows':[{'coin':'BTC','time':1000,'fundingRate':'0.02'}, {'coin':'BTC','time':2000}]})
    assert state.next_funding('BTC') == 1001
    with pytest.raises(ValueError):
        state.ingest({**row,'rows':[{'coin':'BTC','time':int((time.time()+3600)*1000)}]})
    assert state.next_funding('BTC') == 1001
    state.close()
