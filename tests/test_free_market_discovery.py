import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from src.research import free_market_discovery as free
from src.research.data_miners import DataMinerPool, RateLimited
from src.research.solana_miners import holder_structure_miner


def client(payload, status=200, headers=None):
    async def get(url, **options):
        assert options['allow_redirects'] is False
        return status, json.dumps(payload), headers or {}
    return SimpleNamespace(get=get)


def networks(*names):
    return {'data': [{'id': name, 'type': 'network', 'attributes': {'name': name}} for name in names]}


def pool(network='new_chain', address='pool'):
    return {'id': network + '_' + address, 'attributes': {'address': address,
        'base_token_price_usd': 'NaN', 'reserve_in_usd': '17', 'name': 'Not an identity'},
        'relationships': {'base_token': {'data': {'id': network + '_base_with_suffix'}},
            'quote_token': {'data': {'id': network + '_quote'}}, 'dex': {'data': {'id': 'some-dex'}}}}


@pytest.mark.asyncio
async def test_cursor_waits_for_canonical_and_pending_bytes_survive_restart(tmp_path):
    path = tmp_path/'free.sqlite'
    book = free.FreeMarketDiscovery(path)
    rows = await book.fetch(client(networks('new_chain')))
    digest = rows[0]['digest']
    assert book.report()['networks'] == 0
    book.close()
    book = free.FreeMarketDiscovery(path)
    async def no_network(*args, **kwargs):
        raise AssertionError('pending response must be replayed without refetch')
    restored = await book.fetch(SimpleNamespace(get=no_network))
    assert restored[0]['digest'] == digest
    with pytest.raises(ValueError, match='reference'):
        book.acknowledge(restored, '')
    book.acknowledge(restored, 'canonical:one')
    assert book.report()['networks'] == 1 and book.report()['pending'] == 0
    assert book.db.execute("SELECT count(*) FROM jobs WHERE network='new_chain'").fetchone()[0] == 3
    book.acknowledge(restored, 'canonical:one')
    assert book.report()['pages'] == 1
    book.close()


@pytest.mark.asyncio
async def test_catalog_tail_and_new_networks_not_fixed_or_capped(tmp_path, monkeypatch):
    now = [1_000_000]
    monkeypatch.setattr(free.time, 'time', lambda: now[0])
    book = free.FreeMarketDiscovery(tmp_path/'free.sqlite')
    book.db.execute("DELETE FROM jobs WHERE role<>'networks'")
    book.db.commit()
    for page in range(1, 4):
        body = networks(*['network_' + str(i) for i in range((page-1)*100, page*100)])
        body['links'] = {'next': free.page_url('networks', '', page+1) if page < 3 else None}
        rows = await book.fetch(client(body))
        assert rows[0]['page'] == page
        observations = book.discovery_records(rows[0], 'canonical:' + str(page))
        assert len(observations) == 100
        assert all(x['chain_family'] == 'UNKNOWN' and not x['execution_eligible'] for x in observations)
        book.acknowledge(rows, 'canonical:' + str(page))
        now[0] += 20
    assert book.report()['networks'] == 300
    assert book.db.execute("SELECT count(*) FROM jobs WHERE role<>'networks'").fetchone()[0] == 900
    book.close()


@pytest.mark.parametrize('target', ['https://evil.invalid/api/v2/networks?page=2',
    'https://api.geckoterminal.com/api/v2/networks?page=1',
    'https://api.geckoterminal.com/api/v2/networks?page=3',
    'https://api.geckoterminal.com/api/v2/networks?page=2&key=x',
    'https://api.geckoterminal.com/api/v2/networks/solana/dexes?page=2'])
def test_pagination_never_becomes_an_arbitrary_fetch_or_repeated_page(target):
    with pytest.raises(ValueError):
        free.next_page({'data': [], 'links': {'next': target}}, free.page_url('networks', '', 1), 1)


@pytest.mark.asyncio
async def test_global_directory_backlog_cannot_starve_fresh_and_trending_pool_roles(tmp_path, monkeypatch):
    now = [1_000_000.]
    monkeypatch.setattr(free.time, 'time', lambda: now[0])
    book = free.FreeMarketDiscovery(tmp_path/'fair.sqlite')
    with book.db:
        book.db.execute('DELETE FROM jobs')
        for network in ('asia', 'africa', 'europe', 'americas'):
            for role in ('dexes', 'new_pools', 'trending_pools'):
                book.db.execute('INSERT INTO jobs(role,network,page,due) VALUES(?,?,1,0)', (role,network))
    selected = []
    for index in range(6):
        rows = await book.fetch(client({'data': []}))
        selected.append(rows[0]['role'])
        book.acknowledge(rows, 'canonical:fair-'+str(index))
        now[0] += 20
    assert set(selected[:3]) == {'dexes', 'new_pools', 'trending_pools'}
    assert set(selected[3:]) == {'dexes', 'new_pools', 'trending_pools'}
    book.close()


def test_cross_chain_tokens_keep_exact_prefix_semantics_and_unknown_values():
    page = {'role': 'new_pools', 'http_status': 200, 'network': 'new_chain',
        'observed_at': 50, 'source_url': 'fixture', 'digest': 'd', 'payload': {'data': [pool()]}}
    row = free.pool_rows(page)[0]
    assert row['token'] == 'base_with_suffix'
    assert row['chain_id'] == 'geckoterminal:new_chain'
    assert row['quote_token'] == 'quote' and row['price_usd'] is None
    assert row['liquidity_usd'] == 17 and not row['execution_eligible']
    page['payload']['data'][0]['relationships']['base_token']['data']['id'] = 'other_base'
    assert free.pool_rows(page) == []


@pytest.mark.asyncio
async def test_provider_page_refusal_is_a_retained_gap_not_completed_history(tmp_path):
    book = free.FreeMarketDiscovery(tmp_path/'free.sqlite')
    with book.db:
        book.db.execute('DELETE FROM jobs')
        book.db.execute("INSERT INTO jobs(role,network,page,due) VALUES('new_pools','new_chain',11,0)")
    rows = await book.fetch(client({'errors': ['public pagination boundary']}, status=400))
    assert rows[0]['http_status'] == 400
    book.acknowledge(rows, 'canonical:refusal')
    assert book.report()['provider_refused_pages'] == 1
    job = book.db.execute('SELECT page,last_error FROM jobs').fetchone()
    assert job[0] == 1 and 'HTTP_400' in job[1]
    book.close()


@pytest.mark.asyncio
async def test_free_provider_cooldown_persists_across_restart(tmp_path):
    path = tmp_path/'free.sqlite'
    book = free.FreeMarketDiscovery(path)
    with pytest.raises(RateLimited):
        await book.fetch(client({}, 429, {'Retry-After': '600'}))
    book.close()
    book = free.FreeMarketDiscovery(path)
    assert await book.fetch(client(networks('must_not_fetch'))) == []
    assert book.report()['pages'] == 0
    book.close()


@pytest.mark.asyncio
async def test_failed_canonical_retention_never_advances_free_cursor(tmp_path):
    book = free.FreeMarketDiscovery(tmp_path/'free.sqlite')
    def fail(*args):
        raise OSError('disk full')
    pool_ = DataMinerPool(retain_batch=fail)
    book.register(pool_, client(networks('new_chain')))
    result = (await pool_.run_due(free.time.time() + 100))[0]
    assert result.status == 'DATA_BLOCKED'
    assert book.report()['pending'] == 1 and book.report()['networks'] == 0
    book.close()


def test_discovered_foreign_pool_cannot_overwrite_solana_or_regime_state():
    from src.main import MemecoinQuantDesk
    desk = SimpleNamespace(_market_context={'sol_usd': 100}, phantom_perps=None)
    MemecoinQuantDesk._ingest_mined_records(desk, free.MINER,
        [{'provider': 'geckoterminal', 'discovery_kind': 'pool_observation',
          'network_id': 'eth', 'token': '0xabc'},
         {'provider': 'geckoterminal', 'discovery_kind': 'source_page'}])
    assert desk._market_context == {'sol_usd': 100}


@pytest.mark.asyncio
async def test_holder_transport_failure_is_not_successful_empty_data():
    async def fail(*args):
        raise RuntimeError('private endpoint URL must not reach failure summary')
    with pytest.raises(RuntimeError, match='DATA_BLOCKED') as error:
        await holder_structure_miner(SimpleNamespace(request=fail), lambda: ['mint'])()
    assert 'private endpoint URL' not in str(error.value)


def test_observed_buyers_and_launchlab_creators_seed_unlabelled_wallet_targets():
    from src.main import MemecoinQuantDesk
    desk = SimpleNamespace(wallet_intel=SimpleNamespace(elite_wallets={}), _recent_funders={},
        _actor_entries={'mint': [SimpleNamespace(wallet='buyer'), SimpleNamespace(wallet='buyer')]},
        _mineable_tokens=lambda: ['mint'], _latest_curve_state={}, _latest_pool_state={},
        dataset_builder=SimpleNamespace(active_episodes={'mint': SimpleNamespace(deployer='creator')}))
    assert MemecoinQuantDesk._tracked_wallets(desk) == ['buyer']
    assert MemecoinQuantDesk._known_deployers(desk) == ['creator']


@pytest.mark.asyncio
async def test_bad_schema_is_retained_and_does_not_starve_other_network_jobs(tmp_path, monkeypatch):
    now = [1_000_000]
    monkeypatch.setattr(free.time, 'time', lambda: now[0])
    book = free.FreeMarketDiscovery(tmp_path/'free.sqlite')
    with book.db:
        book.db.execute('DELETE FROM jobs')
        book.db.executemany("INSERT INTO jobs(role,network,page,due) VALUES('new_pools',?,1,0)", [('a',), ('b',)])
    rows = await book.fetch(client({'wrong': []}))
    assert rows[0]['network'] == 'a' and rows[0]['parse_status'] != 'OK'
    book.acknowledge(rows, 'canonical:bad-schema')
    now[0] += 20
    rows = await book.fetch(client({'data': []}))
    assert rows[0]['network'] == 'b'
    assert book.report()['unparsed_pages'] == 1
    book.close()


@pytest.mark.asyncio
async def test_runtime_archives_before_frontier_discovery_and_cursor_ack(tmp_path):
    from src.runtime.research_runtime import ResearchRuntime
    book = free.FreeMarketDiscovery(tmp_path/'free.sqlite')
    rows = await book.fetch(client(networks('new_chain')))
    sequence = []
    def retain(kind, payload, now):
        sequence.append('canonical')
        assert payload['records'][0]['raw_body']
        assert book.report()['pending'] == 1
        return 'canonical:original-page'
    def discover(record):
        sequence.append('frontier')
        assert record['data_source'] == 'geckoterminal:public-api'
        assert record['source_response_ref'] == 'canonical:original-page'
        assert book.report()['networks'] == 0
    desk = SimpleNamespace(unknown_engine=SimpleNamespace(_record=retain),
        free_discovery=book, _retain_chain_scout_record=discover)
    ref = ResearchRuntime._retain_mined_batch(desk, free.MINER, rows, free.time.time())
    assert sequence == ['canonical', 'frontier'] and ref == 'canonical:original-page'
    assert book.report()['networks'] == 1 and book.report()['pending'] == 0
    book.close()


@pytest.mark.asyncio
async def test_gdelt_manifest_race_refreshes_only_actual_published_url(monkeypatch):
    from src.research import world_miners as world
    urls = ['https://data.gdeltproject.org/gdeltv2/20261025000000.gkg.csv.zip',
            'https://data.gdeltproject.org/gdeltv2/20261025001500.gkg.csv.zip']
    calls = []
    manifest_count = 0
    async def get(_client, url):
        nonlocal manifest_count
        calls.append(url)
        if url == world.GDELT_LASTUPDATE_URL:
            url_ = urls[min(manifest_count, 1)]
            manifest_count += 1
            return ('1 hash ' + url_).encode()
        if url == urls[0]:
            raise world.ArchiveUnavailable('not published yet')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            line = '\t'.join(['row', 'date', '', 'source', 'https://news.invalid/solana', '', '', 'crypto', '', '', '', 'person', '', 'org'])
            z.writestr('gkg.csv', '\n'.join([line]*250))
        return buf.getvalue()
    monkeypatch.setattr(world, '_get_bytes', get)
    rows = await world.gdelt_world_news_miner(None)()
    assert len(rows) == 250 and all(row['archive_url'] == urls[1] for row in rows)
    assert calls == [world.GDELT_LASTUPDATE_URL, urls[0], world.GDELT_LASTUPDATE_URL, urls[1]]
    assert world.parse_lastupdate('1 hash https://evil.invalid/gdeltv2/x.gkg.csv.zip') == {}
