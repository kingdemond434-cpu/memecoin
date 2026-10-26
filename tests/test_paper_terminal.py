"""Private read-only monitor: truthful staleness, bounded I/O and nonblocking views."""
import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from src.terminal import server


@pytest.mark.parametrize('url', ['https://127.0.0.1:18081', 'http://localhost:18081',
    'http://example.com:80', 'http://127.0.0.1', 'http://user:pass@127.0.0.1:12',
    'http://127.0.0.1:12/path', 'http://127.0.0.1:12?key=secret', 'http://127.0.0.1:12#x'])
def test_only_explicit_literal_loopback_sources(url):
    with pytest.raises(ValueError):server.loopback_url(url)


def test_snapshot_has_no_execution_authority_and_events_require_observed_change():
    hub=server.Hub({'spot':'http://127.0.0.1:12'})
    before={'available':False, 'data':{}}
    data={'portfolio':{'portfolio_value':100}, 'unknown_opportunity_engine':{'canonical_evidence':{'records':20}}}
    after={'available':True,'data':data}
    hub.observe('spot',before,after,10)
    assert len(hub.events)==1 and 'diagnostics received' in hub.events[0]['text']
    hub.observe('spot',after,after,11)
    assert len(hub.events)==1
    later={'available':True,'data':{**data,'unknown_opportunity_engine':{'canonical_evidence':{'records':23}}}}
    hub.observe('spot',after,later,12)
    assert hub.events[-1]['text'].startswith('+3 evidence')
    assert hub.events[-1]['kind']=='DERIVED_SERVICE_TELEMETRY'
    hub.observe('spot',later,after,13)
    assert len(hub.events)==2  # A reset is not represented as a loss or negative collection.
    assert hub.snapshot()['execution_authority'] is False
    assert hub.history[-1]['model_equity_usd']==100
    assert hub.history[-1]['perps_funding_records'] is None


@pytest.mark.asyncio
async def test_live_loopback_read_and_failure_preserve_last_good_data_without_error_strings():
    mode={'fail':False}
    async def endpoint(request):
        if mode['fail']:return web.Response(status=429,text='secret-provider-key')
        return web.json_response({'live_submission_locked':True} if request.path=='/health' else {'portfolio':{'portfolio_value':11}})
    upstream=web.Application();upstream.router.add_get('/{path}',endpoint)
    async with TestServer(upstream) as source:
        hub=server.Hub({'spot':str(source.make_url('')).rstrip('/')})
        import aiohttp
        async with aiohttp.ClientSession(trust_env=False) as session:
            await hub.refresh(session,'spot')
            frame=hub.frames['spot'];assert frame['data']['portfolio']['portfolio_value']==11
            mode['fail']=True;await hub.refresh(session,'spot')
        failed=hub.snapshot()['sources']['spot']
        assert failed['data']==frame['data'] and failed['fetched_at']==frame['fetched_at']
        assert failed['last_error_type']=='ValueError' and failed['cache_age_seconds']>=0
        assert 'secret-provider-key' not in json.dumps(hub.snapshot())


@pytest.mark.asyncio
async def test_source_refresh_cannot_block_cached_views_or_add_order_routes():
    entered,release=asyncio.Event(),asyncio.Event()
    app=server.create_app(poll=False);hub=app[server.HUB]
    async def slow(session,url):
        entered.set();await release.wait();return {'status':'healthy'}
    hub.read=slow
    async with TestClient(TestServer(app)) as client:
        task=asyncio.create_task(hub.refresh(None,'spot'))
        try:
            await asyncio.wait_for(entered.wait(),1)
            response=await asyncio.wait_for(client.get('/api/snapshot'),1)
            assert (await response.json())['sources']['spot']['refreshing']
            health=await asyncio.wait_for(client.get('/health'),1)
            assert (await health.json())['execution_authority'] is False
            assert "connect-src 'self'" in response.headers['Content-Security-Policy']
            assert (await client.post('/api/snapshot',json={'order':'buy'})).status==405
            assert (await client.post('/orders',json={})).status==404
            assert (await client.get('/api/snapshot',headers={'Host':'untrusted.example'})).status==403
            assert (await client.get('/assets/server.py')).status==404
            for path in ('/','/assets/terminal.js','/assets/terminal.css'):
                assert (await client.get(path)).status==200
        finally:
            release.set();await task


@pytest.mark.asyncio
@pytest.mark.parametrize('body', ['[]','{"value": NaN}', '{"value": Infinity}'])
async def test_invalid_upstream_envelopes_rejected(body):
    async def endpoint(request):return web.Response(text=body,content_type='application/json')
    app=web.Application();app.router.add_get('/',endpoint)
    import aiohttp
    async with TestServer(app) as source, aiohttp.ClientSession(trust_env=False) as session:
        with pytest.raises(ValueError):await server.Hub({}).read(session,source.make_url('/'))


@pytest.mark.asyncio
async def test_redirect_not_followed_and_body_bounded(monkeypatch):
    reached=[]
    async def redirect(request):raise web.HTTPFound('/target')
    async def target(request):reached.append(True);return web.json_response({'a':'too much data'})
    app=web.Application();app.router.add_get('/',redirect);app.router.add_get('/target',target)
    import aiohttp
    async with TestServer(app) as source, aiohttp.ClientSession(trust_env=False) as session:
        with pytest.raises(ValueError):await server.Hub({}).read(session,source.make_url('/'))
        assert reached==[]
        monkeypatch.setattr(server,'MAX_BYTES',8)
        with pytest.raises(ValueError,match='envelope exceeded'):
            await server.Hub({}).read(session,source.make_url('/target'))
