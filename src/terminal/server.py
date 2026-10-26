"""Read-only loopback monitor; no signer, connection-file import or order API."""
from __future__ import annotations
import argparse
import asyncio
from collections import deque
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

ASSETS = Path(__file__).parent/'assets'
MAX_BYTES = 8*1024*1024
HUB = web.AppKey('paper_terminal_hub', object)


def loopback_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1','::1'}
            or parsed.username or parsed.password or parsed.path not in {'','/'}
            or parsed.query or parsed.fragment or parsed.port is None):
        raise ValueError('explicit loopback HTTP source required')
    return value.rstrip('/')


def number(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None


def at_path(value, *keys):
    for key in keys:
        value = value.get(key) if isinstance(value,dict) else None
    return value


class Hub:
    def __init__(self, sources, *, interval=5.):
        self.sources = {name:loopback_url(url) for name,url in sources.items()}
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('positive finite polling interval required')
        self.interval = interval
        self.frames = {name:{'available':False,'health':{},'data':{},'fetched_at':None,
                             'last_error_type':None,'refreshing':False} for name in sources}
        self.events = deque(maxlen=256)
        self.history = deque(maxlen=720)
        self.started_at = time.time()
        self.event_count = 0
        self.tasks = []

    def event(self, role, text, source, observed_at):
        self.event_count += 1
        self.events.append({'id':self.event_count,'role':role,'text':text,'source':source,
                            'observed_at':observed_at,'kind':'DERIVED_SERVICE_TELEMETRY'})

    def observe(self, name, before, after, at):
        if not before.get('available'):
            self.event('world' if name=='world' else 'verifier',
                       f'{name} diagnostics received; source scope remains explicit',name,at)
        if name != 'spot':
            return
        old,new = before.get('data',{}),after['data']
        fields = [('verifier',('unknown_opportunity_engine','canonical_evidence','records'),'evidence records retained'),
                  ('narrative',('neural_sentiment','canonical_windows'),'sentiment windows committed; uncalibrated research'),
                  ('scanner',('native_fork_probe','targets'),'native-probe targets retained'),
                  ('perps',('phantom_perps','funding_records'),'public underlying-perps funding observations recorded'),
                  ('risk',('native_fork_probe','canonical_dispositions','FORK_ENVIRONMENT_UNKNOWN'),'local-probe outcomes remain unknown')]
        for role,path,label in fields:
            previous,current = number(at_path(old,*path)),number(at_path(new,*path))
            if previous is not None and current is not None and current>previous:
                self.event(role,f'+{int(current-previous):,} {label}',name,at)
        self.history.append({'observed_at':at,
            'model_equity_usd':number(at_path(new,'portfolio','portfolio_value')),
            'perps_funding_records':number(at_path(new,'phantom_perps','funding_records')),
            'canonical_records':number(at_path(new,'unknown_opportunity_engine','canonical_evidence','records'))})

    async def read(self, session, url):
        async with session.get(url,allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError('upstream HTTP unavailable')
            raw=bytearray()
            async for chunk in response.content.iter_chunked(65536):
                raw.extend(chunk)
                if len(raw)>MAX_BYTES:
                    raise ValueError('upstream envelope exceeded')
        def invalid(value):raise ValueError('nonfinite upstream data')
        result=json.loads(raw,parse_constant=invalid)
        if not isinstance(result,dict):raise ValueError('upstream object required')
        return result

    async def refresh(self, session, name):
        before=self.frames[name]
        self.frames[name]={**before,'refreshing':True}
        began=time.perf_counter()
        try:
            results=await asyncio.gather(self.read(session,self.sources[name]+'/health'),
                                         self.read(session,self.sources[name]+'/status'),return_exceptions=True)
            for result in results:
                if isinstance(result,BaseException):raise result
            health,data=results
            frame={'available':True,'health':health,'data':data,'fetched_at':time.time(),
                   'last_error_type':None,'refreshing':False,'response_seconds':time.perf_counter()-began}
            self.observe(name,before,frame,frame['fetched_at'])
            self.frames[name]=frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Error strings can contain provider credentials. Never forward them.
            self.frames[name]={**before,'last_error_type':type(exc).__name__,'refreshing':False}

    async def loop(self, session, name):
        while True:
            began=time.monotonic()
            await self.refresh(session,name)
            await asyncio.sleep(max(.25,self.interval-(time.monotonic()-began)))

    def snapshot(self):
        now=time.time()
        return {'schema':'read-only-paper-terminal-v1','observed_at':now,'started_at':self.started_at,
            'execution_authority':False,'event_scope':'recent derived service telemetry, not AI deliberation',
            'history_scope':'observed status snapshots since this monitor started, not confirmed fill history',
            'event_count_since_start':self.event_count,
            'sources':{name:{**frame,'cache_age_seconds':max(0.,now-frame['fetched_at'])
                       if frame['fetched_at'] is not None else None} for name,frame in self.frames.items()},
            'events':list(self.events),'history':list(self.history)}


@web.middleware
async def headers(request,handler):
    # Bind is loopback-only; also reject browser DNS-rebinding Host headers.
    try: host=urlsplit('http://'+request.host).hostname
    except ValueError:host=None
    if host not in {'127.0.0.1','::1','localhost'}:raise web.HTTPForbidden()
    response=await handler(request)
    response.headers.update({'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
        'Content-Security-Policy':"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        'Referrer-Policy':'no-referrer'})
    return response


def create_app(*,spot='http://127.0.0.1:18081',world='http://127.0.0.1:18082',interval=5.,poll=True):
    app=web.Application(middlewares=[headers]);hub=Hub({'spot':spot,'world':world},interval=interval);app[HUB]=hub
    async def snapshot(request):return web.json_response(hub.snapshot())
    async def health(request):return web.json_response({'status':'healthy','read_only':True,'execution_authority':False,
        'upstream_available':{name:frame['available'] and not frame['last_error_type']
                              for name,frame in hub.frames.items()}})
    async def asset(request):
        name=request.match_info.get('name','index.html')
        if name not in {'index.html','terminal.css','terminal.js'}:raise web.HTTPNotFound()
        return web.FileResponse(ASSETS/name)
    app.router.add_get('/health',health);app.router.add_get('/api/snapshot',snapshot)
    app.router.add_get('/',asset);app.router.add_get('/assets/{name}',asset)
    async def lifecycle(app):
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=8)) as session:
            if poll:hub.tasks=[asyncio.create_task(hub.loop(session,name)) for name in hub.sources]
            try:yield
            finally:
                for task in hub.tasks:task.cancel()
                await asyncio.gather(*hub.tasks,return_exceptions=True)
    app.cleanup_ctx.append(lifecycle)
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18083)
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('valid local port required')
    web.run_app(create_app(),host='127.0.0.1',port=args.port,print=None)


if __name__=='__main__':main()
