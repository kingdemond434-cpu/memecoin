"""Independent public program-log recording; no order or signing operation."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace

from aiohttp import web
from solders.pubkey import Pubkey
import websockets

from src.chains.rpc_manager import ChainRegistry,RPCManager,RPCEndpointConfig
from src.chains.yellowstone_grpc import PumpFunMonitor,PumpSwapMonitor,RaydiumMonitor,scoped_program_data
from src.chains.launchlab import LAUNCHLAB_PROGRAM
from src.chains.meteora_dbc import DBC_PROGRAM
from src.research.program_tape import ProgramEventTape

CAPTURE=web.AppKey('capture',object)


def program_ids(extra=()):
    values=[PumpFunMonitor.PUMP_FUN_PROGRAM,PumpSwapMonitor.PUMP_AMM_PROGRAM,
        RaydiumMonitor.RAYDIUM_AMM_V4,RaydiumMonitor.RAYDIUM_CPMM,RaydiumMonitor.RAYDIUM_CLMM,
        RaydiumMonitor.METEORA_DLMM,RaydiumMonitor.METEORA_DYNAMIC_AMM,RaydiumMonitor.ORCA_WHIRLPOOL,
        LAUNCHLAB_PROGRAM,DBC_PROGRAM,*extra]
    result=list(dict.fromkeys(values))
    for value in result:Pubkey.from_string(value)
    return result


class ProgramCapture:
    def __init__(self,rpc,tape,programs):
        self.rpc,self.tape,self.programs=rpc,tape,list(programs)
        self.task=None;self.running=False;self.connected=False;self.status='NOT_STARTED'
        self.last_error=None;self.errors=0;self.connections=0;self.last_received_at=None
        self.subscribed=[];self.refused={};self.deferred={};self.observed_notifications=0
        self.last_refusal_status=None;self.last_refusal_retry_at=None

    async def transition(self,status,**facts):
        self.status=status
        await self.tape.enqueue('source_transition',{'status':status,'observed_at':time.time(),**facts})

    def _until(self,url):
        shared=getattr(self.rpc,'_shared_cooldowns',None)
        deadline=shared.until(RPCEndpointConfig(url=url)) if shared else 0.
        return max(deadline,self.deferred.get(url,0.))

    def _defer(self,url,status,headers=None):
        delay=RPCManager._retry_after_seconds(SimpleNamespace(headers=headers or {}),
            3600. if status in {401,403} else 60.)
        until=time.time()+delay;self.deferred[url]=max(until,self.deferred.get(url,0.))
        self.last_refusal_status=status;self.last_refusal_retry_at=self.deferred[url]
        shared=getattr(self.rpc,'_shared_cooldowns',None)
        if shared:shared.defer(RPCEndpointConfig(url=url),until)
        return delay

    async def start(self):
        self.tape.start();self.running=True
        self.task=asyncio.create_task(self._run(),name='permanent-raw-program-capture')

    async def _run(self):
        cursor=0
        while self.running:
            urls=[url for url in self.rpc.get_ws_urls() if self._until(url)<=time.time()]
            if not urls:
                if self.status!='DATA_BLOCKED_NO_ELIGIBLE_WS':await self.transition('DATA_BLOCKED_NO_ELIGIBLE_WS')
                await asyncio.sleep(5);continue
            url=urls[cursor%len(urls)];retry=5.
            try:
                await self.transition('CONNECTING')
                async with websockets.connect(url,open_timeout=15,ping_interval=20,ping_timeout=20,
                        close_timeout=5,max_size=8*1024*1024) as socket:
                    self.connections+=1;self.refused={};self.subscribed=[]
                    requested={number:program for number,program in enumerate(self.programs,1)}
                    active={};answered=set()
                    for number,program in requested.items():
                        await socket.send(json.dumps({'jsonrpc':'2.0','id':number,'method':'logsSubscribe',
                            'params':[{'mentions':[program]},{'commitment':'processed'}]}))
                    # Notifications can arrive while later subscription ACKs
                    # are pending. Dispatch accepted subscriptions immediately.
                    while self.running:
                        message=json.loads(await asyncio.wait_for(socket.recv(),timeout=45))
                        if message.get('id') in requested:
                            number=message['id'];answered.add(number)
                            error=message.get('error')
                            if error:
                                code=error.get('code') if isinstance(error,dict) else None
                                if code in {429,-32029}:
                                    retry=self._defer(url,429)
                                    raise ConnectionRefusedError('subscription quota refusal')
                                self.refused[requested[number]]=code if type(code) is int else None
                            elif type(message.get('result')) is int:
                                active[message['result']]=requested[number]
                                self.subscribed=list(active.values())
                            if len(answered)==len(requested):
                                if not active:
                                    retry=self._defer(url,403)
                                    raise ConnectionRefusedError('no authorized program subscriptions')
                                self.connected=True
                                await self.transition('PARTIAL_RPC_WS' if self.refused else 'RPC_WS',
                                    accepted_programs=len(active),refused_programs=len(self.refused))
                            continue
                        params=message.get('params') or {};program=active.get(params.get('subscription'))
                        if program is None:continue
                        result=params.get('result') or {};value=result.get('value') or {}
                        self.observed_notifications+=1;self.last_received_at=time.time()
                        received_ns=time.time_ns()
                        # Retain the complete received log notification even
                        # when this version cannot decode any program payload.
                        await self.tape.enqueue('transaction',{'program':program,
                            'received_ns':received_ns,'notification':params,
                            'source_commitment':'processed','source':'authorized_rpc_ws',
                            'execution_authority':False})
                        # Failed transactions remain observed failed attempts;
                        # their logs are not successful fills or settled trades.
                        for index,raw in scoped_program_data(value.get('logs') or [],program):
                            await self.tape.enqueue('program_data',{'program':program,
                                'signature':value.get('signature'),'slot':(result.get('context') or {}).get('slot'),
                                'log_index':index,'data':raw,'received_ns':received_ns,
                                'transaction_error':value.get('err'),'source_commitment':'processed',
                                'source':'authorized_rpc_ws','execution_authority':False})
            except asyncio.CancelledError:raise
            except Exception as exc:
                self.errors+=1;self.last_error=type(exc).__name__
                response=getattr(exc,'response',exc)
                status=getattr(response,'status_code',None)
                if status in {401,403,429,503}:retry=max(retry,self._defer(url,status,getattr(response,'headers',None)))
                await self.transition('DISCONNECTED',error_type=type(exc).__name__)
                cursor+=1
            finally:self.connected=False
            # Refused providers stay deferred; other explicitly configured
            # providers can serve the same public data during that window.
            await asyncio.sleep(min(5.,retry))

    async def stop(self):
        self.running=False
        if self.task:
            self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        await self.transition('STOPPED')
        await self.tape.stop()
        await self.rpc.stop()

    def report(self):
        return {'schema':'independent-program-capture-v1','status':self.status,'connected':self.connected,
            'configured_programs':list(self.programs),'subscribed_programs':list(self.subscribed),
            'refused_program_codes':dict(self.refused),'connections_this_run':self.connections,
            'errors_this_run':self.errors,'last_error_type':self.last_error,
            'last_refusal_status':self.last_refusal_status,'last_refusal_retry_at':self.last_refusal_retry_at,
            'last_notification_at':self.last_received_at,'notifications_this_run':self.observed_notifications,
            'tape':self.tape.report(),'transport':'RPC_WS_ONLY','http_backfill':'NOT_IMPLEMENTED',
            'coverage_scope':'received logs for configured programs; no complete chain or launch census',
            'all_programs_coverage':False,'execution_authority':False,'real_transactions_submitted':0}


@web.middleware
async def local_only(request,handler):
    from urllib.parse import urlsplit
    try:host=urlsplit('http://'+request.host).hostname
    except ValueError:host=None
    if host not in {'127.0.0.1','::1','localhost'}:raise web.HTTPForbidden()
    response=await handler(request)
    response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
    return response


def create_app(capture):
    app=web.Application(middlewares=[local_only]);app[CAPTURE]=capture
    async def status(request):return web.json_response(capture.report())
    async def health(request):
        report=capture.report()
        return web.json_response({'status':'healthy' if capture.running and not capture.tape.last_error else 'degraded',
            'source_status':capture.status,'committed_program_records':report['tape']['committed_by_kind'].get('program_data',0),
            'execution_authority':False,'live_submission_locked':True})
    app.router.add_get('/health',health);app.router.add_get('/status',status)
    async def lifecycle(app):
        await capture.start()
        try:yield
        finally:await capture.stop()
    app.cleanup_ctx.append(lifecycle)
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/chains.yaml')
    parser.add_argument('--state-dir',required=True)
    parser.add_argument('--port',type=int,default=18084)
    parser.add_argument('--program',action='append',default=[])
    parser.add_argument('--verify-only',action='store_true',help='Verify a committed tape snapshot and exit without connecting')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('unprivileged loopback port required')
    tape_path=Path(args.state_dir)/'program-events.sqlite'
    if args.verify_only:
        print(json.dumps(ProgramEventTape.verify(tape_path)))
        return
    registry=ChainRegistry(args.config)
    chain=registry.get_chain('solana')
    if chain is None:parser.error('configured Solana RPC required')
    tape=ProgramEventTape(tape_path)
    capture=ProgramCapture(RPCManager(chain),tape,program_ids(args.program))
    web.run_app(create_app(capture),host='127.0.0.1',port=args.port,print=None)


if __name__=='__main__':main()
