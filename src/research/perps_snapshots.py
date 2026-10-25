"""Unauthenticated, sampled public marks and keyed oracle/context snapshots.

One short-lived socket per pass. This is not a claim to continuous tick capture.
"""
import asyncio
import base64
import json
import math
import time
import zlib
from decimal import Decimal

from src.research.data_miners import RateLimited

WS = 'wss://api.hyperliquid.xyz/ws'
MAX_FRAME = 8*1024*1024  # parser memory envelope, not a market universe ceiling


def _marks(frame):
    if frame.get('channel')!='fastAssetCtxs' or not isinstance(frame.get('data'),str):
        raise ValueError('documented fast mark snapshot required')
    raw = base64.b64decode(frame['data'],validate=True)
    decoder = zlib.decompressobj(-15)
    body = decoder.decompress(raw,MAX_FRAME+1)
    if len(body)>MAX_FRAME or decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
        raise ValueError('invalid or oversized mark frame')
    values = json.loads(body,parse_float=str)
    if not isinstance(values,dict): raise ValueError('keyed public mark frame required')
    marks = {}
    for coin,ctx in values.items():
        if not isinstance(coin,str) or not isinstance(ctx,dict): raise ValueError('invalid keyed mark')
        px = Decimal(str(ctx.get('markPx','NaN')))
        if px.is_finite() and px>0: marks[coin]=str(px)
    return marks


async def _fetch_snapshots(state,http):
    now = time.time()
    with state.lock,state.db:
        state.db.execute('DELETE FROM pp_ws_attempts WHERE at<=?',(now-60,))
        connections,messages = state.db.execute('SELECT count(*),COALESCE(sum(subscriptions),0) FROM pp_ws_attempts').fetchone()
        if connections>=24 or messages>=1800: return []
        quantum = min(149,max(1,math.ceil(len(state.market_names())/3)),1799-messages)
        coins = state.acquisition.targets('ws_ctx',quantum)
        state.db.execute('INSERT INTO pp_ws_attempts VALUES(?,?)',(now,len(coins)+1))
        state.db.executemany("INSERT INTO acquisition_service VALUES('ws_ctx',?,?,'') ON CONFLICT(role,target) DO UPDATE SET attempted=excluded.attempted",((coin,now) for coin in coins))
    rows,seen = [],set()
    have_marks = False
    def observation(role,frame):
        return {'provider':'hyperliquid','product':'perpetual','perps_role':role,
                'observed_at':time.time(),'frame':frame,'execution_authority':False,
                'timestamp_semantics':'snapshot receipt; provider event time unknown'}
    session = await http.session()
    try:
        async with asyncio.timeout(8):
            async with session.ws_connect(WS,heartbeat=15,max_msg_size=MAX_FRAME) as socket:
                await socket.send_json({'method':'subscribe','subscription':{'type':'fastAssetCtxs'}})
                for coin in coins:
                    await socket.send_json({'method':'subscribe','subscription':{'type':'activeAssetCtx','coin':coin}})
                while not have_marks or len(seen)<len(coins):
                    frame = await socket.receive_json()
                    if frame.get('channel')=='fastAssetCtxs' and not have_marks:
                        _marks(frame)  # malformed data cannot become a retained measurement
                        rows.append(observation('mark_snapshot',frame))
                        have_marks = True
                    elif frame.get('channel')=='activeAssetCtx':
                        payload = frame.get('data',{})
                        coin = payload.get('coin')
                        if coin not in coins or coin in seen or not isinstance(payload.get('ctx'),dict):
                            continue
                        seen.add(coin)
                        rows.append(observation('asset_context',frame))
    except Exception as exc:
        # Keep already received snapshots when another response or the socket
        # fails. Failure is explicit and retained alongside those observations.
        rows.append({'provider':'hyperliquid','product':'perpetual','perps_role':'request_error',
            'request_role':'snapshot','error':type(exc).__name__,'observed_at':time.time(),
            'execution_authority':False})
        if getattr(exc,'status',None)==429:
            raise RateLimited('public perps WebSocket refused connection') from None
    with state.lock,state.db:
        for coin in coins:
            state.db.execute("UPDATE acquisition_service SET error=? WHERE role='ws_ctx' AND target=?",
                             ('' if coin in seen else 'SNAPSHOT_NOT_RECEIVED',coin))
    return rows
