"""Permanent received-program observations, independent of trading authority.

The queue bounds active memory, not retained history. It applies backpressure
instead of silently dropping admitted observations. Only committed batches are
reported as retained. Raw receipt clocks are not execution or finality proof.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

SCHEMA = 'received-program-tape-v1'
GENESIS = '0'*64


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def freeze(value):
    if isinstance(value,(bytes,bytearray,memoryview)):
        return {'encoding':'base64','data':base64.b64encode(bytes(value)).decode()}
    if isinstance(value,dict):
        if any(not isinstance(key,str) for key in value):raise ValueError('string tape field names required')
        return {key:freeze(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [freeze(item) for item in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('unsupported raw observation type')


class ProgramEventTape:
    def __init__(self,path,*,queue_size=4096,batch_size=128,batch_delay=.02):
        if type(queue_size) is not int or queue_size<1 or type(batch_size) is not int or batch_size<1:
            raise ValueError('positive active queue and commit batch sizes required')
        if not 0<=batch_delay<=1:raise ValueError('finite commit delay within one second required')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=20,check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS records(sequence INTEGER PRIMARY KEY,observation_id TEXT UNIQUE NOT NULL,
            event_id TEXT NOT NULL,kind TEXT NOT NULL,body TEXT NOT NULL,hash TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS program_event_identity ON records(event_id,sequence);
          CREATE TABLE IF NOT EXISTS tape_meta(id INTEGER PRIMARY KEY CHECK(id=1),sequence INTEGER NOT NULL,
            last_hash TEXT NOT NULL,counts TEXT NOT NULL,clean_shutdown INTEGER NOT NULL);
        ''')
        self.db.execute('INSERT OR IGNORE INTO tape_meta VALUES(1,0,?,\'{}\',1)',(GENESIS,))
        sequence,seal,counts,clean=self.db.execute('SELECT sequence,last_hash,counts,clean_shutdown FROM tape_meta WHERE id=1').fetchone()
        last=self.db.execute('SELECT sequence,hash FROM records ORDER BY sequence DESC LIMIT 1').fetchone()
        if (last or (0,GENESIS))!=(sequence,seal):raise ValueError('tape checkpoint and tail disagree')
        self.previous_shutdown_clean=bool(clean)
        self.db.execute('UPDATE tape_meta SET clean_shutdown=0 WHERE id=1');self.db.commit()
        self.lock=threading.RLock();self.queue=asyncio.Queue(maxsize=queue_size)
        self.batch_size,self.batch_delay=batch_size,batch_delay
        self.worker=None;self.closed=False;self.accepted=0;self.backpressure=0
        self.admission_cancelled=0;self.input_errors=0;self.write_errors=0;self.last_error=None
        self.inflight=0;self.last_commit_at=None;self.last_commit_seconds=None
        self.sequence,self.last_hash,self.counts=sequence,seal,json.loads(counts)

    def start(self):
        if self.closed:raise RuntimeError('tape is closed')
        if self.worker is None:self.worker=asyncio.create_task(self._writer(),name='durable-program-tape')

    async def enqueue(self,kind,update):
        if self.closed:raise RuntimeError('tape is closed')
        try:
            if kind not in {'program_data','transaction','source_transition'}:raise ValueError('unknown tape observation kind')
            received=time.time_ns();payload=freeze(update)
            identity=([payload.get('program'),payload.get('signature'),payload.get('log_index'),payload.get('data')]
                      if kind=='program_data' else payload)
            event_id=digest([kind,identity])
            body={'schema':SCHEMA,'kind':kind,'capture_received_at_ns':received,
                'event_id':event_id,'payload':payload,
                'scope':'received source observation; not a confirmed fill or complete market census'}
            body['observation_id']=digest(body)
            canonical(body)  # Reject nonfinite/unserializable data before queue admission.
        except (ValueError,TypeError,AttributeError):
            self.input_errors+=1;raise
        self.start()
        if self.queue.full():self.backpressure+=1
        try:await self.queue.put(body)
        except asyncio.CancelledError:
            self.admission_cancelled+=1;raise
        self.accepted+=1
        return body['observation_id']

    def _append(self,batch):
        began=time.perf_counter()
        with self.lock,self.db:
            self.db.execute('BEGIN IMMEDIATE')
            sequence,previous,raw=self.db.execute('SELECT sequence,last_hash,counts FROM tape_meta WHERE id=1').fetchone()
            counts=json.loads(raw)
            for observation in batch:
                existing=self.db.execute('SELECT body FROM records WHERE observation_id=?',(observation['observation_id'],)).fetchone()
                if existing:
                    saved=json.loads(existing[0]);saved.pop('sequence');saved.pop('previous_hash')
                    if saved!=observation:raise ValueError('retained tape observation identity changed')
                    continue
                sequence+=1;body={**observation,'sequence':sequence,'previous_hash':previous}
                previous=digest(body)
                self.db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',
                    (sequence,body['observation_id'],body['event_id'],body['kind'],canonical(body),previous))
                counts[body['kind']]=counts.get(body['kind'],0)+1
            self.db.execute('UPDATE tape_meta SET sequence=?,last_hash=?,counts=? WHERE id=1',
                            (sequence,previous,canonical(counts)))
        # Publish counters only after SQLite commit has returned.
        self.sequence,self.last_hash,self.counts=sequence,previous,counts
        self.last_commit_at=time.time();self.last_commit_seconds=time.perf_counter()-began

    async def _writer(self):
        while True:
            batch=[await self.queue.get()]
            if self.batch_delay:await asyncio.sleep(self.batch_delay)
            while len(batch)<self.batch_size:
                try:batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:break
            self.inflight=len(batch)
            while True:
                operation=asyncio.create_task(asyncio.to_thread(self._append,batch))
                try:
                    await asyncio.shield(operation);self.last_error=None;break
                except asyncio.CancelledError:
                    await asyncio.gather(operation,return_exceptions=True)
                    raise
                except Exception as exc:
                    self.write_errors+=1;self.last_error=type(exc).__name__
                    await asyncio.sleep(1)  # Retain the same batch for retry.
            for _ in batch:self.queue.task_done()
            self.inflight=0

    async def flush(self):
        await self.queue.join()

    async def stop(self):
        self.closed=True
        await self.flush()
        if self.worker:
            self.worker.cancel();await asyncio.gather(self.worker,return_exceptions=True)
        def close():
            with self.lock,self.db:self.db.execute('UPDATE tape_meta SET clean_shutdown=1 WHERE id=1')
            self.db.close()
        await asyncio.to_thread(close)

    def report(self):
        return {'status':'DEGRADED' if self.last_error else 'OBSERVED' if self.sequence else 'NOT_OBSERVED',
            'schema':SCHEMA,'committed_records':self.sequence,'committed_by_kind':dict(self.counts),
            'accepted_this_run':self.accepted,'queued':self.queue.qsize(),'inflight':self.inflight,
            'backpressure_events_this_run':self.backpressure,'admission_cancellations_this_run':self.admission_cancelled,
            'input_errors_this_run':self.input_errors,'write_errors_this_run':self.write_errors,
            'last_error_type':self.last_error,'last_commit_at':self.last_commit_at,
            'last_commit_seconds':self.last_commit_seconds,'previous_shutdown_clean':self.previous_shutdown_clean,
            'uncommitted_loss_after_unclean_shutdown':'UNKNOWN' if not self.previous_shutdown_clean else None,
            'history_count_limit':None,'all_programs_coverage':False,'execution_authority':False}

    @staticmethod
    def verify(path):
        db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
        try:
            db.execute('BEGIN');previous=GENESIS;sequence=0
            for number,raw,seal in db.execute('SELECT sequence,body,hash FROM records ORDER BY sequence'):
                body=json.loads(raw)
                if number!=sequence+1 or body['sequence']!=number or body['previous_hash']!=previous or digest(body)!=seal:
                    raise ValueError('tape sequence/hash chain failed')
                sequence,previous=number,seal
            checkpoint=db.execute('SELECT sequence,last_hash FROM tape_meta WHERE id=1').fetchone()
            if checkpoint!=(sequence,previous):raise ValueError('tape tail/checkpoint verification failed')
            return {'status':'PASS','records':sequence,'last_hash':previous,'scope':'owner-local committed snapshot integrity; not external attestation'}
        finally:db.close()
