"""Durable all-source document sentiment research, never order authority.

Inputs are already canonical public/authorized observations. Inference runs
offline in a separate, credential-free Python process with pinned public model
bytes. Every token window is queued; resource batches never trim the universe.
Raw softmax scores are neither calibrated trading probabilities nor truth tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

MINER = 'local:neural_sentiment'
SCHEMA = 'offline-document-sentiment-v1'
REVISION = '9d9ac661fd7b0b48535a1fc99b20ae6947629e65'
FILES = {
    'onnx/model_quantized.onnx': '9570fdd726d952dbaf559ba0b9f38a523554f08a8038d62d9263a90dcbb3cd3e',
    'tokenizer.json': 'a4a4fa4b10a2a40361529954532f10257fda53cb22177554a862bcc661b7e4d8',
    'config.json': '89c0158e3725d30d852258638c64057228ff93d676f2c3e22ac20025a89653c7',
}
LANGUAGES = ('en', 'ar', 'de', 'es', 'fr', 'ja', 'zh', 'id', 'hi', 'it', 'ms', 'pt')
TRANSPORT_FIELDS = frozenset(('_miner', '_enriches', '_endpoint', '_fetched_at',
                             '_request_started_at', '_scheduled_at'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class OfflineWorker:
    def __init__(self, python, model_root, *, intra_op_threads=1):
        self.inference_profile = inference_profile(intra_op_threads)
        self.python, self.model_root = str(python), str(model_root)
        self.process = None

    async def infer(self, text, offset):
        if self.process is None or self.process.returncode is not None:
            environment = {k: v for k, v in os.environ.items()
                           if k.upper() in {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
            environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            self.process = await asyncio.create_subprocess_exec(self.python, '-m', __name__,
                '--worker', self.model_root, '--threads', str(self.inference_profile['intra_op_threads']),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                env=environment, limit=1024 * 1024)
        try:
            self.process.stdin.write((canonical({'text': text, 'offset': offset})+'\n').encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), 30)
            result = json.loads(line)
            if not result.get('ok') or result.get('inference_profile') != self.inference_profile:
                raise ValueError('offline inference unavailable')
            return result['result']
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.returncode is None:
                process.kill()
            await process.wait()


def inference_profile(threads):
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError('NLP thread count must be a positive integer within logical CPU capacity')
    return {'provider': 'CPUExecutionProvider', 'intra_op_threads': threads, 'inter_op_threads': 1}


class SentimentResearch:
    def __init__(self, path, *, python=None, model_root=None, worker=None, ledger_path=None, intra_op_threads=1):
        profile = inference_profile(intra_op_threads)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=20)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS inputs(id TEXT PRIMARY KEY,body TEXT NOT NULL,
              offset INTEGER NOT NULL DEFAULT 0,complete INTEGER NOT NULL DEFAULT 0,
              served REAL NOT NULL DEFAULT 0,created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,input_id TEXT NOT NULL,
              body TEXT NOT NULL,ref TEXT,available REAL);
            CREATE TABLE IF NOT EXISTS replay(id INTEGER PRIMARY KEY CHECK(id=1),
              offset INTEGER NOT NULL,previous TEXT NOT NULL,sequence INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS scheduler(id INTEGER PRIMARY KEY CHECK(id=1),ticket INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS inputs_pending ON inputs(complete,served,created,id);
            CREATE INDEX IF NOT EXISTS results_pending ON results(ref);
            CREATE TABLE IF NOT EXISTS dispatch(id INTEGER PRIMARY KEY CHECK(id=1),cursor INTEGER NOT NULL);
            INSERT OR IGNORE INTO dispatch VALUES(1,0);
            CREATE TABLE IF NOT EXISTS inference_cache(id TEXT PRIMARY KEY,body TEXT NOT NULL,sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metrics(name TEXT PRIMARY KEY,value INTEGER NOT NULL);
        ''')
        if 'origin' not in {row[1] for row in self.db.execute('PRAGMA table_info(inputs)')}:
            self.db.execute("ALTER TABLE inputs ADD COLUMN origin TEXT NOT NULL DEFAULT 'LIVE'")
            self.db.execute("UPDATE inputs SET origin='REPLAY' WHERE json_extract(body,'$.source_job') IS NULL")
        self.db.execute('CREATE INDEX IF NOT EXISTS inputs_lanes ON inputs(complete,origin,served,created,id)')
        if 'arrived' not in {row[1] for row in self.db.execute('PRAGMA table_info(inputs)')}:
            self.db.execute('ALTER TABLE inputs ADD COLUMN arrived INTEGER NOT NULL DEFAULT 0')
            self.db.execute('UPDATE inputs SET arrived=rowid')
        self.db.execute('CREATE INDEX IF NOT EXISTS inputs_fresh ON inputs(complete,origin,arrived DESC,served,id)')
        self.db.execute('INSERT OR IGNORE INTO replay VALUES(1,0,?,0)', ('0'*64,))
        # Old unserved rows used zero, which lets an endless stream of new
        # inputs leapfrog the remaining windows of an already-started post.
        self.db.execute('UPDATE inputs SET served=created WHERE served=0')
        self.db.execute('INSERT OR IGNORE INTO scheduler SELECT 1,CAST(COALESCE(MAX(served),0) AS INTEGER) FROM inputs')
        self.db.commit()
        # Hot-path delivery is a point lookup of already ACKed output. It must
        # not wait behind the writer connection's replay/report/fsync lock.
        # WAL gives this separate read-only connection committed snapshots;
        # uncommitted annotations and ACKs remain invisible.
        self.observation_db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro',
            uri=True, check_same_thread=False, timeout=3)
        self.observation_db.execute('PRAGMA query_only=ON')
        self.observation_lock = threading.RLock()
        # Full queue counts are intentionally separate from both the writer
        # and the latency-sensitive primary-key observation lookup.
        self.report_db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro',
            uri=True, check_same_thread=False, timeout=3)
        self.report_db.execute('PRAGMA query_only=ON')
        self.report_lock = threading.RLock()
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.lock = threading.RLock()
        self.worker = worker or (OfflineWorker(python, model_root, intra_op_threads=intra_op_threads)
                                 if python and model_root else None)
        self.inference_profile = getattr(self.worker, 'inference_profile', profile)
        self.last_error = None
        self.last_intake_error = None
        self.intake_failures = 0
        self.replay_error = None
        self.last_replay_seconds = None
        self.last_fetch_profile = None
        self.inference_code_sha256 = hashlib.sha256(
            (inspect.getsource(worker_main)+inspect.getsource(inference_profile)).encode()).hexdigest()

    def _next_ticket(self):
        # Called inside the queue transaction. It survives process restarts
        # and cannot move backwards when the host wall clock is corrected.
        return self.db.execute('UPDATE scheduler SET ticket=ticket+1 WHERE id=1 RETURNING ticket').fetchone()[0]

    def enqueue(self, payload, *, source_job, source_ref, tokens=(), origin=None):
        with self.lock, self.db:
            return self._enqueue_locked(payload, source_job=source_job, source_ref=source_ref,
                                        tokens=tokens, origin=origin)

    def _enqueue_locked(self, payload, *, source_job, source_ref, tokens=(), origin=None):
        """Insert inside the caller's locked transaction, never commit a batch."""
        origin = origin or ('LIVE' if source_job else 'REPLAY')
        if origin not in {'LIVE', 'REPLAY'}:
            raise ValueError('invalid observation intake origin')
        if not source_ref:
            raise ValueError('canonical source reference required')
        text = payload['text']
        if not isinstance(text, str) or not text.strip():
            return None
        observed = float(payload['observed_at'])
        if not math.isfinite(observed):
            raise ValueError('finite observation time required')
        body = {'source_job': source_job, 'source_ref': source_ref,
            'source_id': payload['source_id'], 'platform': payload.get('platform'),
            'language': payload.get('language') or 'UNKNOWN', 'observed_at': observed,
            'event_at': payload.get('event_at'), 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'tokens': sorted(set(tokens)), 'model_revision': REVISION}
        identity = digest([source_ref, REVISION])
        previous = self.db.execute('SELECT body FROM inputs WHERE id=?', (identity,)).fetchone()
        if previous and json.loads(previous[0])['text_sha256'] != body['text_sha256']:
            raise ValueError('canonical source text changed')
        if previous:
            if origin == 'LIVE':
                self.db.execute("UPDATE inputs SET origin='LIVE' WHERE id=?", (identity,))
            return identity
        queued_at = time.time()
        ticket = self._next_ticket()
        self.db.execute('INSERT OR IGNORE INTO inputs(id,body,created,served,origin,arrived) VALUES(?,?,?,?,?,?)',
                        (identity, canonical(body), queued_at, ticket, origin, ticket))
        return identity

    def _metric(self, name):
        self.db.execute('INSERT INTO metrics VALUES(?,1) ON CONFLICT(name) DO UPDATE SET value=value+1', (name,))

    def _job(self, seen):
        with self.lock:
            cursor = self.db.execute('SELECT cursor FROM dispatch WHERE id=1').fetchone()[0]
            preferred = 'LIVE' if cursor % 4 < 3 else 'REPLAY'
            exclusion = ' AND id NOT IN ('+','.join('?' for _ in seen)+')' if seen else ''
            for origin in (preferred, 'REPLAY' if preferred == 'LIVE' else 'LIVE'):
                # Two fresh live slots, one age-fair live slot, one replay
                # slot. No age threshold drops a post or erases a long tail.
                order = 'arrived DESC,served,id' if origin == preferred == 'LIVE' and cursor % 4 < 2 else 'served,created,id'
                row = self.db.execute('SELECT id,body,offset,origin FROM inputs WHERE complete=0 AND origin=? '
                    'AND id NOT IN (SELECT input_id FROM results WHERE ref IS NULL)'+exclusion+
                    ' ORDER BY '+order+' LIMIT 1', (origin, *seen)).fetchone()
                if row:
                    return (*row, (cursor+1) % 4)
        return None

    def reconcile(self, byte_quantum=1024*1024):
        """Verify/replay the append-only canonical prefix, resuming across restarts.

        This recovers raw observations whose immediate NLP intake failed and
        backfills earlier sources. A quantum bounds one pass, not total history.
        An incomplete last line is retried without advancing the checkpoint.
        """
        if self.ledger_path is None or not self.ledger_path.exists():
            return
        from src.research.trade_evidence import _verified_row
        with self.lock:
            offset, previous, sequence = self.db.execute('SELECT offset,previous,sequence FROM replay').fetchone()
        checkpoint = (offset, previous, sequence)
        inputs = []
        if self.ledger_path.stat().st_size < offset:
            raise ValueError('canonical source ledger was truncated')
        boundary = offset+byte_quantum
        with self.ledger_path.open('rb') as handle:
            handle.seek(offset)
            while handle.tell() < boundary:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b'\n'):
                    break
                if line.strip():
                    row, previous = _verified_row(line, previous, sequence+1)
                    sequence += 1
                    work = row.get('payload', {})
                    if row['event_type'] == 'research_typed_work' and work.get('work_kind') == 'NARRATIVE_OBSERVATION':
                        payload = work['point_payload']
                        tokens = payload.get('token_addresses') or ([payload['token_id']] if payload.get('token_id') else [])
                        inputs.append((payload, f"evidence:{row['hash']}:{start}", tokens))
                offset = handle.tell()
        if offset == checkpoint[0]:
            self.replay_error = None
            return
        # Verified inputs and their source cursor commit together. A replay
        # pass must not fsync once per historical post or rewind a concurrent
        # pass that has already advanced the cursor.
        with self.lock, self.db:
            if not self.db.in_transaction:
                self.db.execute('BEGIN IMMEDIATE')
            current = self.db.execute('SELECT offset,previous,sequence FROM replay').fetchone()
            if current != checkpoint:
                return
            for payload, ref, tokens in inputs:
                self._enqueue_locked(payload, source_job=None, source_ref=ref, tokens=tokens)
            self.db.execute('UPDATE replay SET offset=?,previous=?,sequence=? WHERE id=1',
                            (offset, previous, sequence))
            if offset != checkpoint[0]:
                self._metric('replay_commits')
        self.replay_error = None

    async def _disk(self, operation, *args, **kwargs):
        """Drain an initiated SQLite operation before cancellation can close its owner."""
        task=asyncio.create_task(asyncio.to_thread(operation,*args,**kwargs))
        try:return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task,return_exceptions=True)
            raise

    def _pending_results(self):
        with self.lock:
            return self.db.execute('SELECT body FROM results WHERE ref IS NULL ORDER BY rowid LIMIT 256').fetchall()

    def _cached_window(self, cache_id, offset):
        with self.lock:
            cached=self.db.execute('SELECT body,sha256 FROM inference_cache WHERE id=?',(cache_id,)).fetchone()
        if not cached:return None
        try:
            cache=json.loads(cached[0])
            if digest(cache)!=cached[1]:raise ValueError('cached inference changed')
            self._validate_window(cache['result'],offset)
            return cache
        except (ValueError,KeyError,TypeError):
            with self.lock,self.db:self._metric('cache_corruptions')
            return None

    def _commit_window(self, cache_id, cache, cache_hit, metric, row, identity, cursor):
        with self.lock,self.db:
            if not cache_hit:
                self.db.execute('INSERT OR REPLACE INTO inference_cache VALUES(?,?,?)',
                    (cache_id,canonical(cache),digest(cache)))
            self._metric(metric)
            self.db.execute('INSERT INTO results VALUES(?,?,?,NULL,NULL)',
                (row['annotation_id'],identity,canonical(row)))
            self.db.execute('UPDATE inputs SET served=? WHERE id=?',(self._next_ticket(),identity))
            self.db.execute('UPDATE dispatch SET cursor=? WHERE id=1',(cursor,))

    def _failed_window(self, identity, metric, cursor):
        with self.lock,self.db:
            if metric is not None:self._metric(metric)
            self.db.execute('UPDATE inputs SET served=? WHERE id=?',(self._next_ticket(),identity))
            self.db.execute('UPDATE dispatch SET cursor=? WHERE id=1',(cursor,))

    async def fetch(self):
        began=time.perf_counter()
        profile={'disk_wait_seconds':0.,'inference_wait_seconds':0.,'returned_windows':0}
        try:
            rows=await self._fetch_windows(profile)
            profile['returned_windows']=len(rows)
            return rows
        finally:
            profile['elapsed_seconds']=max(0.,time.perf_counter()-began)
            self.last_fetch_profile=profile

    async def _fetch_windows(self, profile):
        if self.worker is None:
            return []
        async def disk(operation,*args):
            started=time.perf_counter()
            try:return await self._disk(operation,*args)
            finally:profile['disk_wait_seconds']+=max(0.,time.perf_counter()-started)
        replay_started = time.perf_counter()
        try:
            await disk(self.reconcile)
        except Exception as exc:
            self.replay_error = type(exc).__name__
        finally:
            self.last_replay_seconds = time.perf_counter()-replay_started
        pending = await disk(self._pending_results)
        if pending:
            return [json.loads(row[0]) for row in pending]
        rows, seen = [], []
        deadline = time.perf_counter()+2
        worker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # These are active-work quanta, never source/history admission limits.
        # Cache hits can drain quickly; expensive new inference is bounded.
        for _ in range(256):
            if seen and time.perf_counter() >= deadline:
                break
            job = await disk(self._job, seen)
            if job is None:
                break
            identity, raw, offset, origin, cursor = job
            seen.append(identity)
            body = json.loads(raw)
            inference_metric = None
            try:
                started = time.perf_counter()
                cache_id = digest([REVISION, FILES, self.inference_code_sha256, self.inference_profile,
                                   body['text_sha256'], offset])
                cache = await disk(self._cached_window,cache_id,offset)
                cache_hit = cache is not None
                if cache is None:
                    inferred=time.perf_counter()
                    try:result = await self.worker.infer(body['text'], offset)
                    finally:profile['inference_wait_seconds']+=max(0.,time.perf_counter()-inferred)
                    self._validate_window(result, offset)
                    cache = {'result': result, 'inferred_at': time.time(),
                        'inference_worker_sha256': worker_sha, 'inference_code_sha256': self.inference_code_sha256}
                else:
                    result = cache['result']
                inference_metric = 'cache_hits' if cache_hit else 'worker_inferences'
                inference_seconds = time.perf_counter()-started
                self._validate_window(result, offset)
                row = {key: value for key, value in body.items() if key != 'text'}
                row.update(schema=SCHEMA, provider='offline_neural_sentiment',
                    annotation_kind='document_sentiment_window', input_id=identity,
                    **result, created_at=time.time(), weights_sha256=FILES['onnx/model_quantized.onnx'],
                    worker_request_seconds=inference_seconds,
                    worker_source_sha256=worker_sha, input_origin=origin,
                    inference_cache_hit=cache_hit, inference_cache_key=cache_id,
                    actual_inference_at=cache['inferred_at'], actual_inference_worker_sha256=cache['inference_worker_sha256'],
                    inference_code_sha256=self.inference_code_sha256,
                    inference_profile=self.inference_profile,
                    calibrated=False, trading_skill='UNMEASURED', execution_authority=False,
                    language_training_scope=body['language'].split('-')[0].lower() in LANGUAGES,
                    scope='document sentiment; not factual credibility, insider identity or token-return probability')
                row['annotation_id'] = digest(row)
                await disk(self._commit_window,cache_id,cache,cache_hit,inference_metric,row,identity,cursor)
                rows.append(row)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Worker failures never erase raw input or print text/credentials.
                self.last_error = type(exc).__name__
                # Count completed work even if its result transaction rolled
                # back; the raw input remains pending for actual retry.
                await disk(self._failed_window,identity,inference_metric,cursor)
        return rows

    @staticmethod
    def _validate_window(result, offset):
        start, stop, total = (result.get(k) for k in ('token_start', 'token_stop', 'token_count'))
        if (any(type(n) is not int for n in (start, stop, total)) or start != offset
                or not 0 <= start <= stop <= total or (stop == start and total != 0)
                or stop-start > 510):
            raise ValueError('invalid token window')
        scores = result.get('raw_model_scores', {})
        if (set(scores) != {'positive', 'negative', 'neutral'} or
                any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())
                or abs(sum(scores.values())-1) > .0001):
            raise ValueError('invalid raw model scores')

    def acknowledge(self, records, ref):
        if not ref:
            raise ValueError('canonical annotation reference required')
        with self.lock, self.db:
            for record in records:
                raw = {k: v for k, v in record.items() if k not in TRANSPORT_FIELDS}
                found = self.db.execute('SELECT body,input_id,ref FROM results WHERE id=?',
                                        (record.get('annotation_id'),)).fetchone()
                if record.get('_miner', MINER) != MINER or found is None or json.loads(found[0]) != raw:
                    raise ValueError('annotation differs from retained offline result')
                if found[2] is not None:
                    continue
                self.db.execute('UPDATE results SET ref=?,available=? WHERE id=?',
                                (ref, time.time(), record['annotation_id']))
                self.db.execute('UPDATE inputs SET offset=?,complete=? WHERE id=?',
                    (record['token_stop'], int(record['token_stop'] == record['token_count']), found[1]))

    def report(self):
        read_started=time.time()
        began=time.perf_counter()
        with self.report_lock,self.report_db:
            # One committed snapshot across all counters, even if intake/ACK
            # commits while this potentially long read is in progress.
            self.report_db.execute('BEGIN')
            groups=self.report_db.execute('SELECT complete,origin,count(*),min(created) '
                'FROM inputs GROUP BY complete,origin').fetchall()
            if any(flag not in (0,1) for flag,_,_,_ in groups):
                raise ValueError('invalid queue completion flag')
            total=sum(row[2] for row in groups)
            complete=sum(row[2] for row in groups if row[0]==1)
            pending=[row for row in groups if row[0]==0]
            oldest=min((row[3] for row in pending),default=None)
            lanes={origin:{'pending':count,'oldest_pending_seconds':max(0,time.time()-at)}
                for _,origin,count,at in pending}
            retained,acked=self.report_db.execute('SELECT count(*),count(ref) FROM results').fetchone()
            replay_offset=self.report_db.execute('SELECT offset FROM replay').fetchone()[0]
            metrics=dict(self.report_db.execute('SELECT name,value FROM metrics'))
            cache_entries=self.report_db.execute('SELECT count(*) FROM inference_cache').fetchone()[0]
        return {'status': 'OFFLINE_RESEARCH' if self.worker else 'NOT_CONFIGURED',
            'model_revision': REVISION, 'inference_profile': self.inference_profile,
            'inputs': total, 'completed_inputs': complete or 0,
            'pending_inputs': total-(complete or 0), 'oldest_pending_seconds': max(0, time.time()-oldest) if oldest else None,
            'retained_windows': retained, 'canonical_windows': acked,
            'awaiting_canonical_ack': retained-acked, 'last_error_type': self.last_error,
            'last_intake_error_type': self.last_intake_error, 'intake_failures_this_run': self.intake_failures,
            'replay_error_type': self.replay_error, 'replayed_canonical_bytes': replay_offset,
            'last_replay_seconds': self.last_replay_seconds,
            'last_fetch_profile': self.last_fetch_profile,
            'database_read_started_at':read_started,
            'database_read_elapsed_seconds':max(0,time.perf_counter()-began),
            'queue_lanes': lanes, 'scheduling': '2 fresh live : 1 age-fair live : 1 replay; unused capacity shared; no input expiration',
            'inference_cache_entries': cache_entries, 'inference_metrics': metrics,
            'source_ledger_backlog_bytes': max(0, self.ledger_path.stat().st_size-replay_offset)
                if self.ledger_path and self.ledger_path.exists() else None,
            'language_training_scope': LANGUAGES, 'trading_skill': 'UNMEASURED',
            'execution_authority': False, 'calibrated': False,
            'scope': 'full token windows of retained source inputs; intake failures and resource backlog remain coverage debt'}

    def observation(self, annotation_id, *, as_of):
        with self.observation_lock:
            row = self.observation_db.execute('SELECT body,ref,available FROM results WHERE id=? AND '
                'ref IS NOT NULL AND available<=?', (annotation_id, as_of)).fetchone()
        return {**json.loads(row[0]), 'source_ref_annotation': row[1], 'available_at': row[2]} if row else None

    def register(self, pool):
        from src.research.data_miners import Enriches, MinerSpec
        return {MINER: pool.register(MinerSpec(MINER, Enriches.NARRATIVE, 2,
            endpoint='local:offline_pinned_onnx',
            resource_class='LOCAL_CPU',
            detail='all-source durable sentiment windows; uncalibrated research only'), self.fetch)}

    async def stop(self):
        if self.worker:
            await self.worker.stop()

    def close(self):
        with self.lock:
            with self.report_lock,self.observation_lock:
                self.report_db.close()
                self.observation_db.close()
                self.db.close()


def worker_main(root, *, intra_op_threads=1):
    """Only pinned public weights/tokenizer data; no remote model code/downloads."""
    profile = inference_profile(intra_op_threads)
    root = Path(root)
    for name, expected in FILES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('pinned model integrity mismatch')
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = profile['intra_op_threads']
    options.inter_op_num_threads = profile['inter_op_threads']
    model = ort.InferenceSession(str(root/'onnx/model_quantized.onnx'), options,
                                 providers=['CPUExecutionProvider'])
    tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    labels = json.loads((root/'config.json').read_text())['id2label']
    for line in sys.stdin:
        try:
            request = json.loads(line)
            encoded = tokenizer.encode(request['text'], add_special_tokens=False)
            offset, total = request['offset'], len(encoded.ids)
            if type(offset) is not int or not 0 <= offset <= total:
                raise ValueError('invalid window offset')
            stop = min(total, offset+510)
            ids = [tokenizer.token_to_id('[CLS]'), *encoded.ids[offset:stop], tokenizer.token_to_id('[SEP]')]
            inputs = {'input_ids': np.asarray([ids], dtype=np.int64),
                      'attention_mask': np.ones((1, len(ids)), dtype=np.int64)}
            logits = model.run(None, {i.name: inputs[i.name] for i in model.get_inputs()})[0][0]
            scores = np.exp(logits-logits.max())
            scores /= scores.sum()
            result = {'token_start': offset, 'token_stop': stop, 'token_count': total,
                'char_start': encoded.offsets[offset][0] if stop > offset else 0,
                'char_stop': encoded.offsets[stop-1][1] if stop > offset else 0,
                'raw_model_scores': {labels[str(i)]: float(v) for i, v in enumerate(scores)}}
            SentimentResearch._validate_window(result, offset)
            response = {'ok': True, 'result': result, 'inference_profile': profile}
        except Exception as exc:
            response = {'ok': False, 'error_type': type(exc).__name__}
        print(canonical(response), flush=True)


if __name__ == '__main__':
    if len(sys.argv) not in (3, 5) or sys.argv[1] != '--worker' or (len(sys.argv) == 5 and sys.argv[3] != '--threads'):
        raise SystemExit('offline worker arguments required')
    worker_main(sys.argv[2], intra_op_threads=int(sys.argv[4]) if len(sys.argv) == 5 else 1)
