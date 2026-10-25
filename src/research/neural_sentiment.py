"""Durable all-source document sentiment research, never order authority.

Inputs are already canonical public/authorized observations. Inference runs
offline in a separate, credential-free Python process with pinned public model
bytes. Every token window is queued; resource batches never trim the universe.
Raw softmax scores are neither calibrated trading probabilities nor truth tests.
"""
from __future__ import annotations

import asyncio
import hashlib
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
    def __init__(self, python, model_root):
        self.python, self.model_root = str(python), str(model_root)
        self.process = None

    async def infer(self, text, offset):
        if self.process is None or self.process.returncode is not None:
            environment = {k: v for k, v in os.environ.items()
                           if k.upper() in {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
            environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            self.process = await asyncio.create_subprocess_exec(self.python, '-m', __name__,
                '--worker', self.model_root, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                env=environment, limit=1024 * 1024)
        try:
            self.process.stdin.write((canonical({'text': text, 'offset': offset})+'\n').encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), 30)
            result = json.loads(line)
            if not result.get('ok'):
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


class SentimentResearch:
    def __init__(self, path, *, python=None, model_root=None, worker=None, ledger_path=None):
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
        ''')
        self.db.execute('INSERT OR IGNORE INTO replay VALUES(1,0,?,0)', ('0'*64,))
        self.db.commit()
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.lock = threading.RLock()
        self.worker = worker or (OfflineWorker(python, model_root) if python and model_root else None)
        self.last_error = None
        self.last_intake_error = None
        self.intake_failures = 0
        self.replay_error = None

    def enqueue(self, payload, *, source_job, source_ref, tokens=()):
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
        with self.lock, self.db:
            previous = self.db.execute('SELECT body FROM inputs WHERE id=?', (identity,)).fetchone()
            if previous and json.loads(previous[0])['text_sha256'] != body['text_sha256']:
                raise ValueError('canonical source text changed')
            self.db.execute('INSERT OR IGNORE INTO inputs(id,body,created) VALUES(?,?,?)',
                            (identity, canonical(body), time.time()))
        return identity

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
                        self.enqueue(payload, source_job=None,
                            source_ref=f"evidence:{row['hash']}:{start}", tokens=tokens)
                offset = handle.tell()
        # Inputs are idempotent if a crash precedes this single checkpoint.
        with self.lock, self.db:
            self.db.execute('UPDATE replay SET offset=?,previous=?,sequence=? WHERE id=1',
                            (offset, previous, sequence))
        self.replay_error = None

    async def fetch(self):
        if self.worker is None:
            return []
        try:
            await asyncio.to_thread(self.reconcile)
        except Exception as exc:
            self.replay_error = type(exc).__name__
        with self.lock:
            pending = self.db.execute('SELECT body FROM results WHERE ref IS NULL ORDER BY rowid LIMIT 4').fetchall()
            jobs = self.db.execute('SELECT id,body,offset FROM inputs WHERE complete=0 '
                'AND id NOT IN (SELECT input_id FROM results WHERE ref IS NULL) '
                'ORDER BY served,created,id LIMIT 4').fetchall()
        if pending:
            return [json.loads(row[0]) for row in pending]
        rows = []
        for identity, raw, offset in jobs:
            body = json.loads(raw)
            try:
                result = await self.worker.infer(body['text'], offset)
                self._validate_window(result, offset)
                row = {key: value for key, value in body.items() if key != 'text'}
                row.update(schema=SCHEMA, provider='offline_neural_sentiment',
                    annotation_kind='document_sentiment_window', input_id=identity,
                    **result, created_at=time.time(), weights_sha256=FILES['onnx/model_quantized.onnx'],
                    worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    calibrated=False, trading_skill='UNMEASURED', execution_authority=False,
                    language_training_scope=body['language'].split('-')[0].lower() in LANGUAGES,
                    scope='document sentiment; not factual credibility, insider identity or token-return probability')
                row['annotation_id'] = digest(row)
                with self.lock, self.db:
                    self.db.execute('INSERT INTO results VALUES(?,?,?,NULL,NULL)',
                                    (row['annotation_id'], identity, canonical(row)))
                    self.db.execute('UPDATE inputs SET served=? WHERE id=?', (time.time(), identity))
                rows.append(row)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Worker failures never erase raw input or print text/credentials.
                self.last_error = type(exc).__name__
                with self.lock, self.db:
                    self.db.execute('UPDATE inputs SET served=? WHERE id=?', (time.time(), identity))
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
        with self.lock:
            total, complete = self.db.execute('SELECT count(*),sum(complete) FROM inputs').fetchone()
            retained, acked = self.db.execute('SELECT count(*),count(ref) FROM results').fetchone()
            oldest = self.db.execute('SELECT min(created) FROM inputs WHERE complete=0').fetchone()[0]
            replay_offset = self.db.execute('SELECT offset FROM replay').fetchone()[0]
        return {'status': 'OFFLINE_RESEARCH' if self.worker else 'NOT_CONFIGURED',
            'model_revision': REVISION, 'inputs': total, 'completed_inputs': complete or 0,
            'pending_inputs': total-(complete or 0), 'oldest_pending_seconds': max(0, time.time()-oldest) if oldest else None,
            'retained_windows': retained, 'canonical_windows': acked,
            'awaiting_canonical_ack': retained-acked, 'last_error_type': self.last_error,
            'last_intake_error_type': self.last_intake_error, 'intake_failures_this_run': self.intake_failures,
            'replay_error_type': self.replay_error, 'replayed_canonical_bytes': replay_offset,
            'source_ledger_backlog_bytes': max(0, self.ledger_path.stat().st_size-replay_offset)
                if self.ledger_path and self.ledger_path.exists() else None,
            'language_training_scope': LANGUAGES, 'trading_skill': 'UNMEASURED',
            'execution_authority': False, 'calibrated': False,
            'scope': 'full token windows of retained source inputs; intake failures and resource backlog remain coverage debt'}

    def observation(self, annotation_id, *, as_of):
        with self.lock:
            row = self.db.execute('SELECT body,ref,available FROM results WHERE id=? AND '
                'ref IS NOT NULL AND available<=?', (annotation_id, as_of)).fetchone()
        return {**json.loads(row[0]), 'source_ref_annotation': row[1], 'available_at': row[2]} if row else None

    def register(self, pool):
        from src.research.data_miners import Enriches, MinerSpec
        return {MINER: pool.register(MinerSpec(MINER, Enriches.NARRATIVE, 2,
            endpoint='local:offline_pinned_onnx',
            detail='all-source durable sentiment windows; uncalibrated research only'), self.fetch)}

    async def stop(self):
        if self.worker:
            await self.worker.stop()

    def close(self):
        with self.lock:
            self.db.close()


def worker_main(root):
    """Only pinned public weights/tokenizer data; no remote model code/downloads."""
    root = Path(root)
    for name, expected in FILES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('pinned model integrity mismatch')
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
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
            response = {'ok': True, 'result': result}
        except Exception as exc:
            response = {'ok': False, 'error_type': type(exc).__name__}
        print(canonical(response), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('offline worker arguments required')
    worker_main(sys.argv[2])
