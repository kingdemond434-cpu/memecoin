"""Durable, fair native honeypot research; canonical ACK before consumption."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import threading
import time
import zlib

from src.research.data_miners import Enriches, MinerSpec
from src.research.fork_simulation import SCHEMA, ForkUnavailable, acquire, digest, run_isolated

MINER = 'chain:local_fork_roundtrip'


class ForkProbe:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=3)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,body TEXT NOT NULL,due REAL NOT NULL,attempts INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS frames(id TEXT PRIMARY KEY,body BLOB NOT NULL,created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS blobs(id TEXT PRIMARY KEY,body BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,job TEXT NOT NULL,body TEXT NOT NULL,created REAL NOT NULL,ref TEXT,available REAL);
          CREATE INDEX IF NOT EXISTS pending ON results(ref,created);
          CREATE INDEX IF NOT EXISTS queue ON jobs(due,id);
          CREATE INDEX IF NOT EXISTS by_token ON results(json_extract(body,'$.token'),available DESC) WHERE ref IS NOT NULL;
        ''')

    def _store_frame(self, frame):
        identity = digest(frame)
        archived = json.loads(json.dumps(frame, allow_nan=False))
        with self.lock, self.db:
            # Large executable images are content-addressed once, rather than
            # repeating megabytes of identical SBF in every sampled bank.
            for value in archived['accounts'].values():
                if value and len(value['data'][0]) > 100_000:
                    encoded = value['data'][0]
                    blob_id = digest(encoded)
                    self.db.execute('INSERT OR IGNORE INTO blobs VALUES(?,?)',
                                    (blob_id, zlib.compress(encoded.encode())))
                    value['data'] = {'base64_blob_sha256': blob_id}
            body = json.dumps(archived, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            self.db.execute('INSERT OR IGNORE INTO frames VALUES(?,?,?)',
                            (identity, zlib.compress(body), time.time()))
        return identity

    def _read_frame(self, identity):
        from src.research.fork_simulation import MAX_FRAME_BYTES
        with self.lock:
            row = self.db.execute('SELECT body FROM frames WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise ForkUnavailable('retained fork input missing')
        decoder = zlib.decompressobj()
        body = decoder.decompress(row[0], MAX_FRAME_BYTES * 2 + 1)
        if len(body) > MAX_FRAME_BYTES * 2 or not decoder.eof or decoder.unused_data:
            raise ForkUnavailable('retained fork input corrupt/oversized')
        frame = json.loads(body)
        for value in frame['accounts'].values():
            if value and isinstance(value.get('data'), dict):
                blob_id = value['data']['base64_blob_sha256']
                with self.lock:
                    blob = self.db.execute('SELECT body FROM blobs WHERE id=?', (blob_id,)).fetchone()
                if blob is None:
                    raise ForkUnavailable('retained SBF blob missing')
                decoder = zlib.decompressobj()
                raw = decoder.decompress(blob[0], MAX_FRAME_BYTES * 2 + 1)
                if len(raw) > MAX_FRAME_BYTES * 2 or not decoder.eof or decoder.unused_data:
                    raise ForkUnavailable('retained SBF blob corrupt/oversized')
                encoded = raw.decode()
                if digest(encoded) != blob_id:
                    raise ForkUnavailable('retained SBF blob hash mismatch')
                value['data'] = [encoded, 'base64']
        if digest(frame) != identity:
            raise ForkUnavailable('retained fork input hash mismatch')
        return frame

    async def fetch(self, rpc, targets):
        now = time.time()
        # Batches/cadence bound current resource use; the durable token universe
        # has no top-N or lifetime ceiling. Old and new targets share the queue.
        with self.lock, self.db:
            for target in targets():
                body = json.dumps(target, sort_keys=True, allow_nan=False)
                self.db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,0)', (digest(target), body, now))
            pending = self.db.execute('SELECT body FROM results WHERE ref IS NULL ORDER BY created LIMIT 1').fetchone()
            job = self.db.execute('SELECT id,body FROM jobs WHERE due<=? ORDER BY due,id LIMIT 1', (now,)).fetchone()
            if pending:
                return [json.loads(pending[0])]
            if job:
                self.db.execute('UPDATE jobs SET due=?,attempts=attempts+1 WHERE id=?', (now + 300, job[0]))
        if job is None:
            return []
        identity, target = job[0], json.loads(job[1])
        frame_id = None
        try:
            frame = await asyncio.wait_for(acquire(rpc, target), timeout=50)
            frame_id = await asyncio.to_thread(self._store_frame, frame)
            # Execute the archived, hash-checked input, not a mutable live object.
            frame = await asyncio.to_thread(self._read_frame, frame_id)
            result = await run_isolated(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Provider exception strings may contain credential-bearing URLs.
            result = {'status': 'FORK_ENVIRONMENT_UNKNOWN', 'sell_simulation_pass': None,
                      'reason': str(exc)[:300] if isinstance(exc, ForkUnavailable) else type(exc).__name__}
        row = {**result, 'provider': 'local_svm', 'fork_kind': 'native_roundtrip',
            'schema': SCHEMA, 'token': target['token'], 'target': target,
            'input_sha256': frame_id, 'created_at': time.time(),
            'input_archive': 'native_fork_probe.sqlite:frames',
            'real_transaction_submitted': False, 'execution_authority': False,
            'scope': 'sampled local model; no future sell or profitability guarantee'}
        row['probe_id'] = digest(row)
        with self.lock, self.db:
            self.db.execute('INSERT INTO results VALUES(?,?,?,?,NULL,NULL)',
                (row['probe_id'], identity, json.dumps(row, allow_nan=False), row['created_at']))
        return [row]

    def acknowledge(self, records, ref):
        if not ref:
            raise ValueError('canonical reference required')
        with self.lock, self.db:
            for row in records:
                original = self.db.execute('SELECT body FROM results WHERE id=?', (row.get('probe_id'),)).fetchone()
                if original is None or json.loads(original[0]) != row:
                    raise ValueError('canonical fork result does not match archived result')
                self.db.execute('UPDATE results SET ref=COALESCE(ref,?),available=COALESCE(available,?) WHERE id=?',
                                (ref, time.time(), row['probe_id']))

    def latest(self, token, *, as_of):
        with self.lock:
            row = self.db.execute('SELECT body,ref,available FROM results WHERE ref IS NOT NULL AND '
                "json_extract(body,'$.token')=? AND available<=? ORDER BY available DESC LIMIT 1", (token, as_of)).fetchone()
        if row is None:
            return None
        return {**json.loads(row[0]), 'source_ref': row[1], 'available_at': row[2]}

    def report(self):
        with self.lock:
            jobs = self.db.execute('SELECT count(*),sum(attempts) FROM jobs').fetchone()
            statuses = dict(self.db.execute("SELECT json_extract(body,'$.status'),count(*) FROM results WHERE ref IS NOT NULL GROUP BY 1"))
            pending = self.db.execute('SELECT count(*) FROM results WHERE ref IS NULL').fetchone()[0]
        return {'status': 'LOCAL_FORK_RESEARCH', 'targets': jobs[0], 'attempts': jobs[1] or 0,
                'canonical_dispositions': statuses, 'awaiting_canonical_ack': pending,
                'supported_adapters': ['pump_curve', 'pumpswap'],
                'other_venues': 'UNSUPPORTED_UNKNOWN', 'live_authority': False,
                'input_archive': 'native_fork_probe.sqlite:frames',
                'runtime_parity': 'exact validator parity unverified',
                'scope': 'sequential buy/sell in sampled local SVM; failures are not proven honeypots'}

    def register(self, pool, rpc, targets):
        async def fetch():
            return await self.fetch(rpc, targets)
        return {MINER: pool.register(MinerSpec(MINER, Enriches.SUPPLY_CONTROL, 60,
            endpoint='rpc:getMultipleAccounts + local LiteSVM',
            detail='fair durable native round-trip probes; sampled bank, virtual funds, no submission'), fetch)}

    def close(self):
        with self.lock:
            self.db.close()
