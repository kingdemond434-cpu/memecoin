"""Durable, fair native honeypot research; canonical ACK before consumption."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import zlib

from src.research.data_miners import Enriches, MinerSpec
from src.research.fork_simulation import SCHEMA, ForkUnavailable, acquire, digest, run_isolated

MINER = 'chain:local_fork_roundtrip'
TRANSPORT_FIELDS = frozenset({'_miner', '_enriches', '_endpoint', '_fetched_at',
                              '_request_started_at', '_scheduled_at'})


def reviewed_native_target(event):
    """Candidate identity only; actual bank/owner/route checks still follow."""
    from solders.pubkey import Pubkey
    from src.chains.launchlab import LAUNCHLAB_PROGRAM
    from src.chains.meteora_dbc import DBC_PROGRAM
    from src.chains.pump_route import WSOL_MINT
    venues={'raydium_launchlab':LAUNCHLAB_PROGRAM,'meteora_dbc':DBC_PROGRAM}
    if (event.get('type') != 'token_launch' or event.get('venue') not in venues
            or event.get('program') != venues[event['venue']] or event.get('quote_mint') != WSOL_MINT):
        return None
    for key in ('token', 'pool'):
        Pubkey.from_string(event[key])
    return {'token': event['token'], 'venue': event['venue'], 'pool': event['pool'],
            'quote_budget': 1_000_000}


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
          CREATE TABLE IF NOT EXISTS fork_venue_service(venue TEXT PRIMARY KEY,last_ticket INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS native_target_sources(ref TEXT PRIMARY KEY,job TEXT NOT NULL,
            available REAL NOT NULL,ingested REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS native_sources_by_job ON native_target_sources(job,ref);
        ''')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(jobs)')}
        if 'arrived' not in columns:
            self.db.execute('ALTER TABLE jobs ADD COLUMN arrived INTEGER NOT NULL DEFAULT 0')
            self.db.execute('UPDATE jobs SET arrived=rowid')
        if 'first_seen' not in columns:
            # Legacy retry deadlines do not establish original queue arrival.
            self.db.execute('ALTER TABLE jobs ADD COLUMN first_seen REAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS fork_dispatch '
                        '(id INTEGER PRIMARY KEY CHECK(id=1),cursor INTEGER NOT NULL,ticket INTEGER NOT NULL)')
        self.db.execute('INSERT OR IGNORE INTO fork_dispatch SELECT 1,0,COALESCE(MAX(arrived),0) FROM jobs')
        self.db.execute('CREATE INDEX IF NOT EXISTS untried_arrival ON jobs(arrived DESC,id) WHERE attempts=0')
        self.db.execute('CREATE INDEX IF NOT EXISTS retry_deadline ON jobs(due,id) WHERE attempts>0')
        self.db.commit()

    def _enqueue_targets_locked(self, targets, now):
        admitted = []
        for target in targets:
            body = json.dumps(target, sort_keys=True, allow_nan=False)
            identity = digest(target)
            inserted = self.db.execute('INSERT OR IGNORE INTO jobs(id,body,due,attempts,arrived,first_seen) '
                'VALUES(?,?,?,0,(SELECT ticket+1 FROM fork_dispatch WHERE id=1),?)',
                (identity, body, now, now))
            if inserted.rowcount:
                self.db.execute('UPDATE fork_dispatch SET ticket=ticket+1 WHERE id=1')
            admitted.append((identity, bool(inserted.rowcount)))
        return admitted

    def enqueue_native_launch(self, event, *, source_ref, available_at):
        """Persist an unexecuted reviewed target after its canonical source append."""
        target = reviewed_native_target(event)
        if target is None:
            return None
        if not isinstance(source_ref, str) or not re.fullmatch('native-launch:[0-9a-f]{64}', source_ref):
            raise ValueError('canonical native-launch hash reference required')
        if type(available_at) not in (int, float) or not math.isfinite(available_at):
            raise ValueError('finite canonical source availability required')
        now = time.time()
        with self.lock, self.db:
            identity, inserted = self._enqueue_targets_locked([target], now)[0]
            previous = self.db.execute('SELECT job,available FROM native_target_sources WHERE ref=?',
                                       (source_ref,)).fetchone()
            if previous is not None and previous != (identity, available_at):
                raise ValueError('canonical native source identity or availability changed')
            self.db.execute('INSERT OR IGNORE INTO native_target_sources VALUES(?,?,?,?)',
                            (source_ref, identity, available_at, now))
        return {'target_id': identity, 'inserted': inserted}

    def _reserve_job(self, now):
        """Caller holds the write transaction; tickets survive restart/clock correction."""
        cursor = self.db.execute('SELECT cursor FROM fork_dispatch WHERE id=1').fetchone()[0]
        preferred = ('FRESH_UNTRIED', 'FRESH_UNTRIED', 'OLDEST_UNTRIED', 'DUE_RETRY')[cursor % 4]
        plans = {'FRESH_UNTRIED': ('attempts=0', 'arrived DESC,id'),
                 'OLDEST_UNTRIED': ('attempts=0', 'arrived,id'),
                 'DUE_RETRY': ('attempts>0', 'due,id')}
        for lane in dict.fromkeys((preferred, 'OLDEST_UNTRIED', 'DUE_RETRY')):
            predicate, order = plans[lane]
            venue = "COALESCE(NULLIF(json_extract(j.body,'$.venue'),''),'UNKNOWN')"
            job = self.db.execute('SELECT j.id,j.body,j.first_seen,'+venue+' FROM jobs j '
                'LEFT JOIN fork_venue_service v ON v.venue='+venue+' WHERE due<=? AND '+predicate+
                ' ORDER BY COALESCE(v.last_ticket,0),'+order+' LIMIT 1', (now,)).fetchone()
            if job:
                self.db.execute('UPDATE jobs SET due=?,attempts=attempts+1 WHERE id=?', (now+300, job[0]))
                self.db.execute('UPDATE fork_dispatch SET cursor=? WHERE id=1', ((cursor+1) % 4,))
                self.db.execute('INSERT INTO fork_venue_service VALUES(?, '
                    '(SELECT COALESCE(MAX(last_ticket),0)+1 FROM fork_venue_service)) '
                    'ON CONFLICT(venue) DO UPDATE SET last_ticket=excluded.last_ticket', (job[3],))
                sources = [{'source_ref': ref, 'available_at': at, 'ingested_at': ingested}
                    for ref, at, ingested in self.db.execute(
                        'SELECT ref,available,ingested FROM native_target_sources WHERE job=? ORDER BY ref', (job[0],))]
                return (*job, lane, sources)
        return None

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
            self._enqueue_targets_locked(targets(), now)
            pending = self.db.execute('SELECT body FROM results WHERE ref IS NULL ORDER BY created LIMIT 1').fetchone()
            if pending:
                return [json.loads(pending[0])]
            job = self._reserve_job(now)
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
            'dispatch_lane': job[4], 'dispatch_venue': job[3], 'target_first_queued_at': job[2],
            'native_source_refs': job[5],
            'queue_age_at_attempt_seconds': max(0., now-job[2]) if job[2] is not None else None,
            'queue_clock_regressed': now < job[2] if job[2] is not None else None,
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
                raw = {key: value for key, value in row.items() if key not in TRANSPORT_FIELDS}
                if row.get('_miner', MINER) != MINER or original is None or json.loads(original[0]) != raw:
                    raise ValueError('canonical fork result does not match archived result')
                self.db.execute('UPDATE results SET ref=COALESCE(ref,?),available=COALESCE(available,?) WHERE id=?',
                                (ref, time.time(), row['probe_id']))

    def latest(self, token, *, as_of):
        with self.lock:
            row = self.db.execute('SELECT body,ref,available FROM results WHERE ref IS NOT NULL AND '
                "json_extract(body,'$.token')=? AND available<=? ORDER BY available DESC LIMIT 1", (token, as_of)).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        if any(source['available_at'] > as_of or source['ingested_at'] > as_of
               for source in result.get('native_source_refs', [])):
            return None
        return {**result, 'source_ref': row[1], 'available_at': row[2]}

    def report(self):
        with self.lock:
            jobs = self.db.execute('SELECT count(*),sum(attempts) FROM jobs').fetchone()
            statuses = dict(self.db.execute("SELECT json_extract(body,'$.status'),count(*) FROM results WHERE ref IS NOT NULL GROUP BY 1"))
            pending = self.db.execute('SELECT count(*) FROM results WHERE ref IS NULL').fetchone()[0]
            venues = {}
            for venue, count, attempts, untried in self.db.execute(
                    "SELECT json_extract(body,'$.venue'),count(*),sum(attempts),sum(attempts=0) FROM jobs GROUP BY 1"):
                venues[venue or 'UNKNOWN'] = {'targets': count, 'attempts': attempts,
                    'untried_targets': untried, 'canonical_dispositions': {}}
            for venue, status, count in self.db.execute("SELECT json_extract(body,'$.target.venue'),"
                    "json_extract(body,'$.status'),count(*) FROM results WHERE ref IS NOT NULL GROUP BY 1,2"):
                row = venues.setdefault(venue or 'UNKNOWN', {'targets': 0, 'attempts': 0,
                    'untried_targets': 0, 'canonical_dispositions': {}})
                row['canonical_dispositions'][status] = count
            for venue in ('pump_curve', 'pumpswap', 'raydium_launchlab','meteora_dbc'):
                venues.setdefault(venue, {'targets': 0, 'attempts': 0,
                    'untried_targets': 0, 'canonical_dispositions': {}})
            for row in venues.values():
                row['coverage_state'] = ('OBSERVED_LOCAL_ROUNDTRIPS' if row['canonical_dispositions'].get(
                    'ROUNDTRIP_SUCCEEDED_IN_FORK', 0) else 'CANONICAL_DISPOSITIONS_ONLY' if row[
                    'canonical_dispositions'] else 'TARGETS_PENDING' if row['targets'] else 'NO_TARGETS_OBSERVED')
            venue_service = dict(self.db.execute('SELECT venue,last_ticket FROM fork_venue_service'))
            source_links, source_targets = self.db.execute(
                'SELECT count(*),count(DISTINCT job) FROM native_target_sources').fetchone()
        return {'status': 'LOCAL_FORK_RESEARCH', 'targets': jobs[0], 'attempts': jobs[1] or 0,
                'canonical_dispositions': statuses, 'awaiting_canonical_ack': pending,
                'by_venue': venues,
                'scheduling': '2 fresh untried : 1 oldest untried : 1 due retry; empty lanes share capacity',
                'venue_scheduling': 'least recently serviced eligible venue within each lane; persistent monotone tickets',
                'venue_service_tickets': venue_service,
                'canonical_native_source_links': source_links, 'canonical_native_linked_targets': source_targets,
                'supported_adapters': ['pump_curve', 'pumpswap', 'raydium_launchlab','meteora_dbc'],
                'adapter_subsets': {'raydium_launchlab': 'funding status, SPL/Token-2022 base, WSOL quote; transfer hooks/other layouts unknown',
                    'meteora_dbc':'ordinary unmigrated VirtualPool/PoolConfig, SPL/Token-2022 base, WSOL quote; hooks/other quotes/layouts unknown'},
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
