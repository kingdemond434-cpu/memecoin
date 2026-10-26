"""Durable public-source intake, independent of the research consumer's speed.

Accepted observations remain until an exact leased body has a canonical work
reference. This is at-least-once delivery, not a claim of gap-free upstream
coverage. SQLite/disk failures backpressure the producer instead of evicting.
"""
import hashlib
import json
import math
import shutil
import sqlite3
import threading
from pathlib import Path

from src.collectors.event_source import Event, SourceClass


def encode(event):
    body = event.to_dict()
    for key in ('observation_lag', 'content_hash', 'schema_version'):
        body.pop(key)
    for key in ('source_at', 'observed_at'):
        if not math.isfinite(body[key]):
            raise ValueError('finite source clocks required')
    return json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)


def decode(body):
    value = json.loads(body)
    value['source_class'] = SourceClass(value['source_class'])
    return Event(**value)


def seal(body):
    return hashlib.sha256(body.encode()).hexdigest()


def identity(event, body):
    metadata = event.metadata or {}
    durable = metadata.get('onion_observation_id') or metadata.get('onion_directory_observation_id')
    return seal(json.dumps([event.source_id, durable])) if isinstance(durable,str) and durable else seal(body)


class SourceJournal:
    def __init__(self, path):
        self.path = Path(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS intake(
            id TEXT PRIMARY KEY, source TEXT NOT NULL, body TEXT, body_sha256 TEXT NOT NULL, accepted REAL NOT NULL,
            first_delivery REAL, due REAL NOT NULL, generation INTEGER NOT NULL DEFAULT 0,
            canonical_job TEXT, canonical_ref TEXT, acknowledged REAL);
          CREATE INDEX IF NOT EXISTS intake_ready ON intake(due,source,accepted) WHERE body IS NOT NULL;
          CREATE INDEX IF NOT EXISTS intake_source ON intake(source,due,accepted) WHERE body IS NOT NULL;
          CREATE INDEX IF NOT EXISTS intake_age ON intake(source,accepted) WHERE body IS NOT NULL;
          CREATE TABLE IF NOT EXISTS source_service(source TEXT PRIMARY KEY,ticket INTEGER NOT NULL,pending INTEGER NOT NULL);
          CREATE INDEX IF NOT EXISTS source_turn ON source_service(ticket);
          CREATE TABLE IF NOT EXISTS intake_counts(id INTEGER PRIMARY KEY CHECK(id=1),accepted INTEGER NOT NULL,acknowledged INTEGER NOT NULL);
          INSERT OR IGNORE INTO intake_counts VALUES(1,0,0);
        ''')

    def append(self, events, now):
        # Serialize/validate the entire batch before beginning a transaction.
        # The caller retains this same batch if either validation or IO fails.
        if not math.isfinite(now):
            raise ValueError('finite admission clock required')
        rows = [(identity(event,body), event.source_id, body, seal(body), now, now)
                for event in events for body in (encode(event),)]
        with self.lock, self.db:
            for event_id, source, body, body_hash, accepted, due in rows:
                inserted = self.db.execute('INSERT OR IGNORE INTO intake(id,source,body,body_sha256,accepted,due) VALUES(?,?,?,?,?,?)',
                                          (event_id, source, body, body_hash, accepted, due))
                if inserted.rowcount:
                    self.db.execute('INSERT INTO source_service VALUES(?,0,1) '
                        'ON CONFLICT(source) DO UPDATE SET pending=pending+1', (source,))
                    self.db.execute('UPDATE intake_counts SET accepted=accepted+1 WHERE id=1')
        return len(rows)

    def lease(self, now, lease_seconds=300):
        if not math.isfinite(now) or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError('finite positive lease required')
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            source = self.db.execute('SELECT source FROM source_service s WHERE pending>0 AND EXISTS('
                'SELECT 1 FROM intake i WHERE i.source=s.source AND i.body IS NOT NULL AND i.due<=?) '
                'ORDER BY ticket,source LIMIT 1', (now,)).fetchone()
            if source is None:
                return None
            row = self.db.execute('SELECT id,body,accepted,first_delivery,generation,source,body_sha256 '
                'FROM intake WHERE source=? AND body IS NOT NULL AND due<=? '
                'ORDER BY due,accepted,rowid LIMIT 1', (source[0],now)).fetchone()
            if row is None:
                return None
            event_id, body, accepted, first, generation, source, body_hash = row
            if seal(body) != body_hash:
                raise ValueError('source journal body integrity mismatch')
            event = decode(body)
            if identity(event, body) != event_id:
                raise ValueError('source journal identity mismatch')
            first = now if first is None else first
            self.db.execute('UPDATE intake SET first_delivery=?,generation=?,due=? WHERE id=?',
                            (first, generation+1, now+lease_seconds, event_id))
            self.db.execute('UPDATE source_service SET ticket=(SELECT COALESCE(MAX(ticket),0)+1 '
                            'FROM source_service) WHERE source=?', (source,))
            receipt = {'id': event_id, 'body_sha256': body_hash, 'accepted_at': accepted,
                       'first_delivered_at': first}
            return event, receipt, generation+1

    def _match(self, event, receipt, generation):
        body = encode(event)
        event_id = identity(event, body)
        if event_id != receipt['id'] or seal(body) != receipt['body_sha256']:
            raise ValueError('source journal ACK body mismatch')
        row = self.db.execute('SELECT body,generation,accepted,first_delivery FROM intake WHERE id=?',
                              (event_id,)).fetchone()
        if (not row or row[0] is None or row[1] != generation or
                seal(row[0]) != receipt['body_sha256'] or
                row[2] != receipt['accepted_at'] or row[3] != receipt['first_delivered_at']):
            raise ValueError('source journal lease is stale or already acknowledged')
        return event_id

    def acknowledge(self, event, receipt, generation, job, reference, now):
        if not math.isfinite(now) or now < receipt['first_delivered_at']:
            raise ValueError('canonical ACK cannot precede delivery')
        if not isinstance(job, str) or not job.startswith('work-') or not isinstance(reference, str) or not reference:
            raise ValueError('canonical research work and evidence reference required')
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            identity = self._match(event, receipt, generation)
            self.db.execute('UPDATE intake SET body=NULL,canonical_job=?,canonical_ref=?,acknowledged=? WHERE id=?',
                            (job, reference, now, identity))
            self.db.execute('UPDATE intake_counts SET acknowledged=acknowledged+1 WHERE id=1')
            self.db.execute('UPDATE source_service SET pending=pending-1 WHERE source=?', (event.source_id,))

    def retry(self, event, receipt, generation, now):
        if not math.isfinite(now):
            raise ValueError('finite retry deadline required')
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            identity = self._match(event, receipt, generation)
            self.db.execute('UPDATE intake SET due=? WHERE id=?', (now+5, identity))

    def report(self, now):
        with self.lock:
            counts = self.db.execute('SELECT accepted,acknowledged FROM intake_counts WHERE id=1').fetchone()
            sources = self.db.execute('SELECT source,pending,(SELECT min(accepted) FROM intake i '
                'WHERE i.source=s.source AND i.body IS NOT NULL) FROM source_service s WHERE pending>0').fetchall()
            due = self.db.execute('SELECT count(*) FROM intake WHERE body IS NOT NULL AND due<=?', (now,)).fetchone()[0]
        return {'status': 'DURABLE_AT_LEAST_ONCE', 'accepted_observations': counts[0],
            'pending_events': counts[0]-counts[1], 'canonical_acknowledged': counts[1],
            'due_events': due, 'pending_by_source': {row[0]:row[1] for row in sources},
            'oldest_pending_age_by_source': {row[0]:max(0.,now-row[2]) for row in sources},
            'database_bytes': self.path.stat().st_size,
            'wal_bytes': Path(str(self.path)+'-wal').stat().st_size if Path(str(self.path)+'-wal').exists() else 0,
            'disk_free_bytes': shutil.disk_usage(self.path.parent).free,
            'evicted_events': 0, 'upstream_gap_free': False}

    def close(self):
        with self.lock:
            self.db.close()
