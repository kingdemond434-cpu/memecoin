"""Read-only source intake audit. Never emits source bodies or identifiers."""
import argparse
import json
import math
import sqlite3
import time
from pathlib import Path

from src.collectors.source_journal import decode, identity, seal


def audit(path, *, now=None, sample_limit=100):
    if not isinstance(sample_limit, int) or not 1 <= sample_limit <= 10000:
        raise ValueError('sample_limit must be between 1 and 10000')
    now = time.time() if now is None else now
    if not math.isfinite(now):
        raise ValueError('finite audit clock required')
    # mode=ro prevents accidental creation; a single read transaction keeps all
    # reconciliation queries on the same WAL snapshot while producers continue.
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        integrity = db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        total, pending, acked, oldest = db.execute('SELECT count(*), '
            'coalesce(sum(body IS NOT NULL),0), coalesce(sum(acknowledged IS NOT NULL),0), '
            'min(CASE WHEN body IS NOT NULL THEN accepted END) FROM intake').fetchone()
        counters = db.execute('SELECT accepted,acknowledged FROM intake_counts WHERE id=1').fetchone()
        malformed = db.execute("SELECT count(*) FROM intake WHERE "
            "(body IS NULL AND (acknowledged IS NULL OR canonical_job IS NULL OR "
            "canonical_job NOT LIKE 'work-%' OR canonical_ref IS NULL OR canonical_ref='')) OR "
            "(body IS NOT NULL AND (acknowledged IS NOT NULL OR canonical_job IS NOT NULL OR canonical_ref IS NOT NULL)) OR "
            "(acknowledged IS NOT NULL AND (first_delivery IS NULL OR acknowledged<first_delivery))").fetchone()[0]
        mismatches = db.execute('WITH actual AS (SELECT source,count(*) n FROM intake '
            'WHERE body IS NOT NULL GROUP BY source) SELECT count(*) FROM '
            '(SELECT s.source FROM source_service s LEFT JOIN actual a ON s.source=a.source '
            'WHERE s.pending != coalesce(a.n,0) UNION ALL SELECT a.source FROM actual a '
            'LEFT JOIN source_service s ON a.source=s.source WHERE s.source IS NULL)').fetchone()[0]
        sampled = failures = 0
        for event_id, body, digest in db.execute('SELECT id,body,body_sha256 FROM intake '
                'WHERE body IS NOT NULL ORDER BY accepted,rowid LIMIT ?', (sample_limit,)):
            sampled += 1
            try:
                valid = seal(body) == digest and identity(decode(body), body) == event_id
            except (ValueError, TypeError, KeyError):
                valid = False
            failures += not valid
        ready = db.execute('SELECT count(*) FROM intake WHERE body IS NOT NULL AND due<=?', (now,)).fetchone()[0]
        checks = {'sqlite_quick_check': integrity, 'counters_match': counters == (total, acked),
                  'receipt_states_valid': malformed == 0, 'source_counts_match': mismatches == 0,
                  'sampled_pending_bodies_valid': failures == 0}
        return {'observed_at': now, 'status': 'PASS' if all(checks.values()) else 'FAIL',
                'checks': checks, 'accepted': total, 'acknowledged': acked, 'pending': pending,
                'ready': ready, 'oldest_pending_age_seconds': None if oldest is None else max(0, now-oldest),
                'sampled_pending_bodies': sampled, 'invalid_sampled_bodies': failures,
                'invalid_receipt_states': malformed, 'source_count_mismatches': mismatches,
                'canonical_references_externally_verified': False,
                'scope': 'journal snapshot consistency and bounded oldest-body sample; not canonical-chain verification or upstream coverage'}
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--sample-limit', type=int, default=100)
    args = parser.parse_args()
    try:
        result = audit(args.database, sample_limit=args.sample_limit)
    except Exception as exc:
        # Private SQLite errors may contain paths; publish the exception type only.
        result = {'status': 'ERROR', 'error_type': type(exc).__name__}
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
