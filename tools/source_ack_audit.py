"""Bounded read-only verification of journal ACK targets; no source content output."""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from src.research.trade_evidence import _canonical


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    db.execute('PRAGMA query_only=ON')
    db.execute('BEGIN')
    return db


def audit_acknowledgements(journal_path, frontier_path, ledger_path, *, sample_limit=100):
    if type(sample_limit) is not int or not 1 <= sample_limit <= 10000:
        raise ValueError('bounded positive sample required')
    journal = readonly(journal_path)
    try:
        # Journal snapshot first: its ACKs were committed after the frontier and
        # canonical rows. Taking the frontier snapshot later cannot omit them.
        rows = journal.execute('SELECT id,body_sha256,accepted,first_delivery,canonical_job,canonical_ref '
            'FROM intake WHERE acknowledged IS NOT NULL ORDER BY acknowledged DESC,id LIMIT ?',
            (sample_limit,)).fetchall()
    finally:
        journal.close()
    failures = Counter()
    with closing(readonly(frontier_path)) as frontier, Path(ledger_path).open('rb') as ledger:
        ledger.seek(0, 2)
        extent = ledger.tell()
        for event_id, body_hash, accepted, delivered, job, reference in rows:
            binding = frontier.execute('SELECT kind,payload_ref FROM jobs WHERE job_id=?', (job,)).fetchone()
            if binding != ('NARRATIVE_OBSERVATION', reference):
                failures['frontier_binding'] += 1
                continue
            try:
                prefix, digest, location = reference.split(':')
                offset = int(location)
                if prefix != 'evidence' or len(digest) != 64 or not 0 <= offset < extent:
                    raise ValueError('invalid reference')
                if offset:
                    ledger.seek(offset-1)
                    if ledger.read(1) != b'\n':
                        raise ValueError('not a row boundary')
                ledger.seek(offset)
                line = ledger.readline(min(8*1024*1024+1, extent-offset))
                if len(line) > 8*1024*1024 or not line.endswith(b'\n'):
                    raise ValueError('unbounded or incomplete row')
                row = json.loads(line)
                actual = row.pop('hash')
                if actual != digest or hashlib.sha256(_canonical(row)).hexdigest() != digest:
                    raise ValueError('row hash mismatch')
                if row['event_type'] != 'research_typed_work' or row['payload']['work_kind'] != 'NARRATIVE_OBSERVATION':
                    raise ValueError('wrong canonical kind')
                payload = row['payload']['point_payload']
                receipt = payload['source_intake_receipt']
                expected = {'id':event_id, 'body_sha256':body_hash, 'accepted_at':accepted,
                            'first_delivered_at':delivered}
                if receipt != expected or payload['research_available_at'] != max(payload['observed_at'], delivered):
                    raise ValueError('receipt or availability mismatch')
            except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
                failures['canonical_row_or_receipt'] += 1
    return {'status': 'FAIL' if failures else ('PASS' if rows else 'NO_ACK_SAMPLE'),
            'sampled_acknowledgements':len(rows), 'verified_acknowledgements':len(rows)-sum(failures.values()),
            'failures':dict(failures), 'sample_policy':'newest acknowledged rows',
            'full_prefix_chain_verified':False, 'upstream_gap_free':False,
            'scope':'sampled committed frontier binding, canonical row self-hash and exact intake receipt; not full-prefix chain verification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path)
    parser.add_argument('frontier',type=Path)
    parser.add_argument('ledger',type=Path)
    parser.add_argument('--sample-limit',type=int,default=100)
    args = parser.parse_args()
    try:
        result = audit_acknowledgements(args.journal,args.frontier,args.ledger,sample_limit=args.sample_limit)
    except Exception as exc:
        result = {'status':'ERROR','error_type':type(exc).__name__}
    print(json.dumps(result,sort_keys=True))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
