"""Fixed-tail, indexed read-only ACK binding sample; never a whole-store audit."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

from src.research.trade_evidence import _canonical


class _Budget(RuntimeError):
    pass


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def audit_recent_acknowledgements(journal_path, frontier_path, ledger_path, *,
        sample_limit=20, tail_bytes=262144, max_lines=128, vm_budget=20000,
        deadline_seconds=30, clock=time.monotonic, end_offset=None):
    if (type(sample_limit) is not int or not 1 <= sample_limit <= 100
            or type(tail_bytes) is not int or not 1024 <= tail_bytes <= 1048576
            or type(max_lines) is not int or not 1 <= max_lines <= 256
            or type(vm_budget) is not int or not 100 <= vm_budget <= 100000
            or type(deadline_seconds) not in (int, float)
            or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 60):
        raise ValueError('Bounded ACK sample policy required')
    if end_offset is not None and (type(end_offset) is not int or end_offset < 0):
        raise ValueError('Bounded canonical window cursor required')
    start = clock(); operations = 0; stop = None
    result = {'status': 'NO_ACK_TARGET_SAMPLE', 'candidate_rows': 0,
        'verified_acknowledgements': 0, 'pending_receipts': 0,
        'non_target_rows': 0, 'lines_inspected': 0, 'bytes_read': 0,
        'failures': {}, 'boundary_row_not_sampled': False, 'sqlite_integrity_verified': False,
        'full_prefix_chain_verified': False, 'durable_owner_application_verified': False,
        'upstream_gap_free': False, 'physical_io_cap_claimed': False,
        'scope': 'bounded canonical tail candidates and exact committed ACK targets only; not newest-ACK census, store integrity, full chain, owner replay or coverage'}
    failures = Counter()
    verified_clocks = []
    def guard():
        if clock()-start >= deadline_seconds:
            raise _Budget('deadline')
    def progress():
        nonlocal operations, stop
        operations += 100
        if operations > vm_budget:
            stop = 'sql_vm_budget'; return 1
        if clock()-start >= deadline_seconds:
            stop = 'deadline'; return 1
        return 0
    def connect(path):
        guard()
        db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=1)
        try:
            db.set_progress_handler(progress, 100)
            db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
            return db
        except BaseException:
            db.close(); raise
    def indexed(db, table, sql, parameters):
        guard()
        plan = [row[3] for row in db.execute('EXPLAIN QUERY PLAN '+sql, parameters)]
        if (any(re.search(r'\bSCAN\b', row, re.I) for row in plan)
                or not any(re.search(r'\bSEARCH(?: TABLE)? '+table+r'\b', row, re.I) for row in plan)):
            raise _Budget('unindexed_'+table+'_lookup')
        guard()
        return db.execute(sql, parameters).fetchone()
    try:
        guard()
        with Path(ledger_path).open('rb') as ledger:
            ledger.seek(0, 2); extent = ledger.tell()
            end = extent if end_offset is None else end_offset
            if end > extent:raise ValueError('Cursor exceeds current ledger extent')
            offset = max(0, end-tail_bytes)
            ledger.seek(offset); data = ledger.read(end-offset)
        result.update(bytes_read=len(data), ledger_extent=extent, window_end=end,
                      next_window_end=offset if offset else None,
                      leading_line_discarded=offset > 0, incomplete_tail=bool(data) and not data.endswith(b'\n'))
        if offset:
            first = data.find(b'\n')
            if first < 0:
                result['boundary_row_not_sampled']=True
                data = b''
            else:
                offset += first+1; data = data[first+1:]
                if offset < end:result['next_window_end']=offset
                else:result['boundary_row_not_sampled']=True
        rows = []
        for line in data.splitlines(keepends=True):
            if line.endswith(b'\n'):
                rows.append((offset, line))
            offset += len(line)
        result['line_budget_exhausted'] = len(rows) > max_lines
        targets = []
        with ExitStack() as receipt_handles:
            journal = None
            for offset, line in reversed(rows[-max_lines:]):
                guard(); result['lines_inspected'] += 1
                try:
                    row = json.loads(line)
                    if row.get('event_type') != 'research_typed_work':continue
                    work = row.get('payload', {})
                    if work.get('work_kind') != 'NARRATIVE_OBSERVATION':continue
                    point = work['point_payload']; receipt = point.get('source_intake_receipt')
                    if receipt is None:continue
                    result['candidate_rows'] += 1
                    digest = row.pop('hash')
                    if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                            or hashlib.sha256(_canonical(row)).hexdigest() != digest):
                        raise ValueError('candidate hash')
                    event_id = receipt['id']
                    if not isinstance(event_id, str) or not 1 <= len(event_id) <= 256:
                        raise ValueError('receipt identity')
                    if journal is None:
                        journal = receipt_handles.enter_context(closing(connect(journal_path)))
                    target = indexed(journal, 'intake',
                        'SELECT id,body_sha256,accepted,first_delivery,canonical_job,canonical_ref,acknowledged,body IS NULL FROM intake WHERE id=?', (event_id,))
                    if target is None:
                        failures['missing_journal_receipt'] += 1; continue
                    event_id, body_hash, accepted, delivered, job, reference, acked, stripped = target
                    if acked is None:
                        result['pending_receipts'] += 1; continue
                    exact_ref = 'evidence:'+digest+':'+str(offset)
                    expected = {'id':event_id,'body_sha256':body_hash,
                                'accepted_at':accepted,'first_delivered_at':delivered}
                    if (receipt != expected or not all(_finite(v) for v in (accepted,delivered,acked,point.get('observed_at'), point.get('research_available_at'), receipt.get('accepted_at'), receipt.get('first_delivered_at')))
                            or not isinstance(body_hash,str) or not re.fullmatch('[0-9a-f]{64}',body_hash)
                            or not accepted <= delivered <= acked or stripped != 1
                            or point.get('research_available_at') != max(point['observed_at'],delivered)
                            or not isinstance(job,str) or not job.startswith('work-')):
                        raise ValueError('receipt clocks/state')
                    if not isinstance(reference,str) or not re.fullmatch(r'evidence:[0-9a-f]{64}:[0-9]+',reference):
                        raise ValueError('ACK reference format')
                    if reference != exact_ref:
                        # Retained retry rows are not themselves proof of ACK.
                        # Their receipt/state must still be internally valid.
                        result['non_target_rows'] += 1; continue
                    targets.append((job,reference,accepted,delivered,acked))
                    if len(targets) >= sample_limit:break
                except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
                    failures['canonical_candidate_or_receipt'] += 1
        # Journal ACK snapshot precedes frontier snapshot, as ACK follows the
        # durable frontier/canonical writes. No full-table selection is needed.
        if targets:
            with closing(connect(frontier_path)) as frontier:
                for job, reference, accepted, delivered, acked in targets:
                    binding = indexed(frontier, 'jobs',
                        'SELECT kind,payload_ref FROM jobs WHERE job_id=?',(job,))
                    if binding != ('NARRATIVE_OBSERVATION',reference):
                        failures['frontier_binding'] += 1
                    else:
                        result['verified_acknowledgements'] += 1
                        verified_clocks.append((accepted,delivered,acked))
        guard()
        result['status'] = ('FAIL_SAMPLED_BINDINGS' if failures else
            'SAMPLED_BINDINGS_VERIFIED' if result['verified_acknowledgements'] else 'NO_ACK_TARGET_SAMPLE')
    except _Budget as exc:
        result.update(status='RESOURCE_BLOCKED',budget_reason=str(exc))
    except sqlite3.OperationalError:
        if stop is None:raise
        result.update(status='RESOURCE_BLOCKED',budget_reason=stop)
    def measured(values):
        return {'count':len(values),'min':min(values) if values else None,
                'max':max(values) if values else None}
    result['journal_timings'] = {
        'intake_to_first_delivery_seconds':measured([d-a for a,d,k in verified_clocks]),
        'first_delivery_to_ack_seconds':measured([k-d for a,d,k in verified_clocks]),
        'intake_to_ack_seconds':measured([k-a for a,d,k in verified_clocks])}
    result['ack_clock_range'] = measured([k for a,d,k in verified_clocks])
    result['latency_scope'] = 'journal wall-clock ranges for verified window-selected targets only; not p99, pure owner-loop delay, causality or sustainable capacity'
    result.update(failures=dict(failures), sql_progress_instructions_observed=operations, sql_progress_interval=100,
                  elapsed_seconds=clock()-start)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path);parser.add_argument('frontier',type=Path);parser.add_argument('ledger',type=Path)
    parser.add_argument('--sample-limit',type=int,default=20);parser.add_argument('--tail-bytes',type=int,default=262144)
    parser.add_argument('--end-offset',type=int,default=None)
    args=parser.parse_args()
    try:result=audit_recent_acknowledgements(args.journal,args.frontier,args.ledger,sample_limit=args.sample_limit,tail_bytes=args.tail_bytes,end_offset=args.end_offset)
    except Exception as exc:result={'status':'ERROR','error_type':type(exc).__name__}
    print(json.dumps(result,sort_keys=True))
    return 0 if result['status']=='SAMPLED_BINDINGS_VERIFIED' else 1


if __name__=='__main__':raise SystemExit(main())
