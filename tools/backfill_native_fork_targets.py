"""Recover reviewed research targets from an owner-certified sealed journal.

Dry run by default. This queues current-bank research, never reconstructs an
old bank or authorizes an order. The certificate is an owner-local integrity
checkpoint, not external notarization. The running collector need not stop.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time

from src.research.evidence_certificate import validate_certificate
from src.research.fork_probe import ForkProbe, reviewed_native_target, digest
from src.research.trade_evidence import TradeEvidenceLedger, _verified_row


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def backfill(ledger, certificate, *, database=None, apply=False):
    ledger, certificate = Path(ledger).resolve(), Path(certificate).resolve()
    if apply and database is None:
        raise ValueError('explicit research database required for apply')
    if database is not None and Path(database).resolve() in (ledger, certificate):
        raise ValueError('research database cannot replace source artifacts')
    certificate_bytes = certificate.read_bytes()
    cert = json.loads(certificate_bytes.decode('utf-8-sig'))
    validate_certificate(cert)
    before = _stamp(ledger)
    if before[2] != cert['canonical_bytes']:
        raise ValueError('a sealed exact-extent journal is required')
    # The shared recovery path hashes every certified byte and checks the
    # actual final row. No mere row count or unverified tail is trusted.
    verified = TradeEvidenceLedger(ledger, recovery_certificate=certificate)
    if verified.sequence != cert['canonical_records']:
        raise ValueError('certified source record count changed')
    started = time.perf_counter()
    report = {'status': 'DRY_RUN', 'applied': apply, 'source_records': verified.sequence,
              'source_certificate_sha256': cert['certificate_sha256'], 'native_rows': 0,
              'source_verification_seconds': verified.recovery_seconds,
              'reviewed_source_rows': 0, 'other_or_unsupported': 0, 'new_targets': 0,
              'scope': 'owner-certified source recovery; current-bank research only; no live authority or historical-fill proof'}
    # Spill proposals to a temporary SQLite file, rather than capping the
    # source universe or retaining its entire candidate set in RAM.
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as handle:
        staged_path = Path(handle.name)
    staged = probe = None
    try:
        staged = sqlite3.connect(staged_path)
        staged.execute('CREATE TABLE proposals(ref TEXT PRIMARY KEY,event TEXT,available REAL,target TEXT)')
        staged.execute('CREATE INDEX targets ON proposals(target)')
        with staged, ledger.open('rb') as source:
            for line in source:
                if b'"native_launch"' not in line:
                    continue
                row = json.loads(line)
                if row.get('event_type') != 'native_launch':
                    continue
                row, _ = _verified_row(line, row['previous_hash'], row['sequence'])
                report['native_rows'] += 1
                event = row['payload']
                target = reviewed_native_target(event)
                if target is None:
                    report['other_or_unsupported'] += 1
                    continue
                at = row['timestamp']
                if type(at) not in (int, float) or not math.isfinite(at):
                    raise ValueError('invalid recorded native source availability')
                selected = {key: event[key] for key in ('type','venue','program','quote_mint','token','pool')}
                staged.execute('INSERT INTO proposals VALUES(?,?,?,?)',
                    ('native-launch:'+row['hash'],json.dumps(selected),at,digest(target)))
                report['reviewed_source_rows'] += 1
        if _stamp(ledger) != before or certificate.read_bytes() != certificate_bytes:
            raise ValueError('sealed source or certificate changed during recovery')
        report['reviewed_targets'] = staged.execute('SELECT count(DISTINCT target) FROM proposals').fetchone()[0]
        if apply:
            probe = ForkProbe(database)
            for ref, raw, at in staged.execute('SELECT ref,event,available FROM proposals ORDER BY rowid'):
                result = probe.enqueue_native_launch(json.loads(raw), source_ref=ref, available_at=at)
                report['new_targets'] += int(result['inserted'])
            report['status'] = 'APPLIED'
        report['source_unchanged'] = _stamp(ledger) == before
        report['scan_apply_seconds'] = time.perf_counter()-started
        return report
    finally:
        if probe is not None:
            probe.close()
        if staged is not None:
            staged.close()
        staged_path.unlink(missing_ok=True)
        Path(str(staged_path)+'-journal').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir() or args.output.exists() or args.output.resolve() in {args.ledger.resolve(), args.certificate.resolve(),
                                                       args.database.resolve() if args.database else None}:
        parser.error('a new receipt path outside source/database files is required')
    report = backfill(args.ledger, args.certificate, database=args.database, apply=args.apply)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
