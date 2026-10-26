"""Verify a sealed canonical backup, optionally reusing a certified prefix.

The prefix must match its earlier full verification certificate AND match the
new backup byte for byte. Every new row is then hash-chain verified. This saves
reparsing a multi-gigabyte history while the paper service is stopped. It never
writes the ledger or trusts a mere record count. Certificates are owner-local
corruption checks, not external notarization against a malicious administrator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from src.research.trade_evidence import GENESIS, _verified_row

SCHEMA = 'canonical-backup-certificate-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_certificate(value):
    body = {key: item for key, item in value.items() if key != 'certificate_sha256'}
    if (value.get('schema') != SCHEMA or value.get('status') != 'PASS'
            or value.get('hash_chain_verified') is not True
            or fingerprint(body) != value.get('certificate_sha256')):
        raise ValueError('invalid canonical prefix certificate')
    for key in ('canonical_bytes', 'canonical_records'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('invalid canonical prefix extent')
    for key in ('canonical_sha256', 'last_hash'):
        if not re.fullmatch('[0-9a-f]{64}', value.get(key, '')):
            raise ValueError('invalid canonical prefix digest')


def verify(path, *, prefix=None, certificate=None):
    path = Path(path)
    if (prefix is None) != (certificate is None):
        raise ValueError('prefix backup and certificate must be provided together')
    started, before = time.perf_counter(), path.stat()
    current_digest = hashlib.sha256()
    previous, count, prefix_count = GENESIS, 0, 0
    with path.open('rb') as current:
        if prefix is not None:
            prefix = Path(prefix)
            validate_certificate(certificate)
            prefix_before = prefix.stat()
            extent = certificate['canonical_bytes']
            if extent != prefix_before.st_size or extent > before.st_size:
                raise ValueError('canonical prefix extent differs from certificate or backup')
            old_digest = hashlib.sha256()
            with prefix.open('rb') as old:
                remaining = extent
                while remaining:
                    chunk = old.read(min(4*1024*1024, remaining))
                    if not chunk or current.read(len(chunk)) != chunk:
                        raise ValueError('new backup differs from the certified prefix')
                    remaining -= len(chunk)
                    old_digest.update(chunk)
                    current_digest.update(chunk)
            prefix_after = prefix.stat()
            if (old_digest.hexdigest() != certificate['canonical_sha256'] or
                    (prefix_after.st_size, prefix_after.st_mtime_ns) !=
                    (prefix_before.st_size, prefix_before.st_mtime_ns)):
                raise ValueError('certified prefix changed')
            previous, count = certificate['last_hash'], certificate['canonical_records']
            prefix_count = count
        while current.tell() < before.st_size:
            line = current.readline(before.st_size-current.tell())
            if not line or not line.endswith(b'\n'):
                raise ValueError('sealed canonical backup ends with an incomplete row')
            current_digest.update(line)
            if line.strip():
                _, previous = _verified_row(line, previous, count+1)
                count += 1
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('canonical backup changed during verification')
    result = {'schema': SCHEMA, 'status': 'PASS', 'hash_chain_verified': True,
        'verification_mode': 'CERTIFIED_PREFIX_AND_NEW_ROWS' if prefix is not None else 'FULL_HASH_CHAIN',
        'canonical_bytes': before.st_size, 'canonical_records': count,
        'canonical_sha256': current_digest.hexdigest(), 'last_hash': previous,
        'certified_prefix_records': prefix_count, 'newly_verified_records': count-prefix_count,
        'elapsed_seconds': time.perf_counter()-started,
        'scope': 'sealed owner-local backup verification; not external notarization'}
    result['certificate_sha256'] = fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--certificate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    certificate = json.loads(args.certificate.read_text(encoding='utf-8-sig')) if args.certificate else None
    result = verify(args.ledger, prefix=args.prefix, certificate=certificate)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
