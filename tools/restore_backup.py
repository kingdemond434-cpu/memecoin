"""Restore one verified gzip backup to a new isolated file, never over live state."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

CHUNK = 1024 * 1024


def restore_archive(archive, destination, *, expected_sha256, expected_bytes, sqlite_check=False):
    """Bound expansion, verify bytes and optional SQLite integrity, publish exclusively."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected lowercase SHA-256 required')
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError('nonnegative integer restored byte count required')
    archive, destination = Path(archive), Path(destination)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError('regular closed archive required')
    if archive.resolve() == destination.resolve() or destination.exists():
        raise ValueError('new distinct restore path required')
    before = archive.stat()
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
    digest, size = hashlib.sha256(), 0
    try:
        with gzip.open(archive, 'rb') as compressed, temporary.open('xb') as output:
            while chunk := compressed.read(min(CHUNK, expected_bytes - size + 1)):
                size += len(chunk)
                if size > expected_bytes:
                    raise ValueError('archive exceeds expected restored size')
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError('restored bytes differ from trusted receipt')
        restored_digest = hashlib.sha256()
        with temporary.open('rb') as restored:
            while chunk := restored.read(CHUNK):
                restored_digest.update(chunk)
        if restored_digest.hexdigest() != expected_sha256:
            raise ValueError('restored file read-back digest mismatch')
        after = archive.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('archive changed during restore')
        if sqlite_check:
            db = sqlite3.connect(temporary.resolve().as_uri() + '?mode=ro', uri=True)
            try:
                db.execute('PRAGMA query_only=ON')
                if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                    raise ValueError('restored SQLite quick_check failed')
            finally:
                db.close()
        os.link(temporary, destination)
        return {'status': 'PASS', 'sha256': expected_sha256, 'restored_bytes': size,
                'sqlite_quick_check': 'PASS' if sqlite_check else 'NOT_REQUESTED',
                'archive_preserved': True, 'existing_destination_replaced': False,
                'scope': 'isolated single-file restore; not service recovery or canonical chain validation'}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-bytes', required=True, type=int)
    parser.add_argument('--sqlite-check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore_archive(args.archive, args.destination,
          expected_sha256=args.expected_sha256, expected_bytes=args.expected_bytes,
          sqlite_check=args.sqlite_check)))


if __name__ == '__main__':
    main()
