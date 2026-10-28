"""Losslessly compress a closed backup file; never remove the original.

A trusted SHA-256 from a prior backup receipt is mandatory. Re-read every
uncompressed byte before publishing, without extracting or changing the source.
This checks byte recovery, not SQLite/chain integrity or a full restore drill.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

CHUNK = 1024 * 1024


def verify_archive(path, expected_sha256):
    """Read the complete gzip stream and validate its uncompressed SHA-256."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected lowercase SHA-256 required')
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('restored byte digest mismatch')
    return size


def compress_backup(source, destination, *, expected_sha256):
    """Publish a verified new archive exclusively; preserve source and peers."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected lowercase SHA-256 required')
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError('regular closed backup file required')
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('new distinct archive path required')
    before = source.stat()
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
    digest, size = hashlib.sha256(), 0
    try:
        with source.open('rb') as original, temporary.open('xb') as output:
            with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=1, mtime=0) as compressed:
                while chunk := original.read(CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
                    compressed.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError('source differs from prior backup receipt')
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('source changed during compression')
        if verify_archive(temporary, expected_sha256) != size:
            raise ValueError('restored byte size mismatch')
        # Hard-link publication is exclusive on Windows and POSIX; no existing
        # archive may be overwritten by a racing publisher. Same directory/fs.
        os.link(temporary, destination)
        return {'status': 'PASS', 'sha256': expected_sha256, 'restored_bytes': size,
                'compressed_bytes': destination.stat().st_size,
                'original_preserved': True, 'space_reclaimed_bytes': 0,
                'scope': 'lossless byte recovery; not full database/service restore'}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(compress_backup(args.source, args.destination,
                                    expected_sha256=args.expected_sha256)))


if __name__ == '__main__':
    main()
