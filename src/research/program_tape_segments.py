"""Sealed segments and verified compressed archives of the program tape.

The tape is permanent, so one SQLite file grows without bound: on 2026-11-10
it was 98 GB on a disk shared with other desks. Nothing here drops a record.
The active file is sealed at a size threshold and a new file continues the
SAME sequence and hash chain. A sealed file is then rewritten as a gzip of its
exact row bodies, and the SQLite copy is removed only after that archive has
been read back and its whole chain recomputed.

Every other column of a row (sequence, ids, kind, hash) is derivable from the
body, so the archive is lossless; it sheds the two indexes and the duplicated
columns, which is most of the file.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

GENESIS = '0'*64
MANIFEST_SCHEMA = 'program-tape-archive/1'
#: Archiving stops, and discards its partial output, the moment free space
#: falls to this, so compaction can never be the thing that fills the disk.
#: It is checked while writing rather than estimated up front, because the
#: segment that most needs archiving is the one larger than the free space.
ARCHIVE_HEADROOM_BYTES = 4 * 1024**3
_CHECK_EVERY = 2048


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sealed_path(active, first, last):
    active = Path(active)
    return active.with_name(f'{active.stem}.{first:016d}-{last:016d}.sealed.sqlite')


def archive_paths(sealed):
    stem = Path(sealed).name[:-len('.sealed.sqlite')]
    parent = Path(sealed).parent
    return parent / f'{stem}.jsonl.gz', parent / f'{stem}.manifest.json'


def list_segments(active):
    """Closed segments of `active`, oldest first, each archived or sealed."""
    active = Path(active)
    found = {}
    for path in active.parent.glob(f'{active.stem}.*-*.manifest.json'):
        first, last = _range(path.name, active.stem)
        if first is not None:
            found[first] = {'first': first, 'last': last, 'kind': 'archive', 'path': path}
    for path in active.parent.glob(f'{active.stem}.*-*.sealed.sqlite'):
        first, last = _range(path.name, active.stem)
        # A sealed file whose archive is complete is only awaiting removal.
        if first is not None and first not in found:
            found[first] = {'first': first, 'last': last, 'kind': 'sealed', 'path': path}
    return [found[key] for key in sorted(found)]


def _range(name, stem):
    try:
        first, last = name[len(stem) + 1:].split('.')[0].split('-')
        return int(first), int(last)
    except ValueError:
        return None, None


def read_tail(segment):
    """(sequence, last_hash, counts) at the end of a closed segment."""
    if segment['kind'] == 'archive':
        manifest = json.loads(segment['path'].read_text(encoding='utf-8'))
        return manifest['last_sequence'], manifest['last_hash'], manifest['counts']
    db = sqlite3.connect(segment['path'].resolve().as_uri() + '?mode=ro', uri=True)
    try:
        sequence, seal, counts = db.execute(
            'SELECT sequence,last_hash,counts FROM tape_meta WHERE id=1').fetchone()
        return sequence, seal, json.loads(counts)
    finally:
        db.close()


def read_base(db):
    """The chain position a SQLite segment starts after."""
    present = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                         "AND name='tape_base'").fetchone()
    row = db.execute('SELECT sequence,hash FROM tape_base WHERE id=1').fetchone() if present else None
    return tuple(row) if row else (0, GENESIS)


def _check(raw, sequence, previous):
    body = json.loads(raw)
    if body['sequence'] != sequence + 1 or body['previous_hash'] != previous:
        raise ValueError('tape sequence/hash chain failed')
    return sequence + 1, _digest(body), body['kind']


def verify_archive(archive, manifest):
    """Read the archive back and recompute its chain against its manifest."""
    sequence, previous = manifest['base_sequence'], manifest['base_hash']
    with gzip.open(archive, 'rt', encoding='utf-8', newline='\n') as handle:
        for line in handle:
            sequence, previous, _ = _check(line[:-1], sequence, previous)
    if (sequence, previous) != (manifest['last_sequence'], manifest['last_hash']):
        raise ValueError('archive tail does not match its manifest')
    return sequence, previous


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class _LowDisk(Exception):
    pass


class _Stopped(Exception):
    pass


def _remove_source(sealed):
    for suffix in ('', '-wal', '-shm'):
        Path(str(sealed) + suffix).unlink(missing_ok=True)


def archive_segment(sealed, *, duty=1.0, usage=shutil.disk_usage, sleep=time.sleep,
                    stop=lambda: False):
    """Rewrite one sealed segment as a verified archive, then remove it.

    `duty` is the fraction of wall time spent working, so the recorder that
    shares this process's CPU budget is never starved by its own compaction.
    Returns the manifest, or a dict with `deferred` when disk is too tight or
    `stop()` turned true; either way the sealed source is left untouched.
    """
    sealed = Path(sealed)
    archive, manifest_path = archive_paths(sealed)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if archive.exists() and _sha256(archive) == manifest['archive_sha256']:
            _remove_source(sealed)       # finished before; only removal was left
            return manifest
        manifest_path.unlink()           # incomplete: rebuild from the source
    size = sealed.stat().st_size
    low = {'deferred': 'LOW_DISK', 'source_bytes': size}
    if usage(sealed.parent).free < ARCHIVE_HEADROOM_BYTES:
        return low
    partial = archive.with_name(archive.name + '.partial')
    db = sqlite3.connect(sealed.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        base_sequence, base_hash = read_base(db)
        sequence, previous = base_sequence, base_hash
        kinds = {}
        began = time.perf_counter()
        with open(partial, 'wb') as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode='wb', compresslevel=6, mtime=0) as handle:
                for number, raw, seal in db.execute(
                        'SELECT sequence,body,hash FROM records ORDER BY sequence'):
                    sequence, previous, kind = _check(raw, sequence, previous)
                    if number != sequence or seal != previous:
                        raise ValueError('tape sequence/hash chain failed')
                    kinds[kind] = kinds.get(kind, 0) + 1
                    handle.write(raw.encode('utf-8') + b'\n')
                    if sequence % _CHECK_EVERY:
                        continue
                    if stop():
                        raise _Stopped()
                    if usage(sealed.parent).free < ARCHIVE_HEADROOM_BYTES:
                        raise _LowDisk()
                    if duty < 1:
                        worked = time.perf_counter() - began
                        sleep(worked * (1 - duty) / duty)
                        began = time.perf_counter()
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        tail, seal, counts = db.execute(
            'SELECT sequence,last_hash,counts FROM tape_meta WHERE id=1').fetchone()
        if (tail, seal) != (sequence, previous):
            raise ValueError('tape tail/checkpoint verification failed')
    except _LowDisk:
        partial.unlink(missing_ok=True)
        return low
    except _Stopped:
        partial.unlink(missing_ok=True)
        return {'deferred': 'STOPPED', 'source_bytes': size}
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        db.close()
    manifest = {
        'schema': MANIFEST_SCHEMA, 'base_sequence': base_sequence, 'base_hash': base_hash,
        'first_sequence': base_sequence + 1, 'last_sequence': sequence, 'last_hash': previous,
        'records': sequence - base_sequence, 'records_by_kind': kinds,
        'counts': json.loads(counts), 'source_bytes': size,
        'archive_bytes': partial.stat().st_size, 'archive_sha256': _sha256(partial),
        'format': 'gzip; one exact row body per line, in sequence order',
    }
    try:
        verify_archive(partial, manifest)    # the source is removed only after this
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, archive)
    pending = manifest_path.with_name(manifest_path.name + '.partial')
    with open(pending, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, manifest_path)
    _remove_source(sealed)
    return manifest


def _verify_sqlite(path, sequence, previous):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        if read_base(db) != (sequence, previous):
            raise ValueError('segment does not continue the previous segment')
        for number, raw, seal in db.execute('SELECT sequence,body,hash FROM records ORDER BY sequence'):
            sequence, previous, _ = _check(raw, sequence, previous)
            if number != sequence or seal != previous:
                raise ValueError('tape sequence/hash chain failed')
        if tuple(db.execute('SELECT sequence,last_hash FROM tape_meta WHERE id=1').fetchone()) != (sequence, previous):
            raise ValueError('tape tail/checkpoint verification failed')
        return sequence, previous
    finally:
        db.close()


def verify_tape(active):
    """Verify every closed segment and the active file as ONE chain."""
    active = Path(active)
    sequence, previous = 0, GENESIS
    segments = list_segments(active)
    for segment in segments:
        if segment['kind'] == 'archive':
            manifest = json.loads(segment['path'].read_text(encoding='utf-8'))
            if (manifest['base_sequence'], manifest['base_hash']) != (sequence, previous):
                raise ValueError('segment does not continue the previous segment')
            archive = segment['path'].with_name(
                segment['path'].name[:-len('.manifest.json')] + '.jsonl.gz')
            if _sha256(archive) != manifest['archive_sha256']:
                raise ValueError('archive bytes do not match their manifest')
            sequence, previous = verify_archive(archive, manifest)
        else:
            sequence, previous = _verify_sqlite(segment['path'], sequence, previous)
    if active.exists():
        sequence, previous = _verify_sqlite(active, sequence, previous)
    return {'status': 'PASS', 'records': sequence, 'last_hash': previous,
            'closed_segments': len(segments),
            'archived_segments': sum(1 for item in segments if item['kind'] == 'archive'),
            'scope': 'owner-local committed snapshot integrity; not external attestation'}


def segment_report(active):
    segments = list_segments(active)
    def total(kind, of):
        return sum(of(item['path']) for item in segments if item['kind'] == kind)
    def archive_size(manifest):
        target = manifest.with_name(manifest.name[:-len('.manifest.json')] + '.jsonl.gz')
        return target.stat().st_size if target.exists() else 0
    return {'sealed': sum(1 for item in segments if item['kind'] == 'sealed'),
            'archived': sum(1 for item in segments if item['kind'] == 'archive'),
            'sealed_bytes': total('sealed', lambda path: path.stat().st_size),
            'archive_bytes': total('archive', archive_size)}
