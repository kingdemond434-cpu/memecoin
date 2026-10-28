"""Retire one older SQLite backup only after verified archive and restore.

Never operate on live state, canonical prefixes, certificates or the two newest
certified recovery directories. Trusted owner-local prior verification receipts
are mandatory; this is not external authentication or full service recovery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from tools.compress_backup import compress_backup
from tools.restore_backup import restore_archive

DATABASES = frozenset(('champion_state.population.sqlite', 'free_market_discovery.sqlite',
    'frontier.sqlite', 'native_fork_probe.sqlite', 'native_pool_attribution.sqlite',
    'neural_sentiment.sqlite', 'phantom_perps.sqlite', 'public_claim_links.sqlite',
    'public_onion_frontier.sqlite', 'source_intake.sqlite'))


def _verified_receipt(path):
    receipt = json.loads(Path(path).read_text())
    if receipt.get('status') != 'PASS' or set(receipt.get('sqlite', {})) != DATABASES or not all(value is True for value in receipt['sqlite'].values()):
        raise ValueError('complete passing prior database verification required')
    return receipt


def _save(path, value, *, exclusive=False):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('x', encoding='utf-8') as output:
            json.dump(value, output, indent=2)
            output.write('\n'); output.flush(); os.fsync(output.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def consolidate(source, *, backups_root, verification_receipt, protected_receipts, retirement_receipt):
    """Consolidate a closed older backup, preserving two newer verified copies."""
    source, root = Path(source), Path(backups_root).resolve()
    receipt_path = Path(retirement_receipt)
    # Resolved path boundaries are checked before any removal or publication.
    if source.is_symlink() or source.resolve() != source.absolute() or source.parent.parent.resolve() != root:
        raise ValueError('direct nonsymlink backup file under intended root required')
    if source.name not in DATABASES or not re.fullmatch(r'report-deployment-backup-\d{8}-\d{6}', source.parent.name):
        raise ValueError('only an identified SQLite recovery copy may be retired')
    if receipt_path.exists() or receipt_path.resolve() == source.resolve():
        raise ValueError('new recovery receipt required')
    prior = _verified_receipt(verification_receipt)
    if Path(prior['backup']).resolve() != source.parent.resolve():
        raise ValueError('prior receipt does not identify this backup')
    protected = [_verified_receipt(p) for p in protected_receipts]
    protected_paths = {Path(p['backup']).resolve() for p in protected}
    if len(protected_paths) != 2 or any(p.parent != root or p.name <= source.parent.name for p in protected_paths):
        raise ValueError('two distinct newer verified recovery directories required')
    certified = []
    for folder in root.glob('report-deployment-backup-*'):
        if not folder.is_dir() or folder.is_symlink():
            continue
        try:
            certificate = json.loads((folder/'canonical-certificate.json').read_text())
            integrity = json.loads((folder/'integrity.json').read_text())
        except (OSError, ValueError):
            continue
        if certificate.get('hash_chain_verified') is True and integrity.get('hash_chain_verified') is True:
            certified.append(folder.resolve())
    if set(sorted(certified, key=lambda p: p.name)[-2:]) != protected_paths:
        raise ValueError('protected receipts must identify the newest two certified backups')
    if any(not all((p/name).is_file() for name in DATABASES) for p in protected_paths):
        raise ValueError('protected recovery database copies missing')
    expected = prior['files'][source.name]
    if type(expected.get('bytes')) is not int or source.stat().st_size != expected['bytes']:
        raise ValueError('source extent differs from prior receipt')
    if shutil.disk_usage(root).free < expected['bytes']*2 + 12*1024**3:
        raise ValueError('insufficient archive/restore headroom and runtime reserve')
    archive = source.with_suffix(source.suffix + '.gz')
    if not archive.exists():
        compress_backup(source, archive, expected_sha256=expected['sha256'])
    scratch = source.with_name(source.name + '.' + uuid.uuid4().hex + '.restore')
    try:
        restored = restore_archive(archive, scratch, expected_sha256=expected['sha256'],
                                   expected_bytes=expected['bytes'], sqlite_check=True)
        # Re-read source after the restore; only the same closed recovery bytes
        # may be retired. No live row or merely matching file name suffices.
        before = source.stat(); digest = hashlib.sha256()
        with source.open('rb') as stream:
            while chunk := stream.read(1024*1024):
                digest.update(chunk)
        after = source.stat()
        if digest.hexdigest() != expected['sha256'] or (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('source changed or differs from trusted recovery bytes')
        result = {'status': 'RESTORE_VERIFIED_ORIGINAL_RETAINED', 'source': str(source.resolve()),
            'archive': str(archive.resolve()), 'sha256': expected['sha256'], 'restored_bytes': expected['bytes'],
            'sqlite_quick_check': restored['sqlite_quick_check'], 'protected_backups': sorted(map(str, protected_paths)),
            'verification_receipt': str(Path(verification_receipt).resolve()),
            'restore_command': 'python -m tools.restore_backup ARCHIVE NEW_ISOLATED_FILE --expected-sha256 SHA256 --expected-bytes RESTORED_BYTES --sqlite-check',
            'full_service_restore_drill': False, 'canonical_files_changed': False}
        _save(receipt_path, result, exclusive=True)
        # Explicit single-file removal only, after trusted restore and durable
        # recovery receipt. The archive and newest two complete copies remain.
        source.unlink()
        result.update(status='ORIGINAL_RETIRED_ARCHIVE_RESTORE_VERIFIED',
                      net_backup_bytes_saved=expected['bytes']-archive.stat().st_size)
        _save(receipt_path, result)
        return result
    finally:
        scratch.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--backups-root', required=True, type=Path)
    p.add_argument('--verification-receipt', required=True, type=Path)
    p.add_argument('--protected-receipt', required=True, action='append', type=Path)
    p.add_argument('--retirement-receipt', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(consolidate(a.source, backups_root=a.backups_root,
        verification_receipt=a.verification_receipt, protected_receipts=a.protected_receipt,
        retirement_receipt=a.retirement_receipt)))


if __name__ == '__main__':
    main()
