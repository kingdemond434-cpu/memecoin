"""Bounded metadata composition; no backup authentication or deployment authority."""
import copy
import re


def _cursor(value):
    if (not isinstance(value, dict) or set(value) != {'offset', 'sequence', 'last_hash'}
            or any(type(value.get(k)) is not int or value[k] < 0 for k in ('offset', 'sequence'))
            or not isinstance(value.get('last_hash'), str)
            or re.fullmatch('[0-9a-f]{64}', value['last_hash']) is None):
        raise ValueError('Exact canonical cursor required')
    return dict(value)


def compose_recovery_plan(anchor, segments):
    """Describe at most 32 contiguous historical receipts without touching files.

    Caller must authenticate the retained baseline and receipts separately.
    Paths are references only; no readback, current existence or chain proof.
    """
    start = _cursor(anchor)
    if type(segments) is not list or not 1 <= len(segments) <= 32:
        raise ValueError('One to 32 segment receipts required')
    cursor = start
    entries = []
    paths = set()
    for receipt in segments:
        if not isinstance(receipt, dict):
            raise ValueError('Segment receipt required')
        first = _cursor(receipt.get('start_cursor'))
        end = _cursor(receipt.get('next_cursor'))
        size = receipt.get('bytes')
        digest = receipt.get('sha256')
        path = receipt.get('pending_path')
        if (receipt.get('status') != 'SINGLE_CANONICAL_SEGMENT_READBACK_VERIFIED'
                or receipt.get('pending_published') is not True
                or receipt.get('deployment_backup_accepted') is not False
                or first != cursor or end['sequence'] <= first['sequence']
                or type(size) is not int or not 0 < size <= 1048576
                or end['offset'] - first['offset'] != size
                or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                or not isinstance(path, str) or not 1 <= len(path) <= 1024
                or '\x00' in path or not path.endswith('.segment.pending')):
            raise ValueError('Invalid or noncontiguous segment receipt')
        # Treat case and separator aliases conservatively for Windows paths.
        key = path.replace('\\', '/').casefold()
        if any(part in ('.', '..') for part in key.split('/')) or key in paths:
            raise ValueError('Duplicate or ambiguous segment reference')
        paths.add(key)
        entries.append(dict(path=path, bytes=size, sha256=digest,
                            start_cursor=first, next_cursor=end))
        cursor = end
    return copy.deepcopy(dict(status='CANONICAL_RECOVERY_INPUT_PLAN_ONLY',
        anchor=start, next_cursor=cursor, segments=entries,
        segment_bytes=sum(item['bytes'] for item in entries),
        segment_records=cursor['sequence']-start['sequence'],
        baseline_authenticated=False, segment_contents_revalidated=False,
        full_prefix_chain_verified=False, cross_store_checkpoint_verified=False,
        recovery_verified=False, deployment_backup_accepted=False,
        source_write_requested=False))
