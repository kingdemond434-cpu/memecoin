"""Compare sanitized service receipts; no runtime access or source data export."""
import argparse
import json
import math
from pathlib import Path

from src.runtime.source_timing import PHASES


def _number(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('invalid numeric receipt field')
    if not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError('invalid numeric receipt field')
    return value


def compare(before, after):
    elapsed = _number(after['observed_at']) - _number(before['observed_at'])
    if elapsed <= 0:
        raise ValueError('receipt clocks must strictly advance')
    result = {'elapsed_seconds': elapsed, 'counters': {}, 'phases': {},
              'scope': 'separate_snapshots_not_atomic_or_end_to_end_latency',
              'process_continuity': 'UNVERIFIED', 'throughput_capacity_verified': False}
    for group, fields in (('journal', ('accepted', 'acknowledged')),
                          ('neural', ('pending_inputs', 'canonical_windows'))):
        a, b = before.get(group, {}), after.get(group, {})
        for field in fields:
            if field not in a or field not in b:
                continue
            delta = _number(b[field], True) - _number(a[field], True)
            gauge = field == 'pending_inputs'
            result['counters'][group + '.' + field] = (
                {'status': 'RESET_OR_INVALID_INTERVAL'} if delta < 0 and not gauge else
                {'status': 'GAUGE_DELTA' if gauge else 'MONOTONE_DELTA',
                 'delta': delta, 'per_second': delta / elapsed})
    journal = result['counters']
    accepted, ack = (journal.get('journal.' + name, {}) for name in ('accepted', 'acknowledged'))
    if accepted.get('status') == ack.get('status') == 'MONOTONE_DELTA':
        result['pending_journal_delta'] = accepted['delta'] - ack['delta']
    for phase in PHASES:
        a = before.get('source_phase_timing', {}).get('phases', {}).get(phase)
        b = after.get('source_phase_timing', {}).get('phases', {}).get(phase)
        if a is None or b is None:
            continue  # No invented zero baseline for a newly appearing phase.
        names = ('attempts', 'completed', 'interrupted', 'total_seconds')
        values = [[_number(row[name], name != 'total_seconds') for name in names] for row in (a, b)]
        delta = [right - left for left, right in zip(*values)]
        if any(row[0] != row[1] + row[2] for row in values) or any(value < 0 for value in delta):
            result['phases'][phase] = {'status': 'RESET_OR_INVALID_INTERVAL'}
            continue
        count, completed, interrupted, seconds = delta
        result['phases'][phase] = {'status': 'MONOTONE_DELTA_CONTINUITY_UNVERIFIED',
            'attempts': count, 'completed': completed, 'interrupted': interrupted,
            'recorded_seconds': seconds,
            'seconds_per_recorded_attempt': seconds / count if count else None}
    # Parent delivery_ack overlaps children; never sum phases or difference lifetime max/last.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        print(json.dumps({'status': 'UNAVAILABLE', 'error_type': type(exc).__name__}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
