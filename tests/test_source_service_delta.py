import copy
import json

import pytest

from tools.source_service_delta import compare


def receipts():
    a = {'observed_at': 10, 'journal': {'accepted': 100, 'acknowledged': 20},
         'neural': {'pending_inputs': 50, 'canonical_windows': 2},
         'source_phase_timing': {'phases': {'delivery_ack': dict(attempts=2, completed=1,
             interrupted=1, total_seconds=4, max_seconds=3, last_seconds=1)}}}
    b = copy.deepcopy(a)
    b['observed_at'] = 20
    b['journal'] = {'accepted': 150, 'acknowledged': 30}
    b['neural'] = {'pending_inputs': 40, 'canonical_windows': 3}
    b['source_phase_timing']['phases']['delivery_ack'].update(attempts=4, completed=3, total_seconds=10)
    return a, b


def test_rates_gauge_and_overlap_without_mutation_or_invented_latency():
    a, b = receipts(); original = copy.deepcopy((a, b))
    result = compare(a, b)
    assert result['pending_journal_delta'] == 40
    assert result['counters']['journal.accepted']['per_second'] == 5
    assert result['counters']['neural.pending_inputs']['delta'] == -10
    phase = result['phases']['delivery_ack']
    assert phase['seconds_per_recorded_attempt'] == 3
    assert 'max_seconds' not in phase and 'last_seconds' not in phase
    assert result['process_continuity'] == 'UNVERIFIED'
    assert not result['throughput_capacity_verified']
    assert (a, b) == original


def test_reset_excludes_backlog_and_phase_average():
    a, b = receipts(); b['journal']['acknowledged'] = 1
    b['source_phase_timing']['phases']['delivery_ack']['total_seconds'] = 1
    result = compare(a, b)
    assert 'pending_journal_delta' not in result
    assert result['phases']['delivery_ack'] == {'status': 'RESET_OR_INVALID_INTERVAL'}


def test_missing_phase_is_not_zero_baseline_and_unknown_fields_not_exported():
    a, b = receipts(); a['source_phase_timing']['phases'].clear()
    b['source_phase_timing']['phases']['PRIVATE_SENTINEL'] = {'body': 'PRIVATE_SENTINEL'}
    b['private_url'] = 'PRIVATE_SENTINEL'
    result = compare(a, b)
    assert result['phases'] == {}
    assert 'PRIVATE_SENTINEL' not in json.dumps(result)


@pytest.mark.parametrize('bad', [True, -1, float('nan'), float('inf'), 'PRIVATE_SENTINEL'])
def test_malformed_counters_fail_closed(bad):
    a, b = receipts(); b['journal']['accepted'] = bad
    with pytest.raises(ValueError):compare(a, b)


@pytest.mark.parametrize('clock', [10, 9])
def test_clock_order_required(clock):
    a, b = receipts(); b['observed_at'] = clock
    with pytest.raises(ValueError):compare(a, b)


def test_inconsistent_attempt_counts_rejected_and_zero_sample_has_no_average():
    a, b = receipts()
    b['source_phase_timing']['phases'] = copy.deepcopy(a['source_phase_timing']['phases'])
    assert compare(a, b)['phases']['delivery_ack']['seconds_per_recorded_attempt'] is None
    b['source_phase_timing']['phases']['delivery_ack']['completed'] = 0
    assert compare(a, b)['phases']['delivery_ack']['status'] == 'RESET_OR_INVALID_INTERVAL'
