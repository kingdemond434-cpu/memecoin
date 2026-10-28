import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.research.dataset_builder import LaunchEpisode, PointInTimeDatasetBuilder
from src.research.fork_probe import ForkProbe
from src.runtime.ingestion import MinedRecordIngestion
from src.runtime.native_fork_delivery import NativeForkOwnerDelivery


def seed(probe, identity, token, *, available=10, acknowledged=True, ingested=6):
    body = {'probe_id': identity, 'token': token, 'provider': 'local_svm',
            'status': 'FORK_ENVIRONMENT_UNKNOWN', 'native_source_refs': [
                {'available_at': 5, 'ingested_at': ingested}]}
    with probe.lock, probe.db:
        probe.db.execute('INSERT INTO results VALUES(?,?,?,?,?,?)',
            (identity, 'job-'+identity, json.dumps(body), 3,
             'canonical:'+identity if acknowledged else None, available if acknowledged else None))
    return body


def builder():
    result = PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
    result.active_episodes = {'active': LaunchEpisode('active','solana',1,'creator','venue','pool','quote')}
    result.completed_episodes = {'completed': LaunchEpisode('completed','solana',1,'creator','venue','pool','quote')}
    return result


def test_actual_owner_counts_real_episode_application_and_missing_episode(tmp_path):
    probe = ForkProbe(tmp_path/'fork.sqlite')
    try:
        records = [seed(probe,'older','active'),seed(probe,'done','completed'),seed(probe,'unknown','unknown')]
        seed(probe,'newer','active',available=20)
        dataset = builder(); desk = SimpleNamespace(fork_probe=probe,dataset_builder=dataset)
        with patch('src.runtime.ingestion.time.time',return_value=30):
            MinedRecordIngestion._ingest_mined_records(desk,'chain:local_fork_roundtrip',records)
        report = desk._native_fork_owner_delivery.report()
        assert report['counts']['records_seen']==3
        assert report['counts']['in_memory_applied']==2
        assert report['counts']['episode_unavailable']==1
        older = dataset.active_episodes['active'].market_observations[0]
        assert older['probe_id']=='older' and older['source_ref']=='canonical:older'
        assert older['timestamp']==10 and older['status']=='FORK_ENVIRONMENT_UNKNOWN'
        assert len(dataset.completed_episodes['completed'].market_observations)==1
        assert not report['durable_acceptance'] and not report['full_replay_verified']
        assert not report['real_fill'] and not report['promotion_eligible']
    finally:probe.close()


def test_ack_identity_and_native_pit_failures_never_count_as_applied(tmp_path):
    probe = ForkProbe(tmp_path/'fork.sqlite')
    try:
        older=seed(probe,'older','active')
        pending=seed(probe,'pending','active',acknowledged=False)
        future=seed(probe,'future','active',ingested=40)
        dataset=builder();desk=SimpleNamespace(fork_probe=probe,dataset_builder=dataset)
        with patch('src.runtime.ingestion.time.time',return_value=30):
            MinedRecordIngestion._ingest_mined_records(desk,'chain:local_fork_roundtrip',[
                {'provider':'local_svm','token':'active'}, {**older,'token':'wrong'}, pending, future])
        counts=desk._native_fork_owner_delivery.report()['counts']
        assert counts['records_seen']==4 and counts['invalid_identity']==1 and counts['proof_unavailable']==3
        assert counts['in_memory_applied']==counts['episode_unavailable']==0
        assert dataset.active_episodes['active'].market_observations==[]
    finally:probe.close()


@pytest.mark.parametrize('error,key',[(RuntimeError,'application_errors'),(KeyboardInterrupt,'interrupted')])
def test_sink_failure_propagates_and_never_becomes_acceptance(error,key):
    tracker=NativeForkOwnerDelivery()
    def fail(*args):raise error('isolated sink fixture')
    with pytest.raises(error):
        tracker.deliver({'token':'active','probe_id':'id'},
            probe=SimpleNamespace(latest=lambda *a,**kw:{'probe_id':'id','available_at':10}),
            builder=SimpleNamespace(record_market_observation=fail),as_of=30)
    counts=tracker.report()['counts'];assert counts[key]==1 and counts['in_memory_applied']==0


def test_normal_legacy_sink_return_is_unconfirmed_and_report_is_snapshot():
    tracker=NativeForkOwnerDelivery()
    tracker.deliver({'token':'active','probe_id':'id'},
        probe=SimpleNamespace(latest=lambda *a,**kw:{'probe_id':'id','available_at':10}),
        builder=SimpleNamespace(record_market_observation=lambda *a:None),as_of=30)
    report=tracker.report();assert report['counts']['application_unconfirmed']==1
    report['counts']['in_memory_applied']=999
    assert tracker.report()['counts']['in_memory_applied']==0


def test_explicit_dataset_result_is_in_memory_only_and_keeps_input_timestamp():
    dataset=builder();observation={'timestamp':10,'fact':'public fixture'}
    assert dataset.record_market_observation('missing',observation) is False
    assert dataset.record_market_observation('active',observation) is True
    assert observation=={'timestamp':10,'fact':'public fixture'}
    assert dataset.active_episodes['active'].market_observations==[observation]


@pytest.mark.parametrize('error,key',[(RuntimeError,'lookup_errors'),(KeyboardInterrupt,'interrupted')])
def test_lookup_failure_propagates_without_application(error,key):
    tracker=NativeForkOwnerDelivery();seen=[]
    def fail(*a,**kw):raise error('isolated lookup fixture')
    with pytest.raises(error):
        tracker.deliver({'token':'active','probe_id':'id'},probe=SimpleNamespace(latest=fail),
            builder=SimpleNamespace(record_market_observation=lambda *a:seen.append(a)),as_of=30)
    assert tracker.report()['counts'][key]==1 and seen==[]


def test_mismatched_proof_never_reaches_episode():
    tracker=NativeForkOwnerDelivery();seen=[]
    tracker.deliver({'token':'active','probe_id':'id'},
        probe=SimpleNamespace(latest=lambda *a,**kw:{'probe_id':'different','available_at':10}),
        builder=SimpleNamespace(record_market_observation=lambda *a:seen.append(a)),as_of=30)
    assert tracker.report()['counts']['identity_mismatch']==1 and seen==[]
