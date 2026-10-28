import json
from types import SimpleNamespace
from unittest.mock import patch

from src.research.fork_probe import ForkProbe
from src.runtime.ingestion import MinedRecordIngestion


def retain(probe, identity, available, *, token='token-one', acknowledged=True, ingested=5):
    body = {'probe_id': identity, 'token': token, 'provider': 'local_svm',
            'status': 'FORK_ENVIRONMENT_UNKNOWN', 'native_source_refs': [
                {'available_at': 4, 'ingested_at': ingested}]}
    with probe.lock, probe.db:
        probe.db.execute('INSERT INTO results VALUES(?,?,?,?,?,?)',
            (identity, 'job-'+identity, json.dumps(body), 3,
             'canonical:'+identity if acknowledged else None, available if acknowledged else None))
    return body


def test_queued_identity_is_not_shadowed_by_newer_result(tmp_path):
    probe=ForkProbe(tmp_path/'fork.sqlite')
    try:
        older=retain(probe,'old',10); retain(probe,'new',20)
        assert probe.latest('token-one',as_of=30)['probe_id']=='new'
        assert probe.latest('token-one',as_of=15)['probe_id']=='old'
        assert probe.latest('token-one',as_of=30,probe_id='old')['probe_id']=='old'
        seen=[]
        desk=SimpleNamespace(fork_probe=probe,dataset_builder=SimpleNamespace(
            record_market_observation=lambda token,body:seen.append((token,body))))
        with patch('src.runtime.ingestion.time.time',return_value=30):
            MinedRecordIngestion._ingest_mined_records(desk,'chain:local_fork_roundtrip',[older])
        assert len(seen)==1 and seen[0][0]=='token-one'
        assert seen[0][1]['probe_id']=='old' and seen[0][1]['source_ref']=='canonical:old'
        assert seen[0][1]['type']=='native_fork_roundtrip' and seen[0][1]['timestamp']==10
        assert seen[0][1]['status']=='FORK_ENVIRONMENT_UNKNOWN'  # No forced success.
    finally:
        probe.close()


def test_exact_binding_keeps_ack_token_and_native_pit_guards(tmp_path):
    probe=ForkProbe(tmp_path/'fork.sqlite')
    try:
        retain(probe,'old',10); retain(probe,'pending',20,acknowledged=False)
        retain(probe,'future-source',12,ingested=40)
        assert probe.latest('other-token',as_of=30,probe_id='old') is None
        assert probe.latest('token-one',as_of=9,probe_id='old') is None
        assert probe.latest('token-one',as_of=30,probe_id='unknown') is None
        assert probe.latest('token-one',as_of=30,probe_id='pending') is None
        assert probe.latest('token-one',as_of=30,probe_id='future-source') is None
        assert probe.latest('token-one',as_of=40,probe_id='future-source')['available_at']==12
        seen=[]
        desk=SimpleNamespace(fork_probe=probe,dataset_builder=SimpleNamespace(
            record_market_observation=lambda *args:seen.append(args)))
        with patch('src.runtime.ingestion.time.time',return_value=30):
            MinedRecordIngestion._ingest_mined_records(desk,'chain:local_fork_roundtrip',[
                {'provider':'local_svm','token':'token-one'},
                {'provider':'local_svm','token':'token-one','probe_id':'pending'},
                {'provider':'local_svm','token':'other-token','probe_id':'old'}])
        assert seen==[]
    finally:
        probe.close()
