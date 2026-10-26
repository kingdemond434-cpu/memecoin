"""Source intake reads canonical bindings without loading diagnostic job fields."""
from contextlib import contextmanager

import pytest

from src.research.frontier import FrontierIndex


def test_payload_reference_is_fresh_across_connections_and_restart(tmp_path):
    path = tmp_path / 'frontier.sqlite'
    first, second = FrontierIndex(path), FrontierIndex(path)
    assert second.job_payload_ref('work-source') is None
    first.enqueue('NARRATIVE_OBSERVATION', 'evidence:source', job_id='work-source',
                  axes={'data_source': 'source'}, evidence_refs=['evidence:source'], now=1)
    assert second.job_payload_ref('work-source') == second.job('work-source')['payload_ref']
    first.close(); second.close()
    reopened = FrontierIndex(path)
    assert reopened.job_payload_ref('work-source') == 'evidence:source'
    reopened.close()
    with pytest.raises(RuntimeError, match='closed'):
        reopened.job_payload_ref('work-source')


def test_payload_lookup_executes_one_query_without_diagnostic_tables(tmp_path, monkeypatch):
    frontier = FrontierIndex(tmp_path / 'frontier.sqlite')
    frontier.enqueue('NARRATIVE_OBSERVATION', 'evidence:source', job_id='work-source', now=1)
    original = frontier._db
    queries = []
    @contextmanager
    def traced():
        with original() as db:
            db.set_trace_callback(queries.append)
            yield db
    monkeypatch.setattr(frontier, '_db', traced)
    assert frontier.job_payload_ref('work-source') == 'evidence:source'
    selects = [q for q in queries if q.startswith('SELECT')]
    assert len(selects) == 1
    assert 'coordinates' not in selects[0] and 'job_tags' not in selects[0]
    assert frontier.job_payload_ref('missing') is None
    frontier.close()

@pytest.mark.asyncio
async def test_missing_reference_retries_source_without_canonical_ack(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.test_population_callbacks import desk
    from src.collectors.event_source import Event, SourceClass, SourceMesh

    value = desk(tmp_path)
    mesh = SourceMesh(journal_path=tmp_path / 'intake.sqlite')
    value.source_mesh = mesh
    value.global_config = {'ops_state_dir': str(tmp_path)}
    value.social_intel = SimpleNamespace()
    value._running = True
    value.transports = {}
    await mesh.publish_batch([Event('feed', SourceClass.NEWS, 1, 2, text='public fixture')])
    monkeypatch.setattr(value.unknown_engine.frontier, 'job_payload_ref', lambda job: None)
    original_retry = mesh.retry_event
    retried = []
    async def retry(event):
        await original_retry(event)
        retried.append(event)
        value._running = False
    monkeypatch.setattr(mesh, 'retry_event', retry)
    monkeypatch.setattr(value, '_index_source_event', lambda *a, **k: pytest.fail('missing reference indexed'))
    try:
        import asyncio
        await asyncio.wait_for(value._source_consumer_loop(), 30)
        report = mesh.journal.report(10)
        assert report['canonical_acknowledged'] == 0
        assert report['pending_events'] == 1 and len(retried) == 1
        assert mesh.journal.db.execute('SELECT body FROM intake').fetchone()[0] is not None
    finally:
        await mesh.stop()
        mesh.close()
        if getattr(value, 'public_claim_links', None):
            value.public_claim_links.close()
        value.unknown_engine.frontier.close()
