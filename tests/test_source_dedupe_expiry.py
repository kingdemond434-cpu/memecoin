"""Streaming suppression expires without erasing within-window provenance."""
from src.collectors.event_source import Event, SourceClass, SourceMesh


def event(source='early', text='launch', **metadata):
    return Event(source, SourceClass.WEB, 100., 100., text=text, metadata=metadata)


def test_live_repeat_keeps_first_source_until_window_then_redelivers():
    mesh=SourceMesh(dedupe_window=60)
    first=event();repeat=event('later')
    mesh._publish(first,100)
    assert mesh.drain()==[first]
    mesh._publish(repeat,160)
    assert mesh.drain()==[]
    assert mesh.repeaters_of(first.content_hash)==['early','later']
    mesh._publish(repeat,160.01)
    assert mesh.drain()==[repeat]
    assert mesh.repeaters_of(first.content_hash)==['later']


def test_idle_health_expires_stale_content_without_dropping_queued_event():
    mesh=SourceMesh(dedupe_window=60)
    first=event();mesh._publish(first,100)
    report=mesh.health(161)
    assert report['dedupe_cached_contents']==0
    assert report['pending_events']==1
    assert mesh.drain()==[first]


def test_clock_regression_does_not_hide_older_entries_behind_newer_ones():
    mesh=SourceMesh(dedupe_window=60)
    newer=event(text='newer');older=event(text='older')
    mesh._publish(newer,200);mesh._publish(older,100)
    mesh.health(161)
    assert mesh.repeaters_of(newer.content_hash)==['early']
    assert mesh.repeaters_of(older.content_hash)==[]
    mesh.health(261)
    assert mesh.health(261)['dedupe_cached_contents']==0


def test_durable_outbox_redelivery_is_not_changed_by_content_expiration():
    mesh=SourceMesh(dedupe_window=60)
    durable=event(onion_observation_id='snapshot:1')
    mesh._publish(durable,100);mesh._publish(durable,101)
    assert mesh.drain()==[durable]
    mesh._publish(durable,102)
    assert mesh.drain()==[durable]
    assert mesh.health(103)['dedupe_cached_contents']==0


def test_live_window_state_does_not_grow_with_total_stream_lifetime():
    mesh=SourceMesh(dedupe_window=10)
    for tick in range(10000):
        mesh._publish(event(text=str(tick)),float(tick))
        mesh.drain()
    assert mesh.health(9999)['dedupe_cached_contents']==11
    assert mesh.repeaters_of(event(text='9999').content_hash)==['early']
