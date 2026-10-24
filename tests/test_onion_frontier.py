import asyncio
import hashlib

import pytest

from src.collectors.onion_frontier import OnionFrontierTransport
from src.collectors.transports import TransportError
from tests.test_onion_collection import URL, Pages, host


def run(awaitable):
    return asyncio.run(awaitable)


def test_discovery_exceeds_old_host_and_page_caps_and_survives_restart(tmp_path):
    path = tmp_path / 'frontier.sqlite'
    frontier = OnionFrontierTransport(path)
    roots = ['http://' + host(n) + '/' for n in range(1, 41)]
    frontier.enqueue(roots + [URL + 'thread/' + str(n) for n in range(1600)], 'frontier_job:parent', 1)
    assert frontier.report()['discovered_hosts'] == 40
    assert frontier.report()['pages'] == 1640
    run(frontier.stop())
    restored = OnionFrontierTransport(path)
    try:
        assert restored.report()['pending_pages'] == 1640
        assert restored.report()['page_count_limit'] is None
        assert restored.report()['crawl_depth_limit'] is None
    finally:
        run(restored.stop())


def test_outbox_restarts_and_children_require_canonical_parent(tmp_path):
    path = tmp_path / 'frontier.sqlite'
    links = ''.join(f'<a href="/thread/{n}">same</a>' for n in range(300))
    pages = Pages(['<p>public claimed launch</p>' + links])
    frontier = OnionFrontierTransport(path, client_factory=lambda _: pages)
    frontier.enqueue([URL], 'frontier_job:origin', 1)
    first = run(frontier.fetch())
    assert len(first[0]['discovered_onion_links']) == 300
    assert frontier.report()['pages'] == 1
    run(frontier.stop())
    restored = OnionFrontierTransport(path, client_factory=lambda _: pytest.fail('outbox redelivery made a network request'))
    try:
        assert run(restored.fetch()) == first
        with pytest.raises(TransportError):
            restored.acknowledge(first[0]['observation_id'], None)
        restored.acknowledge(first[0]['observation_id'], 'durable-observation')
        assert restored.report()['pages'] == 301
        restored.acknowledge(first[0]['observation_id'], 'durable-observation')
        assert restored.report()['pages'] == 301
        assert restored.db.execute('SELECT parent,depth,urls.source_ref FROM urls JOIN origins USING(url) WHERE depth=1 LIMIT 1').fetchone()[:] == (URL, 1, 'frontier_job:durable-observation')
    finally:
        run(restored.stop())


def test_one_denied_host_does_not_starve_another_and_retry_after_is_retained(tmp_path):
    class Denied(Pages):
        async def get(self, url):
            return 429, 'slow down', {'Retry-After': '7200'}
    other = 'http://' + host(2) + '/'
    pages = Pages(['<p>insider claim</p>'])
    frontier = OnionFrontierTransport(tmp_path / 'frontier.sqlite',
        client_factory=lambda selected: Denied([]) if selected == host() else pages)
    try:
        frontier.enqueue([URL, other], 'frontier_job:origin', 1)
        assert run(frontier.fetch()) == []
        assert frontier.report()['observed_pages'] == 0
        refused = frontier.db.execute('SELECT last_error,next_at FROM urls WHERE url=?', (URL,)).fetchone()
        assert refused['last_error'] == 'HTTP_429'
        assert refused['next_at'] > 7200
        assert run(frontier.fetch())[0]['url'] == other
    finally:
        run(frontier.stop())


def test_legacy_migration_is_idempotent_and_preserves_historical_debt(tmp_path):
    frontier = OnionFrontierTransport(tmp_path / 'frontier.sqlite')
    try:
        state = {'admitted': {'a': {'url': URL, 'source_ref': 'frontier_job:legacy', 'discovered_at': 1}},
                 'pending': {}, 'capacity_debt': 7}
        frontier.migrate(state)
        frontier.migrate(state)
        assert frontier.report()['pages'] == 1
        assert frontier.report()['legacy_candidate_capacity_debt'] == 7
    finally:
        run(frontier.stop())


def test_access_challenge_is_deferred_without_claim_observation(tmp_path):
    frontier = OnionFrontierTransport(tmp_path / 'frontier.sqlite',
        client_factory=lambda _: Pages(['Please complete the CAPTCHA']))
    try:
        frontier.enqueue([URL, 'http://127.0.0.1/'], 'frontier_job:origin', 1)
        assert run(frontier.fetch()) == []
        assert frontier.report()['pages'] == 1
        assert frontier.report()['observed_pages'] == 0
        assert frontier.report()['deferred_pages'] == 1
    finally:
        run(frontier.stop())


def test_unacknowledged_legacy_observation_is_imported_before_new_fetch(tmp_path):
    from src.collectors.onion import OnionResearchTransport
    source_id = 'onion:legacy'
    state_path = tmp_path / (hashlib.sha256(source_id.encode()).hexdigest() + '.json')
    old = OnionResearchTransport(source_id, URL, source_ref='frontier_job:legacy',
        client=Pages(['<p>claimed launch</p><a href="/next">next</a>']), state_path=state_path)
    pending = run(old.fetch())
    frontier = OnionFrontierTransport(tmp_path / 'frontier.sqlite', client_factory=lambda _: pytest.fail('legacy outbox was skipped'))
    try:
        frontier.migrate({'admitted': {'old': {'source_id': source_id, 'url': URL,
            'source_ref': 'frontier_job:legacy', 'discovered_at': 1}}}, tmp_path)
        imported = run(frontier.fetch())[0]
        assert imported['observation_id'] == pending[0]['observation_id']
        assert imported['observed_at'] == pending[0]['observed_at']
        assert frontier.report()['pages'] == 2
        frontier.acknowledge(imported['observation_id'], 'canonical:recovered')
        assert frontier.report()['pending_canonical_delivery'] == 0
        assert state_path.exists()
    finally:
        run(frontier.stop())
