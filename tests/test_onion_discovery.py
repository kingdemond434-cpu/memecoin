import asyncio
import copy
from types import SimpleNamespace

from src.collectors.adapters import onion_directory_source
from src.collectors.event_source import Event, EventSource, SourceClass, SourceMesh
from src.collectors.onion import OnionDirectoryTransport
from src.collectors.onion_discovery import COMPONENT, candidate_urls, discover, drain, restore
from src.collectors.transports import HttpClient, stop_transports
from tests.test_onion_collection import URL, Pages, host


class Framework:
    def __init__(self):
        self.state = {}
        self.writes = 0

    def get_research_component(self, name):
        return copy.deepcopy(self.state.get(name))

    def checkpoint_research_component(self, name, state):
        self.writes += 1
        self.state[name] = copy.deepcopy(state)


def desk(path, framework=None):
    return SimpleNamespace(unknown_engine=SimpleNamespace(framework=framework or Framework()),
        http_client=HttpClient(), global_config={"onion_auto_discovery": True, "onion_frontier_path": path},
        transports={}, source_fetchers={}, source_mesh=SourceMesh())


def event():
    return Event("public:source", SourceClass.WEB, 1., 2.,
                 text="Public forum link " + URL,
                 metadata={"discovered_onion_links": ["http://" + host(2) + "/discussion"]})


def test_links_survive_restart_without_source_caps_or_framework_rewrites(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMECOIN_TOR_PROXY_URL", "socks5://127.0.0.1:9050")
    current = desk(tmp_path / 'onions.sqlite')
    discover(current, event(), None)
    assert not current.transports
    discover(current, event(), "durable-origin-job")
    frontier = current.onion_frontier
    assert len(current.transports) == 1
    assert frontier.report()['discovered_hosts'] == 2
    assert {row[0] for row in frontier.db.execute('SELECT source_ref FROM urls')} == {'frontier_job:durable-origin-job'}
    discover(current, event(), "same-observation-again")
    assert len(current.transports) == 1
    assert current.unknown_engine.framework.writes == 0
    drain(current)
    assert len(current.transports) == 1
    restored = desk(tmp_path / 'onions.sqlite', current.unknown_engine.framework)
    assert restore(restored) == 2
    asyncio.run(stop_transports(current.transports, current.http_client))
    asyncio.run(stop_transports(restored.transports, restored.http_client))


def test_missing_proxy_retains_candidates_and_disabled_discovery_opens_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMECOIN_TOR_PROXY_URL", raising=False)
    current = desk(tmp_path / 'onions.sqlite')
    discover(current, event(), "durable-origin-job")
    assert not current.transports
    assert current.onion_frontier.report()['pending_pages'] == 2
    current.global_config["onion_auto_discovery"] = False
    assert restore(current) == 0
    asyncio.run(current.onion_frontier.stop())


def test_public_announcement_page_exposes_link_provenance_to_discovery():
    client = Pages(['<p>Public announcement</p><a href="' + URL + '">discussion</a>'])
    transport = OnionDirectoryTransport("directory:public", "https://example.test/forum", client)
    source = onion_directory_source(transport.source_id, transport)
    record = asyncio.run(source.collect())[0]
    assert record.source_class is SourceClass.WEB
    assert candidate_urls(record) == [URL]
    assert record.metadata["publication_time_known"] is False


def test_bare_forum_and_telegram_onion_addresses_are_discovered():
    bare = Event("public:chat", SourceClass.CHAT, 1., 2., text="Public forum " + host() + "/thread")
    assert candidate_urls(bare) == [URL + "thread"]


def test_unaccepted_durable_claim_can_be_redelivered_through_mesh_dedupe():
    mesh = SourceMesh()
    claim = Event("onion:source", SourceClass.WEB, 1., 2., text="claimed launch",
                  metadata={"onion_observation_id": "stable-observation"})
    assert mesh._admit(claim, 2.)
    assert mesh._admit(claim, 3.)
    ordinary = Event("public:source", SourceClass.WEB, 1., 2., text="ordinary post")
    assert mesh._admit(ordinary, 2.)
    assert not mesh._admit(ordinary, 3.)


def test_dynamically_discovered_source_really_starts_after_mesh_start():
    class OneShot(EventSource):
        async def poll(self):
            return [Event(self.source_id, SourceClass.WEB, 1., 2., text="unique-discovered-content")]
    async def run():
        mesh = SourceMesh()
        await mesh.start()
        try:
            source = OneShot("onion:new", SourceClass.WEB)
            await asyncio.to_thread(mesh.add, source)
            mesh.add(source)
            received = await asyncio.wait_for(mesh.next_event(), timeout=1)
            assert received.source_id == "onion:new"
            assert len(mesh.sources) == len(mesh._producers) == 1
        finally:
            await mesh.stop()
    asyncio.run(run())
