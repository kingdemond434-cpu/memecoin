import asyncio
import base64
import hashlib
from pathlib import Path

import pytest

from src.collectors.adapters import onion_source
from src.collectors.event_source import SourceClass
from src.collectors.onion import OnionResearchTransport, TorHttpClient, onion_host, onion_url
from src.collectors.registry import SourceDeclaration, build_sources, load_declarations
from src.collectors.transports import HttpClient, TransportError, build_transports, stop_transports


def host(seed=1):
    key = bytes([seed]) * 32
    checksum = hashlib.sha3_256(b".onion checksum" + key + b"\x03").digest()[:2]
    return base64.b32encode(key + checksum + b"\x03").decode().lower() + ".onion"


URL = "http://" + host() + "/"


class Pages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    async def get(self, url):
        self.calls.append(url)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return 200, result, {}

    async def close(self):
        pass


@pytest.mark.parametrize("url", ["https://example.com/", "http://localhost/",
    "http://127.0.0.1/", "file:///tmp/data", "http://short.onion/",
    "http://" + "a" * 56 + ".onion/", URL.replace("http://", "http://user:pass@"),
    URL.replace(".onion/", ".onion:8080/"), URL + "bad\npath", URL + "\\bad"])
def test_invalid_unapproved_and_credential_urls_are_refused(url):
    with pytest.raises(TransportError):
        onion_url(url)


def test_checksum_and_origin_binding():
    assert onion_host(host())
    assert onion_url(URL + "#fragment") == URL
    with pytest.raises(TransportError):
        onion_url("http://" + host(2), {host()})


def test_ordinary_client_never_attempts_onion_dns(monkeypatch):
    async def forbidden_session(*args):
        pytest.fail("ordinary TCP session opened for onion")
    monkeypatch.setattr(HttpClient, "session", forbidden_session)
    with pytest.raises(TransportError):
        asyncio.run(HttpClient().get(URL))


def test_forum_depth_and_external_link_candidates_are_bounded():
    external = "http://" + host(2) + "/forum"
    pages = Pages([f'<p>insider wallet launch</p><script>evil()</script>'
                   f'<a href="/thread">thread</a><a href="{external}">source</a>',
                   '<p>dev follow-up</p><a href="/page2">more</a>', '<p>insider wallet launch</p>'])
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin", client=pages,
                                       max_pages=2, max_depth=1)
    first = asyncio.run(transport.fetch())
    second = asyncio.run(transport.fetch())
    assert first[0]["observation_kind"] == "initial_snapshot"
    assert first[0]["published_at"] is None and "evil()" not in first[0]["text"]
    assert second[0]["url"] == URL + "thread"
    assert transport.report()["pages"] == 2
    assert external in transport.report()["candidate_links"]
    assert external not in pages.calls


def test_identical_content_is_not_new_evidence_and_failure_does_not_consume_changes():
    pages = Pages(['<a href="/thread">new</a>', '<a href="/thread">new</a>',
                   TransportError("offline"), '<a href="/thread">new</a>', '<p>thread</p>'])
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin", client=pages)
    asyncio.run(transport.fetch())
    transport.queue.rotate(1)  # revisit the same root: no new content
    assert asyncio.run(transport.fetch()) == []
    transport.pages_per_poll = 2
    snapshot = dict(transport.hashes)
    with pytest.raises(TransportError):
        asyncio.run(transport.fetch())
    assert transport.hashes == snapshot


def test_second_page_failure_rolls_back_first_page_delivery():
    pages = Pages(['<a href="/thread">root</a>', '<p>new thread</p>', TransportError("offline")])
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin", client=pages)
    asyncio.run(transport.fetch())
    transport.pages_per_poll = 2
    before = list(transport.queue)
    with pytest.raises(TransportError):
        asyncio.run(transport.fetch())
    assert URL + "thread" not in transport.hashes
    assert list(transport.queue) == before


def test_durable_outbox_replays_original_observation_until_canonical_handoff(tmp_path):
    state = tmp_path / "onion.json"
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin",
        client=Pages(["<p>claimed dev launch</p>"]), state_path=state)
    first = asyncio.run(transport.fetch())
    resumed = OnionResearchTransport("onion:forum", URL, source_ref="public-origin",
        client=Pages(["<p>claimed dev launch</p>"]), state_path=state)
    assert asyncio.run(resumed.fetch()) == first
    assert resumed.client.calls == []
    with pytest.raises(TransportError):
        resumed.acknowledge(first[0]["observation_id"], None)
    resumed.acknowledge(first[0]["observation_id"], "canonical-job-id")
    assert asyncio.run(resumed.fetch()) == []
    with pytest.raises(TransportError, match="checkpoint invalid"):
        OnionResearchTransport("onion:another", URL, source_ref="public-origin", client=Pages([]), state_path=state)


@pytest.mark.parametrize("body", ["<script>run()</script>", "Please complete the CAPTCHA", "Sign in to view"])
def test_access_challenges_do_not_count_as_observed_forum_content(body):
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin", client=Pages([body]))
    with pytest.raises(TransportError, match="public content unavailable"):
        asyncio.run(transport.fetch())
    assert transport.report()["status"] == "NOT_OBSERVED"


def test_runtime_retains_claim_provenance_and_acknowledges_only_durable_intake(tmp_path):
    from types import SimpleNamespace
    from src.runtime.source_intelligence import SourceIntelligence
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin",
        client=Pages(["<p>insider wallet launch</p>"]), state_path=tmp_path / "state.json")
    event = asyncio.run(onion_source("onion:forum", transport).collect())[0]
    captured = []
    def enqueue(kind, payload, **kwargs):
        captured.append(payload)
        return None if len(captured) == 1 else "canonical-job-id"
    desk = SimpleNamespace(transports={"onion:forum": transport}, _enqueue_population_work=enqueue)
    assert SourceIntelligence._record_population_narrative(desk, event) is None
    assert transport.pending
    assert SourceIntelligence._record_population_narrative(desk, event) == "canonical-job-id"
    assert not transport.pending
    assert captured[0] == captured[1]
    assert captured[0]["public_source_evidence"]["url"] == URL
    assert captured[0]["public_source_evidence"]["claim_status"] == "UNVERIFIED"


def test_adapter_preserves_unverified_claims_without_official_authenticity():
    pages = Pages(['<p>insider dev presale wallet 0x' + 'a' * 40 + '</p>'])
    transport = OnionResearchTransport("onion:forum", URL, source_ref="public-origin", client=pages)
    events = asyncio.run(onion_source("onion:forum", transport).collect())
    event = events[0]
    assert event.source_class is SourceClass.WEB
    assert event.source_at == event.observed_at
    assert event.metadata["publication_time_known"] is False
    assert event.metadata["claim_status"] == "UNVERIFIED"
    assert event.metadata["claim_topics"] == ["launch", "wallet", "claimed_insider"]
    assert event.metadata["evm_address_mentions"] == ["0x" + "a" * 40]
    assert event.metadata["execution_authority"] is False


def test_registry_reports_missing_proxy_and_dread_endpoint_honestly(monkeypatch):
    monkeypatch.delenv("MEMECOIN_TOR_PROXY_URL", raising=False)
    declaration = SourceDeclaration("onion:test", "onion", options={"url": URL, "source_ref": "public-origin"})
    transports, report, _ = build_transports([declaration])
    assert not transports and report.built == 0
    assert "Tor proxy" in str(report.unbuilt)
    pending = load_declarations(str(Path(__file__).parents[1] / "config/sources.onion.example.yaml"))
    assert len(pending) == 1
    transports, report, _ = build_transports(pending)
    assert not transports and report.unavailable[0][0] == "onion:dread-public-discussions"


def test_factory_adapter_and_shutdown_are_wired(monkeypatch):
    monkeypatch.setenv("MEMECOIN_TOR_PROXY_URL", "socks5://127.0.0.1:9050")
    declaration = SourceDeclaration("onion:test", "onion", options={"url": URL, "source_ref": "public-origin"})
    transports, report, client = build_transports([declaration])
    sources, registry = build_sources([declaration], transports)
    assert report.built == registry.ready == 1
    assert sources[0].source_class is SourceClass.WEB
    assert sources[0].poll_timeout_seconds == 45
    asyncio.run(stop_transports(transports, client))


def test_actual_socks5_handshake_sends_hostname_and_rejects_cross_origin_redirect():
    async def run():
        requested = []
        async def socks(reader, writer):
            try:
                header = await reader.readexactly(2)
                assert header[0] == 5
                await reader.readexactly(header[1])
                writer.write(b"\x05\x00")
                await writer.drain()
                request = await reader.readexactly(4)
                assert request == b"\x05\x01\x00\x03"  # domain name, never an IP
                size = (await reader.readexactly(1))[0]
                requested.append((await reader.readexactly(size)).decode())
                await reader.readexactly(2)
                writer.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x50")
                await writer.drain()
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 302 Found\r\nLocation: http://example.com/\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(socks, "127.0.0.1", 0)
        client = TorHttpClient("socks5://127.0.0.1:" + str(server.sockets[0].getsockname()[1]), host())
        try:
            with pytest.raises(TransportError):
                await client.get(URL)
            assert requested == [host()]
        finally:
            await client.close()
            server.close()
            await server.wait_closed()
    asyncio.run(run())


def test_ordinary_http_redirect_cannot_send_an_onion_to_local_dns():
    async def run():
        async def redirect(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(("HTTP/1.1 302 Found\r\nLocation: " + URL +
                              "\r\nContent-Length: 0\r\n\r\n").encode())
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(redirect, "127.0.0.1", 0)
        client = HttpClient(timeout_s=1)
        try:
            with pytest.raises(TransportError, match="onion redirects"):
                await client.get("http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]))
        finally:
            await client.close()
            server.close()
            await server.wait_closed()
    asyncio.run(run())
