"""Source repairs must preserve access boundaries and named coverage gaps."""

import asyncio
import json
from pathlib import Path

import pytest

from src.collectors.registry import SourceDeclaration, build_sources, load_declarations
from src.collectors.transports import (
    MastodonTimelineTransport, RssTransport, TransportError, build_transports,
)
from src.research.source_catalogue import REGIONAL_NEWS


class Client:
    def __init__(self, status=200, body="[]"):
        self.status, self.response = status, body
        self.requests = []

    async def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.status, self.response, {}


def test_verified_odaily_substitution_uses_publisher_subdomain_and_preserves_language():
    declarations = load_declarations(str(Path(__file__).resolve().parents[1] / "config/sources.yaml"))
    source = next(row for row in declarations if row.source_id == "rss:cn-odaily")
    assert source.options["url"] == "https://rss.odaily.news/rss/newsflash"
    assert source.language == "zh" and source.region == "cn"
    endpoint = next(row for row in REGIONAL_NEWS if row.name == "odaily_cn")
    assert endpoint.url == source.options["url"]
    xml = """<rss><channel><item><guid>control-1</guid><title>测试新闻</title>
             <link>https://www.odaily.news/zh-CN/newsflash/control</link>
             <pubDate>Tue, 20 Oct 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    transport = RssTransport(source.source_id, source.options["url"], Client(body=xml), language=source.language)
    records = asyncio.run(transport.fetch())
    assert len(records) == 1 and records[0]["language"] == "zh"
    assert records[0]["published_epoch"] is not None


def test_three_unavailable_declarations_cannot_count_as_ready_or_attempt_network():
    declarations = load_declarations(str(Path(__file__).resolve().parents[1] / "config/sources.yaml"))
    unavailable_ids = {"mastodon:mastodon-social", "mastodon:mastodon-online", "rss:cn-techflow"}
    selected = [row for row in declarations if row.source_id in unavailable_ids]
    client = Client()
    transports, transport_report, _ = build_transports(selected, client)
    assert transports == {}
    assert transport_report.built == 0
    assert {source for source, _ in transport_report.unavailable} == unavailable_ids
    assert all("UNAVAILABLE:" in reason for _, reason in transport_report.unbuilt)
    # Even a supplied fetcher must not turn explicitly unavailable coverage
    # into READY. Source config holds the observed coverage disposition.
    sources, source_report = build_sources(selected, fetchers={name: lambda: [] for name in unavailable_ids})
    assert sources == [] and source_report.ready == 0
    assert source_report.by_state == {"UNAVAILABLE": 3}
    assert {source for source, _ in source_report.problems} == unavailable_ids
    assert client.requests == []


def test_mastodon_authenticated_user_422_is_explicit_unavailable_not_invalid_limit():
    client = Client(422, json.dumps({"error": "This method requires an authenticated user"}))
    transport = MastodonTimelineTransport("mastodon:control", "https://mastodon.social", client)
    with pytest.raises(TransportError, match="UNAVAILABLE: HTTP 422.*requires authorization"):
        asyncio.run(transport())
    assert "limit=40" in client.requests[0][0]  # the documented valid maximum
    assert "Authorization" not in client.requests[0][1].get("headers", {})
    assert transport.failures == 1 and transport.records == 0
    assert "UNAVAILABLE" in transport.last_error


def test_other_mastodon_422_remains_a_real_error_without_false_auth_diagnosis():
    client = Client(422, json.dumps({"error": "unknown validation failure"}))
    transport = MastodonTimelineTransport("m", "https://example.test", client)
    with pytest.raises(TransportError, match="HTTP 422") as caught:
        asyncio.run(transport.fetch())
    assert "requires authorization" not in str(caught.value)


def test_public_mastodon_instances_still_use_existing_transport():
    declaration = SourceDeclaration("mastodon:public", "mastodon", options={"instance": "https://example.test"})
    transports, report, _ = build_transports([declaration], Client())
    assert declaration.source_id in transports and report.built == 1
    assert not report.unavailable
