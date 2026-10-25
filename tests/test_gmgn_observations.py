"""Owner-authorized public-address queries cannot acquire trading authority."""
import json
import time
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock

import pytest

from src.research.data_miners import DataMinerPool, RateLimited
from src.research.gmgn_observations import GMGNReadClient, ORIGIN, READS, register_gmgn_miners

ADDRESS = "So11111111111111111111111111111111111111112"
KEY = "fixture-gmgn-test-key"


@pytest.mark.asyncio
async def test_read_query_has_fresh_auth_and_cannot_forward_key_to_redirect():
    http = AsyncMock()
    http.get.return_value = (200, json.dumps({"code": 0, "data": {"tag": "insider", "echo": KEY}}), {})
    record = await GMGNReadClient(http, credential=lambda:KEY).observe("token_holders", ADDRESS)
    url = http.get.call_args.args[0]
    assert urlsplit(url).netloc == urlsplit(ORIGIN).netloc
    assert KEY not in url
    assert parse_qs(urlsplit(url).query)["address"] == [ADDRESS]
    assert abs(int(parse_qs(urlsplit(url).query)["timestamp"][0]) - time.time()) < 5
    assert http.get.call_args.kwargs["headers"]["X-APIKEY"] == KEY
    assert http.get.call_args.kwargs["allow_redirects"] is False
    assert record["claim_status"] == "PROVIDER_CLAIM_UNVERIFIED"
    assert record["execution_authority"] is False
    assert KEY not in json.dumps(record)


@pytest.mark.asyncio
async def test_no_key_and_unapproved_operations_make_no_request():
    http = AsyncMock()
    client = GMGNReadClient(http, credential=lambda:"")
    with pytest.raises(RuntimeError, match="GMGN_API_KEY missing"):
        await client.observe("token_info", ADDRESS)
    for operation in ("swap", "wallet_holdings", "https://elsewhere.example", "follow_wallet"):
        with pytest.raises(ValueError, match="unreviewed"):
            await client.observe(operation, ADDRESS)
    http.get.assert_not_called()


@pytest.mark.asyncio
async def test_provider_cooldown_is_shared_across_operations():
    http = AsyncMock()
    http.get.return_value = (429, "not JSON", {"Retry-After": "180"})
    client = GMGNReadClient(http, credential=lambda:KEY)
    for operation in ("token_info", "creator_launches"):
        with pytest.raises(RateLimited):
            await client.observe(operation, ADDRESS)
    assert http.get.await_count == 1
    assert client.cooldown >= time.monotonic() + 175


@pytest.mark.asyncio
async def test_error_does_not_echo_provider_headers_body_or_credentials():
    http = AsyncMock()
    http.get.side_effect = RuntimeError(KEY)
    client = GMGNReadClient(http, credential=lambda:KEY)
    with pytest.raises(RuntimeError) as exc:
        await client.observe("wallet_activity", ADDRESS)
    assert KEY not in str(exc.value)


def test_all_read_miners_report_missing_key_instead_of_mining_without_it(monkeypatch):
    monkeypatch.delenv("GMGN_API_KEY", raising=False)
    pool = DataMinerPool()
    registered = register_gmgn_miners(pool, http=AsyncMock(), watched_tokens=lambda:[ADDRESS],
                                    watched_wallets=lambda:[ADDRESS], deployers=lambda:[ADDRESS])
    assert len(registered) == len(READS) == 8
    assert not any(registered.values())
    assert all(spec.requires_env == ("GMGN_API_KEY",) for spec in pool._specs.values())
    assert not hasattr(GMGNReadClient, "sign") and not hasattr(GMGNReadClient, "trade")
