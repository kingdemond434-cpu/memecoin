"""Raw window comparisons and complete target rotation, not marketing scores."""
import json
from types import SimpleNamespace

import pytest

from src.research.solana_miners import (
    DEXSCREENER_BATCH, dexscreener_pairs_miner, dexscreener_window_measurements,
    holder_structure_miner, token_metadata_miner,
)


def test_comparable_windows_preserve_units_and_missingness():
    result = dexscreener_window_measurements({
        "liquidity": {"usd": 1000}, "volume": {"m5": 500, "h1": 6000},
        "txns": {"m5": {"buys": 30, "sells": 10}, "h1": {"buys": 0, "sells": 0}},
        "priceChange": {"m5": 25},
    })
    assert result["m5"]["transactions_per_minute"] == 8
    assert result["m5"]["buy_fraction"] == 0.75
    assert result["m5"]["transaction_imbalance"] == 0.5
    assert result["m5"]["volume_to_liquidity"] == 0.5
    assert result["m5"]["volume_usd_per_minute"] == result["h1"]["volume_usd_per_minute"] == 100
    assert result["h1"]["buy_fraction"] is None
    assert result["h6"]["transactions_per_minute"] is None
    invalid = dexscreener_window_measurements({"liquidity": {"usd": 0},
        "volume": {"m5": "NaN", "h1": -1}, "txns": {"m5": {"buys": True, "sells": 1}}})
    assert invalid["m5"]["volume_to_liquidity"] is None
    assert invalid["m5"]["transactions_per_minute"] is None
    assert invalid["h1"]["volume_usd_per_minute"] is None


@pytest.mark.asyncio
async def test_pair_miner_reaches_tail_and_keeps_raw_windows():
    watched = ["Mint" + str(i) for i in range(DEXSCREENER_BATCH * 2 + 5)]
    requested = []
    async def get(url):
        targets = url.rsplit("/", 1)[1].split(",")
        requested.extend(targets)
        return 200, json.dumps({"pairs": [{"baseToken": {"address": targets[0]},
            "quoteToken": {"address": "quote"}, "txns": {"h6": {"buys": 100, "sells": 20}},
            "volume": {"h6": 1000}, "boosts": {"active": 10}}]}), {}
    fetch = dexscreener_pairs_miner(SimpleNamespace(get=get), lambda: watched)
    rows = []
    for _ in range(3):
        rows.extend(await fetch())
    assert set(watched) <= set(requested)
    assert rows[0]["transaction_windows"]["h6"]["buys"] == 100
    assert rows[0]["paid_boosts"] == {"active": 10}
    assert rows[0]["quote_status"] == "INDICATIVE_PROVIDER_OBSERVATION"
    assert not any("score" in key for key in rows[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [holder_structure_miner, token_metadata_miner])
async def test_native_enrichment_does_not_starve_targets_beyond_first_work_batch(factory):
    requested = []
    async def request(method, params):
        if method in {"getTokenLargestAccounts", "getAccountInfo"}:
            requested.append(params[0])
        return {"value": {"amount": "100"}} if method == "getTokenSupply" else {"value": []} if method == "getTokenLargestAccounts" else None
    watched = ["Mint" + str(i) for i in range(31)]
    fetch = factory(SimpleNamespace(request=request), lambda: watched, per_pass=12)
    for _ in range(3):
        await fetch()
    assert set(watched) <= set(requested)
