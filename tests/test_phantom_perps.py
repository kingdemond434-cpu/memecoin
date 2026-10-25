"""Separate perpetual contract data, recorded-depth costs and durable funding."""
import asyncio
import time
from decimal import Decimal

import pytest

from src.research.phantom_perps import PhantomPerpsResearch, book_cost_scenario, public_info
from src.research.data_miners import DataMinerPool
from src.runtime.ingestion import MinedRecordIngestion


def catalog(dex="", coin="BTC", at=None):
    return {"provider":"hyperliquid", "product":"perpetual", "perps_role":"catalog",
        "observed_at":time.time() if at is None else at, "dex":dex,
        "meta":{"universe":[{"name":coin,"szDecimals":4,"maxLeverage":40,"marginTableId":1}], "collateralToken":0,
                "marginTables":[[1,{"marginTiers":[{"lowerBound":"0","maxLeverage":40}]}]]},
        "contexts":[{"markPx":"100","oraclePx":"99.5","funding":"0.00001","openInterest":"10"}]}


def book(now):
    return {"coin":"BTC", "time":int(now * 1000),
        "levels":[[{"px":"99","sz":"0.2"},{"px":"98","sz":"10"}],
                  [{"px":"101","sz":"0.2"},{"px":"102","sz":"10"}]]}


def test_cost_walks_displayed_levels_and_labels_unmeasured_components():
    now = time.time()
    result = book_cost_scenario(book(now), 4, observed_at=now, now=now, notional_usdc="30.3", collateral_verified=True)
    assert Decimal(result["quantity"]) == Decimal("0.3")
    assert Decimal(result["buy_notional_usdc"]) == Decimal("30.4")
    assert Decimal(result["sell_notional_usdc"]) == Decimal("29.6")
    assert Decimal(result["roundtrip_cost_usdc"]) == Decimal("0.857")
    assert result["missing_costs"] and result["real_fill"] is False
    assert result["simulated_strategy_fill"] is False
    assert book_cost_scenario(book(now - 20), 4, observed_at=now, now=now, collateral_verified=True)["status"] == "DATA_BLOCKED_STALE_BOOK"
    assert book_cost_scenario(book(now), 4, observed_at=now, now=now)["status"] == "DATA_BLOCKED_COLLATERAL_MAPPING"
    limited = book(now)
    limited["levels"][1] = [{"px":"101","sz":"0.01"}]
    assert book_cost_scenario(limited, 4, observed_at=now, now=now, collateral_verified=True)["status"] == "DATA_BLOCKED_BOOK_DEPTH"


def test_catalog_funding_and_costs_survive_restart_without_spot_or_world_pnl(tmp_path):
    path = tmp_path / "perps.sqlite"
    state = PhantomPerpsResearch(path)
    now = time.time()
    state.ingest({"perps_role":"dex_inventory","observed_at":now,"dexes":["","xyz","new-platform"]})
    state.ingest({"perps_role":"collateral_inventory","observed_at":now,"tokens":[{"index":0,"name":"USDC","isCanonical":True}]})
    state.ingest(catalog())
    state.ingest(catalog("xyz","xyz:TSLA"))
    state.ingest({"perps_role":"book","observed_at":now,"coin":"BTC","book":book(now)})
    funding = {"perps_role":"funding","observed_at":now,"coin":"BTC","start_time":0,
               "rows":[{"coin":"BTC","time":1000,"fundingRate":"0.001"}]}
    state.ingest(funding)
    state.ingest(funding)  # retained-page replay is idempotent
    assert state.next_funding("BTC") == 1001
    state.close()
    state = PhantomPerpsResearch(path)
    report = state.report()
    assert report["active_markets"] == 2 and report["funding_records"] == 1
    assert report["books_recorded"] == 1 and report["real_fills"] == 0
    assert report["simulated_strategy_fills"] == 0 and report["execution_authority"] is False
    assert "dex:new-platform" in state.dex_names()
    assert state.market_names() == ["BTC", "xyz:TSLA"]
    assert state.db.execute("SELECT count(*) FROM catalogs").fetchone()[0] == 2
    state.close()


def test_misaligned_catalog_and_foreign_funding_cannot_corrupt_existing_state(tmp_path):
    state = PhantomPerpsResearch(tmp_path / "perps.sqlite")
    state.ingest(catalog())
    bad = catalog(coin="wrong")
    bad["contexts"] = []
    with pytest.raises(ValueError): state.ingest(bad)
    assert state.market_names() == ["BTC"]
    with pytest.raises(ValueError): state.ingest({"perps_role":"funding","observed_at":time.time(),
        "coin":"BTC","start_time":0,"rows":[{"coin":"ETH","time":1000}]})
    assert state.next_funding("BTC") == 0
    state.close()


@pytest.mark.asyncio
async def test_public_info_cannot_query_accounts_or_submit_orders():
    for request in ({"type":"order"},{"type":"clearinghouseState","user":"wallet"},
                    {"type":"l2Book","coin":"BTC","user":"wallet"}):
        with pytest.raises(ValueError): await public_info(None, request)


@pytest.mark.asyncio
async def test_every_new_underlying_dex_rotates_and_ingestion_is_separate(monkeypatch,tmp_path):
    state = PhantomPerpsResearch(tmp_path / "perps.sqlite")
    observed = []
    async def response(http,payload):
        observed.append(payload)
        if payload["type"] == "perpDexs": return [None,{"name":"xyz"},{"name":"unknown-new"}]
        if payload["type"] == "spotMeta": return {"tokens":[{"index":0,"name":"USDC","isCanonical":True}]}
        if payload["type"] == "metaAndAssetCtxs":
            row = catalog(payload["dex"],(payload["dex"] + ":" if payload["dex"] else "") + "BTC")
            return [row["meta"],row["contexts"]]
        return []
    monkeypatch.setattr("src.research.phantom_perps.public_info",response)
    pool = DataMinerPool(on_records=lambda name,rows:[state.ingest(row) for row in rows])
    state.register(pool,None)
    fetch = pool._callables["perps:catalog"]
    for _ in range(5):
        for row in await fetch():
            # Product data is indexed after canonical retention on the worker;
            # the hot-loop dispatch must have no dependency on perps SQLite.
            state.ingest_batch([row], "fixture:"+str(len(observed)))
            MinedRecordIngestion._ingest_mined_records(type("Desk",(),{})(),"perps:catalog",[row])
    assert {r["dex"] for r in observed if r["type"] == "metaAndAssetCtxs"} == {"","xyz","unknown-new"}
    assert state.report()["active_markets"] == 3
    state.close()
