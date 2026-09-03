"""The materialised actor graph, and the leakage it exists to prevent.

The dangerous failure is not a crash. It is a store that cheerfully answers
"this deployer had forty prior launches" on the day of their first, because it
counted every row on disk instead of every row observable at that moment. Every
as-of test here would pass trivially against such a store if the cut were
removed, so each one is written to fail loudly if it is.
"""

import json

import pytest

from src.research.actor_store import (
    ActorStore, Edge, EdgeKind, DEFAULT_HUB_DEGREE)

DAY = 86_400.0
T0 = 1_700_000_000.0


def _store(**kwargs):
    return ActorStore(**kwargs)


def _funded(funder, wallet, at):
    return Edge(source=funder, target=wallet, kind=EdgeKind.FUNDED,
                observed_at=at)


# --- the as-of cut -------------------------------------------------------

def test_an_edge_observed_later_is_invisible():
    store = _store()
    store.add(Edge(source="dep", target="mintA", kind=EdgeKind.DEPLOYED,
                   observed_at=T0))
    store.add(Edge(source="dep", target="mintB", kind=EdgeKind.DEPLOYED,
                   observed_at=T0 + DAY))
    assert store.prior_launches("dep", T0 - 1) == 0
    assert store.prior_launches("dep", T0) == 1
    assert store.prior_launches("dep", T0 + DAY) == 2


def test_prior_launches_at_the_first_launch_is_zero_not_forty():
    """The exact leakage this store exists to prevent."""
    store = _store()
    for index in range(40):
        store.add(Edge(source="dep", target=f"mint{index}",
                       kind=EdgeKind.DEPLOYED,
                       observed_at=T0 + index * DAY))
    # Standing at the first launch, the deployer has no history.
    assert store.prior_launches("dep", T0 - 1) == 0
    # Standing at the twentieth, it has nineteen -- not forty.
    assert store.prior_launches("dep", T0 + 19 * DAY - 1) == 19


def test_edges_arriving_out_of_order_still_cut_correctly():
    """Bulk history arrives one day-partition at a time, in any order."""
    store = _store()
    for offset in (5, 1, 9, 3, 7):
        store.add(Edge(source="dep", target=f"mint{offset}",
                       kind=EdgeKind.DEPLOYED, observed_at=T0 + offset * DAY))
    assert store.prior_launches("dep", T0 + 4 * DAY) == 2   # offsets 1 and 3
    assert store.prior_launches("dep", T0 + 100 * DAY) == 5


def test_prior_mints_are_the_ones_that_already_existed():
    store = _store()
    store.add(Edge(source="dep", target="old", kind=EdgeKind.DEPLOYED,
                   observed_at=T0))
    store.add(Edge(source="dep", target="new", kind=EdgeKind.DEPLOYED,
                   observed_at=T0 + DAY))
    assert store.prior_mints("dep", T0 + 1) == ["old"]


# --- the first-buyer sequence -------------------------------------------

def test_first_buyers_keep_their_order():
    store = _store()
    store.ingest_launch(mint="m", creator="dep", created_at=T0,
                        buyers=[("w1", T0 + 1), ("w2", T0 + 2),
                                ("w3", T0 + 3)])
    assert store.first_buyers("m", T0 + 10) == ["w1", "w2", "w3"]


def test_first_buyers_are_cut_by_time_too():
    store = _store()
    store.ingest_launch(mint="m", creator="dep", created_at=T0,
                        buyers=[("w1", T0 + 1), ("w2", T0 + 100)])
    assert store.first_buyers("m", T0 + 50) == ["w1"]


def test_first_buyers_respect_the_requested_depth():
    store = _store()
    store.ingest_launch(
        mint="m", creator="dep", created_at=T0,
        buyers=[(f"w{index}", T0 + index) for index in range(50)])
    assert len(store.first_buyers("m", T0 + 1_000, depth=25)) == 25


# --- families and hubs ---------------------------------------------------

def test_wallets_sharing_a_funder_are_one_family():
    store = _store()
    for wallet in ("a", "b", "c"):
        store.add(_funded("funder", wallet, T0))
    walk = store.family("a", T0 + 1)
    assert set(walk.reached) >= {"a", "funder", "b", "c"}
    assert not walk.truncated


def test_an_exchange_hot_wallet_does_not_make_everyone_family():
    store = _store()
    for index in range(DEFAULT_HUB_DEGREE + 10):
        store.add(_funded("exchange", f"w{index}", T0))
    walk = store.family("w0", T0 + 1)
    assert "exchange" in walk.hubs_skipped
    assert set(walk.reached) == {"w0"}, (
        "walking through a hot wallet collapses the whole chain into one "
        "family and reads downstream as zero independent buyers on every "
        "launch at once")


def test_hub_status_is_itself_as_of():
    store = _store()
    store.add(_funded("later_hub", "a", T0))
    store.add(_funded("later_hub", "b", T0))
    for index in range(DEFAULT_HUB_DEGREE):
        store.add(_funded("later_hub", f"x{index}", T0 + 365 * DAY))
    # In 2024 it was an ordinary funder and a and b really were family.
    early = store.family("a", T0 + 1)
    assert "b" in early.reached
    assert early.hubs_skipped == []
    # By the time it had funded a hundred thousand wallets, it is a hub.
    late = store.family("a", T0 + 400 * DAY)
    assert "later_hub" in late.hubs_skipped


def test_a_family_edge_created_later_is_not_visible_earlier():
    store = _store()
    store.add(_funded("f", "a", T0))
    store.add(_funded("f", "b", T0 + 10 * DAY))
    assert "b" not in store.family("a", T0 + DAY).reached
    assert "b" in store.family("a", T0 + 20 * DAY).reached


def test_traversal_depth_is_bounded():
    store = _store()
    for index in range(20):
        store.add(_funded(f"n{index}", f"n{index + 1}", T0))
    walk = store.family("n0", T0 + 1, max_hops=2)
    assert max(walk.reached.values()) <= 2
    assert "n5" not in walk.reached


def test_a_truncated_walk_says_so():
    store = ActorStore(max_nodes=5)
    for index in range(50):
        store.add(_funded("f", f"w{index}", T0))
    walk = store.family("w0", T0 + 1)
    assert walk.truncated


# --- independence collapse ----------------------------------------------

def test_ten_wallets_from_one_family_are_one_buyer():
    store = _store()
    wallets = [f"w{index}" for index in range(10)]
    for wallet in wallets:
        store.add(_funded("family", wallet, T0))
    result = store.shared_family(wallets, T0 + 1)
    assert result["families"] == 1
    assert result["independence"] == pytest.approx(0.1)
    assert result["largest_family"] == 10


def test_genuinely_independent_wallets_stay_independent():
    store = _store()
    wallets = [f"w{index}" for index in range(10)]
    for index, wallet in enumerate(wallets):
        store.add(_funded(f"funder{index}", wallet, T0))
    result = store.shared_family(wallets, T0 + 1)
    assert result["families"] == 10
    assert result["independence"] == pytest.approx(1.0)


def test_independence_is_measured_as_of_the_launch():
    store = _store()
    wallets = ["a", "b"]
    store.add(_funded("f", "a", T0))
    # The link that reveals them as siblings is only observed a month later.
    store.add(_funded("f", "b", T0 + 30 * DAY))
    at_launch = store.shared_family(wallets, T0 + DAY)
    later = store.shared_family(wallets, T0 + 60 * DAY)
    assert at_launch["families"] == 2
    assert later["families"] == 1


def test_no_wallets_is_data_blocked():
    assert _store().shared_family([], T0)["status"] == "DATA_BLOCKED"


# --- ingestion -----------------------------------------------------------

class _RawLaunch:
    def __init__(self, token, creator, created_at, trades=(), transfers=()):
        self.token = token
        self.creator = creator
        self.created_at = created_at
        self.trades = list(trades)
        self.funding_transfers = list(transfers)


def test_bulk_history_becomes_edges():
    store = _store()
    result = store.ingest_raw_launches([
        _RawLaunch("m1", "dep", T0,
                   trades=[{"signer": "w1", "block_timestamp": T0 + 1},
                           {"signer": "w2", "block_timestamp": T0 + 2}],
                   transfers=[{"source": "f", "destination": "w1",
                               "block_timestamp": T0 - 100}]),
    ])
    assert result["edges"] == 4     # deployed + two buys + one funding
    assert store.first_buyers("m1", T0 + 10) == ["w1", "w2"]
    assert store.funders_of("w1", T0) == ["f"]


def test_a_launch_with_no_creation_time_contributes_nothing():
    store = _store()
    result = store.ingest_raw_launches([_RawLaunch("m1", "dep", None)])
    assert result["skipped"] == 1
    assert store.edges == [], (
        "an edge stamped zero would be visible to every as-of query ever made")


def test_malformed_trade_rows_are_skipped_not_fatal():
    store = _store()
    store.ingest_raw_launches([
        _RawLaunch("m1", "dep", T0,
                   trades=["not a dict", {"signer": "", "block_timestamp": 1},
                           {"signer": "w1", "block_timestamp": T0 + 1}])])
    assert store.first_buyers("m1", T0 + 10) == ["w1"]


# --- persistence ---------------------------------------------------------

def test_the_log_round_trips(tmp_path):
    path = tmp_path / "actors.jsonl"
    store = ActorStore(path)
    store.ingest_launch(mint="m", creator="dep", created_at=T0,
                        buyers=[("w1", T0 + 1)],
                        funding=[("f", "w1", T0 - 10)])
    reborn = ActorStore(path)
    assert reborn.load() == 3
    assert reborn.prior_launches("dep", T0 + 1) == 1
    assert reborn.first_buyers("m", T0 + 10) == ["w1"]
    assert reborn.funders_of("w1", T0) == ["f"]


def test_a_truncated_final_line_costs_one_edge_not_the_corpus(tmp_path):
    path = tmp_path / "actors.jsonl"
    store = ActorStore(path)
    store.ingest_launch(mint="m", creator="dep", created_at=T0,
                        buyers=[("w1", T0 + 1)])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"source": "dep", "target": "m2", "kind": "dep')
    reborn = ActorStore(path)
    assert reborn.load() == 2
    assert reborn.prior_launches("dep", T0 + 1) == 1


def test_a_reloaded_store_answers_identically(tmp_path):
    path = tmp_path / "actors.jsonl"
    store = ActorStore(path)
    for index in range(20):
        store.ingest_launch(
            mint=f"m{index}", creator="dep", created_at=T0 + index * DAY,
            buyers=[(f"w{index}", T0 + index * DAY + 1)],
            funding=[("f", f"w{index}", T0 + index * DAY - 10)])
    reborn = ActorStore(path)
    reborn.load()
    for at in (T0, T0 + 5 * DAY, T0 + 19 * DAY, T0 + 100 * DAY):
        assert (store.prior_launches("dep", at)
                == reborn.prior_launches("dep", at))
        assert (store.family("w3", at).reached
                == reborn.family("w3", at).reached)


def test_the_report_describes_the_corpus(tmp_path):
    store = ActorStore(tmp_path / "actors.jsonl")
    store.ingest_launch(mint="m", creator="dep", created_at=T0,
                        launchpad="pump", buyers=[("w1", T0 + 1)])
    report = store.report()
    assert report["edges"] == 3
    assert report["by_kind"]["deployed"] == 1
    assert report["by_kind"]["launched_on"] == 1
    assert report["earliest"] == T0
    assert report["appended"] == 3


# --- wiring --------------------------------------------------------------

def _priced_trades(count=6):
    """Trades reconstruction can actually build a price path from."""
    return [{"signer": f"w{index}", "block_timestamp": T0 + index,
             "timestamp": T0 + index,
             "price_sol_per_token": 1e-7 * (1 + index)}
            for index in range(count)]


def test_backfill_feeds_the_store_when_one_is_attached(tmp_path):
    from src.research.backfill import RawLaunch, run_backfill
    store = ActorStore()
    report = run_backfill(
        [RawLaunch(token="m1", creator="dep", created_at=T0,
                   trades=[{"signer": "w1", "block_timestamp": T0 + 1}])],
        tmp_path, min_trades=1, actor_store=store)
    assert report.actor_edges == 2
    assert store.prior_launches("dep", T0 + 1) == 1


def test_a_launch_too_thin_to_reconstruct_still_counts_as_a_deployment(
        tmp_path):
    """The deployer deployed it. Dropping it understates prior-launch counts.

    Reconstruction needs enough trades to build an episode; the actor graph
    needs only that the launch happened, and those are different bars.
    """
    from src.research.backfill import RawLaunch, run_backfill
    store = ActorStore()
    report = run_backfill(
        [RawLaunch(token="thin", creator="dep", created_at=T0, trades=[])],
        tmp_path, min_trades=5, actor_store=store)
    assert report.reconstructed == 0
    assert store.prior_launches("dep", T0 + 1) == 1


def test_a_raising_store_does_not_end_the_backfill(tmp_path):
    from src.research.backfill import RawLaunch, run_backfill

    class Broken:
        def ingest_raw_launches(self, launches):
            raise RuntimeError("disk full")

    report = run_backfill(
        [RawLaunch(token="m1", creator="dep", created_at=T0,
                   trades=_priced_trades())],
        tmp_path, min_trades=1, actor_store=Broken())
    assert report.actor_edges == 0
    assert report.reconstructed == 1, (
        "the corpus is the point; a graph that cannot be written must not "
        "take the episodes down with it")


def test_backfill_without_a_store_is_unchanged(tmp_path):
    from src.research.backfill import RawLaunch, run_backfill
    report = run_backfill(
        [RawLaunch(token="m1", creator="dep", created_at=T0,
                   trades=_priced_trades())],
        tmp_path, min_trades=1)
    assert report.actor_edges == 0
    assert report.reconstructed == 1
