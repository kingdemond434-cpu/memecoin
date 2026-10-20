"""A corpus of "price went there" is a corpus of screenshots.

The desk records that a token reached 100x and has never recorded whether
anything could have been SOLD there. That gap is the whole difference between
a tail corpus and a scrapbook, and it has to be closed after migration,
because a Pump curve completes at 14.4x from a T0 entry -- every 20x, 100x and
1000x this desk will ever see is a pool event.

Two numbers per mint, and the gap between them is the point:

    max_price_multiple       what the chart printed
    max_executable_multiple  what a position could have taken out
"""

import time

import pytest

from src.chains.pumpswap_curve import PumpSwapPoolState
from src.research.migration_lineage import MigrationLineage, MigrationRecord

T0 = 1_700_000_000.0


def _pool(base=206_900_000_000_000, quote=85_000_000_000, at=None):
    return PumpSwapPoolState(
        pool="pool1", base_mint="b", quote_mint="q", base_reserves=base,
        quote_reserves=quote, virtual_quote_reserves=0, total_fee_bps=100,
        updated_at=at if at is not None else time.time())


def _lineage(**kwargs):
    return MigrationLineage(**kwargs)


#: A position whose entry price equals the pool's opening price, so a
#: `price_multiple` of 4 means the pool really is four times where we came in.
#: Getting this wrong makes every rung unreachable and every ladder empty.
OPENING_PRICE = 85_000_000_000 / 206_900_000_000_000


def _position_for(cost_lamports=300_000_000):
    return int(cost_lamports / OPENING_PRICE), cost_lamports


def _grown(multiple, **kwargs):
    """The same pool after its price has multiplied. Constant product."""
    import math
    base, quote = 206_900_000_000_000, 85_000_000_000
    product = base * quote
    new_quote = int(quote * math.sqrt(multiple))
    return _pool(base=product // new_quote, quote=new_quote, **kwargs)


class TestItPricesTheExitNotThePrice:
    def _tracked(self, tokens=None, cost=300_000_000):
        lineage = _lineage()
        if tokens is None:
            tokens, cost = _position_for(cost)
        lineage.set_reference_position("mint", tokens, cost, at=T0)
        lineage.observe_migration("mint", _pool(at=T0), at=T0)
        return lineage

    def test_a_sample_carries_both_the_printed_and_the_takeable_multiple(self):
        lineage = self._tracked()
        lineage.observe_pool("mint", _grown(4.0), at=T0 + 60)
        sample = lineage.get("mint").samples[-1]
        assert sample.price_multiple is not None
        assert sample.executable_multiple is not None
        assert sample.exitable_fraction is not None

    def test_the_capture_ratio_is_below_one_for_a_position_that_moves_the_pool(self):
        """A large position cannot take out what the chart printed."""
        lineage = self._tracked(tokens=int(_position_for()[0] * 400))
        for multiple in (2.0, 5.0, 20.0):
            lineage.observe_pool("mint", _grown(multiple), at=T0 + multiple)
        record = lineage.get("mint")
        assert record.capture_ratio is not None
        assert record.capture_ratio < 1.0

    def test_a_small_position_captures_more_of_the_same_move(self):
        """Exitability is a property of the PAIR, not of the pool."""
        big = self._tracked(tokens=int(_position_for()[0] * 400))
        small = self._tracked(tokens=int(_position_for()[0] * 0.001))
        for lineage in (big, small):
            for multiple in (2.0, 5.0, 20.0):
                lineage.observe_pool("mint", _grown(multiple), at=T0 + multiple)
        assert (small.get("mint").capture_ratio
                > big.get("mint").capture_ratio)

    def test_the_peak_is_taken_across_the_whole_path_not_the_last_reading(self):
        lineage = self._tracked()
        lineage.observe_pool("mint", _grown(50.0), at=T0 + 10)
        lineage.observe_pool("mint", _grown(2.0), at=T0 + 20)
        record = lineage.get("mint")
        assert record.max_price_multiple > 40


class TestItRefusesRatherThanInfers:
    def test_without_a_reference_position_nothing_is_priced(self):
        lineage = _lineage()
        lineage.observe_pool("mint", _pool(), at=T0)
        record = lineage.get("mint")
        assert record.samples[-1].executable_multiple is None
        assert record.capture_ratio is None

    def test_an_untradeable_pool_produces_no_sample(self):
        lineage = _lineage()
        lineage.set_reference_position("mint", 1_000, 1_000, at=T0)
        assert lineage.observe_pool("mint", _pool(base=0), at=T0) is None

    def test_a_mint_with_no_pool_observation_is_blocked(self):
        lineage = _lineage()
        assert lineage.executable_path("unknown")["status"] == "DATA_BLOCKED"

    def test_an_empty_corpus_says_no_capture_ratio_exists(self):
        report = _lineage().report()
        assert report["status"] == "DATA_BLOCKED"
        assert report["median_capture_ratio"] is None
        assert "no capture ratio exists" in report["detail"]

    def test_the_curve_side_of_the_crossing_is_recorded_separately(self):
        """A migration observed only from the pool cannot say what the curve
        handed over."""
        from src.chains.pump_curve import BondingCurveState
        lineage = _lineage()
        lineage.observe_curve("mint", BondingCurveState(
            virtual_token_reserves=1, virtual_sol_reserves=1,
            real_token_reserves=1_000, real_sol_reserves=85_000_000_000,
            token_total_supply=1, complete=False), at=T0)
        record = lineage.get("mint")
        assert record.curve_sol_reserves == 85_000_000_000
        assert not record.migrated
        lineage.observe_migration("mint", _pool(at=T0), at=T0 + 1)
        assert lineage.get("mint").migrated


class TestTheLadderIsBuiltFromObservations:
    def test_only_rungs_the_price_actually_reached_appear(self):
        lineage = _lineage()
        lineage.set_reference_position("mint", *_position_for(), at=T0)
        lineage.observe_migration("mint", _pool(at=T0), at=T0)
        lineage.observe_pool("mint", _grown(6.0), at=T0 + 30)
        ladder = lineage.executable_path("mint")["ladder"]
        assert "5x" in ladder
        assert "100x" not in ladder

    def test_a_rung_records_the_first_moment_it_was_reached(self):
        lineage = _lineage()
        lineage.set_reference_position("mint", *_position_for(), at=T0)
        lineage.observe_migration("mint", _pool(at=T0), at=T0)
        lineage.observe_pool("mint", _grown(12.0), at=T0 + 10)
        lineage.observe_pool("mint", _grown(30.0), at=T0 + 20)
        ladder = lineage.executable_path("mint")["ladder"]
        assert ladder["10x"]["reached_at"] == T0 + 10


class TestBoundedAndPersistent:
    def test_samples_are_thinned_from_the_middle_keeping_both_ends(self):
        lineage = _lineage(samples_per_mint=8)
        lineage.set_reference_position("mint", *_position_for(), at=T0)
        for index in range(40):
            lineage.observe_pool("mint", _grown(1.0 + index * 0.1),
                                 at=T0 + index)
        samples = lineage.get("mint").samples
        assert len(samples) <= 8
        assert samples[0].at == T0
        assert samples[-1].at == T0 + 39

    def test_mints_are_evicted_oldest_first(self):
        lineage = _lineage(max_mints=3)
        for index in range(10):
            lineage.observe_curve(f"m{index}", None, at=T0 + index)
            lineage.set_reference_position(f"m{index}", 10, 10, at=T0 + index)
        assert lineage.report()["mints_tracked"] <= 3

    def test_a_round_trip_through_disk_preserves_the_capture_ratio(self, tmp_path):
        path = tmp_path / "lineage.json"
        lineage = _lineage(path=str(path))
        lineage.set_reference_position("mint", *_position_for(), at=T0)
        lineage.observe_migration("mint", _pool(at=T0), at=T0)
        lineage.observe_pool("mint", _grown(9.0), at=T0 + 60)
        before = lineage.get("mint").capture_ratio
        assert lineage.save()
        restored = _lineage(path=str(path))
        assert restored.load()
        assert restored.get("mint").capture_ratio == pytest.approx(before)


def test_the_desk_feeds_it_from_pool_updates_and_entries():
    """Testing the corpus alone would have passed every day nothing fed it."""
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"note_pool_state", "note_migration", "note_curve_state",
            "note_entry_position"} <= called
