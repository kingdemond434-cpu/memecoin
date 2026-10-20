import math

from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator


def test_cash_competes_with_open_positions_and_new_candidates():
    allocator = OpportunityAllocator()
    incumbent = Opportunity("held", -0.1, 100, 60, 10000, is_open_position=True)
    candidate = Opportunity("new", -0.01, 100, 60, 10000)
    slate = allocator.rank([incumbent, candidate], cash_available_usd=500,
                           cash_horizon_seconds=60)
    assert slate.best.token == "__CASH__"
    assert {item.token for item in slate.ranked} == {"held", "new", "__CASH__"}
    assert slate.displacements == []
    assert slate.report()["cash"]["ranked_with_positions"]


def test_positive_after_cost_opportunity_can_beat_waiting():
    candidate = Opportunity("new", 0.1, 100, 60, 10000)
    slate = OpportunityAllocator().rank([candidate], cash_available_usd=500,
                                        cash_horizon_seconds=60)
    assert slate.best.token == "new"


def test_unknown_cash_is_not_invented_and_nonfinite_inputs_do_not_rank():
    candidate = Opportunity("new", 0.1, 100, math.nan, 10000)
    slate = OpportunityAllocator().rank([candidate])
    assert slate.best is None
    assert slate.report()["cash"]["capital_usd"] is None
