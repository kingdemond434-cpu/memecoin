"""The veto that rejected 100% of decided launches, and why.

Measured on the live desk 2026-09-04: 678 launches reached a decision and
every one was hard-vetoed -- `sell_route_unavailable` in 359,
`catastrophic_exit_price_impact` in 301. Both came from the ROUTER. The proof
is the second reason: it needs a numeric price impact, and the native
bonding-curve branch returns None for that, so Jupiter answered all 301 and
the curve branch was never reached.

Two causes, both fixed here.

The curve state was being PRUNED while the launch was still being decided.
`_prune_curve_static` kept open positions and anything in `active_tokens`,
which is hard-capped and age-expiring -- but a candidate in
`_candidate_pipeline` makes several RPC round trips, and can leave that set
mid-flight. Its curve state went with it.

And the router was treated as authoritative when it was not. A mint the
router has never indexed answers "no route" out of ignorance, and a price
impact quoted for a venue the desk does not trade cannot veto the launch.
"""

from types import SimpleNamespace

import pytest

from src.detection.rug_detector import RugDetector


class _Quote:
    def __init__(self, output_amount, price_impact_pct):
        self.output_amount = output_amount
        self.price_impact_pct = price_impact_pct
        self.min_output_amount = 0


def _detector(curve, *, quote=None, raises=False):
    detector = RugDetector.__new__(RugDetector)
    detector.curve_route_skips = {}

    def provider(_mint):
        if raises:
            raise RuntimeError("lookup exploded")
        return curve

    detector.curve_state_provider = provider

    class _Router:
        _session = object()

        async def get_quote(self, *args, **kwargs):
            return quote

    detector.quote_provider = _Router() if quote is not None else None
    return detector


MINT_STATE = {"supply": 1_000_000_000_000_000, "decimals": 6}
TRADEABLE = SimpleNamespace(tradeable=True)
MIGRATED = SimpleNamespace(tradeable=False)


async def _route(detector):
    return await detector._solana_sell_route("mint", MINT_STATE)


# --- the native route wins when the curve is known ------------------------

@pytest.mark.asyncio
async def test_a_live_curve_is_sellable_without_asking_the_router():
    result = await _route(_detector(TRADEABLE, quote=_Quote(0, 0.9)))
    assert result["feasible"] is True
    assert result["venue"] == "bonding_curve"
    assert result["price_impact_pct"] is None


# --- the router is authority only for a migrated curve --------------------

@pytest.mark.asyncio
async def test_a_migrated_curve_lets_the_router_veto():
    """The pool is the only venue then, and the router prices it correctly."""
    result = await _route(_detector(MIGRATED, quote=_Quote(0, 0.5)))
    assert result["status"] == "OK"
    assert result["feasible"] is False
    assert result["price_impact_pct"] == 0.5


@pytest.mark.asyncio
async def test_a_migrated_curve_keeps_a_real_price_impact():
    result = await _route(_detector(MIGRATED, quote=_Quote(500, 0.42)))
    assert result["feasible"] is True
    assert result["price_impact_pct"] == 0.42


# --- an unknown curve makes the router non-authoritative ------------------

@pytest.mark.asyncio
async def test_an_unknown_curve_turns_no_route_into_unmeasured():
    """359 hard vetoes came from this. Ignorance is not unsellability."""
    result = await _route(_detector(None, quote=_Quote(0, 0.9)))
    assert result["status"] == "DATA_BLOCKED"
    assert result["feasible"] is None
    assert result["curve_status"] == "no_cached_curve_state"


@pytest.mark.asyncio
async def test_an_unknown_curve_suppresses_the_routers_price_impact():
    """301 hard vetoes came from this, on a venue the desk never trades."""
    result = await _route(_detector(None, quote=_Quote(500, 0.95)))
    assert result["feasible"] is True
    assert result["price_impact_pct"] is None, (
        "a Jupiter impact cannot veto a bonding-curve exit")
    assert result["curve_status"] == "no_cached_curve_state"


@pytest.mark.asyncio
async def test_a_raising_lookup_is_named_not_swallowed():
    detector = _detector(None, quote=_Quote(500, 0.95), raises=True)
    result = await _route(detector)
    assert result["curve_status"].startswith("curve_lookup_raised:")
    assert detector.curve_route_skips


@pytest.mark.asyncio
async def test_every_router_answer_carries_why_the_curve_was_skipped():
    """Without this the veto cannot be read back to its cause."""
    for curve, quote in ((None, _Quote(500, 0.1)),
                         (MIGRATED, _Quote(500, 0.1))):
        result = await _route(_detector(curve, quote=quote))
        assert "curve_status" in result


# --- the prune that caused the unknown curve in the first place -----------

def _desk(active, held, in_flight):
    """A desk shaped enough to run the real prune."""
    from src.main import MemecoinQuantDesk
    desk = MemecoinQuantDesk.__new__(MemecoinQuantDesk)
    desk._curve_static = {token: {} for token in ("a", "b", "c")}
    desk._latest_curve_state = {token: object() for token in ("a", "b", "c")}
    desk._latest_pool_state = {}
    desk.hot_state = SimpleNamespace(active_tokens=set(active))
    desk.elogw_engine = SimpleNamespace(open_positions=dict.fromkeys(held))
    desk._candidate_pipelines = dict.fromkeys(in_flight)
    return desk


def test_a_candidate_mid_decision_keeps_its_curve_state():
    """The bug: a launch left active_tokens while its RPCs were in flight.

    Its curve state went with it, the sell route found nothing cached, and
    the router hard-vetoed a mint the desk could sell natively.
    """
    from src.main import MemecoinQuantDesk
    desk = _desk(active=set(), held=(), in_flight=("b",))
    MemecoinQuantDesk._prune_curve_static(desk)
    assert "b" in desk._latest_curve_state
    assert "a" not in desk._latest_curve_state


def test_an_open_position_still_keeps_its_curve_state():
    from src.main import MemecoinQuantDesk
    desk = _desk(active=set(), held=("c",), in_flight=())
    MemecoinQuantDesk._prune_curve_static(desk)
    assert "c" in desk._latest_curve_state


def test_the_prune_still_bounds_the_dicts():
    """It exists because these were unbounded and OOM-killed the service."""
    from src.main import MemecoinQuantDesk
    desk = _desk(active=set(), held=(), in_flight=())
    dropped = MemecoinQuantDesk._prune_curve_static(desk)
    assert desk._latest_curve_state == {}
    assert dropped >= 3
