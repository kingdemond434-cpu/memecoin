"""The Raptor challenger, and the ways a router comparison lies.

Each test here corresponds to a way a route can look better than it is: a
better quote it does not fill, a fill measured against an absent one, a lucky
streak, a comparison taken at two different moments.
"""

import asyncio

import pytest

from src.execution.jupiter_jito import RouteType
from src.execution.raptor import (
    DEFAULT_MIN_PAIRED_FILLS, PairedObservation, RaptorClient, RaptorShadow,
    RouteObservation, ShadowStatus, observe_both)

MINT = "MintMintMintMintMintMintMintMintMintMintMin"


def _obs(route, *, quoted=None, realised=None, cost=0, landed=None,
         latency=1.0):
    return RouteObservation(
        route=route, mint=MINT, input_amount=1_000_000_000,
        quote_latency_ms=latency, quoted_out=quoted, realised_out=realised,
        landed=(realised is not None) if landed is None else landed,
        total_cost_lamports=cost)


def _pair(index, *, inc_out, cha_out, inc_cost=0, cha_cost=0):
    return PairedObservation(
        key=f"pair-{index}",
        incumbent=_obs("jupiter_v1", quoted=inc_out, realised=inc_out,
                       cost=inc_cost),
        challenger=_obs("raptor", quoted=cha_out, realised=cha_out,
                        cost=cha_cost))


# --- the observation record ---------------------------------------------

def test_an_unfilled_route_has_no_value_rather_than_zero_value():
    observation = _obs("raptor", quoted=1_000, realised=None, landed=False)
    assert observation.quoted
    assert not observation.filled
    assert observation.net_value() is None


def test_net_value_subtracts_what_the_fill_cost():
    observation = _obs("raptor", quoted=1_000, realised=1_000, cost=250)
    assert observation.net_value() == 750.0


def test_a_pair_with_one_unfilled_arm_contributes_no_delta():
    pair = PairedObservation(
        key="p", incumbent=_obs("jupiter_v1", quoted=1_000, realised=1_000),
        challenger=_obs("raptor", quoted=1_100, landed=False))
    assert pair.both_quoted
    assert not pair.both_filled
    assert pair.value_delta() is None
    # The quote still says the challenger looked better. That is the trap.
    assert pair.quote_delta_bps() == pytest.approx(1_000.0)


# --- promotion ----------------------------------------------------------

def test_a_challenger_that_only_ever_quotes_better_is_never_promoted():
    shadow = RaptorShadow(min_paired_fills=5)
    for index in range(200):
        shadow.record(PairedObservation(
            key=str(index),
            incumbent=_obs("jupiter_v1", quoted=1_000, realised=1_000),
            challenger=_obs("raptor", quoted=1_500, landed=False)))
    verdict = shadow.verdict()
    assert verdict["status"] == ShadowStatus.DATA_BLOCKED.value
    assert verdict["paired_fills"] == 0
    assert shadow.quote_summary()["challenger_quote_wins"] == 200
    assert shadow.quote_summary()["promotable"] is False
    assert not shadow.should_route_through_challenger()


def test_a_short_winning_streak_is_data_blocked_not_promoted():
    shadow = RaptorShadow()
    for index in range(20):
        shadow.record(_pair(index, inc_out=1_000, cha_out=1_100))
    verdict = shadow.verdict()
    assert verdict["status"] == ShadowStatus.DATA_BLOCKED.value
    assert str(DEFAULT_MIN_PAIRED_FILLS) in verdict["reason"]
    assert not shadow.should_route_through_challenger()


def test_a_consistent_realised_win_over_enough_pairs_promotes():
    shadow = RaptorShadow(min_paired_fills=50)
    for index in range(60):
        # Not a clean sweep: the challenger loses a fifth of them, so the
        # promotion is the sign test's, not a fixture's.
        if index % 5 == 0:
            shadow.record(_pair(index, inc_out=1_100, cha_out=1_000))
        else:
            shadow.record(_pair(index, inc_out=1_000, cha_out=1_100))
    verdict = shadow.verdict()
    assert verdict["status"] == ShadowStatus.PROMOTED.value
    assert verdict["p_challenger_better"] < 0.01
    assert shadow.should_route_through_challenger()


def test_cost_can_reverse_a_route_that_wins_on_output():
    shadow = RaptorShadow(min_paired_fills=50)
    for index in range(60):
        # Challenger returns more tokens and pays far more to land them.
        shadow.record(_pair(index, inc_out=1_000, cha_out=1_100,
                            inc_cost=10, cha_cost=300))
    verdict = shadow.verdict()
    assert verdict["status"] == ShadowStatus.DEMOTED.value
    assert not shadow.should_route_through_challenger()


def test_demotion_latches_against_a_later_winning_streak():
    shadow = RaptorShadow(min_paired_fills=30)
    for index in range(40):
        shadow.record(_pair(index, inc_out=1_100, cha_out=1_000))
    assert shadow.verdict()["status"] == ShadowStatus.DEMOTED.value
    for index in range(400):
        shadow.record(_pair(1_000 + index, inc_out=1_000, cha_out=9_999))
    verdict = shadow.verdict()
    assert verdict["status"] == ShadowStatus.DEMOTED.value
    assert verdict["latched"] is True
    assert not shadow.should_route_through_challenger()


def test_a_coin_flip_stays_in_shadow():
    shadow = RaptorShadow(min_paired_fills=50)
    for index in range(80):
        if index % 2:
            shadow.record(_pair(index, inc_out=1_000, cha_out=1_010))
        else:
            shadow.record(_pair(index, inc_out=1_010, cha_out=1_000))
    verdict = shadow.verdict()
    assert verdict["status"] == ShadowStatus.SHADOW.value
    assert not shadow.should_route_through_challenger()


def test_ties_are_excluded_rather_than_credited_to_either_arm():
    shadow = RaptorShadow(min_paired_fills=10)
    for index in range(30):
        shadow.record(_pair(index, inc_out=1_000, cha_out=1_000))
    verdict = shadow.verdict()
    assert verdict["decisive_pairs"] == 0
    assert verdict["status"] == ShadowStatus.SHADOW.value


# --- the client ---------------------------------------------------------

def test_quote_parsing_rejects_a_payload_with_no_output():
    for payload in ({}, {"amountOut": None}, {"amountOut": "0"},
                    {"amountOut": "not a number"}, [1, 2, 3]):
        assert RaptorClient._parse_quote(payload, "in", "out", 10, 100) is None


def test_quote_parsing_reads_raptors_field_names():
    quote = RaptorClient._parse_quote(
        {"amountOut": "1234567", "priceImpactPct": "0.42",
         "route": [{"label": "Raydium"}, {"label": "Orca"}],
         "minAmountOut": "1200000", "contextSlot": 99},
        "in", "out", 1_000_000, 100)
    assert quote is not None
    assert quote.output_amount == 1_234_567
    assert quote.min_output_amount == 1_200_000
    assert quote.price_impact_pct == pytest.approx(0.42)
    assert quote.route_type is RouteType.RAPTOR
    assert quote.raw_quote["contextSlot"] == 99


def test_a_missing_minimum_falls_back_to_the_requested_slippage():
    quote = RaptorClient._parse_quote(
        {"amountOut": "1000000"}, "in", "out", 1_000_000, slippage_bps=100)
    assert quote is not None
    assert quote.min_output_amount == 990_000


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


class FakeSession:
    def __init__(self, status=200, payload=None, raises=None):
        self.status = status
        self.payload = payload or {}
        self.raises = raises
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if self.raises:
            raise self.raises
        return FakeResponse(self.status, self.payload)


def test_observe_records_latency_even_when_the_quote_fails():
    session = FakeSession(status=503)
    client = RaptorClient(session=session)
    observation = asyncio.run(client.observe("in", MINT, 1_000_000))
    assert observation.error == "no_quote"
    assert observation.quoted_out is None
    assert observation.quote_latency_ms >= 0.0
    assert session.calls[0][1]["inputMint"] == "in"


def test_a_transport_error_is_data_blocked_not_an_exception():
    client = RaptorClient(session=FakeSession(raises=OSError("refused")))
    quote, latency = asyncio.run(client.get_quote("in", MINT, 1_000_000))
    assert quote is None
    assert latency >= 0.0


def test_observe_extracts_the_route_path_for_the_record():
    session = FakeSession(payload={"amountOut": "500",
                                   "route": [{"label": "Meteora"},
                                             {"ammKey": "abc"}]})
    client = RaptorClient(session=session)
    observation = asyncio.run(client.observe("in", MINT, 1_000_000))
    assert observation.quoted_out == 500
    assert observation.route_path == ["Meteora", "abc"]


# --- the pairing --------------------------------------------------------

def test_both_arms_are_quoted_concurrently_not_in_sequence():
    order = []

    async def slow():
        order.append("incumbent-start")
        await asyncio.sleep(0.02)
        order.append("incumbent-end")
        return _obs("jupiter_v1", quoted=1_000)

    async def quick():
        order.append("challenger-start")
        return _obs("raptor", quoted=1_100)

    pair = asyncio.run(observe_both(slow(), quick(), key="k"))
    assert pair.both_quoted
    # The challenger started before the incumbent finished; a sequential
    # comparison would compare two different markets.
    assert order.index("challenger-start") < order.index("incumbent-end")


def test_one_arm_raising_errors_that_arm_rather_than_losing_the_pair():
    async def broken():
        raise RuntimeError("router down")

    async def fine():
        return _obs("jupiter_v1", quoted=1_000)

    pair = asyncio.run(observe_both(fine(), broken(), key="k"))
    assert pair.incumbent.quoted
    assert "RuntimeError" in pair.challenger.error
    assert not pair.both_quoted
