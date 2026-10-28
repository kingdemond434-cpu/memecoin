import asyncio

import pytest

from src.research.dataset_builder import LaunchEpisode, PointInTimeDatasetBuilder


def desk():
    result=PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
    result.memecoin_state_provider=None
    return result


def episode():
    return LaunchEpisode('token','solana',100,'creator','venue','pool','quote')


def capture(builder, ep, cutoff=120):
    return asyncio.run(builder._capture_token_features(ep, cutoff))


def test_future_latest_risk_fallback_does_not_leak_into_earlier_snapshot():
    ep=episode();ep.risk_report={'timestamp':150,'can_mint':True,'top_10_pct':99,'data_status':'OK'}
    result=capture(desk(),ep)
    assert result['can_mint'] is None and result['top_10_pct'] is None
    assert result['provenance']=='CREATION_NATIVE'


@pytest.mark.parametrize('timestamp',[None,float('nan'),float('inf'),'unparseable',True])
def test_unknown_or_invalid_fallback_clock_is_not_past_knowledge(timestamp):
    ep=episode();ep.risk_report={'timestamp':timestamp,'can_mint':True}
    result=capture(desk(),ep)
    assert result['can_mint'] is None


@pytest.mark.parametrize('clock',['available_at','ingested_at'])
def test_future_knowledge_clock_rejects_old_event_time(clock):
    ep=episode();item={'type':'risk_report','timestamp':110,clock:150,'can_mint':True}
    ep.market_observations=[item];ep.risk_report=item
    assert capture(desk(),ep)['can_mint'] is None


def test_only_eligible_reports_contribute_to_latest_and_concentration_velocity():
    ep=episode();ep.market_observations=[
        {'type':'risk_report','timestamp':105,'top_10_pct':10,'can_mint':True},
        {'type':'risk_report','timestamp':110,'available_at':111,'ingested_at':112,'top_10_pct':20,'can_mint':False},
        {'type':'risk_report','timestamp':118,'available_at':150,'top_10_pct':99,'can_mint':True}]
    ep.risk_report=ep.market_observations[-1]
    result=capture(desk(),ep)
    assert result['can_mint'] is False and result['top_10_pct']==20
    assert result['top_10_delta_pct']==10 and result['top_10_velocity_pct_per_second']==2


def test_timestamped_legacy_fallback_and_exact_cutoff_remain_usable():
    ep=episode();ep.risk_report={'timestamp':120,'available_at':120,'ingested_at':120,'can_mint':False}
    assert capture(desk(),ep)['can_mint'] is False
    assert capture(desk(),ep,119.999)['can_mint'] is None


def test_reloaded_episode_retains_future_fallback_rejection():
    builder=desk();ep=episode();ep.risk_report={'timestamp':150,'can_mint':True}
    restored=builder._episode_from_dict(builder._episode_to_dict(ep))
    assert capture(builder,restored)['can_mint'] is None



@pytest.mark.parametrize('cutoff',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_risk_cutoff_fails_closed(cutoff):
    with pytest.raises(ValueError,match='finite'):
        capture(desk(),episode(),cutoff)
