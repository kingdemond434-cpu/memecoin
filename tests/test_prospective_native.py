"""Synthetic native controls test plumbing; they establish no mainnet edge."""
import copy
from types import SimpleNamespace

import pytest

from src.research.prospective_native import (
    CHECKPOINT,CONTROL,PRODUCER,ProspectiveNativeValidation,ProspectiveNativeForward,
)
from src.research.candidate_replay import ReplayBlocked
from src.runtime.research_runtime import ResearchRuntime
from test_native_forward import costs,curve,fresh_episode
from test_research_validation import desk


def study(desk):
    framework,spec,_,_,ledger,_=desk
    now=spec.frozen_at+100
    validation=ProspectiveNativeValidation(framework,ledger,virtual_seed_usd=1000.,now=now)
    return validation,ProspectiveNativeForward(validation),now


def tick(runner,desk,row,now,**overrides):
    position=next(iter(runner.state['positions'].values()),None)
    options={'state_observed_at':now,'source_ref':'native:synthetic-control',
        'equity_usd':1.,'native_to_usd':100.,'now':now,'fee_bps':100,
        'execution_costs':costs(desk,now,phase='exit' if position else 'entry',
            amount=position['tokens'] if position else 1_000_000)}
    options.update(overrides)
    return runner.tick(row['token'],row,curve(),**options)


def test_prospective_control_opens_and_closes_with_actual_quote_math_without_historical_survivor(desk):
    framework,_,_,historical,_,forward=desk
    before=set(framework.hypotheses)
    validation,runner,now=study(desk);now+=1;row=fresh_episode(now)
    assert historical.report()['historical_receipts']==historical.report()['forward_experiments']==0
    assert validation.report()['forward_experiments']==1 and validation.report()['historical_receipts']==0
    entered=tick(runner,desk,row,now)
    assert entered['detail']['open_positions']==1,entered
    assert next(iter(runner.state['positions'].values()))['equity_usd']==1000.
    # Holding marks are local arithmetic, not repeated fee/simulation RPC work.
    assert runner.cost_requests(row['token'],row,now=now+5,native_state=curve(),fee_bps=100)==[]
    assert len(runner.cost_requests(row['token'],row,now=now+61,native_state=curve(),fee_bps=100))==1
    closed=tick(runner,desk,row,now+61)
    receipt=closed['detail']['results'][0]
    assert receipt['pnl']<0 and receipt['paper'] is True and receipt['real_fill'] is False
    assert receipt['historical_validation']=='UNMEASURED' and not receipt['promotion_eligible']
    assert receipt['verified_live_profit'] is None and receipt['unmeasured_live_costs']
    assert runner.report()['open_positions']==0 and validation.report()['paper_outcomes']==1
    assert framework.get_research_component(CHECKPOINT) and framework.get_research_component(PRODUCER)
    assert set(framework.hypotheses)==before and CONTROL not in framework.hypotheses
    assert not framework.shadow_models and not framework.canary_models and not framework.champions
    assert not forward.entered and not forward.real_fills and not framework.trial_results
    restored=ProspectiveNativeValidation(framework,desk[4],virtual_seed_usd=1000.,now=now+100)
    assert restored.report()['forward_experiments']==restored.report()['paper_outcomes']==1
    assert ProspectiveNativeForward(restored).report()['open_positions']==0


def test_missing_cost_proof_old_cohorts_and_future_state_never_open_a_control(desk):
    validation,runner,now=study(desk);row=fresh_episode(now+1)
    result=tick(runner,desk,row,now+1,execution_costs=None)
    assert result['disposition']=='BLOCKED' and not runner.report()['open_positions']
    result=tick(runner,desk,fresh_episode(now-1),now+1)
    assert result['detail']['status']=='NO_REGISTERED_FRESH_COHORT'
    result=tick(runner,desk,row,now+1,state_observed_at=now+2)
    assert result['disposition']=='BLOCKED' and validation.report()['paper_outcomes']==0
    restored=ProspectiveNativeForward(validation)
    assert restored.report()['observed_ticks']==3
    assert restored.report()['input_refusals']=={'inputs':1,'native_state':1}
    assert restored.report()['last_input_wait']=='native_state'
    assert restored.report()['last_tick_status']=='BLOCKED'


def test_fresh_native_controls_and_open_positions_get_observation_admission(desk):
    validation,runner,now=study(desk)
    app=ResearchRuntime();app.dry_run=True;app.native_prospective_forward=runner
    def episode(at,chain='solana',factory='pump.fun'):
        return SimpleNamespace(created_at=at,chain=chain,factory=factory)
    app.dataset_builder=SimpleNamespace(active_episodes={
        'old':episode(now-1),'fresh':episode(now+1),'future':episode(now+20),
        'foreign':episode(now+2,chain='base'),'unsupported':episode(now+2,factory='other'),
        'expired':episode(now+2),'open':episode(now-100),
    })
    runner.state['positions']['synthetic-open']={'cohort_payload':{'token':'open'}}
    app._market_observed_at={}
    assert app._prospective_native_observation_tokens(now+10)==['open','expired','fresh']
    assert app._prospective_native_observation_tokens(now+63)==['open','future']
    app.dry_run=False
    assert app._prospective_native_observation_tokens(now+10)==[]


@pytest.mark.asyncio
async def test_full_stable_outcome_cohort_does_not_starve_new_paper_control(desk,monkeypatch):
    from src.main import MemecoinQuantDesk
    validation,runner,now=study(desk);now+=10
    app=MemecoinQuantDesk(dry_run_override=True)
    app.dry_run=True;app.native_prospective_forward=runner
    app.global_config={'market_observation_cohort_size':2,'market_observation_budget':2}
    app.dataset_builder=SimpleNamespace(active_episodes={
        token:SimpleNamespace(token=token,created_at=created,chain='solana',factory='pump.fun')
        for token,created in [('old-a',now-100),('old-b',now-90),('fresh',now-1)]})
    app._market_observation_cohort={'old-a','old-b'}
    app._market_observed_at={}
    calls=[]
    async def paper(token,at):calls.append(('paper',token))
    async def outcome(token,at):calls.append(('outcome',token))
    app._native_forward_tick=paper;app._observe_token_market=outcome
    monkeypatch.setattr('src.main.time.time',lambda:now)
    await app._observe_active_markets()
    assert ('paper','fresh') in calls and len(calls)==2
    assert app._market_observation_cohort=={'old-a','old-b'}
    assert any(kind=='outcome' and token in {'old-a','old-b'} for kind,token in calls)
    calls.clear()
    await app._observe_active_markets()
    assert ('paper','fresh') not in calls  # retain the two-second pacing


def test_declaration_tamper_seed_change_and_nonpaper_work_are_rejected(desk):
    validation,_,now=study(desk)
    with pytest.raises(ReplayBlocked,match='virtual seed'):
        ProspectiveNativeValidation(desk[0],desk[4],virtual_seed_usd=2000.,now=now+1)
    assert validation.consume('GAUNTLET',{},[],now+1)['disposition']=='BLOCKED'
    spec=validation._candidate({'candidate_id':CONTROL})
    experiment=next(iter(validation.state['experiments'].values()))
    for payload in ({'scope':'observed_forward'},
                    {'scope':'observed_forward_shadow','record_in_forward_evidence':True},
                    {'scope':'observed_forward_shadow','equity_at_decision_usd':2000.}):
        with pytest.raises(ReplayBlocked):validation._outcome(spec,experiment,payload,[],now+1)
    validation.state['programs'][CONTROL]['program']['quote_budget_lamports']=2_000_000
    with pytest.raises(ReplayBlocked,match='binding changed'):validation._candidate({'candidate_id':CONTROL})


def test_foreign_token_execution_cannot_be_reused_and_blocked_position_survives(desk):
    validation,runner,now=study(desk);now+=1;row=fresh_episode(now)
    tick(runner,desk,row,now)
    position=next(iter(runner.state['positions'].values()));original=position['entry_ref']
    body=validation.read_evidence(original)['payload']
    position['entry_ref']=desk[4].record_reference('research_forward_native_execution',
        {**body,'token':'another-mint'},timestamp=now)
    result=tick(runner,desk,row,now+61)
    assert result['disposition']=='BLOCKED' and validation.report()['paper_outcomes']==0
    assert runner.report()['open_positions']==1
    position['entry_ref']=original
    result=tick(runner,desk,row,now+62)
    assert result['disposition']=='DEEPENING_TASK' and validation.report()['paper_outcomes']==1


def test_runtime_feeds_prospective_producer_without_registering_a_historical_experiment(desk,monkeypatch):
    validation,runner,now=study(desk);now+=1;row=fresh_episode(now);state=curve()
    row['market_observations']=[{'signature':'synthetic-control-source','timestamp':now,
        'virtual_sol_reserves':state.virtual_sol_reserves,'virtual_token_reserves':state.virtual_token_reserves}]
    app=ResearchRuntime();app.dry_run=True
    app.unknown_engine=SimpleNamespace(workflows=SimpleNamespace(validation=desk[3]),enqueue_work=lambda *a,**k:None)
    app.dataset_builder=SimpleNamespace(active_episodes={row['token']:row},_episode_to_dict=lambda _:copy.deepcopy(row))
    app.native_prospective_forward=runner
    app._latest_curve_state={row['token']:state};app._latest_pool_state={};app.trade_evidence=desk[4]
    app.equity_status='DATA_BLOCKED';app.wallet_equity_usd=0.;app.sol_price_usd=100.;app.sol_price_age_s=0.
    app._cost_model=lambda *a:{'status':'OK','assumed':False,'entry_cost':.01}
    from src.chains.rpc_manager import RPCError
    async def refused(*args,**kwargs):raise RPCError({'message':'private-provider-DO_NOT_EXPORT'})
    monkeypatch.setattr('src.research.native_forward.NativeSimulationFeeAdapter.acquire',refused)
    import asyncio
    blocked=asyncio.run(app._native_forward_tick(row['token'],now))
    assert blocked['prospective_control']['disposition']=='BLOCKED' and not runner.report()['open_positions']
    assert 'DO_NOT_EXPORT' not in desk[4].path.read_text()
    async def acquire(self,token,amount,*,buying,now):
        assert amount==1_000_000
        return costs(desk,now,amount=amount)
    monkeypatch.setattr('src.research.native_forward.NativeSimulationFeeAdapter.acquire',acquire)
    result=asyncio.run(app._native_forward_tick(row['token'],now))
    assert result['prospective_control']['detail']['open_positions']==1,result
    assert desk[3].report()['forward_experiments']==0 and not app.native_forward_paper.report()['open_positions']
    # Live mode cannot drive the prospective producer even if an object exists.
    app.dry_run=False
    assert 'prospective_control' not in asyncio.run(app._native_forward_tick(row['token'],now+1))
