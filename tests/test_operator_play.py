"""Operator associations are PIT facts and unfilled plans, never insider proof."""
import csv
import io
import json

import pytest

from src.research.actor_store import ActorStore, EdgeKind, native_node
from src.research.operator_play import OperatorPlayService


@pytest.fixture
def service(tmp_path):
    return OperatorPlayService(tmp_path/'operator.sqlite', actor_store=ActorStore())


def fact(oid, at, **extra):
    return dict(chain_id='solana', chain_family='SOLANA', observation_id=oid,
                event_at=at, observed_at=at, **extra)


def consume(service, kind, payload, now=None):
    return service.consume('OPERATOR_'+kind, payload, ['canonical:'+payload['observation_id']],
                           payload['observed_at'] if now is None else now)


def observed(service, kind, p, now=None):
    result = consume(service, kind, p, now)
    assert result['disposition'] == 'OBSERVED', result
    assert result['detail']['live_authority'] is False
    return result['detail']


def edge(service, source, target, kind, at, **kw):
    return service.actor_store.observe_native(chain_id=kw.pop('chain_id','solana'),
        chain_family=kw.pop('chain_family','SOLANA'), source=source, target=target,
        kind=EdgeKind(kind), event_at=at, observed_at=at,
        observation_id=f'{source}-{target}-{at}', evidence_refs=['canonical:edge'], **kw)


def watch(service, at=10, **kw):
    return observed(service, 'WATCH', fact('watch'+str(at), at, funder='F', **kw))


def funding(service, at=11, oid='fund', now=None, **kw):
    p = fact(oid, at, source='F', target='Dev', amount_raw=100,
             amount_unit='lamports', **kw)
    return observed(service, 'FUNDING', p, now)


def launch(service, token='Token', at=12, **kw):
    return observed(service, 'LAUNCH', fact('launch'+token+str(at), at,
                    wallet='Dev', token=token, **kw))


def test_pit_and_identity_conflicts_survive_restart(service):
    p = fact('fixed', 10, funder='F')
    result = consume(service, 'WATCH', p)
    reopened = OperatorPlayService(service.path)
    assert consume(reopened, 'WATCH', p, 20) == result
    assert consume(reopened, 'WATCH', {**p, 'minimum_raw':200},20)['disposition'] == 'DATA_BLOCKED'
    assert consume(service,'WATCH',fact('future',20,funder='F'),19)['disposition'] == 'DATA_BLOCKED'
    assert consume(service,'WATCH',{**p,'observation_id':'nan','event_at':float('nan')},20)['disposition'] == 'DATA_BLOCKED'


def test_actual_processing_arm_clock_blocks_backlog_fake_snipe(service):
    watch(service)
    arm = funding(service, now=30)['armed'][0]
    assert arm['funding_available_at'] == 11 and arm['armed_at'] == 30
    result = observed(service,'LAUNCH',fact('late-launch',20,wallet='Dev',token='Late'),31)
    assert not result['plans'] and result['missed'][0]['reason'] == 'LAUNCH_PREDATES_ACTUAL_ARMING'
    assert not launch(service,'Later',32)['plans']  # first observed launch consumed, even when missed


def test_freshness_native_units_and_funding_bounds_fail_honestly(service):
    watch(service,minimum_raw=50,maximum_raw=200,freshness='VERIFIED_FIRST_TRANSACTION')
    assert not funding(service)['armed']
    assert not funding(service,oid='history',wallet_history='HAS_HISTORY',wallet_history_evidence_refs=['history'])['armed']
    assert funding(service,oid='fresh',wallet_history='VERIFIED_FIRST_TRANSACTION',wallet_history_evidence_refs=['history'])['armed']
    bad = fact('wei',12,source='F',target='Other',amount_raw=100,amount_unit='wei')
    assert consume(service,'FUNDING',bad)['disposition'] == 'DATA_BLOCKED'
    assert consume(service,'WATCH',fact('float',12,funder='F',minimum_raw=0.1))['disposition'] == 'DATA_BLOCKED'
    assert consume(service,'WATCH',fact('bool',12,funder='F',minimum_raw=True))['disposition'] == 'DATA_BLOCKED'


def test_global_token_dedupe_and_independent_first_buy_creation_policy(service):
    watch(service, trigger='BOTH', first_launch_only=False,first_buy_only=True)
    funding(service)
    first = launch(service,'A',12,trigger='BUY')['plans']
    assert len(first) == 1 and first[0]['filled'] is False
    assert first[0]['status'] == 'AWAITING_QUOTE_COST_SAFETY_AND_SELL_VALIDATION'
    assert not launch(service,'B',13,trigger='BUY')['plans']
    assert launch(service,'B',14,trigger='CREATION')['plans']
    assert not launch(service,'B',15,trigger='CREATION')['plans']
    funding(service,16,oid='another-arm')
    assert not launch(service,'A',17,trigger='BUY')['plans']
    assert not launch(OperatorPlayService(service.path),'A',18,trigger='CREATION')['plans']


def test_funding_before_watch_does_not_retroactively_arm(service):
    watch(service,20)
    assert not funding(service,11,now=21)['armed']


def test_typed_chain_addresses_and_contract_funders(service):
    p = fact('evmwatch',10,funder='0xABC',funder_type='contract')
    p.update(chain_id='evm:1',chain_family='EVM')
    observed(service,'WATCH',p)
    f = {**p,'observation_id':'evmfund','event_at':11,'observed_at':11,
         'source':'0xabc','source_type':'contract','target':'0xDEV','amount_raw':10,'amount_unit':'wei'}
    assert observed(service,'FUNDING',f)['armed']
    assert native_node('solana','wallet','AbC','SOLANA') != native_node('solana','wallet','abc','SOLANA')
    assert not funding(service)['armed']


def test_indexed_prefix_avoids_future_and_preserves_deferred_count(service):
    for at in range(1,9):
        edge(service,'Buyer'+str(at),'T','bought',at,target_type='token')
    token = native_node('solana','token','T','SOLANA')
    prefix,count = service.actor_store.observed_prefix(token,EdgeKind.BOUGHT,5,limit=2,offset=1)
    assert count == 5 and [e.event_at for e in prefix] == [2,3]
    with pytest.raises(ValueError):
        service.actor_store.observed_prefix(token,EdgeKind.BOUGHT,float('nan'))


def test_genealogy_all_known_launches_pages_and_evidenced_associations(service):
    edge(service,'F','Dev','funded',1)
    for at,token in enumerate(['Failed','Runner','Future'],2):
        edge(service,'Dev',token,'deployed',at,target_type='token')
        edge(service,'Early',token,'bought',at,target_type='token')
    p = fact('analysis',20,address='F',as_of=3,page_size=1)
    first = observed(service,'ANALYSIS',p)
    assert first['known_deployment_observations_in_walk'] == 2
    assert first['launches'][0]['token_node'].endswith('"Failed"]')
    next_page = observed(service,'ANALYSIS',{**p,'observation_id':'page2',**first['next_cursor']})
    assert next_page['launches'][0]['token_node'].endswith('"Runner"]')
    assert not first['full_history_verified'] and first['forecast_probability'] is None
    both = observed(service,'ANALYSIS',{**p,'observation_id':'both','page_size':10})
    assert list(both['recurring_observed_buyers'].values()) == [2]
    assert both['launches'][0]['benchmark']['entry'] is None


def test_exchange_hubs_do_not_authenticate_a_common_owner(service):
    service.actor_store.hub_degree=2
    edge(service,'Exchange','Dev','funded',1)
    edge(service,'Exchange','Unrelated','funded',1)
    edge(service,'Unrelated','Other','deployed',2,target_type='token')
    result = observed(service,'ANALYSIS',fact('hub',10,address='Dev',as_of=10))
    assert result['hubs_skipped'] and not result['launches']


def trade(service,oid,at,price,**kw):
    p = fact(oid,at,token='T',wallet='Ordinary',quote_asset='SOL',price=price,side='BUY',**kw)
    return observed(service,'TRADE',p)


def analysis(service,oid,at):
    return observed(service,'ANALYSIS',fact(oid,at,address='Dev',as_of=at))['launches'][0]['benchmark']


def test_benchmark_uses_entry_provenance_clocks_quote_and_horizons(service):
    edge(service,'Dev','T','deployed',1,target_type='token')
    trade(service,'unclassified',2,0.1)
    assert analysis(service,'empty',3)['entry'] is None
    trade(service,'entry',10,2,benchmark_eligible=True,eligibility_evidence_refs=['verified:ordinary-buy'])
    trade(service,'two',12,4)
    trade(service,'five',15,6)
    # Delayed observations are absent from a past snapshot despite a past event clock.
    delayed = fact('lateprice',11,token='T',wallet='X',quote_asset='SOL',price=100,side='BUY')
    delayed['observed_at']=70
    observed(service,'TRADE',delayed)
    wrong_quote = fact('usd',12,token='T',wallet='X',quote_asset='USD',price=900,side='BUY')
    observed(service,'TRADE',wrong_quote)
    early = analysis(service,'early',20)
    assert early['entry']['price'] == 2 and early['peaks']['2'] == 1 and early['peaks']['5'] == 2
    assert early['window_elapsed']['60'] is False and early['full_tape_verified'] is False
    assert analysis(service,'later',80)['peaks']['2'] == 49
    assert 'NOT_EXECUTABLE_PNL' in early['semantics']


def test_ambiguous_first_buy_is_not_fabricated(service):
    edge(service,'Dev','T','deployed',1,target_type='token')
    trade(service,'a',10,1,benchmark_eligible=True,eligibility_evidence_refs=['a'])
    trade(service,'b',10,2,benchmark_eligible=True,eligibility_evidence_refs=['b'])
    assert analysis(service,'ambiguous',20)['status'] == 'ENTRY_ORDER_AMBIGUOUS'


def test_csv_is_descriptive_and_defends_formula_cells(service):
    benchmark={'status':'NO_EVIDENCED_ENTRY_BENCHMARK','entry':{'quote_asset':'=unsafe'},
               'peaks':{str(h):None for h in (2,5,10,30,60)}}
    result = service.csv_report({'launches':[dict(token_node='T',deployers=['D'],event_at=1,available_at=2,benchmark=benchmark)]})
    row = next(csv.DictReader(io.StringIO(result)))
    assert row['quote_asset'] == "'=unsafe" and row['peak_2s_price_excursion'] == ''


def test_actor_children_are_incremental_and_have_no_rpc_dependency(service):
    edge(service,'F','D1','funded',1)
    edge(service,'F','D2','funded',2)
    p = fact('actor',2,address='F',relation={'kind':'funded','target':'D2'})
    children = service.actor_children(p,['fact'])
    assert [c['kind'] for c in children] == ['OPERATOR_ANALYSIS']
    edge(service,'F','D3','funded',3)
    assert service.actor_children({**p,'observed_at':3},['fact']) == []
    assert service.actor_children({**p,'chain_family':'UNKNOWN'},['fact']) == []


def test_token_seed_resolves_only_a_creator_known_at_cutoff(service):
    edge(service,'Creator','T','deployed',10,target_type='token')
    p = fact('token-seed',20,address='T',entity_type='token',as_of=9)
    assert observed(service,'ANALYSIS',p)['resolved_creator_node'] is None
    p.update(observation_id='known-creator',as_of=10)
    report = observed(service,'ANALYSIS',p)
    assert report['resolved_creator_node'] == native_node('solana','wallet','Creator','SOLANA')
    assert report['known_launches_in_page'] == 1


def test_canonical_workflow_produces_and_consumes_a_real_unfilled_plan(tmp_path):
    from test_research_workflows import create
    desk = create(tmp_path)
    job = desk.enqueue_work('OPERATOR_WATCH',fact('canonical-watch',100,funder='F'),now=100)
    desk.run_pass(now=100)
    assert desk.frontier.job(job)['state'] == 'ACKED'
    funding_fact = fact('canonical-funding',101,address='F',amount_raw=100,amount_unit='lamports',
        relation={'kind':'funded','target':'Dev','target_type':'wallet'})
    parent = desk.enqueue_work('ACTOR_OBSERVATION',funding_fact,now=102)
    for _ in range(4):
        desk.run_pass(now=102)
    assert desk.frontier.job(parent)['state'] == 'ACKED'
    launch_fact = fact('canonical-launch',103,address='Dev',
        relation={'kind':'deployed','target':'T','target_type':'token'})
    desk.enqueue_work('ACTOR_OBSERVATION',launch_fact,now=104)
    for _ in range(4):
        desk.run_pass(now=104)
    import sqlite3
    with sqlite3.connect(desk.workflows.operator.path) as db:
        plans = [json.loads(r[0]) for r in db.execute('SELECT body FROM plans')]
    assert len(plans) == 1
    assert plans[0]['watch_armed_at'] == 102 and plans[0]['signal_event_at'] == 103
    assert plans[0]['filled'] is False and plans[0]['live_authority'] is False
    from src.research.trade_evidence import TradeEvidenceLedger
    assert TradeEvidenceLedger.verify(desk.evidence.path)[0]


def test_exporter_reads_a_retained_report_without_database_or_provider_io(service,tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    edge(service,'Dev','T','deployed',1,target_type='token')
    report = observed(service,'ANALYSIS',fact('export',10,address='Dev',as_of=10))
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'detail':report}),encoding='utf-8')
    csv_path = tmp_path/'report.csv'
    tool = Path(__file__).resolve().parents[1]/'tools'/'operator_play_report.py'
    done = subprocess.run([sys.executable,str(tool),'--receipt',str(receipt),'--csv',str(csv_path)],
                          check=True,capture_output=True,text=True)
    assert json.loads(done.stdout)['launch_rows'] == 1
    assert len(list(csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8'))))) == 1


def test_unconfigured_actor_path_does_not_open_sqlite(service,monkeypatch):
    def unavailable():
        raise AssertionError('unconfigured actor path must remain index-only')
    monkeypatch.setattr(service,'_connect',unavailable)
    assert service.actor_children(fact('actor',2,address='F',relation={'kind':'funded','target':'D'}),['canonical']) == []


def test_report_paging_retains_every_known_deployment_and_prefix_debt(service):
    for at in range(1,18):
        edge(service,'Dev',f'T{at}','deployed',at,target_type='token')
    for at in range(1,28):
        edge(service,f'Buyer{at}','T1','bought',at,target_type='token')
    cursor, rows, page = {}, [], 0
    while cursor is not None:
        report = observed(service,'ANALYSIS',fact('page'+str(page),100,address='Dev',as_of=100,page_size=3,**cursor))
        assert report['known_deployment_observations_in_walk'] == 17
        rows.extend(report['launches'])
        cursor=report['next_cursor'];page+=1
        assert page < 20
    assert len({r['token_node'] for r in rows}) == 17
    first = next(r for r in rows if json.loads(r['token_node'][7:])[-1] == 'T1')
    assert first['available_buy_observations'] == 27 and not first['buyer_prefix_complete']
    assert len(first['observed_buyers']) == 25
