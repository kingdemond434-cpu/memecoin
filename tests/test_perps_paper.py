"""Economic invariants for real future-book paper controls, never live orders."""
import json
import time
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from src.research.phantom_perps import PhantomPerpsResearch
from src.research.perps_paper import PerpsPaperLedger, _margin, _walk
from tests.test_phantom_perps import catalog, book


@pytest.fixture
def desk(tmp_path,monkeypatch):
    import src.research.perps_paper as paper
    clock = SimpleNamespace(now=int(time.time()//3600)*3600+1800)
    monkeypatch.setattr(paper,'time',SimpleNamespace(time=lambda:clock.now))
    state = PhantomPerpsResearch(tmp_path/'perps.sqlite')
    state.paper = PerpsPaperLedger(state,now=clock.now-1,controls=False)
    def ingest(role, **fields):
        clock.now += 1
        row = {'provider':'hyperliquid','product':'perpetual','perps_role':role,'observed_at':clock.now,**fields}
        if role=='catalog':
            row = {**catalog(at=clock.now),**fields,'observed_at':clock.now}
        if role=='book':
            row.setdefault('coin','BTC')
            row.setdefault('book',book(clock.now))
        state.ingest_batch([row],'canonical:'+str(clock.now))
        assert state.report()['derived_record_errors']==0
        return row
    ingest('collateral_inventory',tokens=[{'index':0,'name':'USDC','isCanonical':True}])
    ingest('catalog')
    with state.db:
        account = state.paper._create('account',clock.now,'canonical:seed')
    yield SimpleNamespace(state=state,paper=state.paper,clock=clock,ingest=ingest,account=account)
    state.close()


def submit(desk,q,**kwargs):
    with desk.state.db:
        return desk.paper.submit('account','BTC',q,at=desk.clock.now,ref='canonical:intent:'+str(desk.clock.now),**kwargs)


def test_roundtrip_cash_loss_equals_recorded_spread_and_both_fees(desk):
    submit(desk,'0.1')
    desk.ingest('book')
    opened = desk.paper._account('account')
    assert D(opened['positions']['BTC']['quantity'])==D('0.1')
    submit(desk,'-0.1',reduce_only=True)
    desk.ingest('book')
    closed = desk.paper._account('account')
    assert D(closed['fees'])==D('.019')  # 10.1 + 9.9 notional * .00095
    assert D(closed['realized_pnl'])==D('-.2')
    assert D(closed['cash'])==D('99.781')
    assert D(closed['positions']['BTC']['margin'])==0
    assert desk.paper.report()['real_fills']==0
    assert desk.paper.report()['simulated_fills']==2
    assert desk.paper.report()['verified_net_profit'] is None


def test_same_or_old_book_cannot_fill_and_restart_cannot_duplicate_a_fill(desk):
    identity = submit(desk,'0.1')
    at = desk.clock.now
    desk.ingest('book',book=book(at))
    assert desk.paper.report()['simulated_fills']==0
    row = desk.ingest('book')
    assert desk.paper.report()['simulated_fills']==1
    desk.state.ingest_batch([row],'canonical:'+str(desk.clock.now))
    assert desk.paper.report()['simulated_fills']==1
    restored = PerpsPaperLedger(desk.state,controls=False)
    assert restored.study==desk.paper.study
    assert D(restored._account('account')['positions']['BTC']['quantity'])==D('0.1')
    assert desk.state.db.execute('SELECT status FROM pp_orders WHERE id=?',(identity,)).fetchone()[0]=='FILLED'


def test_reduce_only_partial_depth_cannot_flip_and_unfilled_ioc_is_cancelled(desk):
    submit(desk,'0.3')
    desk.ingest('book')
    submit(desk,'-5',reduce_only=True)
    limited = book(desk.clock.now+1)
    limited['levels'][0]=[{'px':'99','sz':'0.1'}]
    desk.ingest('book',book=limited)
    position = desk.paper._account('account')['positions']['BTC']
    assert D(position['quantity'])==D('.2')
    assert desk.paper.report()['orders']['PARTIAL_IOC']==1
    assert desk.paper.report()['orders'].get('PENDING',0)==0
    with pytest.raises(ValueError): submit(desk,'1',reduce_only=True)


def test_entry_rejections_preserve_cash_and_never_create_positions(desk):
    submit(desk,'100',leverage='40',mode='cross')
    desk.ingest('book')
    account = desk.paper._account('account')
    assert account['cash']=='100' and account['positions']=={}
    assert desk.paper.report()['orders']['REJECTED_MARGIN']==1
    with pytest.raises(ValueError): submit(desk,'0.00001')


def test_tier_deductions_are_continuous_and_boundary_uses_next_leverage():
    asset={'marginTableId':100}
    meta={'marginTables':[[100,{'marginTiers':[{'lowerBound':'0','maxLeverage':20},
                                                   {'lowerBound':'1000','maxLeverage':10}]}]]}
    assert _margin('999',asset,meta)==(D('24.975'),D(20))
    assert _margin('1000',asset,meta)==(D('25'),D(10))
    assert _margin('1100',asset,meta)==(D('30'),D(10))
    assert _margin('1100',{'marginTableId':20},{})==(D('27.5'),D(20))
    with pytest.raises(ValueError): _margin('1',{'marginTableId':100},{})


def test_bad_book_or_size_cannot_become_an_executable_price():
    now=time.time()
    crossed=book(now);crossed['levels'][0][0]['px']='500'
    with pytest.raises(ValueError): _walk(crossed,'.1',4)
    with pytest.raises(ValueError): _walk(book(now),'NaN',4)
    assert _walk(book(now),'.3',4,limit='101')==(D('.2'),D('20.2'))


def test_stale_mark_never_certifies_new_margin(desk):
    desk.clock.now+=60
    submit(desk,'0.11')
    desk.ingest('book')
    assert desk.paper.report()['simulated_fills']==0
    assert desk.paper.report()['orders']['UNMEASURED_MARK']==1


def test_repeating_vwap_partial_exits_conserve_exact_recorded_notional(desk):
    submit(desk,'0.3')
    desk.ingest('book')  # entry costs 30.4, repeating average price
    for q in ('-0.1','-0.2'):
        submit(desk,q,reduce_only=True)
        desk.ingest('book')
    account=desk.paper._account('account')
    assert D(account['realized_pnl'])==D('-.7')  # 29.7 proceeds minus 30.4 cost
    assert D(account['cash'])==D('100')+D(account['realized_pnl'])-D(account['fees'])
    assert D(account['positions']['BTC']['entry_notional'])==0


def test_cross_margin_and_isolated_margin_react_independently(desk):
    submit(desk,'0.11',mode='cross')
    desk.ingest('book')
    account=desk.paper._account('account')
    assert D(account['positions']['BTC']['margin'])==0
    risk=desk.paper._risk(account,desk.clock.now)
    assert risk['cross_equity']==D(account['cash'])+D('.11')*100-D('11.11')
    assert risk['initial']==D('5.5')
    assert risk['maintenance']==D('.1375')


def test_mutated_spec_cannot_trade_under_a_previous_study_identity(desk):
    desk.paper._plan['fee_bps']='0'
    with pytest.raises(ValueError,match='new study'):
        submit(desk,'0.1')


def test_cross_backstop_preserves_an_isolated_position(desk):
    submit(desk,'0.11')
    desk.ingest('book')
    isolated=dict(desk.paper._account('account')['positions']['BTC'])
    eth=catalog(dex='other',coin='ETH')
    desk.ingest('catalog',dex=eth['dex'],meta=eth['meta'],contexts=eth['contexts'])
    with desk.state.db:
        desk.paper.submit('account','ETH','20',at=desk.clock.now,ref='canonical:cross',mode='cross',leverage='40')
    depth=book(desk.clock.now+1);depth['coin']='ETH'
    depth['levels']=[[{'px':'99','sz':'100'}],[{'px':'101','sz':'100'}]]
    desk.ingest('book',coin='ETH',book=depth)
    eth['contexts'][0].update(markPx='1',oraclePx='1')
    desk.ingest('catalog',dex=eth['dex'],meta=eth['meta'],contexts=eth['contexts'])
    empty={**depth,'time':int((desk.clock.now+1)*1000),'levels':[[{'px':'1','sz':'0'}],[{'px':'2','sz':'0'}]]}
    desk.ingest('book',coin='ETH',book=empty)
    account=desk.paper._account('account')
    assert D(account['cash'])==0
    assert D(account['positions']['ETH']['quantity'])==0
    assert account['positions']['BTC']==isolated
    assert desk.state.db.execute("SELECT count(*) FROM pp_journal WHERE kind='BACKSTOP_MODEL'").fetchone()[0]==1


def test_large_liquidation_begins_with_twenty_percent(desk):
    with desk.state.db:
        account=desk.paper._account('account')
        account.update(cash='20000',seed='20000')
        desk.paper._save(account)
    submit(desk,'2000',leverage='40')
    depth=book(desk.clock.now+1)
    depth['levels']=[[{'px':'99','sz':'10000'}],[{'px':'101','sz':'10000'}]]
    desk.ingest('book',book=depth)
    crash=catalog();crash['contexts'][0].update(markPx='95',oraclePx='95')
    desk.ingest('catalog',meta=crash['meta'],contexts=crash['contexts'])
    order=json.loads(desk.state.db.execute("SELECT body FROM pp_orders WHERE status='PENDING'").fetchone()[0])
    assert D(order['quantity'])==D('-400')
    assert order['reason']=='OBSERVED_MARK_LIQUIDATION'


def test_report_command_reads_all_accounts_by_pages_without_writing(desk,capsys,monkeypatch):
    from tools.perps_paper_report import main
    import tools.perps_paper_report as report_module
    monkeypatch.setattr(report_module,'time',SimpleNamespace(time=lambda:desk.clock.now))
    submit(desk,'0.1');desk.ingest('book')
    count=desk.state.db.execute('SELECT count(*) FROM pp_journal').fetchone()[0]
    assert main(['--db',str(desk.state.db.execute('PRAGMA database_list').fetchone()[2]),'--study',desk.paper.study,'--limit','1'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['total_accounts']==1 and result['trade_authority'] is False
    assert result['accounts'][0]['model_pnl_usdc'] is not None
    assert result['verified_profit'] is None
    assert desk.state.db.execute('SELECT count(*) FROM pp_journal').fetchone()[0]==count


def test_isolated_gap_loss_cannot_spend_unallocated_cross_cash(desk):
    submit(desk,'0.1',leverage='40')
    desk.ingest('book')
    before = desk.paper._account('account')
    free = D(before['cash'])
    submit(desk,'-0.1',reduce_only=True,leverage='40')
    crash = book(desk.clock.now+1)
    crash['levels']=[[{'px':'1','sz':'10'}],[{'px':'2','sz':'10'}]]
    desk.ingest('book',book=crash)
    after = desk.paper._account('account')
    assert D(after['cash'])==free
    assert D(after['bad_debt'])>0
    assert D(after['positions']['BTC']['quantity'])==0


def test_long_and_short_hourly_funding_are_opposite_and_missing_oracle_is_not_zero(desk):
    # A deterministic future boundary in the test clock; no historical fills.
    boundary=(int(desk.clock.now//3600)+1)*3600
    desk.clock.now=boundary-9
    desk.ingest('catalog')  # oracle 99.5, mark 100
    submit(desk,'0.11')
    desk.ingest('book')
    with desk.state.db:
        desk.paper._create('short',desk.clock.now,'canonical:short')
        desk.paper.submit('short','BTC','-0.11',at=desk.clock.now,ref='canonical:short-intent')
    desk.ingest('book')
    desk.clock.now=boundary+1
    desk.ingest('book')
    assert desk.paper.report()['funding']['UNPRICED']==2
    desk.ingest('funding',coin='BTC',start_time=0,rows=[{'coin':'BTC','time':boundary*1000+47,'fundingRate':'0.001'}])
    long = desk.paper._account('account');short = desk.paper._account('short')
    assert D(long['funding_cashflow'])==D('-.010945')
    assert D(short['funding_cashflow'])==D('.010945')
    assert desk.paper.report()['funding']=={'SAMPLED_SCENARIO':2}
    cash=long['funding_cashflow']
    desk.ingest('funding',coin='BTC',start_time=0,rows=[{'coin':'BTC','time':boundary*1000+47,'fundingRate':'0.001'}])
    assert desk.paper._account('account')['funding_cashflow']==cash


def test_closed_position_still_owes_delayed_funding_and_missing_price_stays_unknown(desk):
    boundary=(int(desk.clock.now//3600)+1)*3600
    desk.clock.now=boundary-40
    desk.ingest('catalog');submit(desk,'0.1');desk.ingest('book')
    desk.clock.now=boundary+1
    submit(desk,'-0.1',reduce_only=True);desk.ingest('book')
    desk.ingest('funding',coin='BTC',start_time=0,rows=[{'coin':'BTC','time':boundary*1000,'fundingRate':'0.001'}])
    assert desk.paper.report()['funding']=={'UNPRICED':1}
    assert desk.paper._account('account')['funding_cashflow']=='0'
    assert desk.paper.report()['verified_net_profit'] is None


@pytest.mark.parametrize('close_before_actual_settlement',[False,True])
def test_funding_uses_actual_millisecond_exposure_not_nominal_hour(desk,close_before_actual_settlement):
    boundary=(int(desk.clock.now//3600)+1)*3600
    desk.clock.now=boundary-8
    desk.ingest('catalog')
    if close_before_actual_settlement:
        submit(desk,'0.1');desk.ingest('book')
        submit(desk,'-0.1',reduce_only=True)
    else:
        submit(desk,'0.1')
    desk.clock.now=boundary-.98
    desk.ingest('book')  # modeled fill at +20ms; funding actually stamped +47ms
    desk.ingest('funding',coin='BTC',start_time=0,
        rows=[{'coin':'BTC','time':boundary*1000+47,'fundingRate':'0.001'}])
    account=desk.paper._account('account')
    if close_before_actual_settlement:
        assert account['funding_cashflow']=='0'
        assert desk.paper.report()['funding']=={'NO_EXPOSURE':1}
    else:
        assert D(account['funding_cashflow'])==D('-.00995')
        assert desk.paper.report()['funding']=={'SAMPLED_SCENARIO':1}


def test_liquidation_requires_future_book_and_backstop_is_a_transfer_not_fake_fill(desk):
    submit(desk,'0.1',leverage='40')
    desk.ingest('book')
    free = D(desk.paper._account('account')['cash'])
    crash = catalog(at=desk.clock.now+1)
    crash['contexts'][0].update(markPx='1',oraclePx='1')
    desk.ingest('catalog',**{k:v for k,v in crash.items() if k in {'meta','contexts','dex'}})
    assert desk.paper.report()['orders']['PENDING']==1
    empty = book(desk.clock.now+1)
    empty['levels']=[[{'px':'1','sz':'0'}],[{'px':'2','sz':'0'}]]
    desk.ingest('book',book=empty)
    account = desk.paper._account('account')
    assert D(account['cash'])==free
    assert D(account['positions']['BTC']['quantity'])==0
    assert desk.paper.report()['simulated_fills']==1
    assert desk.state.db.execute("SELECT count(*) FROM pp_journal WHERE kind='BACKSTOP_MODEL'").fetchone()[0]==1
    assert sum((D(r[0]) for r in desk.state.db.execute("SELECT delta FROM pp_exposure WHERE account='account'")),D(0))==0


def test_real_canonical_pipeline_runs_long_short_controls_without_claiming_alpha(tmp_path):
    state=PhantomPerpsResearch(tmp_path/'controls.sqlite')
    now=time.time()
    rows=[{'provider':'hyperliquid','product':'perpetual','perps_role':'collateral_inventory',
           'observed_at':now,'tokens':[{'index':0,'name':'USDC','isCanonical':True}]},catalog(at=now)]
    state.ingest_batch(rows,'canonical:seed')
    first={'provider':'hyperliquid','product':'perpetual','perps_role':'book','coin':'BTC',
           'observed_at':now,'book':book(now)}
    state.ingest_batch([first],'canonical:first')
    assert state.report()['paper_ledger']['orders']=={'PENDING':2}
    # A test-clock future observation after the latency interval; still inside
    # the measured freshness window and no real exchange submission.
    second={**first,'observed_at':now+1,'book':book(now+1)}
    import src.research.perps_paper as module
    original=module.time
    module.time=SimpleNamespace(time=lambda:now+1)
    try: state.ingest_batch([second],'canonical:second')
    finally: module.time=original
    report=state.report()
    assert report['simulated_control_fills']==2
    assert report['simulated_strategy_fills']==0 and report['real_fills']==0
    assert report['paper_ledger']['open_positions']==2
    state.close()
