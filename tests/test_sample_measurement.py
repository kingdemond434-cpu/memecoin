"""Statistical fixtures verify arithmetic, not a trading edge."""
import math

import pytest

from src.research.sample_measurement import deflated_sharpe_ratio, effective_sample, markout_diagnostics


def test_conditional_dsr_matches_independent_scipy_moment_calculation():
    import numpy as np
    from scipy import stats
    values=np.array([-.05,.02,.03,-.01,.01,.04,.007,-.02,.01,.03])
    sr=float(values.mean()/values.std(ddof=1));n=len(values)
    denominator=1-stats.skew(values,bias=False)*sr+(stats.kurtosis(values,fisher=False,bias=False)-1)/4*sr**2
    expected=math.sqrt(.02)*((1-.5772156649015329)*stats.norm.ppf(1-1/20)
        +.5772156649015329*stats.norm.ppf(1-1/(20*math.e)))
    row=deflated_sharpe_ratio(values,trial_count=20,trial_sharpe_variance=.02,selection_evidence_ref='fixture:trial-register')
    assert row['sharpe_per_observation']==pytest.approx(sr)
    assert row['deflated_sharpe_ratio_iid']==pytest.approx(stats.norm.cdf((sr-expected)*math.sqrt((n-1)/denominator)))
    assert row['deflated_sharpe_ratio_iid']<row['probabilistic_sharpe_ratio_iid']
    assert row['economic_edge']=='UNPROVEN' and 'passed' not in row


def test_missing_selection_and_constant_returns_never_publish_fake_certainty():
    row=deflated_sharpe_ratio([-.1,.2,.1,-.04,.01])
    assert row['deflated_sharpe_ratio_iid'] is None and row['status']=='SELECTION_UNMEASURED'
    for values in ([],[.1,.2],[0]*10,[1]*20):
        row=deflated_sharpe_ratio(values)
        assert row['probabilistic_sharpe_ratio_iid'] is None
        assert row['deflated_sharpe_ratio_iid'] is None


@pytest.mark.parametrize('kwargs',[
    {'trial_count':10}, {'trial_count':True,'trial_sharpe_variance':.1,'selection_evidence_ref':'fixture'},
    {'trial_count':10,'trial_sharpe_variance':0,'selection_evidence_ref':'fixture'},
    {'trial_count':10,'trial_sharpe_variance':float('nan'),'selection_evidence_ref':'fixture'},
])
def test_partial_or_invalid_selection_metadata_is_refused_even_with_no_observations(kwargs):
    with pytest.raises(ValueError):deflated_sharpe_ratio([],**kwargs)


def test_irregular_overlap_and_repeated_launches_are_not_independent_rows():
    row=effective_sample([(0,10),(1,11),(10,20),(20,30)],['a','b','a','b'])
    assert row['raw_rows']==4 and row['maximum_disjoint_windows']==3
    assert row['independent_sample_upper_bound']==2 and row['effective_sample_size'] is None


def test_exact_duplicate_fill_is_counted_once_and_conflicting_duplicates_refused():
    row={'fill_id':'sig','market':'token','interval':[100,105],'net_return':.02}
    report=markout_diagnostics([row,row.copy()])
    assert report['duplicate_observations']==1 and report['selection']['observations']==1
    with pytest.raises(ValueError):markout_diagnostics([row,{**row,'net_return':.03}])


def test_audit_uses_only_cost_complete_fill_markouts_and_keeps_corpus_unchanged(tmp_path):
    import gzip,json,hashlib
    from tools.execution_measurements import audit
    from src.research.execution_measurement import reserve_reference,WSOL
    source=tmp_path/'episodes';source.mkdir()
    attempt={'simulated':False,'filled':True,'landed':True,'success':True,'signature':'sig',
        'measurement_context':{'amount_proof':'confirmed_owner_balance_deltas','signature':'sig',
            'input_mint':WSOL,'output_mint':'mint','input_amount_raw':1000,'output_amount_raw':10,
            'confirmation_observed_at':100.,'arrival_at':99.,'external_costs_complete':True,
            'external_cost_quote_raw':10}}
    tape=[{'execution_reference':reserve_reference('mint',px,1,observed_at=at,source_ref='fixture:'+str(at))}
          for at,px in [(100,100),(105,110)]]
    path=source/'fixture.json.gz'
    with gzip.open(path,'wt') as sink:json.dump({'token':'mint','execution_attempts':[attempt],
        'market_observations':tape},sink)
    original=hashlib.sha256(path.read_bytes()).hexdigest()
    report=audit(source,tmp_path/'receipt.jsonl',as_of=105)
    groups=report['cost_complete_markout_samples'];assert len(groups)==1
    row=next(iter(groups.values()))
    assert row['selection']['observations']==1 and row['realized_profit'] is None
    assert row['selection']['deflated_sharpe_ratio_iid'] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original
    attempt['measurement_context']['external_costs_complete']=False
    with gzip.open(path,'wt') as sink:json.dump({'token':'mint','execution_attempts':[attempt],
        'market_observations':tape},sink)
    assert audit(source,tmp_path/'gross-only.jsonl',as_of=105)['cost_complete_markout_samples']=={}
