from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.research.proposal_rounds import ProposalRoundRunner, HTTPProposalProvider, ProviderConfig
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine


@pytest.mark.asyncio
@pytest.mark.parametrize('providers_present',[False,True])
async def test_unconfigured_round_does_not_scan_evidence_or_create_a_fake_completed_round(tmp_path,providers_present):
    ledger=TradeEvidenceLedger(tmp_path/'evidence.jsonl')
    ledger.record('existing_fact',{'public':True})
    before=ledger.path.read_bytes()
    providers=[]
    if providers_present:
        providers=[HTTPProposalProvider(ProviderConfig('codex','openai','MODEL','KEY',10000,100,2,enabled=True),environ={})]
    runner=ProposalRoundRunner(ledger,providers)
    def forbidden(*args):raise AssertionError('unconfigured round scanned the ledger')
    runner._previous=forbidden
    desk=SimpleNamespace(proposal_runner=runner,training_facts=[{'evidence_id':'retained-public-fact'}])
    result=await UnknownOpportunityEngine.run_proposal_pass(desk)
    assert result['status']=='DATA_BLOCKED' and result['authority']=='none'
    assert result['network_requests']==0 and result['evidence_scan_started'] is False
    assert ledger.path.read_bytes()==before


@pytest.mark.asyncio
@pytest.mark.parametrize('status',['CONFIGURED_NOT_AUTHENTICATED','CUSTOM_ADAPTER_UNVERIFIED'])
async def test_one_available_provider_still_runs_even_when_other_required_providers_are_missing(status):
    run=AsyncMock(return_value={'status':'PARTIAL','experiment_requests':[]})
    runner=SimpleNamespace(configuration_report=lambda:{'providers':[
        {'provider':'codex','status':status},{'provider':'other','status':'DATA_BLOCKED'}]},run_round=run)
    facts=[{'evidence_id':'retained-public-fact'}]
    desk=SimpleNamespace(proposal_runner=runner,training_facts=facts)
    result=await UnknownOpportunityEngine.run_proposal_pass(desk)
    assert result['status']=='PARTIAL'
    run.assert_awaited_once()
    assert run.call_args.kwargs['facts']==facts
