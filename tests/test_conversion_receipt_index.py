import pytest

from src.research.frontier import FrontierIndex
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine
from src.strategies.champion_challenger import ChampionChallengerFramework


def test_legacy_refs_survive_migration_and_are_no_longer_copied_per_job(tmp_path, monkeypatch):
    framework=ChampionChallengerFramework(state_path=str(tmp_path/'framework.json'))
    refs={f'{n:064x}':f'evidence:{n:064x}:{n}' for n in range(2000)}
    framework.checkpoint_research_component('research_conversion_receipts',{'refs':refs})
    frontier=FrontierIndex(tmp_path/'frontier.sqlite')
    UnknownOpportunityEngine(framework,frontier,TradeEvidenceLedger(tmp_path/'evidence.jsonl'))
    state=framework.get_research_component('research_conversion_receipts')
    assert 'refs' not in state and state['migrated_refs']==2000
    frontier.close()
    frontier=FrontierIndex(tmp_path/'frontier.sqlite')
    assert frontier.conversion_receipt(f'{1999:064x}')==refs[f'{1999:064x}']
    frontier.retain_conversion_receipts({f'{2001:064x}':f'evidence:{2001:064x}:2001'})
    assert framework.get_research_component('research_conversion_receipts')==state
    with pytest.raises(ValueError,match='conflicting'):
        frontier.retain_conversion_receipts({f'{1999:064x}':'evidence:other:1'})
    assert frontier.conversion_receipt(f'{1999:064x}')==refs[f'{1999:064x}']
    frontier.close()


def test_interruption_after_index_commit_preserves_legacy_until_restart(tmp_path, monkeypatch):
    framework=ChampionChallengerFramework(state_path=str(tmp_path/'framework.json'))
    refs={'a'*64:'evidence:'+'b'*64+':0'}
    framework.checkpoint_research_component('research_conversion_receipts',{'refs':refs})
    frontier=FrontierIndex(tmp_path/'frontier.sqlite')
    checkpoint=framework.checkpoint_research_component
    monkeypatch.setattr(framework,'checkpoint_research_component',lambda *args:(_ for _ in ()).throw(OSError('injected disk failure')))
    with pytest.raises(OSError): UnknownOpportunityEngine(framework,frontier,TradeEvidenceLedger(tmp_path/'evidence.jsonl'))
    assert framework.get_research_component('research_conversion_receipts')['refs']==refs
    assert frontier.conversion_receipt('a'*64)==refs['a'*64]
    monkeypatch.setattr(framework,'checkpoint_research_component',checkpoint)
    UnknownOpportunityEngine(framework,frontier,TradeEvidenceLedger(tmp_path/'evidence.jsonl'))
    assert 'refs' not in framework.get_research_component('research_conversion_receipts')
    frontier.close()
