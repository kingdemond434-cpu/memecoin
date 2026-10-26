"""Owner-certified local recovery controls; synthetic native events, no RPC."""
import json

import pytest
from solders.pubkey import Pubkey

from src.chains.launchlab import LAUNCHLAB_PROGRAM
from src.chains.pump_route import WSOL_MINT
from src.research.fork_probe import ForkProbe
from src.research.trade_evidence import TradeEvidenceLedger
from tools.backfill_native_fork_targets import backfill
from tools.verify_evidence_backup import verify


def source_at(root):
    ledger_path=root/'sealed.jsonl';certificate=root/'certificate.json'
    event={'type':'token_launch','venue':'raydium_launchlab','program':LAUNCHLAB_PROGRAM,
           'quote_mint':WSOL_MINT,'token':str(Pubkey.new_unique()),'pool':str(Pubkey.new_unique())}
    ledger=TradeEvidenceLedger(ledger_path)
    ledger.record('native_launch',event,timestamp=50.)
    ledger.record('native_launch',event,timestamp=60.)
    ledger.record('native_launch',{**event,'quote_mint':str(Pubkey.new_unique())},timestamp=70.)
    ledger.record('research_typed_work',{'native_launch':event,'marker':'native_launch'},timestamp=80.)
    certificate.write_text(json.dumps(verify(ledger_path)))
    return ledger_path,certificate


def test_dry_run_verifies_all_source_bytes_without_creating_research_database(tmp_path):
    ledger,certificate=source_at(tmp_path);before=ledger.read_bytes();database=tmp_path/'probe.sqlite'
    report=backfill(ledger,certificate,database=database)
    assert report['status']=='DRY_RUN' and not report['applied']
    assert report['source_records']==4 and report['native_rows']==3
    assert report['reviewed_source_rows']==2 and report['reviewed_targets']==1
    assert report['other_or_unsupported']==1 and report['source_unchanged']
    assert not database.exists() and ledger.read_bytes()==before


def test_apply_recovers_one_target_with_two_original_sources_and_is_idempotent(tmp_path):
    ledger,certificate=source_at(tmp_path);database=tmp_path/'probe.sqlite'
    first=backfill(ledger,certificate,database=database,apply=True)
    second=backfill(ledger,certificate,database=database,apply=True)
    assert first['status']==second['status']=='APPLIED'
    assert first['new_targets']==1 and second['new_targets']==0
    probe=ForkProbe(database)
    assert probe.report()['targets']==1 and probe.report()['canonical_native_source_links']==2
    assert [row[0] for row in probe.db.execute('SELECT available FROM native_target_sources ORDER BY available')]==[50.,60.]
    assert probe.report()['canonical_dispositions']=={}  # Enrollment is not a simulated fill.
    probe.close()


def test_modified_source_is_rejected_before_any_target_is_written(tmp_path):
    ledger,certificate=source_at(tmp_path);database=tmp_path/'probe.sqlite'
    raw=ledger.read_bytes().replace(b'raydium_launchlab',b'raydium_launchlax',1)
    ledger.write_bytes(raw)
    with pytest.raises(ValueError):backfill(ledger,certificate,database=database,apply=True)
    assert not database.exists()


def test_growing_live_journal_is_not_treated_as_the_sealed_certificate_extent(tmp_path):
    ledger,certificate=source_at(tmp_path);database=tmp_path/'probe.sqlite'
    TradeEvidenceLedger(ledger).record('later',{'observed':True})
    with pytest.raises(ValueError,match='sealed'):backfill(ledger,certificate,database=database,apply=True)
    assert not database.exists()


def test_apply_cannot_replace_either_source_artifact(tmp_path):
    ledger,certificate=source_at(tmp_path);before=ledger.read_bytes()
    for database in (ledger,certificate):
        with pytest.raises(ValueError,match='replace'):backfill(ledger,certificate,database=database,apply=True)
    assert ledger.read_bytes()==before
