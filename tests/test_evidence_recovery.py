import json

import pytest

from src.research import trade_evidence as t
from src.research.evidence_certificate import fingerprint
from tools.verify_evidence_backup import verify


def fixture(tmp_path,large=False):
    path=tmp_path/'live.jsonl'
    ledger=t.TradeEvidenceLedger(path)
    for index in range(5):
        ledger.record('public_fixture',{'index':index,'text':'x'*180_000 if large and index==4 else 'public text'})
    certificate=verify(path)
    cert=tmp_path/'protected-certificate.json'
    cert.write_text(json.dumps(certificate))
    return ledger,cert,certificate


@pytest.mark.parametrize('large',[False,True])
def test_certified_startup_recovers_prefix_and_verifies_every_later_row(tmp_path,large):
    ledger,cert,certificate=fixture(tmp_path,large)
    for index in range(3): ledger.record('later',{'index':index})
    recovered=t.TradeEvidenceLedger(ledger.path,recovery_certificate=cert)
    assert recovered.sequence==8 and recovered.last_hash==ledger.last_hash
    assert recovered.report()['recovery_mode']=='CERTIFIED_PREFIX_SHA256_AND_VERIFIED_TAIL'
    assert recovered.report()['certified_prefix_records']==5
    assert recovered.report()['recovery_seconds']>=0
    assert recovered.record('after_restart',{})['sequence']==9
    assert t.TradeEvidenceLedger.verify(ledger.path)==(True,'OK',9)


@pytest.mark.parametrize('damage',['prefix','new_tail','truncated','partial_tail','bad_certificate','wrong_count','wrong_last_hash','missing_certificate'])
def test_bad_checkpoint_or_bytes_never_fall_back_to_unverified_append(tmp_path,damage):
    ledger,cert,certificate=fixture(tmp_path)
    ledger.record('new_tail',{'value':7})
    raw=ledger.path.read_bytes()
    if damage=='prefix': ledger.path.write_bytes(raw.replace(b'public_fixture',b'public_fixturE',1))
    elif damage=='new_tail': ledger.path.write_bytes(raw.replace(b'new_tail',b'new_taiL'))
    elif damage=='truncated': ledger.path.write_bytes(raw[:30])
    elif damage=='partial_tail': ledger.path.write_bytes(raw[:-3])
    elif damage=='missing_certificate': cert.unlink()
    else:
        if damage=='bad_certificate': certificate['certificate_sha256']='0'*64
        else:
            certificate['canonical_records' if damage=='wrong_count' else 'last_hash']=7 if damage=='wrong_count' else '0'*64
            certificate['certificate_sha256']=fingerprint({k:v for k,v in certificate.items() if k!='certificate_sha256'})
        cert.write_text(json.dumps(certificate))
    before=ledger.path.read_bytes()
    recovered=t.TradeEvidenceLedger(ledger.path,recovery_certificate=cert)
    assert recovered.report()['status']=='DATA_BLOCKED'
    assert recovered.report()['recovery_mode']=='CERTIFIED_RECOVERY_FAILED'
    assert recovered.record('must_not_append',{}) is None
    assert ledger.path.read_bytes()==before


def test_large_certified_prefix_is_hashed_without_decoding_all_its_rows(tmp_path,monkeypatch):
    path=tmp_path/'live.jsonl'
    ledger=t.TradeEvidenceLedger(path)
    for index in range(1000): ledger.record('retained',{'index':index})
    cert=tmp_path/'certificate.json'
    cert.write_text(json.dumps(verify(path)))
    ledger.record('new_tail',{})
    calls=[]
    original=t._verified_row
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(t,'_verified_row',counted)
    recovered=t.TradeEvidenceLedger(path,recovery_certificate=cert)
    assert recovered.sequence==1001 and len(calls)==2  # certified endpoint and new row


def test_empty_and_blank_prefixes_and_legacy_startup(tmp_path):
    for index,content in enumerate((b'',b'\n\r\n   \n')):
        path=tmp_path/f'empty-{index}.jsonl';path.write_bytes(content)
        cert=tmp_path/f'cert-{index}.json';cert.write_text(json.dumps(verify(path)))
        ledger=t.TradeEvidenceLedger(path,recovery_certificate=cert)
        assert ledger.sequence==0 and ledger.last_hash==t.GENESIS
        assert ledger.record('first',{})['sequence']==1
    plain=t.TradeEvidenceLedger(tmp_path/'fresh.jsonl')
    plain.record('fixture',{})
    assert t.TradeEvidenceLedger(plain.path).sequence==1


def test_concurrent_file_change_is_detected_before_recovery_is_published(tmp_path,monkeypatch):
    ledger,cert,_=fixture(tmp_path)
    original=t.TradeEvidenceLedger._prefix_last_row
    def changed(handle,extent):
        result=original(handle,extent)
        with ledger.path.open('ab') as writer: writer.write(b'\n')
        return result
    monkeypatch.setattr(t.TradeEvidenceLedger,'_prefix_last_row',staticmethod(changed))
    recovered=t.TradeEvidenceLedger(ledger.path,recovery_certificate=cert)
    assert recovered.report()['status']=='DATA_BLOCKED' and recovered.last_hash==''
