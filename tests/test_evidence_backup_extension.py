import json
import shutil

import pytest

from src.research.trade_evidence import TradeEvidenceLedger
from tools.verify_evidence_backup import verify


def fixture(tmp_path):
    path = tmp_path/'live.jsonl'
    ledger = TradeEvidenceLedger(path)
    for index in range(7):
        ledger.record('public_fixture', {'index':index, 'unicode':'日本語'}, timestamp=index+1)
    prefix = tmp_path/'prefix.jsonl'
    shutil.copy2(path, prefix)
    certificate = verify(prefix)
    for index in range(7,12):
        ledger.record('public_fixture', {'index':index}, timestamp=index+1)
    return path, prefix, certificate


def test_certified_prefix_and_new_rows_equal_full_chain_verification(tmp_path):
    path, prefix, certificate = fixture(tmp_path)
    extended = verify(path, prefix=prefix, certificate=certificate)
    full = verify(path)
    for key in ('canonical_records','canonical_bytes','canonical_sha256','last_hash'):
        assert extended[key] == full[key]
    assert extended['certified_prefix_records'] == 7 and extended['newly_verified_records'] == 5
    assert certificate['verification_mode'] == 'FULL_HASH_CHAIN'
    assert extended['verification_mode'] == 'CERTIFIED_PREFIX_AND_NEW_ROWS'
    # Reusing an entire certified snapshot still hashes and compares its bytes.
    assert verify(path, prefix=path, certificate=full)['newly_verified_records'] == 0


@pytest.mark.parametrize('damage', ['prefix','target_prefix','tail','partial_tail','certificate','short_target'])
def test_corruption_and_incomplete_snapshots_are_refused(tmp_path, damage):
    path, prefix, certificate = fixture(tmp_path)
    if damage == 'prefix':
        prefix.write_bytes(prefix.read_bytes().replace(b'public_fixture', b'public_fixturE', 1))
    elif damage == 'target_prefix':
        path.write_bytes(path.read_bytes().replace(b'public_fixture', b'public_fixturE', 1))
    elif damage == 'tail':
        raw = path.read_bytes()
        boundary = prefix.stat().st_size
        path.write_bytes(raw[:boundary]+raw[boundary:].replace(b'public_fixture', b'public_fixturE', 1))
    elif damage == 'partial_tail':
        path.write_bytes(path.read_bytes()[:-3])
    elif damage == 'certificate':
        certificate['canonical_records'] = 42
    else:
        path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(ValueError):
        verify(path, prefix=prefix, certificate=certificate)


def test_empty_backup_and_missing_prefix_arguments(tmp_path):
    path=tmp_path/'empty.jsonl'
    path.write_bytes(b'')
    certificate=verify(path)
    assert certificate['canonical_records'] == 0
    assert verify(path,prefix=path,certificate=certificate)['canonical_records'] == 0
    with pytest.raises(ValueError):
        verify(path,prefix=path)
