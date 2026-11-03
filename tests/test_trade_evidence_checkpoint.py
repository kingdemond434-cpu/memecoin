"""Bounded ledger boot via a verified tail checkpoint.

Spot sat >40 min in its 'prediction' boot phase on 2026-11-03 re-verifying
a 13.5 GB evidence ledger row by row at a 5% CPU cap.
"""
import json

import pytest

from src.research import trade_evidence as t
from tools.verify_evidence_backup import verify


def _mark(path):
    return path.with_name(path.name + ".tail.json")


def _ledger(tmp_path, rows=5):
    path = tmp_path / "trade_evidence.jsonl"
    ledger = t.TradeEvidenceLedger(path)
    for i in range(rows):
        assert ledger.record("candidate", {"i": i}) is not None
    ledger.flush_checkpoint()
    return path, ledger


def test_reopen_resumes_from_checkpoint_without_decoding_history(tmp_path, monkeypatch):
    path, ledger = _ledger(tmp_path, rows=50)
    decoded = []
    real = t._verified_row
    monkeypatch.setattr(t, "_verified_row", lambda *a: decoded.append(1) or real(*a))
    again = t.TradeEvidenceLedger(path)
    assert again.recovery_mode == "TAIL_CHECKPOINT_AND_VERIFIED_TAIL"
    assert (again.sequence, again.last_hash) == (ledger.sequence, ledger.last_hash)
    assert len(decoded) == 1  # only the boundary row
    assert again.record("candidate", {"i": 50})["sequence"] == 51
    assert t.TradeEvidenceLedger.verify(path) == (True, "OK", 51)


def test_rows_after_checkpoint_are_chain_verified(tmp_path):
    path, ledger = _ledger(tmp_path)
    ledger.checkpoint_interval_s = 1e9
    for i in range(3):
        ledger.record("late", {"i": i})
    again = t.TradeEvidenceLedger(path)
    assert again.checkpoint_used and again.sequence == 8 and again.last_hash == ledger.last_hash


@pytest.mark.parametrize("damage", ["prefix_byte", "late_row", "truncate", "bad_hash", "past_eof", "base"])
def test_damage_never_yields_an_unverified_tail(tmp_path, damage):
    path, ledger = _ledger(tmp_path)
    ledger.checkpoint_interval_s = 1e9
    ledger.record("late", {"v": 1})
    raw, mark = path.read_bytes(), json.loads(_mark(path).read_text())
    if damage == "prefix_byte":
        path.write_bytes(raw.replace(b'"i":0', b'"i":9', 1))
    elif damage == "late_row":
        path.write_bytes(raw.replace(b'"v":1', b'"v":2'))
    elif damage == "truncate":
        path.write_bytes(raw[:mark["extent"] - 10])
    else:
        if damage == "bad_hash": mark["segment_sha256"] = "0" * 64
        if damage == "past_eof": mark["extent"] += 10_000
        if damage == "base": mark["base_extent"] = 7
        _mark(path).write_text(json.dumps(mark))
    again = t.TradeEvidenceLedger(path)
    if damage == "prefix_byte":
        # The checkpoint's segment hash catches it; the pre-existing legacy
        # path (unchanged here) then reads only the tail row, as before.
        assert not again.checkpoint_used and again.checkpoint_rejected_type == "ValueError"
    elif damage in ("bad_hash", "past_eof", "base"):
        # checkpoint rejected, full parse of an intact file succeeds
        assert not again.checkpoint_used and again.recovery_mode == "LEGACY_FULL_PARSE"
        assert (again.sequence, again.last_hash) == (ledger.sequence, ledger.last_hash)
    else:
        assert again.last_hash == "" and again.record("x", {}) is None


def test_certified_prefix_still_hashed_and_checkpoint_resumes_tail(tmp_path, monkeypatch):
    path = tmp_path / "live.jsonl"
    ledger = t.TradeEvidenceLedger(path)
    for i in range(5):
        ledger.record("public_fixture", {"i": i})
    cert = tmp_path / "cert.json"
    cert.write_text(json.dumps(verify(path)))
    opened = t.TradeEvidenceLedger(path, recovery_certificate=cert)   # anchors base at cert extent
    for i in range(20):
        opened.record("later", {"i": i})
    opened.flush_checkpoint()
    decoded = []
    real = t._verified_row
    monkeypatch.setattr(t, "_verified_row", lambda *a: decoded.append(1) or real(*a))
    again = t.TradeEvidenceLedger(path, recovery_certificate=cert)
    assert again.report()["recovery_mode"] == "CERTIFIED_PREFIX_SHA256_TAIL_CHECKPOINT_AND_VERIFIED_TAIL"
    assert again.sequence == 25 and again.last_hash == opened.last_hash
    assert len(decoded) <= 2  # certificate boundary row + checkpoint boundary row


def test_bad_certificate_still_blocks_even_with_valid_checkpoint(tmp_path):
    path, ledger = _ledger(tmp_path)
    cert = tmp_path / "cert.json"
    cert.write_text("{}")
    again = t.TradeEvidenceLedger(path, recovery_certificate=cert)
    assert again.report()["recovery_mode"] == "CERTIFIED_RECOVERY_FAILED"
    assert again.record("x", {}) is None
