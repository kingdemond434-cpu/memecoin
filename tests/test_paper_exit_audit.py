import copy
import json
import sqlite3

import pytest
from solders.pubkey import Pubkey

from src.research.fork_probe import ForkProbe
from src.research.fork_simulation import TOKEN_PROGRAM, digest, virtual_owner
from src.research.paper_bank import inventory_keys
from tools.paper_exit_audit import audit


def fixture(path, mutation=None):
    probe=ForkProbe(path,paper_enabled=True)
    target={"venue":"pump_curve","token":str(Pubkey.new_unique())}
    frame={"target":target,"slot":2,"observed_at":200.,"accounts":{}}
    frame["accounts"]={key:None for key in inventory_keys(frame,TOKEN_PROGRAM)}
    frame_id=probe._store_frame(frame)
    owner=str(virtual_owner().pubkey())
    accounts={key:None for key in frame["accounts"]}
    accounts[owner]={"lamports":100000,"data":["","base64"],"owner":"11111111111111111111111111111111","executable":False}
    prior={"target":target,"token_program":TOKEN_PROGRAM,"token_units":123,"last_slot":1,
           "last_observed_at":100.,"entry_frame_sha256":"0"*64,"vm_fees_lamports":5000,"accounts":accounts}
    prior["seal"]=digest(prior)
    current=copy.deepcopy(prior);current.pop("seal")
    current.update(vm_fees_lamports=10000,last_slot=2,last_observed_at=200.,last_frame_sha256=frame_id)
    current["accounts"][owner]["lamports"]-=5000
    current["seal"]=digest(current)
    execution={"status":"PAPER_BANK_EXIT_UNRESOLVED","inventory":current,"slot":2,"observed_at":200.,
        "exit":{"succeeded":False,"error":"InstructionError(1, Custom(6024))","vm_fee_lamports":5000},
        "exit_frame_sha256":frame_id,"input_sha256":digest({**frame,"_paper_request":{"phase":"exit","inventory":prior}}),
        "sell_input_token_units":123,"remaining_token_units":123,"vm_fees_lamports":10000,
        "real_fill":False,"promotion_eligible":False}
    body={"target":target,"input_sha256":frame_id,"paper_control":{"source_frame_sha256":frame_id,
        "previous_canonical_ref":"evidence:prior","execution":execution}}
    if mutation:mutation(body)
    identity=digest(body);body["probe_id"]=identity
    with probe.db:
        probe.db.execute("INSERT INTO results VALUES(?,?,?,?,?,?)",("prior","job",json.dumps({"paper_control":{"execution":{"inventory":prior}}}),100.,"evidence:prior",101.))
        probe.db.execute("INSERT INTO results VALUES(?,?,?,?,?,?)",(identity,"job",json.dumps(body),200.,"evidence:latest",201.))
        probe.db.execute("INSERT INTO paper_bank_inventory VALUES(?,?,?,?,?)",("job",json.dumps(body["paper_control"]["execution"]["inventory"]),"OPEN",identity,260.))
    probe.db.close()
    return identity


def test_archived_exact_request_and_failed_fee_conservation_read_only(tmp_path):
    path=tmp_path/"probe.sqlite";fixture(path)
    before=path.read_bytes();result=audit(path)
    assert result["status"]=="PASS"
    row=result["samples"][0]
    assert row["local_error_code"]==6024
    assert row["retained_token_units"]==123 and row["exit_fee_lamports"]==5000
    assert all(row["checks"].values())
    assert result["protocol_failure_resolved"] is False
    assert path.read_bytes()==before
    output=json.dumps(result)
    assert "InstructionError" not in output and "base64" not in output


@pytest.mark.parametrize("field,category", [
    ("input_sha256","exact_input_binding"),
    ("sell_input_token_units","units_bound"),
    ("vm_fees_lamports","fee_accumulation")])
def test_changed_execution_cannot_pass(field,category,tmp_path):
    def change(row):
        row["paper_control"]["execution"][field]="0"*64 if field=="input_sha256" else 1
    path=tmp_path/"probe.sqlite";fixture(path,change)
    result=audit(path)
    assert result["status"]=="FAIL" and result["failure_categories"][category]==1


def test_missing_frame_and_uncanonical_prior_are_explicit_failures(tmp_path):
    path=tmp_path/"probe.sqlite";fixture(path)
    with sqlite3.connect(path) as db:db.execute("UPDATE results SET ref=NULL WHERE id='prior'")
    result=audit(path)
    assert result["failure_categories"]=={"unique_prior_canonical_inventory":1}
    with sqlite3.connect(path) as db:db.execute("DELETE FROM frames")
    result=audit(path)
    assert result["failure_categories"]=={"diagnostic_complete":1}
    assert "retained fork input" not in json.dumps(result)


def test_bounds_and_missing_database_do_not_create_state(tmp_path):
    path=tmp_path/"absent.sqlite"
    for limit in (0,101,True):
        with pytest.raises(ValueError):audit(path,sample_limit=limit)
    with pytest.raises(sqlite3.OperationalError):audit(path)
    assert not path.exists()


def test_prior_ack_cannot_arrive_after_exit_observation(tmp_path):
    path=tmp_path/"probe.sqlite";fixture(path)
    with sqlite3.connect(path) as db:db.execute("UPDATE results SET available=201 WHERE id='prior'")
    result=audit(path)
    assert result["failure_categories"]=={"prior_canonical_pit":1}


def test_malformed_numeric_diagnostics_do_not_emit_untrusted_text(tmp_path):
    marker="PRIVATE_SENTINEL_MUST_NOT_BE_EXPORTED"
    path=tmp_path/"probe.sqlite"
    fixture(path,lambda row:row["paper_control"]["execution"].update(sell_input_token_units=marker))
    result=audit(path)
    assert result["status"]=="FAIL"
    assert result["samples"][0]["sell_input_token_units"] is None
    assert marker not in json.dumps(result)
