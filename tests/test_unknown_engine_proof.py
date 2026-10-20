import json
from pathlib import Path

import pytest

from tools.prove_unknown_engine import (
    REQUIREMENTS, file_digest, isolated_runtime, main, run_proof,
    scenario_authority, scenario_route, scenario_safety,
)
from src.research.trade_evidence import TradeEvidenceLedger


def test_every_requirement_gets_a_verified_honest_artifact_in_one_output_tree(tmp_path, monkeypatch):
    # This suite avoids recursively launching pytest. A08 must say BLOCKED.
    previous_cwd = Path.cwd()
    manifest = run_proof(tmp_path / "proof", run_pytest=False)
    assert Path.cwd() == previous_cwd
    assert len(manifest["requirements"]) == len(REQUIREMENTS) == 15
    assert {row["requirement"] for row in manifest["requirements"]} == {f"A{i:02d}" for i in range(1, 16)}
    assert manifest["counts"]["FAIL"] == 0
    assert manifest["status"] == "BLOCKED"
    assert manifest["production_status"] == manifest["provider_status"] == manifest["forward_status"] == "UNMEASURED"
    assert manifest["live_capability"] is False
    output = Path(manifest["output_dir"])
    assert TradeEvidenceLedger.verify(output / "acceptance-evidence.jsonl")[0]
    for row in manifest["requirements"]:
        result_path = (output / row["artifact"]).resolve()
        assert result_path.is_relative_to(output)
        result = json.loads(result_path.read_text())
        assert result["canonical_evidence_ref"].startswith("acceptance-evidence.jsonl#sequence=")
        assert len(result["input_sha256"]) == len(result["result_sha256"]) == 64
        assert result["production_evidence"] == "UNMEASURED"
        if result["status"] == "BLOCKED":
            assert result["gap"]
    for name, record in manifest["files"].items():
        target = output / name
        assert target.resolve().is_relative_to(output)
        assert file_digest(target) == record["sha256"]
    assert not (output / "data").exists()


def test_missing_integration_is_blocked_instead_of_passed(tmp_path):
    from tools.prove_unknown_engine import MissingProof
    with pytest.raises(MissingProof, match="explicitly skipped"):
        scenario_safety(tmp_path, run_pytest=False)


def test_actual_buy_sell_paths_refuse_operator_absence_even_with_mock_earned_stage(tmp_path):
    with isolated_runtime(tmp_path):
        result = scenario_authority(tmp_path)
    assert len(result["attempts"]) == 8
    assert result["submissions"] == result["real_signatures"] == 0
    assert {item["side"] for item in result["attempts"]} == {"buy", "sell"}
    assert all(item["simulated"] or "acknowledgement absent" in item["error"]
               for item in result["attempts"])


def test_executable_counterfactual_persists_both_actual_return_and_mark_peak(tmp_path):
    result = scenario_route(tmp_path)
    assert result["executable_cell"]["net_sol"] < result["ideal_mark_return_sol"]
    row = json.loads((tmp_path / "counterfactuals.jsonl").read_text().splitlines()[0])
    assert row["resolution"]["realised_multiple"] < row["resolution"]["peak_multiple"]


def test_existing_proof_directory_is_never_silently_overwritten(tmp_path):
    output = tmp_path / "proof"
    output.mkdir()
    (output / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_proof(output, run_pytest=False)


def test_cli_propagates_blocked_and_failed_statuses(monkeypatch, tmp_path):
    from tools import prove_unknown_engine
    def answer(status):
        return {"output_dir": str(tmp_path), "status": status, "counts": {}, "production_status": "UNMEASURED"}
    for status, expected in (("PASS", 0), ("FAIL", 1), ("BLOCKED", 4)):
        monkeypatch.setattr(prove_unknown_engine, "run_proof", lambda *args, **kwargs: answer(status))
        assert main(["--output-dir", str(tmp_path)]) == expected
