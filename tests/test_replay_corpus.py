import copy
import gzip
import json

import pytest

from src.research.candidate_replay import CandidateReplayHandler, ReplayBlocked, _digest
from src.research.replay_corpus import audit_corpus, freeze_corpus, load_sealed_corpus
from src.research.gauntlet import LATENCY_GRID_S
from test_candidate_replay import episode


WINDOWS = {"train": [1, 950], "validation": [1000, 1950], "test": [2000, 2950]}


def test_audit_recognizes_decoded_pump_program_but_retains_all_execution_gaps(tmp_path):
    from src.chains.idl import PUMP_IDL, program_id
    def native_labels(rows):
        for row in rows:
            row.update(chain="Solana Mainnet", factory=program_id(PUMP_IDL))
    root, rows = corpus(tmp_path, native_labels)
    assert audit_corpus(root)["replay_input_complete"] == 3
    path = next(root.glob("*/*.json.gz"))
    original = path.read_bytes()
    assert freeze_corpus(root, tmp_path / "frozen-native", windows=WINDOWS, now=3000)["status"] == "SEALED_NOT_EVALUATED"
    assert path.read_bytes() == original
    from src.research.replay_corpus import replay_gaps
    row = copy.deepcopy(rows[0]);row.pop("replay_execution")
    gaps = replay_gaps(row)
    assert "native_replay_protocol_mapping_missing" not in gaps
    assert "native_route_depth_fill_cost_missing" in gaps


def corpus(tmp_path, mutate=None):
    root = tmp_path / "source"
    rows = [episode(index) for index in (0, 10, 20)]
    for row in rows:
        cells = row["replay_execution"]["fills_by_latency"]
        for delay in LATENCY_GRID_S:
            cells[str(float(delay))] = copy.deepcopy(cells["1.0"])
    if mutate:
        mutate(rows)
    for index, row in enumerate(rows):
        path = root / "day" / f"{index}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(row).encode()))
    return root, rows


def test_seal_preserves_full_declared_cohort_and_drives_actual_handler(tmp_path):
    root, rows = corpus(tmp_path)
    before = {p: p.read_bytes() for p in root.glob("*/*")}
    snapshot = tmp_path / "sealed"
    manifest = freeze_corpus(root, snapshot, windows=WINDOWS, now=3000)
    assert manifest["status"] == "SEALED_NOT_EVALUATED"
    assert manifest["corpus_hash"] == _digest(rows)
    assert manifest["partition_counts"] == {"train": 1, "validation": 1, "test": 1}
    handler = CandidateReplayHandler(None, snapshot, chronological_split=manifest)
    assert handler._episodes() == rows
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(FileExistsError):
        freeze_corpus(root, snapshot, windows=WINDOWS, now=3000)


@pytest.mark.parametrize("fault", ["cost", "unresolved", "boundary", "duplicate"])
def test_difficult_launch_is_a_block_never_silently_removed(tmp_path, fault):
    def mutate(rows):
        if fault == "cost":
            rows[1].pop("replay_execution")
        elif fault == "unresolved":
            rows[1]["final_outcome"]["status"] = "DATA_BLOCKED"
        elif fault == "boundary":
            rows[1]["market_observations"][-1]["timestamp"] = 1951
        else:
            rows.append(copy.deepcopy(rows[1]))
    root, _ = corpus(tmp_path, mutate)
    manifest = freeze_corpus(root, tmp_path / "blocked", windows=WINDOWS, now=3000)
    assert manifest["status"] == "DATA_BLOCKED"
    assert manifest["blocked"] and "corpus_hash" not in manifest
    assert not list((tmp_path / "blocked").glob("*/*.gz"))


def test_mutation_and_unlisted_launch_cannot_enter_frozen_corpus(tmp_path):
    root, _ = corpus(tmp_path)
    snapshot = tmp_path / "sealed"
    manifest = freeze_corpus(root, snapshot, windows=WINDOWS, now=3000)
    path = snapshot / manifest["files"][0]["path"]
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ReplayBlocked, match="bytes changed"):
        load_sealed_corpus(snapshot, manifest, max_episodes=10)
    path.write_bytes(original)
    (snapshot / "episodes" / "extra.json.gz").write_bytes(original)
    with pytest.raises(ReplayBlocked, match="unlisted"):
        load_sealed_corpus(snapshot, manifest, max_episodes=10)


def test_windows_must_be_disjoint_past_and_snapshot_cannot_mutate_source(tmp_path):
    root, _ = corpus(tmp_path)
    for windows in ({**WINDOWS, "validation": [900, 1950]}, {**WINDOWS, "test": [2000, 3001]}):
        with pytest.raises(ValueError):
            freeze_corpus(root, tmp_path / "invalid", windows=windows, now=3000)
    with pytest.raises(ValueError):
        freeze_corpus(root, root / "snapshot", windows=WINDOWS, now=3000)


def test_audit_reports_missing_costs_without_exposing_or_rewriting_data(tmp_path):
    root, _ = corpus(tmp_path, lambda rows: rows[0].pop("replay_execution"))
    report = audit_corpus(root)
    assert report["status"] == "DATA_BLOCKED"
    assert report["episode_files"] == 3 and report["replay_input_complete"] == 2
    assert report["gaps"]["native_route_depth_fill_cost_missing"] == 1
    assert audit_corpus(root, max_episodes=2)["status"] == "DATA_BLOCKED"
