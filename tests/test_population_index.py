"""Migration, crash boundaries and bounded writes, never performance evidence."""
import sqlite3
from copy import deepcopy

import pytest

from src.research.population_hunter import COMPONENT, PopulationHunter
from src.research.population_index import BUCKETS, STORAGE, PopulationIndex
from src.strategies.champion_challenger import ChampionChallengerFramework


def empty():
    return {"schema": 1, **{name: {} for name in BUCKETS}}


def mention(number=0):
    return dict(narrative_id="public-claim", source_id="public-test", platform="RSS",
                language="en", event_at=10., observed_at=11., text="public unverified launch",
                observation_id=f"mention-{number}")


def test_migration_preserves_every_bucket_and_is_retryable_before_descriptor(tmp_path):
    legacy = empty()
    for bucket in BUCKETS:
        legacy[bucket]["retained-id"] = 11. if bucket == "receipt_observed_at" else {"evidence_refs": ["fixture:original"], "scope": "PAPER"}
    path = tmp_path / "population.sqlite"
    index = PopulationIndex(path)
    index.initialize(legacy)
    index.close()  # crash after migration, before framework descriptor save
    restarted = PopulationIndex(path)
    restarted.initialize(deepcopy(legacy))
    assert restarted.load() == legacy
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == len(BUCKETS)
    restarted.close()


def test_row_conflict_rolls_back_receipt_and_other_new_rows(tmp_path):
    index = PopulationIndex(tmp_path / "population.sqlite")
    index.initialize(empty())
    index.apply({"actors": {"original": {"evidence_refs": ["fixture:a"]}}})
    with pytest.raises(ValueError, match="immutable"):
        index.apply({"receipts": {"new": {"disposition": "OBSERVED"}},
                     "actors": {"new": {}, "original": {"evidence_refs": ["fixture:changed"]}}})
    assert index.load()["receipts"] == {}
    assert set(index.load()["actors"]) == {"original"}
    index.close()


def test_observation_commits_before_return_without_rewriting_framework(tmp_path):
    path = tmp_path / "champion.json"
    framework = ChampionChallengerFramework(state_path=str(path))
    hunter = PopulationHunter(framework)
    before = path.read_bytes()
    assert framework.get_research_component(COMPONENT)["storage"] == STORAGE
    result = hunter.consume("NARRATIVE_OBSERVATION", mention(), ["fixture:canonical"], 11.)
    assert result["disposition"] == "OBSERVED"
    assert path.read_bytes() == before
    assert len(before) < 10000
    committed = PopulationIndex(hunter.index.path, required=True)
    assert len(committed.load()["receipts"]) == len(committed.load()["narratives"]) == 1
    committed.close()
    hunter.close()
    restarted = PopulationHunter(ChampionChallengerFramework(state_path=str(path)))
    assert restarted.consume("NARRATIVE_OBSERVATION", mention(), ["fixture:canonical"], 50.) == result
    assert restarted.report()["receipt_count"] == 1
    restarted.close()


def test_storage_failure_cannot_publish_uncommitted_receipt(tmp_path, monkeypatch):
    hunter = PopulationHunter(ChampionChallengerFramework(state_path=str(tmp_path / "champion.json")))
    before = deepcopy(hunter.state)
    def failed(changes):
        raise sqlite3.OperationalError("simulated storage failure")
    monkeypatch.setattr(hunter.index, "apply", failed)
    with pytest.raises(sqlite3.OperationalError, match="storage failure"):
        hunter.consume("NARRATIVE_OBSERVATION", mention(), ["fixture:canonical"], 11.)
    assert hunter.state == before
    assert hunter.index.load() == before
    hunter.close()


@pytest.mark.parametrize("damage", ["missing", "hash", "path", "schema"])
def test_referenced_index_damage_never_restarts_empty(tmp_path, damage):
    path = tmp_path / "champion.json"
    framework = ChampionChallengerFramework(state_path=str(path))
    hunter = PopulationHunter(framework)
    hunter.consume("NARRATIVE_OBSERVATION", mention(), ["fixture:canonical"], 11.)
    index_path = hunter.index.path
    hunter.close()
    if damage == "missing":
        index_path.unlink()
    elif damage in {"hash", "schema"}:
        with sqlite3.connect(index_path) as db:
            if damage == "hash":
                db.execute("UPDATE records SET payload='{}' WHERE bucket='narratives'")
            else:
                db.execute("UPDATE metadata SET value='999' WHERE key='schema'")
    else:
        descriptor = framework.get_research_component(COMPONENT)
        descriptor["file"] = "elsewhere.sqlite"
        framework.checkpoint_research_component(COMPONENT, descriptor)
    with pytest.raises(RuntimeError):
        PopulationHunter(ChampionChallengerFramework(state_path=str(path)))


def test_real_legacy_framework_migrates_without_changing_receipts_or_report(tmp_path):
    legacy_framework = ChampionChallengerFramework()
    legacy_hunter = PopulationHunter(legacy_framework)
    original = legacy_hunter.consume("NARRATIVE_OBSERVATION", mention(), ["fixture:canonical"], 11.)
    framework = ChampionChallengerFramework(state_path=str(tmp_path / "champion.json"))
    framework.checkpoint_research_component(COMPONENT, legacy_hunter.state)
    migrated = PopulationHunter(framework)
    assert migrated.state == legacy_hunter.state
    assert migrated.report() == legacy_hunter.report()
    assert migrated.consume("NARRATIVE_OBSERVATION", mention(), ["fixture:canonical"], 99.) == original
    assert framework.get_research_component(COMPONENT) == migrated.index.descriptor
    migrated.close()
