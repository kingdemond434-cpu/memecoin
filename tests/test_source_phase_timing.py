import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from src.runtime.source_timing import SourceTiming
from tests.test_population_callbacks import desk
from tests.test_source_intake_nonblocking import event


def test_interrupted_attempts_snapshot_and_fixed_cardinality():
    timing = SourceTiming()
    with patch("src.runtime.source_timing.time.perf_counter", side_effect=[1, 3, 5, 8]):
        with timing.measure("canonical"):
            pass
        with pytest.raises(asyncio.CancelledError):
            with timing.measure("canonical"):
                raise asyncio.CancelledError()
    row = timing.report()["phases"]["canonical"]
    assert row == dict(attempts=2, completed=1, interrupted=1,
                      total_seconds=5, max_seconds=3, last_seconds=3)
    row["attempts"] = 999
    assert timing.report()["phases"]["canonical"]["attempts"] == 2
    with pytest.raises(ValueError):
        with timing.measure("private-source-identity"):
            pass
    assert len(timing.report()["phases"]) == 1


def test_worker_updates_are_not_lost():
    timing = SourceTiming()
    def write(_):
        with timing.measure("neural_intake"):
            pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(200)))
    row = timing.report()["phases"]["neural_intake"]
    assert row["attempts"] == row["completed"] == 200
    assert row["interrupted"] == 0


def test_real_source_intake_records_phases_without_changing_job(tmp_path):
    async def scenario():
        value = desk(tmp_path)
        value.global_config = {"ops_state_dir": str(tmp_path)}
        try:
            job = await value._record_population_narrative_async(event())
            assert job
            phases = value._source_phase_timing.report()["phases"]
            assert set(phases) == {"canonical", "neural_intake", "claim_intake", "owner_hook"}
            assert all(row["attempts"] == row["completed"] == 1 for row in phases.values())
        finally:
            value.public_claim_links.close()
            value.unknown_engine.frontier.close()
    asyncio.run(scenario())
