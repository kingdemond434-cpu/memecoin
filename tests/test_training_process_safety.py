"""Explicitly disabled work must never become a real network backfill."""
from unittest.mock import AsyncMock, patch

import pytest

from src.runtime.training import DEFAULT_JOBS, TrainingSupervisor


@pytest.mark.asyncio
async def test_an_explicit_empty_job_list_never_spawns_a_trainer(tmp_path):
    supervisor = TrainingSupervisor(tmp_path / "episodes", tmp_path / "models", jobs=[])
    with patch("src.runtime.training.asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
        assert await supervisor.run_round() == {}
    spawn.assert_not_called()
    assert supervisor.rounds == 1


def test_omitting_jobs_keeps_the_declared_default_jobs(tmp_path):
    supervisor = TrainingSupervisor(tmp_path / "episodes", tmp_path / "models")
    assert [job.name for job in supervisor.jobs] == [job.name for job in DEFAULT_JOBS]


def test_worker_priority_never_changes_the_parent_process():
    from src.runtime.training import TRAINER_NICE, _lower_priority
    with patch("src.runtime.training.os.setpriority", create=True) as setpriority, \
            patch("src.runtime.training.os.PRIO_PROCESS", 0, create=True):
        _lower_priority(12345)
    setpriority.assert_called_once_with(0, 12345, TRAINER_NICE)
