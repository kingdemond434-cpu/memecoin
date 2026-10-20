"""Restored blocked research must not monopolize the running event loop."""
import asyncio
from types import SimpleNamespace

from src.strategies.champion_challenger import ChampionChallengerFramework
from test_scientific_core import spec


def test_unchanged_block_is_noop_but_new_reason_or_stale_authority_persists(monkeypatch):
    framework = ChampionChallengerFramework()
    framework.submit_hypothesis(spec(exact=False))
    writes = []
    monkeypatch.setattr(framework, "_save_state", lambda: writes.append(True))
    framework.mark_data_blocked("c1", "missing native evidence")
    framework.mark_data_blocked("c1", "missing native evidence")
    assert len(writes) == 1
    framework.mark_data_blocked("c1", "different missing input")
    assert len(writes) == 2
    framework.shadow_models["c1"] = {"hypothesis": framework.hypotheses["c1"]}
    framework.mark_data_blocked("c1", "different missing input")
    assert len(writes) == 3 and "c1" not in framework.shadow_models
    framework.champions["c1"] = SimpleNamespace(status="ACTIVE")
    framework.mark_data_blocked("c1", "different missing input")
    assert len(writes) == 4 and framework.champions["c1"].status == "HIBERNATED"
    framework.mark_data_blocked("c1", "different missing input")
    assert len(writes) == 4


def test_restored_blocked_population_yields_and_does_not_rewrite_unchanged_state(monkeypatch):
    async def scenario():
        framework = ChampionChallengerFramework()
        for index in range(20):
            framework.submit_hypothesis(spec("legacy-"+str(index), exact=False))
            framework.mark_data_blocked("legacy-"+str(index), "exact frozen candidate required before promotion")
        writes, inspected, heartbeat = [], [], []
        monkeypatch.setattr(framework, "_save_state", lambda: writes.append(True))
        original = framework._exact_frozen_valid
        def check(hypothesis):
            inspected.append(hypothesis.hypothesis_id)
            return original(hypothesis)
        monkeypatch.setattr(framework, "_exact_frozen_valid", check)
        async def health():
            await asyncio.sleep(0)
            heartbeat.append(len(inspected))
        await asyncio.gather(framework._evaluate_challengers(), health())
        assert len(inspected) == 20 and not writes
        assert 0 < heartbeat[0] < len(inspected)
        assert not framework.shadow_models and not framework.canary_models and not framework.champions
    asyncio.run(scenario())
