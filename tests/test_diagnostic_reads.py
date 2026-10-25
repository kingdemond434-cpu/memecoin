import asyncio
import json

from src.runtime.diagnostic_reads import diagnostic_reads, diagnostic_value
from src.runtime.training import TrainingSupervisor
from src.research.promotion_gate import PromotionLedger
from src.research.population_hunter import PopulationHunter


def test_report_scope_reuses_reads_but_live_training_checks_remain_fresh(tmp_path):
    training=object.__new__(TrainingSupervisor);training.storage=tmp_path
    (tmp_path/'one.json').write_text('{}')
    with diagnostic_reads():
        assert training.resolved_episodes()==1
        (tmp_path/'two.json').write_text('{}')
        assert training.resolved_episodes()==1
    assert training.resolved_episodes()==2
    (tmp_path/'three.json.gz').write_bytes(b'fixture')
    assert training.resolved_episodes()==3


def test_promotion_history_is_shared_only_inside_one_report(tmp_path):
    ledger=PromotionLedger(tmp_path/'promotion.jsonl')
    ledger.path.write_text(json.dumps({'n':1})+'\n')
    with diagnostic_reads():
        assert ledger.history()==[{'n':1}]
        ledger.path.write_text(json.dumps({'n':1})+'\n'+json.dumps({'n':2})+'\n')
        assert ledger.history()==[{'n':1}]
    assert ledger.history()==[{'n':1},{'n':2}]


def test_parallel_diagnostic_scopes_do_not_share_values():
    async def run():
        async def report(value):
            with diagnostic_reads():
                first=diagnostic_value('same-key',lambda:value)
                await asyncio.sleep(0)
                assert diagnostic_value('same-key',lambda:999)==first==value
        await asyncio.gather(report(1),report(2))
    asyncio.run(run())


def test_inherited_child_tasks_and_workers_do_not_reuse_report_values():
    async def run():
        with diagnostic_reads():
            assert diagnostic_value('same-key',lambda:1)==1
            async def child():return diagnostic_value('same-key',lambda:2)
            assert await asyncio.create_task(child())==2
            assert await asyncio.to_thread(diagnostic_value,'same-key',lambda:3)==3
            assert diagnostic_value('same-key',lambda:999)==1
    asyncio.run(run())


def test_grouped_narrative_report_retains_point_in_time_and_repost_semantics():
    rows=[]
    for n in range(6):
        rows.append({'id':str(n),'narrative_id':'target' if n<4 else 'other','observed_at':n,
            'event_at':n-1,'content_hash':'shared' if n<2 else str(n),'operator_id':str(n),
            'source_id':str(n),'platform':'telegram','region':'global','language':'en',
            'stage':'RUMOR','label':'CLAIM'})
    state={'narratives':{row['id']:row for row in rows}}
    baseline=PopulationHunter._narrative_report(state,'target',1)
    grouped=PopulationHunter._narrative_report(state,'target',1,observations=rows[:4])
    assert grouped==baseline
    assert grouped['observations']==2 and grouped['reposted_content']==1
    assert grouped['independent_confirmation'] is False
