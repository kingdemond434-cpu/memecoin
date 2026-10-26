"""A balanced aggregate dispatch must not hide starvation inside fresh lanes."""
import json

from src.research.fork_probe import ForkProbe


def test_every_venue_gets_fresh_service_under_continuous_arrivals_and_restart(tmp_path):
    path = tmp_path / 'probe.sqlite'
    probe = ForkProbe(path)
    venues = ('pump_curve', 'pumpswap', 'meteora_dbc', 'raydium_launchlab')
    served = {lane: [] for lane in ('FRESH_UNTRIED', 'OLDEST_UNTRIED', 'DUE_RETRY')}
    with probe.lock, probe.db:
        for venue in venues:
            probe._enqueue_targets_locked([
                {'token': f'{venue}-backlog-{n}', 'venue': venue} for n in range(20)], 1.)
    for step in range(32):
        if step == 16:
            probe.close()
            probe = ForkProbe(path)
        with probe.lock, probe.db:
            # All venues continuously have newer candidates, old backlog,
            # and due retries. Neither a frozen clock nor ACK backlog is needed.
            probe._enqueue_targets_locked([
                {'token': f'{venue}-fresh-{step}', 'venue': venue} for venue in venues], 1000.+step)
            probe.db.execute('UPDATE jobs SET due=0 WHERE attempts>0')
            row = probe._reserve_job(1000.+step)
            served[row[4]].append(row[3])
            if row[4] == 'FRESH_UNTRIED':
                assert json.loads(row[1])['token'] == f'{row[3]}-fresh-{step}'
    probe.close()
    assert {lane: len(rows) for lane, rows in served.items()} == {
        'FRESH_UNTRIED': 16, 'OLDEST_UNTRIED': 8, 'DUE_RETRY': 8}
    for lane, rows in served.items():
        assert set(rows[:4]) == set(venues), (lane, rows)
        assert set(rows[4:8]) == set(venues), (lane, rows)
