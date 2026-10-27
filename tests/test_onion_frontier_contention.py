import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import sqlite3
import threading

import pytest

from src.collectors.onion_frontier import OnionFrontierTransport
from tests.test_onion_collection import host

OLD = """SELECT u.* FROM urls u JOIN hosts h ON h.host=u.host
WHERE u.next_at<=? AND h.next_at<=?
ORDER BY h.last_attempt,((u.observed_at IS NULL)=(h.turn % 2)) DESC,
u.next_at,u.discovered_at,u.url LIMIT 1"""


def seed(frontier, hosts=30, pages=8, *, randomize=True):
    rng = random.Random(417)
    frontier.enqueue([f"http://{host(n)}/{i}" for n in range(1, hosts+1)
                      for i in range(pages)], "work-public-parent", 1.)
    with frontier.db:
        for n in range(1, hosts+1):
            frontier.db.execute("UPDATE hosts SET last_attempt=?,next_at=?,turn=? WHERE host=?",
                (rng.randrange(5) if randomize else n, rng.choice([0, 15, 30]) if randomize else 0,
                 rng.randrange(2), host(n)))
            for i in range(pages):
                frontier.db.execute("UPDATE urls SET next_at=?,discovered_at=?,observed_at=? WHERE url=?",
                    (rng.choice([0, 10, 30]) if randomize else 0, rng.randrange(10),
                     rng.choice([None, 1.]), f"http://{host(n)}/{i}"))


def test_host_first_selection_matches_original_total_order_with_ties_and_parity(tmp_path):
    value = OnionFrontierTransport(tmp_path / "frontier.sqlite", workers=1)
    try:
        seed(value)
        for now in (10., 15., 30., 100., 200.):
            for _ in range(30):
                expected = value.db.execute(OLD, (now, now)).fetchone()
                pending, actual = value._lease(now)
                assert not pending
                assert [row["url"] for row in actual] == ([expected["url"]] if expected else [])
    finally:
        asyncio.run(value.stop())


def test_lease_skips_later_hosts_urls_without_scanning_entire_population(tmp_path):
    value = OnionFrontierTransport(tmp_path / "frontier.sqlite")
    try:
        seed(value, hosts=200, pages=20, randomize=False)
        value.db.execute("DROP INDEX onion_host_turn")
        counts = [0]
        def tick():
            counts[0] += 1
            return 0
        value.db.set_progress_handler(tick, 1)
        expected = value.db.execute(OLD, (10., 10.)).fetchone()
        old_steps = counts[0]
        value.db.set_progress_handler(None, 0)
        value.db.execute("CREATE INDEX onion_host_turn ON hosts(last_attempt,host,next_at)")
        counts[0] = 0
        value.db.set_progress_handler(tick, 1)
        _, rows = value._lease(10.)
        new_steps = counts[0]
        value.db.set_progress_handler(None, 0)
        assert rows[0]["url"] == expected["url"]
        # Includes the real new lease's BEGIN/UPDATE/COMMIT, not just its SELECT.
        assert new_steps < old_steps / 2
    finally:
        asyncio.run(value.stop())


def test_report_scan_does_not_block_delivery_or_child_admission(tmp_path, monkeypatch):
    value = OnionFrontierTransport(tmp_path / "frontier.sqlite")
    value.enqueue([f"http://{host(1)}/"], "work-parent", 1.)
    entered, release = threading.Event(), threading.Event()
    connect = sqlite3.connect
    def reader(*args, **kwargs):
        db = connect(*args, **kwargs)
        def trace(sql):
            if sql == "SELECT count(*) FROM outbox":
                entered.set()
                assert release.wait(10)
        db.set_trace_callback(trace)
        return db
    monkeypatch.setattr("src.collectors.onion_frontier.sqlite3.connect", reader)
    with ThreadPoolExecutor(max_workers=2) as pool:
        report = pool.submit(value.report)
        try:
            assert entered.wait(5)
            pool.submit(value.enqueue, [f"http://{host(2)}/"], "work-child", 2.).result(timeout=5)
        finally:
            release.set()
        snapshot = report.result(timeout=5)
    assert snapshot["pages"] == snapshot["discovered_hosts"] == 1
    assert value.report()["pages"] == 2
    asyncio.run(value.stop())
    with pytest.raises(sqlite3.ProgrammingError):
        value.report()
