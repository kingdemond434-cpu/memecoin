"""The permanent tape must stay bounded per file without losing one record.

On 2026-11-10 `program-events.sqlite` was a single 98 GB file on a disk shared
with other desks. These tests pin the only acceptable remedy: seal, continue
the SAME hash chain, archive losslessly, and remove a SQLite segment only
after its archive has been read back and verified.
"""
import asyncio
import gzip
import json
import sqlite3
from collections import namedtuple

import pytest

from src.research import program_tape_segments as segments
from src.research.program_tape import ProgramEventTape

Usage = namedtuple("Usage", "total used free")


def _fill(path, count, *, segment_bytes=None, start=0):
    async def run():
        tape = ProgramEventTape(path, batch_size=4, batch_delay=0, segment_bytes=segment_bytes)
        for index in range(start, start + count):
            await tape.enqueue("program_data", {
                "program": "P", "signature": f"s{index}", "log_index": 0, "data": "x" * 400})
        await tape.flush()
        report = tape.report()
        await tape.stop()
        return report
    return asyncio.run(run())


def _bodies(path):
    db = sqlite3.connect(path)
    try:
        return [row[0] for row in db.execute("SELECT body FROM records ORDER BY sequence")]
    finally:
        db.close()


def test_without_a_threshold_nothing_changes(tmp_path):
    path = tmp_path / "program-events.sqlite"
    report = _fill(path, 40)
    assert report["committed_records"] == 40
    assert segments.list_segments(path) == []
    assert ProgramEventTape.verify(path)["records"] == 40


def test_rotation_continues_one_chain_across_files(tmp_path):
    path = tmp_path / "program-events.sqlite"
    report = _fill(path, 200, segment_bytes=1)       # seal after every commit
    closed = segments.list_segments(path)
    assert report["committed_records"] == 200
    assert report["segments"]["rotations_this_run"] == len(closed) > 1
    assert closed[0]["first"] == 1
    for earlier, later in zip(closed, closed[1:]):
        assert later["first"] == earlier["last"] + 1
    result = ProgramEventTape.verify(path)
    assert result["status"] == "PASS" and result["records"] == 200
    # Cumulative counters survive every seal.
    assert report["committed_by_kind"] == {"program_data": 200}


def test_a_reopened_tape_keeps_appending_to_the_same_chain(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 30, segment_bytes=1)
    report = _fill(path, 30, segment_bytes=1, start=30)
    assert report["committed_records"] == 60
    assert ProgramEventTape.verify(path)["records"] == 60


def test_a_crash_between_seal_and_successor_cannot_restart_at_genesis(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 20)
    # The state a crash leaves: the file renamed, no successor created yet.
    path.rename(segments.sealed_path(path, 1, 20))
    report = _fill(path, 5, start=20)
    assert report["committed_records"] == 25
    assert ProgramEventTape.verify(path)["records"] == 25


def test_the_archive_is_the_exact_rows_and_the_source_goes_only_after(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 50)
    sealed = segments.sealed_path(path, 1, 50)
    path.rename(sealed)
    original = _bodies(sealed)
    manifest = segments.archive_segment(sealed)
    archive, manifest_path = segments.archive_paths(sealed)
    assert not sealed.exists() and manifest_path.exists()
    with gzip.open(archive, "rt", encoding="utf-8", newline="\n") as handle:
        assert [line[:-1] for line in handle] == original
    assert manifest["records"] == 50 and manifest["first_sequence"] == 1
    assert manifest["counts"] == {"program_data": 50}
    assert ProgramEventTape.verify(path)["archived_segments"] == 1


def test_a_corrupt_segment_is_never_archived_or_removed(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 10)
    sealed = segments.sealed_path(path, 1, 10)
    path.rename(sealed)
    db = sqlite3.connect(sealed)
    body = json.loads(db.execute("SELECT body FROM records WHERE sequence=5").fetchone()[0])
    body["payload"]["data"] = "tampered"
    db.execute("UPDATE records SET body=? WHERE sequence=5",
               (json.dumps(body, sort_keys=True, separators=(",", ":")),))
    db.commit(); db.close()
    with pytest.raises(ValueError):
        segments.archive_segment(sealed)
    archive, manifest_path = segments.archive_paths(sealed)
    assert sealed.exists() and not archive.exists() and not manifest_path.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_tight_disk_defers_and_leaves_the_source(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 10)
    sealed = segments.sealed_path(path, 1, 10)
    path.rename(sealed)
    result = segments.archive_segment(sealed, usage=lambda _: Usage(1, 1, 0))
    assert result["deferred"] == "LOW_DISK"
    assert sealed.exists() and not list(tmp_path.glob("*.gz*"))


def test_disk_running_out_mid_archive_discards_the_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(segments, "_CHECK_EVERY", 4)
    path = tmp_path / "program-events.sqlite"
    _fill(path, 40)
    sealed = segments.sealed_path(path, 1, 40)
    path.rename(sealed)
    readings = iter([10 * 1024**3, 10 * 1024**3, 0])
    result = segments.archive_segment(sealed, usage=lambda _: Usage(1, 1, next(readings, 0)))
    assert result["deferred"] == "LOW_DISK"
    assert sealed.exists() and not list(tmp_path.glob("*.gz*"))


def test_a_stop_request_abandons_the_archive_and_keeps_the_source(tmp_path, monkeypatch):
    monkeypatch.setattr(segments, "_CHECK_EVERY", 4)
    path = tmp_path / "program-events.sqlite"
    _fill(path, 40)
    sealed = segments.sealed_path(path, 1, 40)
    path.rename(sealed)
    assert segments.archive_segment(sealed, stop=lambda: True)["deferred"] == "STOPPED"
    assert sealed.exists() and not list(tmp_path.glob("*.gz*"))


def test_an_interrupted_removal_is_finished_not_redone(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 10)
    sealed = segments.sealed_path(path, 1, 10)
    path.rename(sealed)
    keep = sealed.read_bytes()
    segments.archive_segment(sealed)
    sealed.write_bytes(keep)                 # as if the crash came before removal
    archive, _ = segments.archive_paths(sealed)
    before = archive.read_bytes()
    segments.archive_segment(sealed)
    assert not sealed.exists() and archive.read_bytes() == before


def test_a_damaged_archive_fails_whole_tape_verification(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 10)
    sealed = segments.sealed_path(path, 1, 10)
    path.rename(sealed)
    segments.archive_segment(sealed)
    archive, _ = segments.archive_paths(sealed)
    data = bytearray(archive.read_bytes()); data[len(data) // 2] ^= 1
    archive.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        ProgramEventTape.verify(path)


def test_compaction_archives_every_sealed_file_and_the_tape_still_verifies(tmp_path):
    path = tmp_path / "program-events.sqlite"

    async def run():
        tape = ProgramEventTape(path, batch_size=4, batch_delay=0, segment_bytes=1)
        for index in range(60):
            await tape.enqueue("transaction", {"program": "P", "n": index})
        await tape.flush()
        await asyncio.to_thread(tape.compact, duty=1.0)
        report = tape.report()
        await tape.stop()
        return report

    report = asyncio.run(run())
    assert report["segments"]["sealed"] == 0
    assert report["segments"]["archived"] == report["segments"]["archived_this_run"] > 1
    assert not list(tmp_path.glob("*.sealed.sqlite"))
    assert ProgramEventTape.verify(path)["records"] == 60


def test_a_reopen_failure_after_sealing_never_replays_the_committed_batch(tmp_path, monkeypatch):
    """Rotation runs after the commit; a raise there would duplicate the batch."""
    path = tmp_path / "program-events.sqlite"

    async def run():
        tape = ProgramEventTape(path, batch_size=4, batch_delay=0, segment_bytes=1)
        real, calls = tape._open, {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("reopen failed")
            return real()

        monkeypatch.setattr(tape, "_open", flaky)
        for index in range(12):
            await tape.enqueue("transaction", {"program": "P", "n": index})
            await tape.flush()
        report = tape.report()
        await tape.stop()
        return report, calls["n"]

    report, calls = asyncio.run(run())
    assert calls >= 2 and report["committed_records"] == 12
    assert report["write_errors_this_run"] == 0
    assert ProgramEventTape.verify(path)["records"] == 12


def test_stop_survives_a_tape_left_closed_by_a_failed_rotation(tmp_path, monkeypatch):
    path = tmp_path / "program-events.sqlite"

    async def run():
        tape = ProgramEventTape(path, batch_size=4, batch_delay=0, segment_bytes=1)
        monkeypatch.setattr(tape, "_open", lambda: (_ for _ in ()).throw(RuntimeError("no")))
        await tape.enqueue("transaction", {"program": "P", "n": 0})
        await tape.queue.join()
        assert tape.db is None
        await tape.stop()

    asyncio.run(run())


def test_verification_and_hashing_also_honour_a_stop_request(tmp_path, monkeypatch):
    monkeypatch.setattr(segments, "_CHECK_EVERY", 4)
    path = tmp_path / "program-events.sqlite"
    _fill(path, 40)
    sealed = segments.sealed_path(path, 1, 40)
    path.rename(sealed)
    calls = {"n": 0}

    def stop():                       # true only once the write pass is over
        calls["n"] += 1
        return calls["n"] > 10

    assert segments.archive_segment(sealed, stop=stop)["deferred"] == "STOPPED"
    assert sealed.exists() and not list(tmp_path.glob("*.gz*"))
    assert not list(tmp_path.glob("*.manifest.json"))


def test_a_manifest_for_a_different_file_never_authorises_removal(tmp_path):
    path = tmp_path / "program-events.sqlite"
    _fill(path, 10)
    sealed = segments.sealed_path(path, 1, 10)
    path.rename(sealed)
    keep = sealed.read_bytes()
    segments.archive_segment(sealed)
    _, manifest_path = segments.archive_paths(sealed)
    manifest = json.loads(manifest_path.read_text())
    manifest["last_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    sealed.write_bytes(keep)
    rebuilt = segments.archive_segment(sealed)     # rebuilt from the source instead
    assert rebuilt["last_hash"] != "f" * 64 and not sealed.exists()
    assert ProgramEventTape.verify(path)["records"] == 10


def test_low_disk_is_not_retried_until_more_space_is_free(tmp_path):
    path = tmp_path / "program-events.sqlite"
    free = {"bytes": 0}
    usage = lambda _: Usage(1, 1, free["bytes"])

    async def run():
        tape = ProgramEventTape(path, batch_size=4, batch_delay=0, segment_bytes=1)
        for index in range(8):
            await tape.enqueue("transaction", {"program": "P", "n": index})
        await tape.flush()
        tape.compact(duty=1.0, usage=usage)
        first = tape.report()["segments"]
        free["bytes"] = 1024**3 - 1                 # still not more than before + 1 GiB
        tape.compact(duty=1.0, usage=usage)
        blocked = tape.report()["segments"]
        free["bytes"] = 50 * 1024**3
        tape.compact(duty=1.0, usage=usage)
        done = tape.report()["segments"]
        await tape.stop()
        return first, blocked, done

    first, blocked, done = asyncio.run(run())
    assert first["archive_deferred"] == "LOW_DISK" and first["archive_needs_free_bytes"] == 1024**3
    assert blocked["archive_deferred"] == "LOW_DISK" and blocked["archived"] == 0
    assert done["archive_deferred"] is None and done["sealed"] == 0 and done["archived"] > 1


def test_a_crash_after_a_failed_rotation_reopen_is_reported_unclean(tmp_path, monkeypatch):
    path = tmp_path / "program-events.sqlite"

    async def run():
        tape = ProgramEventTape(path, batch_size=4, batch_delay=0, segment_bytes=1)
        real, calls = tape._open, {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("reopen failed")
            return real()

        monkeypatch.setattr(tape, "_open", flaky)
        await tape.enqueue("transaction", {"program": "P", "n": 0})
        await tape.flush()                       # rotation: seal, reopen fails
        tape.segment_bytes = None                # keep the next file active
        await tape.enqueue("transaction", {"program": "P", "n": 1})
        await tape.flush()                       # append reopens, then writes
        tape.worker.cancel()
        await asyncio.gather(tape.worker, return_exceptions=True)
        tape.db.close()                          # crash: no clean stop()

    asyncio.run(run())

    async def reopen():
        tape = ProgramEventTape(path)
        report = tape.report()
        await tape.stop()
        return report

    report = asyncio.run(reopen())
    assert report["previous_shutdown_clean"] is False
    assert report["uncommitted_loss_after_unclean_shutdown"] == "UNKNOWN"
    assert report["committed_records"] == 2
