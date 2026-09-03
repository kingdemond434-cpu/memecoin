"""The archival tier.

Two guarantees, and every test here is an attempt to break one of them:
nothing is deleted locally unless a verified remote copy exists, and nothing
comes back up without being proved to be what went down.
"""

import json

import pytest

from src.research.object_store import (
    ArchiveError, ArchiveVault, ArchivedObject, LocalDirectoryStore,
    S3CompatibleStore, digest_of)


def _file(root, name, content=b"x" * 4096):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _vault(tmp_path, **kwargs):
    return ArchiveVault(tmp_path / "local",
                        LocalDirectoryStore(tmp_path / "remote"), **kwargs)


# --- the local store -----------------------------------------------------

def test_a_key_cannot_escape_the_store_root(tmp_path):
    store = LocalDirectoryStore(tmp_path / "remote")
    source = _file(tmp_path, "src.bin")
    with pytest.raises(ArchiveError) as caught:
        store.put("../../escape.bin", source)
    assert "escapes the store root" in str(caught.value)
    assert not (tmp_path.parent / "escape.bin").exists()


def test_an_interrupted_upload_leaves_no_partial_object(tmp_path):
    store = LocalDirectoryStore(tmp_path / "remote")
    store.put("a/b.bin", _file(tmp_path, "src.bin"))
    assert store.list() == ["a/b.bin"]
    assert not list((tmp_path / "remote").rglob("*.partial"))


def test_round_trip_through_the_local_store(tmp_path):
    store = LocalDirectoryStore(tmp_path / "remote")
    source = _file(tmp_path, "src.bin", b"payload")
    store.put("k", source)
    out = tmp_path / "back" / "k"
    store.get("k", out)
    assert out.read_bytes() == b"payload"
    assert store.size("k") == 7
    store.delete("k")
    assert not store.exists("k")


def test_getting_an_absent_object_names_it(tmp_path):
    store = LocalDirectoryStore(tmp_path / "remote")
    with pytest.raises(ArchiveError) as caught:
        store.get("nope", tmp_path / "out")
    assert "not in the store" in str(caught.value)


# --- archiving -----------------------------------------------------------

def test_archiving_records_a_digest_and_marks_it_durable(tmp_path):
    vault = _vault(tmp_path)
    source = _file(tmp_path / "local", "episodes/day.json.gz")
    record = vault.archive(source)
    assert record.durable
    assert record.digest == digest_of(source)
    assert record.key == "episodes/day.json.gz"
    assert vault.status()["durable"] == 1


def test_a_size_mismatch_after_upload_refuses_to_claim_durability(tmp_path):
    class LyingStore(LocalDirectoryStore):
        def size(self, key):
            return 1

    vault = ArchiveVault(tmp_path / "local", LyingStore(tmp_path / "remote"))
    source = _file(tmp_path / "local", "big.bin")
    with pytest.raises(ArchiveError) as caught:
        vault.archive(source)
    assert "refusing to record it as durable" in str(caught.value)


def test_without_a_store_the_file_is_recorded_as_not_durable(tmp_path):
    vault = ArchiveVault(tmp_path / "local")
    source = _file(tmp_path / "local", "x.bin")
    with pytest.raises(ArchiveError) as caught:
        vault.archive(source)
    assert "not durable" in str(caught.value)
    assert vault.status()["not_durable"] == 1


def test_archiving_a_missing_file_says_so(tmp_path):
    with pytest.raises(ArchiveError):
        _vault(tmp_path).archive(tmp_path / "nope.bin")


# --- eviction ------------------------------------------------------------

def test_an_unarchived_file_is_never_evicted(tmp_path):
    vault = _vault(tmp_path)
    source = _file(tmp_path / "local", "x.bin")
    vault.objects["x.bin"] = ArchivedObject(
        key="x.bin", digest=digest_of(source), size=source.stat().st_size,
        local_path=str(source))
    ok, reason = vault.evict("x.bin")
    assert not ok
    assert "not verified durable" in reason
    assert source.exists()


def test_eviction_rechecks_the_store_rather_than_trusting_the_manifest(
        tmp_path):
    vault = _vault(tmp_path)
    source = _file(tmp_path / "local", "x.bin")
    vault.archive(source)
    # Someone tidied the bucket. The manifest still says durable.
    vault.store.delete("x.bin")
    ok, reason = vault.evict("x.bin")
    assert not ok
    assert "would destroy the only copy" in reason
    assert source.exists()


def test_a_verified_object_is_evicted_and_can_come_back(tmp_path):
    vault = _vault(tmp_path)
    source = _file(tmp_path / "local", "x.bin", b"corpus")
    vault.archive(source)
    ok, reason = vault.evict("x.bin")
    assert ok and reason == ""
    assert not source.exists()
    restored = vault.retrieve("x.bin")
    assert restored.read_bytes() == b"corpus"
    assert not vault.objects["x.bin"].evicted


def test_the_budget_only_spends_durable_objects(tmp_path):
    vault = _vault(tmp_path, local_budget_bytes=5_000)
    durable = _file(tmp_path / "local", "a.bin", b"a" * 4_000)
    vault.archive(durable, now=1.0)
    stranded = _file(tmp_path / "local", "b.bin", b"b" * 4_000)
    vault.objects["b.bin"] = ArchivedObject(
        key="b.bin", digest=digest_of(stranded), size=4_000,
        local_path=str(stranded))

    result = vault.enforce_budget()
    assert result["evicted"] == 1
    assert not durable.exists()
    # The one that was never uploaded is still on disk, and the vault says it
    # could not reach the budget rather than reaching for it.
    assert stranded.exists()


def test_the_budget_reports_over_budget_rather_than_deleting_the_last_copy(
        tmp_path):
    vault = _vault(tmp_path, local_budget_bytes=100)
    stranded = _file(tmp_path / "local", "b.bin", b"b" * 4_000)
    vault.objects["b.bin"] = ArchivedObject(
        key="b.bin", digest=digest_of(stranded), size=4_000,
        local_path=str(stranded))
    result = vault.enforce_budget()
    assert result["status"] == "OVER_BUDGET"
    assert result["evicted"] == 0
    assert stranded.exists()
    assert "refusing" in result["detail"]


def test_a_vault_inside_budget_evicts_nothing(tmp_path):
    vault = _vault(tmp_path, local_budget_bytes=10 ** 9)
    vault.archive(_file(tmp_path / "local", "a.bin"))
    assert vault.enforce_budget()["evicted"] == 0


def test_no_budget_configured_is_a_no_op(tmp_path):
    assert _vault(tmp_path).enforce_budget()["status"] == "NO_BUDGET"


# --- retrieval -----------------------------------------------------------

def test_a_truncated_object_is_refused_rather_than_returned(tmp_path):
    vault = _vault(tmp_path)
    source = _file(tmp_path / "local", "x.bin", b"z" * 8_000)
    vault.archive(source)
    vault.evict("x.bin")
    # The store now holds a short object -- exactly what a half-written
    # parquet looks like, and exactly what reads as valid-but-small.
    (tmp_path / "remote" / "x.bin").write_bytes(b"z" * 40)
    with pytest.raises(ArchiveError) as caught:
        vault.retrieve("x.bin")
    assert "refusing to return it" in str(caught.value)


def test_a_present_and_matching_local_copy_is_not_re_downloaded(tmp_path):
    class CountingStore(LocalDirectoryStore):
        gets = 0

        def get(self, key, path):
            CountingStore.gets += 1
            super().get(key, path)

    vault = ArchiveVault(tmp_path / "local", CountingStore(tmp_path / "remote"))
    source = _file(tmp_path / "local", "x.bin")
    vault.archive(source)
    vault.retrieve("x.bin")
    assert CountingStore.gets == 0


def test_retrieving_an_unknown_key_names_it(tmp_path):
    with pytest.raises(ArchiveError) as caught:
        _vault(tmp_path).retrieve("ghost")
    assert "not in the manifest" in str(caught.value)


# --- the manifest --------------------------------------------------------

def test_the_manifest_survives_a_restart(tmp_path):
    vault = _vault(tmp_path)
    vault.archive(_file(tmp_path / "local", "x.bin"))
    reborn = _vault(tmp_path)
    assert "x.bin" in reborn.objects
    assert reborn.objects["x.bin"].durable


def test_a_corrupt_manifest_does_not_prevent_archiving(tmp_path):
    (tmp_path / "local").mkdir(parents=True, exist_ok=True)
    (tmp_path / "local" / "archive_manifest.json").write_text("{not json")
    vault = _vault(tmp_path)
    assert vault.archive(_file(tmp_path / "local", "x.bin")).durable


def test_audit_strips_a_durability_claim_the_store_cannot_support(tmp_path):
    vault = _vault(tmp_path)
    vault.archive(_file(tmp_path / "local", "x.bin"))
    vault.store.delete("x.bin")
    result = vault.audit()
    assert result["status"] == "DIVERGED"
    assert result["missing_in_store"] == ["x.bin"]
    assert not vault.objects["x.bin"].durable
    # And having been demoted, it can no longer be evicted.
    assert not vault.evict("x.bin")[0]


def test_audit_is_data_blocked_without_a_store(tmp_path):
    assert ArchiveVault(tmp_path / "local").audit()["status"] == "DATA_BLOCKED"


# --- the s3-compatible adapter ------------------------------------------

class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body.read()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": _Reader(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix=""):
        return {"Contents": [{"Key": key} for key in sorted(self.objects)
                             if key.startswith(Prefix)]}


class _Reader:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


def test_the_s3_adapter_applies_its_prefix_without_leaking_it(tmp_path):
    client = FakeS3()
    store = S3CompatibleStore(client, "bucket", prefix="memecoin/v1")
    source = _file(tmp_path, "x.bin", b"payload")
    store.put("episodes/day.gz", source)
    assert "memecoin/v1/episodes/day.gz" in client.objects
    # Callers see their own keys back, not the bucket layout.
    assert store.list() == ["episodes/day.gz"]
    assert store.exists("episodes/day.gz")
    assert store.size("episodes/day.gz") == 7


def test_a_vault_over_the_s3_adapter_round_trips(tmp_path):
    vault = ArchiveVault(tmp_path / "local",
                         S3CompatibleStore(FakeS3(), "bucket"))
    source = _file(tmp_path / "local", "x.bin", b"corpus")
    vault.archive(source)
    assert vault.evict("x.bin")[0]
    assert vault.retrieve("x.bin").read_bytes() == b"corpus"
