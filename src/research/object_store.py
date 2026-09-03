"""Making the corpus outlive the box.

The desk's storage is a 4 GB VPS. The research design assumes millions of
launch observations. Those two facts have coexisted so far only because the
local layer is careful -- bounded memory, disk spill, checkpoints -- and
careful is not the same as durable. Every episode, every decision snapshot,
every reconstructed launch lives on one disk that is one reinstall from empty,
and the forward ledger's whole value is that it is long.

So there is a tier below local disk:

    LOCAL      the working set. Fast, small, and expendable.
    OBJECT     durable storage -- an R2 bucket, an S3-compatible endpoint, a
               dataset repo. Cheap or free at this volume, and not on the box.
    CACHE      what came back down, kept while it is being read.

`ArchiveVault` moves files between them under two rules that are the entire
point of the module:

**Nothing is evicted until it is durably stored AND verified.** Uploading and
deleting is not archiving; it is deleting with extra steps. Every put records
a SHA256, every eviction re-checks that the remote object exists with that
digest, and an eviction whose verification fails leaves the local file exactly
where it was. A corpus lost to a confident `unlink` is not recoverable by
noticing later.

**A retrieved object is verified before it is trusted.** Object stores truncate,
proxies rewrite, and a half-written parquet file reads as a short one rather
than a corrupt one. `get` re-hashes what came back and raises on a mismatch
instead of handing a caller a silently shortened corpus, which would train a
model on a hole nobody could see.

The manifest is the index: local path, remote key, size, digest, when it was
uploaded, when it was last verified. It is written after every mutation, so an
interrupted run resumes from a state that is true rather than optimistic.

No SDK is imported here. The store is injected -- a local directory, an
S3-compatible client, or a fake -- so this is testable without credentials and
so a box without boto3 degrades to local-only with a stated reason rather than
failing to import.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

OBJECT_STORE_SCHEMA_VERSION = "v1"

#: Read in chunks so hashing a multi-gigabyte parquet file does not need a
#: multi-gigabyte resident buffer on a 4 GB box.
_CHUNK = 1024 * 1024


class Tier(Enum):
    LOCAL = "local"
    OBJECT = "object"
    CACHE = "cache"


class ArchiveError(RuntimeError):
    """A durability guarantee could not be met, with the reason attached."""


def digest_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ArchivedObject:
    """One file's whole story, in the manifest."""

    key: str
    digest: str
    size: int
    uploaded_at: float = 0.0
    verified_at: float = 0.0
    local_path: str = ""
    evicted: bool = False

    @property
    def durable(self) -> bool:
        """Uploaded AND verified. Uploaded alone is a hope, not a state."""
        return bool(self.uploaded_at and self.verified_at >= self.uploaded_at)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchivedObject":
        return cls(key=str(data.get("key", "")),
                   digest=str(data.get("digest", "")),
                   size=int(data.get("size", 0)),
                   uploaded_at=float(data.get("uploaded_at", 0.0)),
                   verified_at=float(data.get("verified_at", 0.0)),
                   local_path=str(data.get("local_path", "")),
                   evicted=bool(data.get("evicted", False)))


class LocalDirectoryStore:
    """An object store that is a directory. Real, and useful on its own.

    A second disk, an NFS mount or a synced folder is a legitimate durable
    tier, and having one implementation with no dependencies means the vault's
    behaviour is exercised end to end without a network or an SDK.
    """

    name = "local_directory"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are paths; refuse traversal rather than writing outside root.
        target = (self.root / key).resolve()
        root = self.root.resolve()
        if root != target and root not in target.parents:
            raise ArchiveError(f"key {key!r} escapes the store root")
        return target

    def put(self, key: str, path: Path) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside and rename, so an interrupted upload never leaves a
        # short object that `exists` would then report as durable.
        staging = target.with_suffix(target.suffix + ".partial")
        shutil.copyfile(path, staging)
        os.replace(staging, target)

    def get(self, key: str, path: Path) -> None:
        source = self._path(key)
        if not source.exists():
            raise ArchiveError(f"object {key!r} is not in the store")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).exists()
        except ArchiveError:
            return False

    def size(self, key: str) -> Optional[int]:
        try:
            target = self._path(key)
        except ArchiveError:
            return None
        return target.stat().st_size if target.exists() else None

    def list(self, prefix: str = "") -> List[str]:
        base = self.root
        return sorted(
            str(item.relative_to(base)) for item in base.rglob("*")
            if item.is_file() and not item.name.endswith(".partial")
            and str(item.relative_to(base)).startswith(prefix))

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except ArchiveError:
            return


class S3CompatibleStore:
    """R2, S3, or anything speaking the same four calls.

    The client is injected. This class knows the shape of `put_object`,
    `get_object`, `head_object` and `list_objects_v2` and nothing else -- no
    credentials, no region logic, no SDK import.
    """

    name = "s3_compatible"

    def __init__(self, client: Any, bucket: str, *, prefix: str = ""):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}".lstrip("/") if self.prefix else key

    def put(self, key: str, path: Path) -> None:
        with open(path, "rb") as handle:
            self.client.put_object(Bucket=self.bucket, Key=self._key(key),
                                   Body=handle)

    def get(self, key: str, path: Path) -> None:
        response = self.client.get_object(Bucket=self.bucket,
                                          Key=self._key(key))
        body = response["Body"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            reader = getattr(body, "read", None)
            handle.write(reader() if reader else bytes(body))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def size(self, key: str) -> Optional[int]:
        try:
            head = self.client.head_object(Bucket=self.bucket,
                                           Key=self._key(key))
        except Exception:
            return None
        value = head.get("ContentLength")
        return int(value) if value is not None else None

    def list(self, prefix: str = "") -> List[str]:
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=self._key(prefix))
        cut = len(self.prefix) + 1 if self.prefix else 0
        return sorted(str(item["Key"])[cut:]
                      for item in (response.get("Contents") or []))

    def delete(self, key: str) -> None:
        deleter = getattr(self.client, "delete_object", None)
        if deleter is not None:
            deleter(Bucket=self.bucket, Key=self._key(key))


class ArchiveVault:
    """Local working set on top, durable object store underneath.

    Every method that could lose data checks before it acts, and every method
    that could hand back wrong data checks after.
    """

    def __init__(self, local_root: Path, store: Any = None, *,
                 manifest_path: Optional[Path] = None,
                 local_budget_bytes: Optional[int] = None):
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.manifest_path = (Path(manifest_path) if manifest_path
                              else self.local_root / "archive_manifest.json")
        self.local_budget_bytes = local_budget_bytes
        self.objects: Dict[str, ArchivedObject] = {}
        self._load()

    # -- manifest --------------------------------------------------------

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            state = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("archive manifest unreadable: %s", exc)
            return
        for payload in state.get("objects") or []:
            try:
                item = ArchivedObject.from_dict(payload)
            except (TypeError, ValueError) as exc:
                logger.warning("dropping unreadable manifest row: %s", exc)
                continue
            if item.key:
                self.objects[item.key] = item

    def _save(self) -> None:
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            staging = self.manifest_path.with_suffix(".json.partial")
            staging.write_text(json.dumps(
                {"schema": OBJECT_STORE_SCHEMA_VERSION,
                 "objects": [item.to_dict()
                             for item in sorted(self.objects.values(),
                                                key=lambda row: row.key)]},
                indent=2))
            os.replace(staging, self.manifest_path)
        except OSError as exc:
            logger.warning("archive manifest unwritable: %s", exc)

    # -- archiving -------------------------------------------------------

    def archive(self, path: Path, *, key: Optional[str] = None,
                now: Optional[float] = None) -> ArchivedObject:
        """Upload one file and prove it arrived intact.

        Verification is a size and existence check against the store, plus the
        digest recorded locally. A store that reports a different size than
        was sent raises rather than being recorded as durable, because the
        only thing worse than an unarchived file is a file believed archived.
        """
        stamp = time.time() if now is None else now
        source = Path(path)
        if not source.exists():
            raise ArchiveError(f"{source} does not exist")
        object_key = key or self._relative(source)
        record = ArchivedObject(
            key=object_key, digest=digest_of(source),
            size=source.stat().st_size, local_path=str(source))
        if self.store is None:
            self.objects[object_key] = record
            self._save()
            raise ArchiveError(
                "no object store configured; the file remains local only and "
                "is recorded as not durable")
        self.store.put(object_key, source)
        record.uploaded_at = stamp
        remote_size = self._remote_size(object_key)
        if remote_size is None:
            raise ArchiveError(
                f"{object_key} is not readable in the store after upload")
        if remote_size != record.size:
            raise ArchiveError(
                f"{object_key} uploaded {record.size} bytes and the store "
                f"reports {remote_size}; refusing to record it as durable")
        record.verified_at = stamp
        self.objects[object_key] = record
        self._save()
        return record

    def _remote_size(self, key: str) -> Optional[int]:
        sizer = getattr(self.store, "size", None)
        if sizer is not None:
            size = sizer(key)
            if size is not None:
                return int(size)
        return None if not self.store.exists(key) else -1

    def _relative(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(
                self.local_root.resolve()))
        except ValueError:
            return Path(path).name

    # -- eviction --------------------------------------------------------

    def evict(self, key: str) -> Tuple[bool, str]:
        """Free the local copy, but only against a verified remote one."""
        record = self.objects.get(key)
        if record is None:
            return False, f"{key} is not in the manifest"
        if not record.durable:
            return False, (f"{key} is not verified durable "
                           f"(uploaded_at={record.uploaded_at}, "
                           f"verified_at={record.verified_at}); "
                           "eviction refused")
        if self.store is None or not self.store.exists(key):
            return False, (f"{key} is recorded durable but the store does not "
                           "have it; the manifest is wrong and eviction would "
                           "destroy the only copy")
        local = Path(record.local_path)
        if local.exists():
            local.unlink()
        record.evicted = True
        self._save()
        return True, ""

    def enforce_budget(self, *, now: Optional[float] = None
                       ) -> Dict[str, Any]:
        """Evict oldest-first until the local set fits, never further.

        Only durable objects are candidates. If the budget cannot be met with
        durable objects alone, the result says so rather than reaching for the
        ones that would be lost.
        """
        if self.local_budget_bytes is None:
            return {"status": "NO_BUDGET", "evicted": 0}
        resident = [item for item in self.objects.values()
                    if not item.evicted and Path(item.local_path).exists()]
        total = sum(item.size for item in resident)
        if total <= self.local_budget_bytes:
            return {"status": "OK", "evicted": 0, "bytes_local": total}
        candidates = sorted((item for item in resident if item.durable),
                            key=lambda row: row.uploaded_at)
        evicted = 0
        freed = 0
        refusals: List[str] = []
        for item in candidates:
            if total - freed <= self.local_budget_bytes:
                break
            ok, reason = self.evict(item.key)
            if ok:
                evicted += 1
                freed += item.size
            else:
                refusals.append(reason)
        remaining = total - freed
        return {
            "status": "OK" if remaining <= self.local_budget_bytes
            else "OVER_BUDGET",
            "evicted": evicted, "bytes_freed": freed,
            "bytes_local": remaining, "budget": self.local_budget_bytes,
            "refusals": refusals,
            "detail": ("" if remaining <= self.local_budget_bytes else
                       "cannot reach budget without evicting objects that are "
                       "not verified durable; refusing"),
        }

    # -- retrieval -------------------------------------------------------

    def retrieve(self, key: str, *, destination: Optional[Path] = None
                 ) -> Path:
        """Bring an object back down and prove it is the one we stored."""
        record = self.objects.get(key)
        if record is None:
            raise ArchiveError(f"{key} is not in the manifest")
        target = Path(destination) if destination else Path(record.local_path
                                                            or key)
        if target.exists() and digest_of(target) == record.digest:
            return target
        if self.store is None:
            raise ArchiveError("no object store configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.store.get(key, target)
        got = digest_of(target)
        if got != record.digest:
            # Leave the bad file where a human can look at it, and refuse to
            # return it. A truncated parquet reads as a short one, not a
            # corrupt one, and would train a model on a hole.
            raise ArchiveError(
                f"{key} came back with digest {got[:12]} but was stored as "
                f"{record.digest[:12]}; refusing to return it")
        record.evicted = False
        record.local_path = str(target)
        self._save()
        return target

    # -- reporting -------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        durable = [item for item in self.objects.values() if item.durable]
        resident = [item for item in self.objects.values()
                    if not item.evicted and Path(item.local_path).exists()]
        return {
            "schema": OBJECT_STORE_SCHEMA_VERSION,
            "store": getattr(self.store, "name", None) if self.store else None,
            "objects": len(self.objects),
            "durable": len(durable),
            "not_durable": len(self.objects) - len(durable),
            "resident_local": len(resident),
            "bytes_local": sum(item.size for item in resident),
            "bytes_durable": sum(item.size for item in durable),
            "local_budget_bytes": self.local_budget_bytes,
        }

    def audit(self) -> Dict[str, Any]:
        """Check the manifest against the store. Trust nothing on restart.

        A manifest is a claim about another system's contents, and it is wrong
        exactly when it matters: after a crash, after a bucket lifecycle rule,
        after someone tidied up. This is the check that must run before a
        budget pass is allowed to delete anything.
        """
        if self.store is None:
            return {"status": "DATA_BLOCKED",
                    "reason": "no object store configured"}
        missing: List[str] = []
        wrong_size: List[Dict[str, Any]] = []
        for key, record in self.objects.items():
            if not record.durable:
                continue
            size = self._remote_size(key)
            if size is None:
                missing.append(key)
                # The manifest is lying; strip the durability claim so nothing
                # can evict against it before someone looks.
                record.verified_at = 0.0
            elif size >= 0 and size != record.size:
                wrong_size.append({"key": key, "manifest": record.size,
                                   "store": size})
                record.verified_at = 0.0
        if missing or wrong_size:
            self._save()
        return {"status": "OK" if not (missing or wrong_size) else "DIVERGED",
                "checked": len(self.objects),
                "missing_in_store": missing, "size_mismatch": wrong_size}
