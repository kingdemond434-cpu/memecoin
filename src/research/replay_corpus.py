"""Audit and seal observed replay inputs without deleting difficult launches.

A declared time window is selected by launch time, never by return or safety
label. Every launch in that window must be replayable. Missing native evidence
blocks the whole snapshot instead of silently improving its cohort.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

from src.research.candidate_replay import CandidateReplayHandler, ReplayBlocked, _digest
from src.research.gauntlet import LATENCY_GRID_S

MAX_OBJECT_BYTES = 4_000_000


def _decode_episode(raw: bytes, *, compressed: bool) -> dict:
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            raw = handle.read(MAX_OBJECT_BYTES + 1)
    if len(raw) > MAX_OBJECT_BYTES:
        raise ReplayBlocked("episode exceeds declared object byte bound")
    row = json.loads(raw)
    if not isinstance(row, dict):
        raise ReplayBlocked("episode must be an object")
    return row


def replay_gaps(row: dict) -> list[str]:
    gaps = []
    if not isinstance(row.get("final_outcome"), dict) or row["final_outcome"].get("status") != "OK":
        gaps.append("unresolved_outcome")
    if not row.get("token") or type(row.get("created_at")) not in (int, float) or not math.isfinite(row["created_at"]) or row["created_at"] <= 0:
        gaps.append("launch_identity_or_clock_missing")
    if row.get("chain") != "solana" or row.get("factory") not in {"pump", "pump.fun"}:
        gaps.append("native_replay_protocol_mapping_missing")
    if not row.get("snapshots"):
        gaps.append("point_in_time_snapshots_missing")
    try:
        execution = CandidateReplayHandler._execution(row, {"chain": row.get("chain"), "protocol": row.get("factory")})
        for delay in LATENCY_GRID_S:
            cell = execution["fills_by_latency"].get(str(float(delay)))
            if (not isinstance(cell, dict) or cell.get("evidence_kind") not in {"native_simulation", "observed_transaction"}
                    or not cell.get("source") or type(cell.get("filled")) is not bool
                    or type(cell.get("route_feasible")) is not bool
                    or any(type(cell.get(k)) not in (int, float) or not math.isfinite(cell[k]) or not 0 <= cell[k] < 1
                           for k in ("round_trip_cost_fraction", "failed_cost_fraction"))):
                gaps.append("native_latency_fill_cost_missing")
                break
    except (ReplayBlocked, TypeError, KeyError):
        gaps.append("native_route_depth_fill_cost_missing")
    return gaps


def audit_corpus(root: str | Path, *, max_episodes: int = 10_000) -> dict:
    if type(max_episodes) is not int or not 1 <= max_episodes <= 100_000:
        raise ValueError("explicit episode bound between 1 and 100000 required")
    root = Path(root)
    paths = sorted(root.glob("*/*.json.gz"))
    if len(paths) > max_episodes:
        return {"status": "DATA_BLOCKED", "reason": "corpus exceeds declared episode bound",
                "episode_files": len(paths), "max_episodes": max_episodes}
    gaps, inventory, identities, duplicates = Counter(), [], set(), 0
    ready = 0
    for path in paths:
        source_bytes = path.read_bytes()
        inventory.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(source_bytes).hexdigest()})
        try:
            row = _decode_episode(source_bytes, compressed=True)
            identity = (row.get("chain"), row.get("token"))
            if identity in identities:
                duplicates += 1
            identities.add(identity)
            missing = replay_gaps(row)
            gaps.update(missing)
            ready += not missing
        except (OSError, ValueError, TypeError, EOFError):
            gaps["unreadable_object"] += 1
    return {"status": "INPUTS_PRESENT_NOT_VALIDATED" if paths and not gaps and not duplicates else "DATA_BLOCKED",
            "episode_files": len(paths), "replay_input_complete": ready, "gaps": dict(gaps),
            "duplicate_launch_identities": duplicates, "source_inventory_hash": _digest(inventory),
            "authority": "none", "production_performance": "UNMEASURED"}


def freeze_corpus(root: str | Path, output: str | Path, *, windows: dict,
                  max_episodes: int = 2000, now: float | None = None) -> dict:
    """Freeze all launches inside three explicit disjoint chronological windows.

    Call before constructing candidate variants. Gaps between windows are the
    declared purge intervals. A late outcome is a block, not an exclusion.
    The fresh output contains a complete source inventory and every admission
    failure even when it cannot produce a usable replay manifest.
    """
    if type(max_episodes) is not int or not 1 <= max_episodes <= 100_000:
        raise ValueError("explicit positive bounded max_episodes required")
    stamp = time.time() if now is None else now
    if type(stamp) not in (int, float) or not math.isfinite(stamp):
        raise ValueError("finite capture clock required")
    if set(windows) != {"train", "validation", "test"}:
        raise ValueError("three explicit windows required")
    prior = 0
    for name in ("train", "validation", "test"):
        bounds = windows[name]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not prior < bounds[0] < bounds[1] <= stamp):
            raise ValueError("strictly ordered closed past windows and purge gaps required")
        prior = bounds[1]
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("snapshot must be outside mutable source corpus")
    if output.exists():
        raise FileExistsError("refusing to overwrite a frozen corpus")
    paths = sorted(root.glob("*/*.json.gz"))
    if not paths or len(paths) > max_episodes:
        raise ReplayBlocked("source corpus empty or exceeds explicit work bound")
    inventory, selected, blocked, seen = [], [], [], set()
    counts = Counter()
    for path in paths:
        source_bytes = path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        entry = {"source_path": path.relative_to(root).as_posix(), "source_sha256": source_hash}
        inventory.append(entry)
        try:
            row = _decode_episode(source_bytes, compressed=True)
            # Protect the snapshot from concurrent replacement of a closed episode.
            if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
                raise ReplayBlocked("source object changed during capture")
            created = row.get("created_at")
            if type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
                raise ReplayBlocked("missing launch time prevents window assignment")
            partition = next((name for name, (start, end) in windows.items() if start <= created <= end), None)
            entry["partition"] = partition or "OUTSIDE_DECLARED_WINDOWS"
            if partition is None:
                continue
            identity = (row.get("chain"), row.get("token"))
            missing = replay_gaps(row)
            if identity in seen:
                missing.append("duplicate_launch_identity")
            seen.add(identity)
            end = windows[partition][1]
            if any(type(mark.get("timestamp")) not in (int, float) or not math.isfinite(mark["timestamp"]) or mark["timestamp"] > end
                   for mark in row.get("market_observations", [])):
                missing.append("outcome_crosses_partition_boundary")
            if missing:
                blocked.append({**entry, "reasons": missing})
            else:
                selected.append(row)
                counts[partition] += 1
        except (OSError, ValueError, TypeError, EOFError) as exc:
            blocked.append({**entry, "reasons": [type(exc).__name__]})
    selected.sort(key=lambda row: (row["created_at"], row["token"]))
    status = "SEALED_NOT_EVALUATED" if not blocked and all(counts[name] for name in windows) else "DATA_BLOCKED"
    report = {"schema": "observed-replay-corpus-v1", "status": status, "frozen_at": stamp,
              "windows": windows, "partition_counts": dict(counts), "source_inventory": inventory,
              "source_inventory_hash": _digest(inventory), "blocked": blocked,
              "selection_rule": "all launches by creation time inside declared windows; no outcome-based selection",
              "authority": "none", "production_performance": "UNMEASURED"}
    output.mkdir(parents=True, exist_ok=False)
    if status == "SEALED_NOT_EVALUATED":
        entries = []
        for index, row in enumerate(selected):
            relative = f"episodes/{index:06d}.json.gz"
            raw = gzip.compress(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
            path = output / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        report.update(corpus_hash=_digest(selected), files=entries, train_end=windows["train"][1],
                      validation_end=windows["validation"][1], test_end=windows["test"][1])
    (output / "split.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def load_sealed_corpus(root: Path, split: dict, *, max_episodes: int) -> list[dict]:
    files = split.get("files")
    if not isinstance(files, list) or not 0 < len(files) <= max_episodes:
        raise ReplayBlocked("sealed corpus file inventory absent or exceeds work bound")
    expected, rows = set(), []
    root = root.resolve()
    for entry in files:
        path = root / entry["path"]
        if (not isinstance(entry["path"], str) or path.resolve().parent != (root / "episodes").resolve()
                or path.suffix != ".gz" or path in expected):
            raise ReplayBlocked("noncanonical or duplicate sealed corpus path")
        expected.add(path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ReplayBlocked("sealed episode bytes changed")
        rows.append(_decode_episode(raw, compressed=True))
    if set(root.glob("*/*.json.gz")) != expected:
        raise ReplayBlocked("sealed corpus contains unlisted or missing episodes")
    return sorted(rows, key=lambda row: (row.get("created_at", 0), row.get("token", "")))
