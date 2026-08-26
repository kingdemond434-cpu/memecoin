"""Shared helpers: paths, JSON I/O, slugs, time."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default if default is not None else []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "item"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def esc(text) -> str:
    """HTML-escape a value for safe interpolation into templates."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class Database:
    """All entity collections, loaded from / saved to data/*.json."""

    COLLECTIONS = ("vehicles", "weapons", "characters", "locations", "missions", "news")

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        for name in self.COLLECTIONS:
            setattr(self, name, load_json(self.data_dir / f"{name}.json", []))

    def save(self) -> None:
        for name in self.COLLECTIONS:
            save_json(self.data_dir / f"{name}.json", getattr(self, name))

    def find(self, collection: str, slug: str):
        for item in getattr(self, collection):
            if item.get("slug") == slug:
                return item
        return None

    def upsert(self, collection: str, item: dict) -> dict:
        existing = self.find(collection, item["slug"])
        if existing:
            existing.update(item)
            return existing
        getattr(self, collection).append(item)
        return item
