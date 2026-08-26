"""Stage 1 — Discovery ingestion.

Watches data/discoveries/inbox/ for JSON discovery files and turns each one
into database entities. A single vehicle discovery fans out into:
vehicle record + map location + mission links + a news event that the
downstream stages (newsgen, socialgen, notify) pick up.

Processed files are moved to data/discoveries/processed/ so the stage is
idempotent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .util import Database, load_json, now_iso, slugify, today

VALID_TYPES = ("vehicle", "weapon", "location", "mission", "character")


def process_inbox(db: Database, data_dir: Path) -> list[dict]:
    """Process every discovery in the inbox. Returns a list of events:
    {"kind": <entity type>, "slug": ..., "name": ..., "discovery": <raw>}
    """
    inbox = Path(data_dir) / "discoveries" / "inbox"
    processed_dir = Path(data_dir) / "discoveries" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    events = []
    for path in sorted(inbox.glob("*.json")):
        discovery = load_json(path, {})
        event = _apply_discovery(db, discovery)
        if event:
            events.append(event)
            print(f"[ingest] {event['kind']}: {event['name']} ({event['slug']})")
        else:
            print(f"[ingest] skipped unrecognized discovery file: {path.name}")
        shutil.move(str(path), str(processed_dir / path.name))

    if events:
        db.save()
    return events


def _apply_discovery(db: Database, discovery: dict) -> dict | None:
    dtype = discovery.get("type")
    name = discovery.get("name")
    if dtype not in VALID_TYPES or not name:
        return None

    slug = discovery.get("slug") or slugify(name)
    status = discovery.get("confidence", "rumored")
    source = discovery.get("source", "Unattributed discovery")
    discovered_at = discovery.get("discovered_at") or today()
    data = dict(discovery.get("data") or {})

    location_slug = None
    loc = discovery.get("location")
    if isinstance(loc, dict) and loc.get("name"):
        location_slug = loc.get("slug") or slugify(loc["name"])
        db.upsert("locations", {
            "slug": location_slug,
            "name": loc["name"],
            "category": loc.get("category", "poi"),
            "region": loc.get("region", "Leonida"),
            "x": loc.get("x", 500),
            "y": loc.get("y", 500),
            "status": status,
            "source": source,
            "summary": loc.get("summary", f"First reported alongside the {name} discovery."),
        })

    if dtype == "vehicle":
        record = {
            "slug": slug,
            "name": name,
            "manufacturer": data.get("manufacturer", "Unknown"),
            "class": data.get("class", "Unclassified"),
            "seats": data.get("seats", 2),
            "stats": data.get("stats", {}),
            "price_estimate_usd": data.get("price_estimate_usd"),
            "acquisition": data.get("acquisition", "Unknown"),
            "status": status,
            "source": source,
            "location": location_slug or data.get("location"),
            "missions": discovery.get("missions", []),
            "discovered_at": discovered_at,
            "summary": data.get("summary", f"Newly discovered {data.get('class', '')} vehicle.".strip()),
        }
        db.upsert("vehicles", record)
    elif dtype == "weapon":
        record = {
            "slug": slug,
            "name": name,
            "class": data.get("class", "Unclassified"),
            "stats": data.get("stats", {}),
            "price_estimate_usd": data.get("price_estimate_usd"),
            "acquisition": data.get("acquisition", "Unknown"),
            "status": status,
            "source": source,
            "summary": data.get("summary", "Newly discovered weapon."),
        }
        db.upsert("weapons", record)
    elif dtype == "location":
        location_slug = slug
        record = {
            "slug": slug,
            "name": name,
            "category": data.get("category", "poi"),
            "region": data.get("region", "Leonida"),
            "x": data.get("x", 500),
            "y": data.get("y", 500),
            "status": status,
            "source": source,
            "summary": data.get("summary", "Newly discovered location."),
        }
        db.upsert("locations", record)
    elif dtype == "mission":
        record = {
            "slug": slug,
            "name": name,
            "giver": data.get("giver"),
            "region": data.get("region", "Leonida"),
            "location": location_slug or data.get("location"),
            "vehicles": data.get("vehicles", []),
            "rewards": data.get("rewards", "Unknown"),
            "status": status,
            "source": source,
            "summary": data.get("summary", "Newly discovered mission."),
        }
        db.upsert("missions", record)
    else:  # character
        record = {
            "slug": slug,
            "name": name,
            "role": data.get("role", "Unknown"),
            "status": status,
            "source": source,
            "summary": data.get("summary", "Newly discovered character."),
        }
        db.upsert("characters", record)

    return {
        "kind": dtype,
        "slug": slug,
        "name": name,
        "status": status,
        "source": source,
        "discovered_at": discovered_at,
        "location": location_slug,
        "ingested_at": now_iso(),
        "discovery": discovery,
    }
