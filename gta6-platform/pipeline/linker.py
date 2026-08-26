"""Stage 3 — Cross-linking.

Builds the bidirectional link graph between entities so that every page can
render its "related" sections without any entity storing more than its own
canonical references:

- vehicle.location  -> location.vehicles
- vehicle.missions  -> mission.vehicles (merged both ways)
- mission.location  -> location.missions
- mission.giver     -> character.missions
"""

from __future__ import annotations

from collections import defaultdict

from .util import Database


def build_links(db: Database) -> dict:
    links = {
        "location_vehicles": defaultdict(list),
        "location_missions": defaultdict(list),
        "vehicle_missions": defaultdict(list),
        "mission_vehicles": defaultdict(list),
        "character_missions": defaultdict(list),
    }

    for v in db.vehicles:
        if v.get("location") and db.find("locations", v["location"]):
            links["location_vehicles"][v["location"]].append(v["slug"])
        for m in v.get("missions", []):
            if db.find("missions", m):
                _add(links["vehicle_missions"], v["slug"], m)
                _add(links["mission_vehicles"], m, v["slug"])

    for m in db.missions:
        if m.get("location") and db.find("locations", m["location"]):
            links["location_missions"][m["location"]].append(m["slug"])
        if m.get("giver") and db.find("characters", m["giver"]):
            links["character_missions"][m["giver"]].append(m["slug"])
        for v in m.get("vehicles", []):
            if db.find("vehicles", v):
                _add(links["mission_vehicles"], m["slug"], v)
                _add(links["vehicle_missions"], v, m["slug"])

    return {k: dict(v) for k, v in links.items()}


def _add(bucket: dict, key: str, value: str) -> None:
    if value not in bucket[key]:
        bucket[key].append(value)
