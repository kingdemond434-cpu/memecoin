"""Stage 2 — Enrichment.

Derives ratings and rankings from raw stats so every page and comparison
table stays consistent without manual editing:

- vehicles: overall score (weighted stat blend, top-speed normalized),
  rank within class, rank overall
- weapons: overall score and class rank
"""

from __future__ import annotations

from .util import Database

VEHICLE_WEIGHTS = {"top_speed": 0.35, "acceleration": 0.30, "handling": 0.20, "braking": 0.15}
WEAPON_WEIGHTS = {"damage": 0.45, "fire_rate": 0.30, "range": 0.25}

# Normalization ceiling for top speed (mph -> 0..10 scale).
TOP_SPEED_CEILING = 150.0


def enrich(db: Database) -> None:
    _enrich_vehicles(db)
    _enrich_weapons(db)
    db.save()


def _enrich_vehicles(db: Database) -> None:
    for v in db.vehicles:
        s = v.get("stats") or {}
        speed_score = min(float(s.get("top_speed_mph", 0)) / TOP_SPEED_CEILING, 1.0) * 10
        overall = (
            speed_score * VEHICLE_WEIGHTS["top_speed"]
            + float(s.get("acceleration", 0)) * VEHICLE_WEIGHTS["acceleration"]
            + float(s.get("handling", 0)) * VEHICLE_WEIGHTS["handling"]
            + float(s.get("braking", 0)) * VEHICLE_WEIGHTS["braking"]
        )
        v["overall"] = round(overall, 2)

    _rank(db.vehicles, "rank_overall")
    for cls in {v.get("class") for v in db.vehicles}:
        group = [v for v in db.vehicles if v.get("class") == cls]
        _rank(group, "rank_in_class")


def _enrich_weapons(db: Database) -> None:
    for w in db.weapons:
        s = w.get("stats") or {}
        overall = sum(float(s.get(k, 0)) * wt for k, wt in WEAPON_WEIGHTS.items())
        w["overall"] = round(overall, 2)

    _rank(db.weapons, "rank_overall")
    for cls in {w.get("class") for w in db.weapons}:
        group = [w for w in db.weapons if w.get("class") == cls]
        _rank(group, "rank_in_class")


def _rank(items: list[dict], field: str) -> None:
    for i, item in enumerate(
        sorted(items, key=lambda x: x.get("overall", 0), reverse=True), start=1
    ):
        item[field] = i
