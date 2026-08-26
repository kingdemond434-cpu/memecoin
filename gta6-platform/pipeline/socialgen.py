"""Stage 6 — Short-video (TikTok/Shorts/Reels) script generation.

For every discovery event, writes a ready-to-shoot script to
data/social/<date>-<slug>-short.md. Deterministic templates by default;
optional Claude API polish via ai.py.
"""

from __future__ import annotations

from pathlib import Path

from .util import Database
from . import ai

HOOKS = {
    "vehicle": "STOP scrolling — a new GTA 6 vehicle just surfaced.",
    "weapon": "A new GTA 6 weapon just showed up in the files of the internet.",
    "location": "GTA 6's map just got bigger — new location found.",
    "mission": "We might already know one of GTA 6's missions.",
    "character": "New GTA 6 character alert.",
}


def generate(db: Database, events: list[dict], data_dir: Path, config: dict) -> list[Path]:
    out_dir = Path(data_dir) / "social"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for event in events:
        date = event.get("discovered_at", "")[:10] or "undated"
        path = out_dir / f"{date}-{event['slug']}-short.md"
        if path.exists():
            continue
        path.write_text(_script_for(db, event, config), encoding="utf-8")
        written.append(path)
        print(f"[socialgen] {path.name}")
    return written


def _script_for(db: Database, event: dict, config: dict) -> str:
    kind, name = event["kind"], event["name"]
    hook = HOOKS.get(kind, "New GTA 6 discovery just dropped.")

    beats = [f"Show the source frame: {event['source']}."]
    if kind == "vehicle":
        v = db.find("vehicles", event["slug"]) or {}
        s = v.get("stats") or {}
        beats.append(
            f"Cut to the stat card: ~{s.get('top_speed_mph', '?')} mph, overall "
            f"{v.get('overall', '?')}/10, #{v.get('rank_in_class', '?')} in the "
            f"{v.get('class', '?')} class."
        )
        if event.get("location"):
            loc = db.find("locations", event["location"])
            if loc:
                beats.append(f"Zoom the map to {loc['name']} in {loc['region']} — that's where it was spotted.")
    elif kind == "location":
        loc = db.find("locations", event["slug"]) or {}
        beats.append(f"Zoom the interactive map into {loc.get('region', 'Leonida')} and drop the new pin on screen.")
    beats.append(f"Confidence check: this one is {event['status'].upper()} — say why in one line.")

    body = "\n".join(f"{i}. {b}" for i, b in enumerate(beats, start=2))
    script = f"""# Short-video script — {name}

**Format:** 30–45s vertical (TikTok / Shorts / Reels)
**Status:** {event['status']} | **Source:** {event['source']}

1. HOOK (0–3s): {hook}
{body}
{len(beats) + 2}. CTA: "Full stats, map pin and tracker are already live — link in bio. Follow for every GTA 6 discovery, posted automatically the minute it drops."

**Caption:** {name} just hit the GTA 6 radar — {event['status']}. Full breakdown on VI Central.
**Hashtags:** #GTA6 #GTAVI #Leonida #ViceCity #Rockstar #Gaming
"""
    return ai.polish(script, f"short-video script about a {kind} discovery", config)
