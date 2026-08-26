"""Stage 5 — News generation.

Turns each ingested discovery event into a news post in the feed.
Deterministic templates by default; optional Claude API polish via ai.py.
"""

from __future__ import annotations

from .util import Database, slugify
from . import ai

KIND_LABELS = {
    "vehicle": "New vehicle",
    "weapon": "New weapon",
    "location": "New location",
    "mission": "New mission",
    "character": "New character",
}

STATUS_PHRASES = {
    "confirmed": "confirmed",
    "rumored": "rumored — treat with caution until a clean source lands",
    "speculated": "fan speculation for now",
}


def generate(db: Database, events: list[dict], config: dict) -> list[dict]:
    posts = []
    for event in events:
        post = _post_for(db, event, config)
        if not db.find("news", post["slug"]):
            db.news.append(post)
            posts.append(post)
            print(f"[newsgen] {post['title']}")
    if posts:
        db.save()
    return posts


def _post_for(db: Database, event: dict, config: dict) -> dict:
    kind, name, slug = event["kind"], event["name"], event["slug"]
    date = event.get("discovered_at", "")[:10]
    label = KIND_LABELS.get(kind, "New discovery")
    status_phrase = STATUS_PHRASES.get(event["status"], event["status"])

    lines = [
        f"{label} spotted: {name}. Status: {status_phrase}. Source: {event['source']}.",
    ]

    if kind == "vehicle":
        v = db.find("vehicles", slug) or {}
        s = v.get("stats") or {}
        if s:
            lines.append(
                f"Early numbers put the {name} at ~{s.get('top_speed_mph', '?')} mph top speed "
                f"with an overall rating of {v.get('overall', '?')}/10, ranking "
                f"#{v.get('rank_in_class', '?')} in the {v.get('class', '?')} class."
            )
        if event.get("location"):
            loc = db.find("locations", event["location"])
            if loc:
                lines.append(
                    f"It has been pinned on the map at {loc['name']} ({loc['region']})."
                )
        lines.append(
            f"The {name} page, class comparison table, map pin and tracker entry "
            "were generated automatically the moment this discovery was ingested."
        )
    elif kind == "location":
        loc = db.find("locations", slug) or {}
        lines.append(
            f"The pin is live on the interactive map in {loc.get('region', 'Leonida')} "
            f"under the '{loc.get('category', 'poi')}' layer, and it has been added to the tracker."
        )
    else:
        lines.append("Its database page, cross-links and tracker entry are live now.")

    body = " ".join(lines)
    body = ai.polish(body, f"news post about a {kind} discovery", config)

    return {
        "slug": f"{date}-{slugify(name)}" if date else slugify(name),
        "title": f"{label}: {name}",
        "date": date,
        "kind": "discovery",
        "auto": True,
        "refs": {f"{kind}s": [slug]},
        "body": body,
    }
