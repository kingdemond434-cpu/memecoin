"""Stage 7 — Search index.

Builds a flat client-side search index across every entity and news post.
Embedded into search.html by sitegen (and written to site/data/ for reuse).
"""

from __future__ import annotations

from pathlib import Path

from .util import Database, save_json


def build(db: Database, site_dir: Path) -> list[dict]:
    entries = []

    def add(title, type_, url, text, status=""):
        entries.append({
            "title": title, "type": type_, "url": url,
            "text": (text or "")[:400], "status": status,
        })

    for v in db.vehicles:
        add(v["name"], "vehicle", f"vehicles/{v['slug']}.html",
            f"{v.get('manufacturer', '')} {v.get('class', '')} {v.get('summary', '')}",
            v.get("status", ""))
    for w in db.weapons:
        add(w["name"], "weapon", "weapons.html",
            f"{w.get('class', '')} {w.get('summary', '')}", w.get("status", ""))
    for c in db.characters:
        add(c["name"], "character", "characters.html",
            f"{c.get('role', '')} {c.get('summary', '')}", c.get("status", ""))
    for loc in db.locations:
        add(loc["name"], "location", f"map.html?focus={loc['slug']}",
            f"{loc.get('region', '')} {loc.get('category', '')} {loc.get('summary', '')}",
            loc.get("status", ""))
    for m in db.missions:
        add(m["name"], "mission", f"missions/{m['slug']}.html",
            f"{m.get('region', '')} {m.get('summary', '')}", m.get("status", ""))
    for n in db.news:
        add(n["title"], "news", f"news/{n['slug']}.html", n.get("body", ""))

    save_json(Path(site_dir) / "data" / "search-index.json", entries)
    print(f"[search] indexed {len(entries)} entries")
    return entries
