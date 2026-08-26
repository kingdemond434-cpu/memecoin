"""Stage 4 — Map generation.

Produces:
- a stylized, original SVG base map of Leonida (procedural — no copyrighted
  map art), returned as a string and written to site/assets/map-base.svg
- site/data/map.json with every pin (also embedded into map.html by sitegen)

Coordinates are a 0-1000 x 0-1000 grid, x right, y down.
"""

from __future__ import annotations

import json
from pathlib import Path

from .util import Database, save_json

CATEGORY_COLORS = {
    "poi": "#f5a623",
    "vehicle_spawn": "#38d39f",
    "collectible": "#c77dff",
    "activity": "#4cc9f0",
    "mission": "#ff5d8f",
    "shop": "#ffd166",
}

REGION_LABELS = [
    ("VICE CITY", 640, 520),
    ("LEONIDA KEYS", 560, 890),
    ("GRASSRIVERS", 470, 700),
    ("PORT GELLHORN", 300, 220),
    ("AMBROSIA", 400, 330),
    ("MT. KALAGA", 430, 120),
]


def build_map(db: Database, links: dict, site_dir: Path) -> dict:
    pins = []
    for loc in db.locations:
        pins.append({
            "slug": loc["slug"],
            "name": loc["name"],
            "category": loc.get("category", "poi"),
            "region": loc.get("region", "Leonida"),
            "x": loc.get("x", 500),
            "y": loc.get("y", 500),
            "status": loc.get("status", "speculated"),
            "summary": loc.get("summary", ""),
            "vehicles": [
                {"slug": s, "name": db.find("vehicles", s)["name"]}
                for s in links["location_vehicles"].get(loc["slug"], [])
            ],
            "missions": [
                {"slug": s, "name": db.find("missions", s)["name"]}
                for s in links["location_missions"].get(loc["slug"], [])
            ],
        })

    map_data = {"pins": pins, "categories": CATEGORY_COLORS}
    site_dir = Path(site_dir)
    save_json(site_dir / "data" / "map.json", map_data)

    svg = base_svg()
    assets = site_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "map-base.svg").write_text(svg, encoding="utf-8")
    print(f"[mapgen] {len(pins)} pins across {len({p['region'] for p in pins})} regions")
    return map_data


def base_svg(inline_id: str = "leonida-map") -> str:
    """Original, stylized landmass loosely evoking a Florida-like peninsula."""
    labels = "\n    ".join(
        f'<text class="region-label" x="{x}" y="{y}">{name}</text>'
        for name, x, y in REGION_LABELS
    )
    return f"""<svg id="{inline_id}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet">
  <defs>
    <radialGradient id="sea" cx="50%" cy="45%" r="75%">
      <stop offset="0%" stop-color="#0b3550"/>
      <stop offset="100%" stop-color="#061e30"/>
    </radialGradient>
    <linearGradient id="land" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1d4d3b"/>
      <stop offset="55%" stop-color="#28584a"/>
      <stop offset="100%" stop-color="#2e5f4d"/>
    </linearGradient>
  </defs>
  <rect width="1000" height="1000" fill="url(#sea)"/>
  <!-- mainland peninsula -->
  <path fill="url(#land)" stroke="#0e2c22" stroke-width="4" d="
    M 180 40 L 560 30 L 600 90 L 640 200 L 700 320 L 730 420
    L 720 520 L 680 600 L 620 660 L 560 720 L 500 760
    L 440 740 L 380 680 L 340 600 L 300 500 L 260 380
    L 220 260 L 180 150 Z"/>
  <!-- keys island chain -->
  <g fill="#2e5f4d" stroke="#0e2c22" stroke-width="3">
    <ellipse cx="500" cy="800" rx="34" ry="13" transform="rotate(18 500 800)"/>
    <ellipse cx="555" cy="838" rx="30" ry="11" transform="rotate(22 555 838)"/>
    <ellipse cx="608" cy="872" rx="26" ry="10" transform="rotate(24 608 872)"/>
    <ellipse cx="655" cy="905" rx="22" ry="9" transform="rotate(26 655 905)"/>
  </g>
  <!-- causeway -->
  <path fill="none" stroke="#d8c690" stroke-width="4" stroke-dasharray="10 6"
        d="M 495 762 L 505 798 L 558 836 L 610 870 L 656 903"/>
  <!-- wetlands -->
  <path fill="#245245" opacity="0.85" d="
    M 420 600 Q 470 570 530 610 Q 560 660 520 700 Q 470 730 430 700 Q 400 650 420 600 Z"/>
  <!-- barrier island / vice beach -->
  <path fill="#3a6b57" stroke="#0e2c22" stroke-width="3" d="
    M 690 380 L 706 430 L 704 520 L 688 590 L 672 560 L 678 460 L 676 400 Z"/>
  <!-- highways -->
  <g fill="none" stroke="#c9b46a" stroke-width="3" opacity="0.9">
    <path d="M 250 120 L 300 260 L 360 420 L 430 560 L 500 700 L 500 758"/>
    <path d="M 300 260 L 420 300 L 560 340 L 640 430 L 650 540"/>
    <path d="M 640 430 L 620 520 L 590 610"/>
  </g>
  <!-- lakes -->
  <ellipse cx="380" cy="450" rx="36" ry="24" fill="#0b3550"/>
  <ellipse cx="470" cy="240" rx="22" ry="15" fill="#0b3550"/>
  <g font-family="'Arial Black', Arial, sans-serif" font-size="22" fill="#e8ddb5"
     opacity="0.85" text-anchor="middle" letter-spacing="3">
    {labels}
  </g>
  <g id="pins"></g>
</svg>"""
