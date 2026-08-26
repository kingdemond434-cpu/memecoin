"""Stage 9 — Static site generation.

Renders every page of the site from the database + link graph. All data a
page needs is embedded into the page itself (no fetch calls), so the site
works over file://, python -m http.server, GitHub Pages, or serve.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import mapgen
from .templates import (
    embed_json, layout, link_list, news_card, stat_bar, status_badge,
)
from .util import Database, esc

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def build(db: Database, links: dict, map_data: dict, search_entries: list[dict],
          site_dir: Path, config: dict) -> None:
    site = Path(site_dir)
    for sub in ("vehicles", "missions", "news", "assets", "data"):
        (site / sub).mkdir(parents=True, exist_ok=True)

    for asset in ASSETS_DIR.glob("*"):
        shutil.copy(asset, site / "assets" / asset.name)
    (site / ".nojekyll").write_text("", encoding="utf-8")

    news_sorted = sorted(db.news, key=lambda n: n.get("date", ""), reverse=True)

    _write(site / "index.html", _index_page(db, news_sorted, config))
    _write(site / "vehicles.html", _vehicles_page(db, config))
    for v in db.vehicles:
        _write(site / "vehicles" / f"{v['slug']}.html",
               _vehicle_detail(db, links, v, config))
    _write(site / "weapons.html", _weapons_page(db, config))
    _write(site / "characters.html", _characters_page(db, links, config))
    _write(site / "missions.html", _missions_page(db, config))
    for m in db.missions:
        _write(site / "missions" / f"{m['slug']}.html",
               _mission_detail(db, links, m, config))
    _write(site / "map.html", _map_page(map_data, config))
    _write(site / "tracker.html", _tracker_page(db, config))
    _write(site / "search.html", _search_page(search_entries, config))
    _write(site / "subscribe.html", _subscribe_page(db, config))
    _write(site / "about.html", _about_page(db, config))
    for post in news_sorted:
        _write(site / "news" / f"{post['slug']}.html",
               _news_detail(db, post, config))
    _write(site / "feed.xml", _rss(news_sorted, config))
    print(f"[sitegen] {len(list(site.rglob('*.html')))} pages -> {site}")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------- pages

def _index_page(db: Database, news_sorted: list[dict], config: dict) -> str:
    stats = [
        (len(db.vehicles), "vehicles"), (len(db.locations), "map pins"),
        (len(db.missions), "missions"), (len(db.weapons), "weapons"),
        (len(db.characters), "characters"), (len(db.news), "news posts"),
    ]
    stat_html = "".join(
        f'<div class="stat-tile"><div class="stat-num">{n}</div><div class="stat-name">{esc(label)}</div></div>'
        for n, label in stats
    )
    chain = " ".join(
        f'<span class="chain-step">{esc(s)}</span>'
        for s in ["discovery lands", "page created", "map pinned", "stats ranked",
                  "missions linked", "short-video script", "news post", "subscribers alerted"]
    )
    cards = "".join(news_card(p) for p in news_sorted[:6])
    content = f"""
<section class="hero">
  <h1>{esc(config.get('site_name', ''))}</h1>
  <p class="tagline">{esc(config.get('tagline', ''))}</p>
  <p>{esc(config.get('description', ''))}</p>
  <div class="stat-grid">{stat_html}</div>
</section>
<section>
  <h2>The machine</h2>
  <p class="chain">{chain}</p>
  <p class="muted">Every arrow above is a pipeline stage — no manual steps. Drop a discovery in the inbox and the whole site updates.</p>
</section>
<section>
  <h2>Latest news</h2>
  <div class="card-grid">{cards}</div>
</section>
"""
    return layout(config, "Home", "index.html", content)


def _vehicles_page(db: Database, config: dict) -> str:
    rows = [{
        "slug": v["slug"], "name": v["name"],
        "manufacturer": v.get("manufacturer", ""), "class": v.get("class", ""),
        "top_speed": (v.get("stats") or {}).get("top_speed_mph", 0),
        "overall": v.get("overall", 0), "seats": v.get("seats", ""),
        "price": v.get("price_estimate_usd"), "status": v.get("status", ""),
        "acquisition": v.get("acquisition", ""),
    } for v in db.vehicles]
    classes = sorted({r["class"] for r in rows})
    content = f"""
<h1>Vehicle database</h1>
<p class="muted">{len(rows)} vehicles tracked. Filter, sort, and click through for stats, comparisons and map locations.</p>
<div class="controls">
  <input id="veh-search" type="search" placeholder="Search name or manufacturer…">
  <select id="veh-class"><option value="">All classes</option>{''.join(f'<option>{esc(c)}</option>' for c in classes)}</select>
  <select id="veh-status"><option value="">Any status</option><option>confirmed</option><option>rumored</option><option>speculated</option></select>
  <select id="veh-sort">
    <option value="overall">Sort: overall</option>
    <option value="top_speed">Sort: top speed</option>
    <option value="price">Sort: price</option>
    <option value="name">Sort: name</option>
  </select>
</div>
<div class="table-wrap"><table id="veh-table">
  <thead><tr><th>Vehicle</th><th>Manufacturer</th><th>Class</th><th>Top speed</th><th>Overall</th><th>Seats</th><th>Est. price</th><th>Status</th></tr></thead>
  <tbody></tbody>
</table></div>
"""
    scripts = embed_json("vehicles-data", rows) + '\n<script src="assets/db.js"></script>'
    return layout(config, "Vehicles", "vehicles.html", content, extra_scripts=scripts)


def _vehicle_detail(db: Database, links: dict, v: dict, config: dict) -> str:
    s = v.get("stats") or {}
    bars = "".join([
        stat_bar("Top speed", s.get("top_speed_mph", 0), 150, f"{s.get('top_speed_mph', '?')} mph"),
        stat_bar("Acceleration", s.get("acceleration", 0)),
        stat_bar("Handling", s.get("handling", 0)),
        stat_bar("Braking", s.get("braking", 0)),
        stat_bar("Overall", v.get("overall", 0)),
    ])
    rivals = sorted(
        [r for r in db.vehicles if r.get("class") == v.get("class")],
        key=lambda r: r.get("overall", 0), reverse=True,
    )
    rival_rows = ""
    for r in rivals:
        highlight = ' class="highlight"' if r["slug"] == v["slug"] else ""
        rival_rows += (
            f'<tr{highlight}>'
            f'<td>#{r.get("rank_in_class", "?")}</td>'
            f'<td><a href="{r["slug"]}.html">{esc(r["name"])}</a></td>'
            f'<td>{(r.get("stats") or {}).get("top_speed_mph", "?")} mph</td>'
            f'<td>{r.get("overall", "?")}</td>'
            f'<td>{status_badge(r.get("status", ""))}</td></tr>'
        )
    loc = db.find("locations", v.get("location")) if v.get("location") else None
    loc_html = (
        f'<p><strong>Spotted at:</strong> <a href="../map.html?focus={esc(loc["slug"])}">'
        f'{esc(loc["name"])}</a> · {esc(loc["region"])}</p>' if loc else
        '<p class="muted">No map location yet.</p>'
    )
    missions = [db.find("missions", m) for m in links["vehicle_missions"].get(v["slug"], [])]
    price = f"${v['price_estimate_usd']:,}" if v.get("price_estimate_usd") else "Unknown"
    content = f"""
<p><a href="../vehicles.html">← All vehicles</a></p>
<h1>{esc(v['name'])} {status_badge(v.get('status', ''))}</h1>
<p class="muted">{esc(v.get('manufacturer', ''))} · {esc(v.get('class', ''))} · #{v.get('rank_in_class', '?')} in class · #{v.get('rank_overall', '?')} overall</p>
<p>{esc(v.get('summary', ''))}</p>
<div class="two-col">
  <section><h2>Stats</h2>{bars}</section>
  <section>
    <h2>Details</h2>
    <p><strong>Est. price:</strong> {esc(price)}</p>
    <p><strong>Seats:</strong> {esc(v.get('seats', '?'))}</p>
    <p><strong>Acquisition:</strong> {esc(v.get('acquisition', 'Unknown'))}</p>
    <p><strong>Source:</strong> {esc(v.get('source', ''))}</p>
    <p><strong>First tracked:</strong> {esc(v.get('discovered_at', ''))}</p>
    {loc_html}
    <h2>Appears in missions</h2>
    {link_list([m for m in missions if m], lambda m: f"../missions/{m['slug']}.html")}
  </section>
</div>
<section>
  <h2>{esc(v.get('class', ''))} class comparison</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Rank</th><th>Vehicle</th><th>Top speed</th><th>Overall</th><th>Status</th></tr></thead>
    <tbody>{rival_rows}</tbody>
  </table></div>
</section>
"""
    return layout(config, v["name"], "vehicles.html", content, root="../")


def _weapons_page(db: Database, config: dict) -> str:
    rows = "".join(
        f'<tr><td>{esc(w["name"])}</td><td>{esc(w.get("class", ""))}</td>'
        f'<td>{(w.get("stats") or {}).get("damage", "?")}</td>'
        f'<td>{(w.get("stats") or {}).get("fire_rate", "?")}</td>'
        f'<td>{(w.get("stats") or {}).get("range", "?")}</td>'
        f'<td>{w.get("overall", "?")}</td>'
        f'<td>{esc(w.get("acquisition", ""))}</td>'
        f'<td>{status_badge(w.get("status", ""))}</td></tr>'
        for w in sorted(db.weapons, key=lambda x: x.get("overall", 0), reverse=True)
    )
    content = f"""
<h1>Weapon database</h1>
<p class="muted">{len(db.weapons)} weapons tracked, ranked by overall score.</p>
<div class="table-wrap"><table>
  <thead><tr><th>Weapon</th><th>Class</th><th>Damage</th><th>Fire rate</th><th>Range</th><th>Overall</th><th>Acquisition</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
"""
    return layout(config, "Weapons", "weapons.html", content)


def _characters_page(db: Database, links: dict, config: dict) -> str:
    cards = ""
    for c in db.characters:
        missions = [db.find("missions", m) for m in links["character_missions"].get(c["slug"], [])]
        cards += f"""<div class="card" id="{esc(c['slug'])}">
  <h3>{esc(c['name'])} {status_badge(c.get('status', ''))}</h3>
  <p class="muted">{esc(c.get('role', ''))}</p>
  <p>{esc(c.get('summary', ''))}</p>
  <h4>Gives missions</h4>
  {link_list([m for m in missions if m], lambda m: f"missions/{m['slug']}.html")}
</div>"""
    content = f"""
<h1>Characters</h1>
<p class="muted">{len(db.characters)} characters tracked.</p>
<div class="card-grid">{cards}</div>
"""
    return layout(config, "Characters", "characters.html", content)


def _missions_page(db: Database, config: dict) -> str:
    rows = "".join(
        f'<tr><td><a href="missions/{esc(m["slug"])}.html">{esc(m["name"])}</a></td>'
        f'<td>{esc(m.get("region", ""))}</td>'
        f'<td>{esc((db.find("characters", m.get("giver")) or {}).get("name", "Unknown"))}</td>'
        f'<td>{esc(m.get("rewards", ""))}</td>'
        f'<td>{status_badge(m.get("status", ""))}</td></tr>'
        for m in db.missions
    )
    content = f"""
<h1>Missions &amp; jobs</h1>
<p class="muted">{len(db.missions)} tracked. Story missions are unannounced — entries marked <em>speculated</em> are fan projections used to demo the cross-linking system.</p>
<div class="table-wrap"><table>
  <thead><tr><th>Mission</th><th>Region</th><th>Giver</th><th>Rewards</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
"""
    return layout(config, "Missions", "missions.html", content)


def _mission_detail(db: Database, links: dict, m: dict, config: dict) -> str:
    giver = db.find("characters", m.get("giver"))
    loc = db.find("locations", m.get("location")) if m.get("location") else None
    vehicles = [db.find("vehicles", v) for v in links["mission_vehicles"].get(m["slug"], [])]
    content = f"""
<p><a href="../missions.html">← All missions</a></p>
<h1>{esc(m['name'])} {status_badge(m.get('status', ''))}</h1>
<p>{esc(m.get('summary', ''))}</p>
<div class="two-col">
  <section>
    <h2>Details</h2>
    <p><strong>Region:</strong> {esc(m.get('region', ''))}</p>
    <p><strong>Giver:</strong> {f'<a href="../characters.html#{esc(giver["slug"])}">{esc(giver["name"])}</a>' if giver else 'Unknown'}</p>
    <p><strong>Rewards:</strong> {esc(m.get('rewards', 'Unknown'))}</p>
    <p><strong>Source:</strong> {esc(m.get('source', ''))}</p>
    {f'<p><strong>Location:</strong> <a href="../map.html?focus={esc(loc["slug"])}">{esc(loc["name"])}</a></p>' if loc else ''}
  </section>
  <section>
    <h2>Featured vehicles</h2>
    {link_list([v for v in vehicles if v], lambda v: f"../vehicles/{v['slug']}.html")}
  </section>
</div>
"""
    return layout(config, m["name"], "missions.html", content, root="../")


def _map_page(map_data: dict, config: dict) -> str:
    legend = "".join(
        f'<label class="legend-item"><input type="checkbox" data-category="{esc(cat)}" checked>'
        f'<span class="legend-dot" style="background:{esc(color)}"></span>{esc(cat.replace("_", " "))}</label>'
        for cat, color in map_data["categories"].items()
    )
    content = f"""
<h1>Interactive map of Leonida</h1>
<p class="muted">Drag to pan, scroll to zoom, click a pin for details. Check off collected pins — progress is saved in your browser and counts toward your <a href="tracker.html">tracker</a>. Stylized fan map, not game cartography.</p>
<div class="legend">{legend}</div>
<div class="map-frame" id="map-frame">
{mapgen.base_svg()}
</div>
<aside id="pin-panel" class="pin-panel" hidden></aside>
"""
    scripts = embed_json("map-data", map_data) + '\n<script src="assets/map.js"></script>'
    return layout(config, "Map", "map.html", content, extra_scripts=scripts)


def _tracker_page(db: Database, config: dict) -> str:
    tracker = {
        "vehicles": [{"slug": v["slug"], "name": v["name"],
                      "detail": v.get("class", ""), "url": f"vehicles/{v['slug']}.html"}
                     for v in db.vehicles],
        "missions": [{"slug": m["slug"], "name": m["name"],
                      "detail": m.get("region", ""), "url": f"missions/{m['slug']}.html"}
                     for m in db.missions],
        "locations": [{"slug": p["slug"], "name": p["name"],
                       "detail": p.get("region", ""), "url": f"map.html?focus={p['slug']}"}
                      for p in db.locations],
    }
    content = """
<h1>Your tracker</h1>
<p class="muted">Personal progress, saved in this browser (no account needed). Run <code>serve.py</code> and use the sync buttons to keep progress across devices.</p>
<div id="tracker-root"></div>
<div class="controls">
  <button id="trk-export">Export progress</button>
  <button id="trk-import">Import progress</button>
  <button id="trk-sync-up">Sync to server</button>
  <button id="trk-sync-down">Load from server</button>
  <span id="trk-msg" class="muted"></span>
</div>
"""
    scripts = embed_json("tracker-data", tracker) + '\n<script src="assets/tracker.js"></script>'
    return layout(config, "Tracker", "tracker.html", content, extra_scripts=scripts)


def _search_page(entries: list[dict], config: dict) -> str:
    content = """
<h1>Search everything</h1>
<p class="muted">One box across vehicles, weapons, characters, locations, missions and news.</p>
<div class="controls"><input id="search-box" type="search" placeholder="Try 'banshee', 'keys', 'raul'…" autofocus></div>
<div id="search-results"></div>
"""
    scripts = embed_json("search-data", entries) + '\n<script src="assets/search.js"></script>'
    return layout(config, "Search", "search.html", content, extra_scripts=scripts)


def _subscribe_page(db: Database, config: dict) -> str:
    content = f"""
<h1>GTA 6 updates, the moment they land</h1>
<p>Every discovery is ingested, published and dispatched automatically. Subscribe and the pipeline alerts you in the same run that creates the page.</p>
<form id="subscribe-form" class="subscribe-form">
  <input type="email" id="sub-email" placeholder="you@example.com" required>
  <fieldset>
    <legend>Topics</legend>
    <label><input type="checkbox" name="topic" value="news" checked> News</label>
    <label><input type="checkbox" name="topic" value="vehicles" checked> Vehicles</label>
    <label><input type="checkbox" name="topic" value="map" checked> Map &amp; locations</label>
  </fieldset>
  <button type="submit">Subscribe</button>
  <p id="sub-msg" class="muted"></p>
</form>
<p class="muted">Prefer feeds? The whole site ships an <a href="feed.xml">RSS feed</a>. The subscribe form needs <code>serve.py</code> (or any backend wired to <code>data/subscribers.json</code>); on the static build it falls back to showing the RSS link. {len(db.news)} posts published so far.</p>
"""
    scripts = '<script src="assets/app.js"></script>'
    return layout(config, "Subscribe", "subscribe.html", content, extra_scripts=scripts)


def _about_page(db: Database, config: dict) -> str:
    content = f"""
<h1>How this site runs itself</h1>
<p>{esc(config.get('site_name', ''))} is generated end-to-end by a Python pipeline. There is no CMS and no manual publishing step.</p>
<ol class="pipeline-list">
  <li><strong>Ingest</strong> — a discovery JSON lands in <code>data/discoveries/inbox/</code> (from a scraper, a tip form, or a human) and becomes database entities.</li>
  <li><strong>Enrich</strong> — overall scores and class rankings are recomputed for every vehicle and weapon.</li>
  <li><strong>Link</strong> — vehicles, missions, locations and characters are cross-referenced both ways.</li>
  <li><strong>Map</strong> — every location becomes a pin on the stylized Leonida map.</li>
  <li><strong>News</strong> — a post is written for each discovery and pushed to the feed and RSS.</li>
  <li><strong>Social</strong> — a 30–45s vertical-video script is drafted per discovery.</li>
  <li><strong>Search &amp; tracker</strong> — the index and personal checklists are rebuilt.</li>
  <li><strong>Notify</strong> — subscribers are alerted by email/webhook in the same run.</li>
</ol>
<p>An optional AI stage can polish the generated copy via the Claude API, but the platform is fully operational without it.</p>
<h2>Status labels</h2>
<p>{status_badge('confirmed')} shown in official material · {status_badge('rumored')} reported but unverified · {status_badge('speculated')} fan speculation.</p>
<p class="muted">{esc(config.get('disclaimer', ''))}</p>
"""
    return layout(config, "About", "about.html", content)


def _news_detail(db: Database, post: dict, config: dict) -> str:
    ref_links = ""
    ref_urls = {
        "vehicles": lambda s: f"../vehicles/{s}.html",
        "missions": lambda s: f"../missions/{s}.html",
        "locations": lambda s: f"../map.html?focus={s}",
        "characters": lambda s: f"../characters.html#{s}",
        "weapons": lambda s: "../weapons.html",
    }
    for coll, slugs in (post.get("refs") or {}).items():
        for slug in slugs:
            entity = db.find(coll, slug) if coll in Database.COLLECTIONS else None
            if entity:
                ref_links += f'<a class="chip" href="{ref_urls[coll](slug)}">{esc(entity["name"])}</a> '
    content = f"""
<p><a href="../index.html">← All news</a></p>
<h1>{esc(post['title'])}</h1>
<p class="muted">{esc(post.get('date', ''))} · {esc(post.get('kind', ''))}{' · generated automatically' if post.get('auto') else ''}</p>
<article class="news-body"><p>{esc(post.get('body', ''))}</p></article>
{f'<section><h2>Related</h2><p>{ref_links}</p></section>' if ref_links else ''}
"""
    return layout(config, post["title"], "index.html", content, root="../")


def _rss(news_sorted: list[dict], config: dict) -> str:
    items = "".join(f"""
  <item>
    <title>{esc(p['title'])}</title>
    <link>news/{esc(p['slug'])}.html</link>
    <guid isPermaLink="false">{esc(p['slug'])}</guid>
    <pubDate>{esc(p.get('date', ''))}</pubDate>
    <description>{esc(p.get('body', '')[:500])}</description>
  </item>""" for p in news_sorted)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>{esc(config.get('site_name', ''))}</title>
  <description>{esc(config.get('description', ''))}</description>
  <link>index.html</link>{items}
</channel></rss>
"""
