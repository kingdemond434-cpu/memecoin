"""HTML templates for the static site. Pure string templating — no deps.

Every page goes through layout(). Pages in subdirectories pass root="../"
so asset and nav links stay relative (the site works from file:// too).
"""

from __future__ import annotations

import json

from .util import esc

NAV = [
    ("index.html", "Home"),
    ("vehicles.html", "Vehicles"),
    ("map.html", "Map"),
    ("missions.html", "Missions"),
    ("weapons.html", "Weapons"),
    ("characters.html", "Characters"),
    ("tracker.html", "Tracker"),
    ("search.html", "Search"),
    ("subscribe.html", "Subscribe"),
]

STATUS_CLASS = {"confirmed": "ok", "rumored": "warn", "speculated": "spec"}


def embed_json(element_id: str, data) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{payload}</script>'


def status_badge(status: str) -> str:
    cls = STATUS_CLASS.get(status, "spec")
    return f'<span class="badge badge-{cls}">{esc(status)}</span>'


def stat_bar(label: str, value: float, maximum: float = 10.0, display=None) -> str:
    pct = max(0, min(100, round(float(value or 0) / maximum * 100)))
    shown = display if display is not None else value
    return (
        '<div class="stat-row">'
        f'<span class="stat-label">{esc(label)}</span>'
        f'<span class="stat-track"><span class="stat-fill" style="width:{pct}%"></span></span>'
        f'<span class="stat-value">{esc(shown)}</span>'
        "</div>"
    )


def layout(config: dict, title: str, active: str, content: str,
           root: str = "", extra_scripts: str = "") -> str:
    site = esc(config.get("site_name", "VI Central"))
    nav_html = ""
    for href, label in NAV:
        active_cls = ' class="active"' if href == active else ""
        nav_html += f'<a href="{root}{href}"{active_cls}>{esc(label)}</a>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — {site}</title>
<meta name="description" content="{esc(config.get('description', ''))}">
<link rel="stylesheet" href="{root}assets/style.css">
<link rel="alternate" type="application/rss+xml" title="{site} feed" href="{root}feed.xml">
</head>
<body>
<header class="topbar">
  <a class="brand" href="{root}index.html">{site}<span class="brand-vi">VI</span></a>
  <nav>{nav_html}</nav>
</header>
<main>
{content}
</main>
<footer>
  <p>{esc(config.get('disclaimer', ''))}</p>
  <p>Generated automatically by the VI Central pipeline · <a href="{root}about.html">How it works</a> · <a href="{root}feed.xml">RSS</a></p>
</footer>
{extra_scripts}
</body>
</html>
"""


def news_card(post: dict, root: str = "") -> str:
    kind = esc(post.get("kind", "news"))
    return f"""<a class="card news-card" href="{root}news/{esc(post['slug'])}.html">
  <span class="badge badge-kind">{kind}</span>{'<span class="badge badge-auto">auto</span>' if post.get('auto') else ''}
  <h3>{esc(post['title'])}</h3>
  <p class="muted">{esc(post.get('date', ''))}</p>
  <p>{esc(post.get('body', '')[:180])}…</p>
</a>"""


def link_list(items: list[dict], url_fn, root: str = "") -> str:
    if not items:
        return '<p class="muted">None linked yet.</p>'
    links = "".join(
        f'<li><a href="{root}{url_fn(i)}">{esc(i["name"])}</a></li>' for i in items
    )
    return f'<ul class="link-list">{links}</ul>'
