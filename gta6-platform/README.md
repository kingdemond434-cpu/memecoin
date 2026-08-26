# VI Central — the AI-operated GTA VI everything platform

GTABase + MapGenie + wiki + news + subscriptions, built for GTA VI from day
one — and run entirely by an automated Python pipeline. One discovery in,
everything out:

```
DISCOVERY INBOX (data/discoveries/inbox/*.json)
    └─ scraper / tip form / manual drop
           ↓
INGEST                 vehicle · weapon · location · mission · character
    ├─ entity record created/updated
    └─ map location pinned
           ↓
ENRICH                 overall scores, class ranks, overall ranks
           ↓
LINKER                 vehicles ↔ missions ↔ locations ↔ characters
           ↓
MAPGEN                 stylized Leonida SVG + pin data
           ↓
NEWSGEN                auto news post (optional Claude API polish)
           ↓
SOCIALGEN              30–45s TikTok/Shorts/Reels script per discovery
           ↓
SEARCH + SITEGEN       search index, every page, RSS feed
           ↓
NOTIFY                 outbox record + optional webhook + optional SMTP
```

Zero dependencies — Python 3.10+ stdlib only. No CMS, no manual publishing,
no AI required at runtime (an optional Claude API copy-polish stage exists
but is off by default).

## Quick start

```bash
cd gta6-platform

# 1. Build everything from the seed database
python run_pipeline.py

# 2. Serve the site + subscription/account APIs
python serve.py            # http://localhost:8000

# 3. Watch the machine work: drop a discovery in the inbox
cp data/discoveries/examples/ocelot-jastic.json data/discoveries/inbox/
python run_pipeline.py
# -> vehicle page created, map pinned, stats ranked, comparison updated,
#    news posted, TikTok script drafted, subscribers alerted. One command.
```

For hands-off local operation, run the watcher instead of step 3's manual
run — it fires the pipeline whenever a file lands in the inbox:

```bash
python watch.py
```

## Fully automated operation (no human, no Claude)

`.github/workflows/gta6-pipeline.yml` runs the tests and the pipeline on
every push that touches `gta6-platform/data/**` (plus a daily heartbeat),
then commits the regenerated site back to the branch. Pushing a discovery
JSON to the inbox from anywhere — a phone, a scraper, a bot — is enough to
publish everything. Point GitHub Pages at `gta6-platform/site/` to host it.

## What the site includes

| Area | Benchmark | Implementation |
|------|-----------|----------------|
| Vehicle database | GTABase | `vehicles.html` — filter by class/status, sort by speed/price/overall; per-vehicle pages with stat bars and auto class-comparison tables |
| Interactive map | MapGenie | `map.html` — pan/zoom SVG of Leonida, category layers, pin popups cross-linked to vehicles/missions, collected state |
| Tracker | MapGenie | `tracker.html` — per-browser checklists + progress bars; export/import; optional server sync via `serve.py` (`/api/progress`) |
| Wiki / database | GTA Wiki | weapons, characters, missions with statuses (confirmed / rumored / speculated) and sources |
| News + feed | GTABase news | auto-generated discovery posts, hand-written posts, RSS (`feed.xml`) |
| Subscriptions | — | `subscribe.html` → `/api/subscribe` → `data/subscribers.json` → notify stage (outbox always; webhook/SMTP via env vars) |
| Short-form video | — | `data/social/*-short.md` ready-to-shoot scripts per discovery |
| Search | — | `search.html`, one index across every entity and post |

## Discovery format

```json
{
  "type": "vehicle",
  "name": "Ocelot Jastic",
  "confidence": "rumored",
  "source": "Frame analysis, promo still batch #7",
  "data": { "manufacturer": "Ocelot", "class": "Super", "stats": { "top_speed_mph": 136, "acceleration": 9.2, "handling": 8.8, "braking": 8.1 } },
  "location": { "name": "Vice City Valet Plaza", "region": "Vice City", "x": 620, "y": 440, "category": "vehicle_spawn" }
}
```

`type` may be `vehicle`, `weapon`, `location`, `mission`, or `character`.
Anything omitted gets sensible defaults; re-ingesting the same slug updates
the record instead of duplicating it.

## Configuration

`config/platform.json`:

- `notify.dry_run` — `true` by default; alerts are logged and written to
  `data/outbox/` but nothing external is sent. Set `false` and export
  `GTA6_WEBHOOK_URL` (Discord-compatible) and/or `GTA6_SMTP_*` to go live.
- `ai.enabled` — `false` by default. Set `true` with `ANTHROPIC_API_KEY`
  present (and `pip install anthropic`) to have news posts and video scripts
  polished by the Claude API. Every stage falls back to deterministic
  templates, so the platform never depends on it.

## Tests

```bash
cd gta6-platform
python -m unittest discover -s tests -v
```

Covers the full chain (discovery → page/pin/news/script/alert) and
idempotency (re-runs create nothing twice).

## Disclaimer

Unofficial fan project. Not affiliated with or endorsed by Rockstar Games or
Take-Two Interactive. Entries are labeled confirmed / rumored / speculated;
the map is stylized original art, not game cartography.
