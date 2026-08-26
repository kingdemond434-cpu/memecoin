#!/usr/bin/env python3
"""VI Central — one-command automated pipeline.

    python run_pipeline.py            # process inbox, rebuild everything
    python run_pipeline.py --help

Chain: ingest discoveries -> enrich stats -> cross-link -> map -> news ->
short-video scripts -> search index -> static site -> subscriber alerts.

Stdlib only. Idempotent: re-running with an empty inbox just rebuilds the
site from the current database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import enrich, ingest, linker, mapgen, newsgen, notify, search_index, sitegen, socialgen
from pipeline.util import PLATFORM_DIR, Database, load_json


def run(data_dir: Path, site_dir: Path, config_path: Path) -> dict:
    config = load_json(config_path, {})
    db = Database(data_dir)

    events = ingest.process_inbox(db, data_dir)
    enrich.enrich(db)
    links = linker.build_links(db)
    map_data = mapgen.build_map(db, links, site_dir)
    posts = newsgen.generate(db, events, config)
    scripts = socialgen.generate(db, events, data_dir, config)
    entries = search_index.build(db, site_dir)
    sitegen.build(db, links, map_data, entries, site_dir, config)
    alerts = notify.dispatch(posts, data_dir, config)

    print(
        f"[done] discoveries={len(events)} news={len(posts)} "
        f"social_scripts={len(scripts)} alerts={len(alerts)} "
        f"vehicles={len(db.vehicles)} pins={len(db.locations)}"
    )
    return {"events": events, "posts": posts, "alerts": alerts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PLATFORM_DIR / "data")
    parser.add_argument("--site", type=Path, default=PLATFORM_DIR / "site")
    parser.add_argument("--config", type=Path,
                        default=PLATFORM_DIR / "config" / "platform.json")
    args = parser.parse_args()
    run(args.data, args.site, args.config)


if __name__ == "__main__":
    main()
