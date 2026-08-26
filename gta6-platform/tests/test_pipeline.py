"""End-to-end pipeline test: seed data + one discovery in a temp dir,
run the full chain, and assert every downstream artifact exists.

    python -m unittest discover -s tests -v   (from gta6-platform/)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_DIR))

from run_pipeline import run  # noqa: E402


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gta6-test-"))
        self.data = self.tmp / "data"
        self.site = self.tmp / "site"
        shutil.copytree(PLATFORM_DIR / "data", self.data)
        # stage the example vehicle discovery as a fresh inbox item
        example = self.data / "discoveries" / "examples" / "ocelot-jastic.json"
        shutil.copy(example, self.data / "discoveries" / "inbox" / "ocelot-jastic.json")
        # drop any previously-processed copy so ingest treats it as new
        for coll in ("vehicles", "news"):
            path = self.data / f"{coll}.json"
            items = json.loads(path.read_text())
            items = [i for i in items if "jastic" not in i.get("slug", "")]
            path.write_text(json.dumps(items))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_chain(self):
        result = run(self.data, self.site, PLATFORM_DIR / "config" / "platform.json")

        # ingest: discovery consumed, vehicle + location created
        self.assertEqual(len(result["events"]), 1)
        self.assertFalse(list((self.data / "discoveries" / "inbox").glob("*.json")))
        vehicles = json.loads((self.data / "vehicles.json").read_text())
        jastic = next(v for v in vehicles if v["slug"] == "ocelot-jastic")
        self.assertEqual(jastic["status"], "rumored")

        # enrich: scores and ranks computed
        self.assertGreater(jastic["overall"], 0)
        self.assertIn("rank_in_class", jastic)

        # map: pin exists for the discovery's location
        map_data = json.loads((self.site / "data" / "map.json").read_text())
        pin_slugs = {p["slug"] for p in map_data["pins"]}
        self.assertIn("vice-city-valet-plaza", pin_slugs)

        # news + alerts generated
        self.assertEqual(len(result["posts"]), 1)
        self.assertTrue(result["posts"][0]["auto"])
        self.assertEqual(len(result["alerts"]), 1)
        alert_files = list((self.data / "outbox").glob("*ocelot-jastic-alert.json"))
        self.assertEqual(len(alert_files), 1)

        # social script drafted
        scripts = list((self.data / "social").glob("*ocelot-jastic-short.md"))
        self.assertEqual(len(scripts), 1)
        self.assertIn("HOOK", scripts[0].read_text())

        # site pages rendered
        for page in ("index.html", "vehicles.html", "map.html", "tracker.html",
                     "search.html", "subscribe.html", "about.html", "feed.xml",
                     "vehicles/ocelot-jastic.html"):
            self.assertTrue((self.site / page).exists(), f"missing {page}")

        # detail page links back to the map pin and shows the class table
        detail = (self.site / "vehicles" / "ocelot-jastic.html").read_text()
        self.assertIn("map.html?focus=vice-city-valet-plaza", detail)
        self.assertIn("class comparison", detail)

        # search index covers the new vehicle
        entries = json.loads((self.site / "data" / "search-index.json").read_text())
        self.assertTrue(any(e["title"] == "Ocelot Jastic" for e in entries))

    def test_idempotent_rerun(self):
        first = run(self.data, self.site, PLATFORM_DIR / "config" / "platform.json")
        second = run(self.data, self.site, PLATFORM_DIR / "config" / "platform.json")
        self.assertEqual(len(first["posts"]), 1)
        self.assertEqual(len(second["events"]), 0)
        self.assertEqual(len(second["posts"]), 0)
        news = json.loads((self.data / "news.json").read_text())
        jastic_posts = [n for n in news if "ocelot-jastic" in n["slug"]]
        self.assertEqual(len(jastic_posts), 1)


if __name__ == "__main__":
    unittest.main()
