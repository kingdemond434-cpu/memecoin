#!/usr/bin/env python3
"""VI Central dev/production-lite server (stdlib only).

Serves the generated static site plus the two personalization APIs:

    POST /api/subscribe            {"email": ..., "topics": [...]}
        -> appends to data/subscribers.json (picked up by the next
           pipeline run's notify stage)
    POST /api/progress             {"progress": {...}, "token": optional}
        -> stores tracker progress under an account token, returns the token
    GET  /api/progress?token=...   -> returns stored progress

    python serve.py [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pipeline.util import PLATFORM_DIR, load_json, now_iso, save_json

DATA_DIR = PLATFORM_DIR / "data"
SITE_DIR = PLATFORM_DIR / "site"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not 0 < length <= 65536:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        if self.path.startswith("/api/progress"):
            token = ""
            if "token=" in self.path:
                token = self.path.split("token=", 1)[1].split("&")[0]
            if not re.fullmatch(r"[a-f0-9]{32}", token or ""):
                return self._json(400, {"error": "valid token required"})
            with _lock:
                accounts = load_json(DATA_DIR / "accounts.json", {})
            account = accounts.get(token)
            if account is None:
                return self._json(404, {"error": "unknown token"})
            return self._json(200, {"token": token, "progress": account.get("progress", {})})
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/subscribe":
            body = self._read_body()
            email = str(body.get("email", "")).strip().lower()
            if not EMAIL_RE.fullmatch(email):
                return self._json(400, {"error": "invalid email"})
            topics = [t for t in body.get("topics", []) if isinstance(t, str)][:10]
            with _lock:
                subs = load_json(DATA_DIR / "subscribers.json", [])
                existing = next((s for s in subs if s.get("email") == email), None)
                if existing:
                    existing["topics"] = topics or existing.get("topics", [])
                else:
                    subs.append({
                        "email": email,
                        "confirmed": True,
                        "topics": topics or ["news"],
                        "subscribed_at": now_iso(),
                    })
                save_json(DATA_DIR / "subscribers.json", subs)
            return self._json(200, {"ok": True})

        if self.path == "/api/progress":
            body = self._read_body()
            progress = body.get("progress")
            if not isinstance(progress, dict) or len(progress) > 5000:
                return self._json(400, {"error": "invalid progress payload"})
            token = str(body.get("token") or "")
            if not re.fullmatch(r"[a-f0-9]{32}", token):
                token = secrets.token_hex(16)
            with _lock:
                accounts = load_json(DATA_DIR / "accounts.json", {})
                accounts[token] = {"progress": progress, "updated_at": now_iso()}
                save_json(DATA_DIR / "accounts.json", accounts)
            return self._json(200, {"token": token, "ok": True})

        return self._json(404, {"error": "unknown endpoint"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not (SITE_DIR / "index.html").exists():
        print("Site not built yet — run `python run_pipeline.py` first.")
        return
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"VI Central serving http://localhost:{args.port} (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
