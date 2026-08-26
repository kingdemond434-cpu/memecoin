"""Stage 8 — Subscriber alerts.

For each auto-generated news post, dispatches alerts to subscribers:

- always: writes an alert record to data/outbox/ (audit trail)
- optional: generic JSON webhook (Discord-compatible) when the env var named
  by config.notify.webhook_url_env is set
- optional: SMTP email when the SMTP env vars are set

`dry_run: true` in config (the default) logs instead of sending, so the
pipeline is safe to run anywhere.
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.mime.text import MIMEText
from pathlib import Path

from .util import load_json, now_iso, save_json


def dispatch(posts: list[dict], data_dir: Path, config: dict) -> list[dict]:
    if not posts:
        return []

    ncfg = (config or {}).get("notify") or {}
    dry_run = ncfg.get("dry_run", True)
    subscribers = [
        s for s in load_json(Path(data_dir) / "subscribers.json", [])
        if s.get("confirmed")
    ]

    alerts = []
    for post in posts:
        alert = {
            "created_at": now_iso(),
            "post": post["slug"],
            "title": post["title"],
            "body": post["body"],
            "recipients": len(subscribers),
            "channels": [],
        }
        if _send_webhook(post, ncfg, dry_run):
            alert["channels"].append("webhook")
        if _send_email(post, subscribers, ncfg, dry_run):
            alert["channels"].append("email")

        save_json(Path(data_dir) / "outbox" / f"{post['slug']}-alert.json", alert)
        alerts.append(alert)
        mode = "DRY-RUN" if dry_run else "SENT"
        print(f"[notify] {mode} '{post['title']}' -> {len(subscribers)} subscriber(s), "
              f"channels: {alert['channels'] or ['outbox only']}")
    return alerts


def _send_webhook(post: dict, ncfg: dict, dry_run: bool) -> bool:
    url = os.environ.get(ncfg.get("webhook_url_env", ""), "")
    if not url:
        return False
    if dry_run:
        return True
    payload = json.dumps({"content": f"**{post['title']}**\n{post['body']}"}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=15).read()
        return True
    except Exception as e:
        print(f"[notify] webhook failed: {e}")
        return False


def _send_email(post: dict, subscribers: list[dict], ncfg: dict, dry_run: bool) -> bool:
    scfg = ncfg.get("smtp") or {}
    host = os.environ.get(scfg.get("host_env", ""), "")
    if not host or not subscribers:
        return False
    if dry_run:
        return True
    port = int(os.environ.get(scfg.get("port_env", ""), "587"))
    user = os.environ.get(scfg.get("user_env", ""), "")
    password = os.environ.get(scfg.get("password_env", ""), "")
    from_addr = scfg.get("from_addr", "alerts@localhost")
    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            if user:
                server.login(user, password)
            for sub in subscribers:
                msg = MIMEText(post["body"])
                msg["Subject"] = post["title"]
                msg["From"] = from_addr
                msg["To"] = sub["email"]
                server.send_message(msg)
        return True
    except Exception as e:
        print(f"[notify] email failed: {e}")
        return False
