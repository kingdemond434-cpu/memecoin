"""Escalation that reaches a person, with a paper trail when it cannot.

Escalations were appended to a log file on the node. Nothing read it. A fault
that wakes nobody is indistinguishable from no fault, and the whole point of
detecting a silent stream at 3am is that somebody finds out before morning.

Telegram is used because the desk already holds an authorised session for it
and needs no new credential. Messages go to Saved Messages -- the account's own
chat -- so no bot, no chat id, and nothing is shared with anyone else.

Three rules.

**Never block the supervisor.** Alerting runs after the corrective action, with
a short timeout, and a failure to alert is logged rather than raised. A node
that cannot report a fault must still be able to fix it.

**Deduplicate, or the alert becomes the fault.** The same message every two
minutes for six hours is how people learn to ignore the channel. One message
per distinct fault per window, and a single line when it clears.

**Always leave the paper trail.** Every escalation is appended to disk whether
or not it was delivered, because the log is what an audit reads and the
message is only what a human sees.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("alert")

ALERT_SCHEMA_VERSION = "v1"

#: One message per distinct fault per this window. Long enough that a flapping
#: fault does not become a flapping notification.
DEFAULT_DEDUPE_S = 3_600.0

#: Telegram must not hold up a supervisor pass.
DEFAULT_TIMEOUT_S = 15.0

SESSION_PATH = Path("data/telegram/collector")


@dataclass
class Delivery:
    sent: bool
    channel: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"sent": self.sent, "channel": self.channel, "detail": self.detail}


class Alerter:
    """Sends what matters, once, and records everything."""

    def __init__(self, log_path: Optional[Path] = None, *,
                 state_path: Optional[Path] = None,
                 dedupe_s: float = DEFAULT_DEDUPE_S,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self.log_path = Path(log_path) if log_path else None
        self.state_path = Path(state_path) if state_path else None
        self.dedupe_s = float(dedupe_s)
        self.timeout_s = float(timeout_s)
        self.sent: Dict[str, float] = {}
        self.deliveries: List[Delivery] = []
        self._load()

    def escalate(self, key: str, message: str,
                 now: Optional[float] = None) -> Optional[Delivery]:
        """One fault. Returns None when suppressed as a duplicate."""
        moment = time.time() if now is None else now
        self._append(key, message, moment)
        last = self.sent.get(key)
        if last is not None and moment - last < self.dedupe_s:
            return None
        self.sent[key] = moment
        delivery = self._deliver(f"AURUM DESK\n\n{message}")
        self.deliveries.append(delivery)
        self._save()
        return delivery

    def clear(self, key: str, now: Optional[float] = None) -> None:
        """A fault that resolved. Sends one line so the channel closes the loop."""
        if key not in self.sent:
            return
        del self.sent[key]
        self._append(key, "resolved", time.time() if now is None else now)
        self._deliver(f"AURUM DESK\n\nrecovered: {key}")
        self._save()

    # --- delivery --------------------------------------------------------

    def _deliver(self, text: str) -> Delivery:
        session = SESSION_PATH.with_suffix(".session")
        if not session.exists():
            return Delivery(False, "telegram",
                            "no authorised session; run "
                            "python -m src.research.telegram_authorize")
        api_id = os.getenv("TELEGRAM_API_ID", "").strip()
        api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
        if not api_id or not api_hash:
            return Delivery(False, "telegram", "TELEGRAM credentials absent")
        try:
            return asyncio.run(
                asyncio.wait_for(self._send(str(api_id), api_hash, text),
                                 timeout=self.timeout_s))
        except asyncio.TimeoutError:
            return Delivery(False, "telegram", f"timed out after {self.timeout_s}s")
        except Exception as exc:
            # A node that cannot report a fault must still be able to fix it.
            return Delivery(False, "telegram", f"{type(exc).__name__}: {exc}")

    async def _send(self, api_id: str, api_hash: str, text: str) -> Delivery:
        from telethon import TelegramClient

        client = TelegramClient(str(SESSION_PATH), int(api_id), api_hash,
                                receive_updates=False)
        await client.connect()
        try:
            if not await client.is_user_authorized():
                return Delivery(False, "telegram", "session is not authorised")
            # "me" is Saved Messages: the account's own chat. Nothing is
            # shared with anyone else, and no bot or chat id is needed.
            await client.send_message("me", text)
            return Delivery(True, "telegram")
        finally:
            await client.disconnect()

    # --- the paper trail -------------------------------------------------

    def _append(self, key: str, message: str, now: float) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as handle:
                handle.write(json.dumps({"at": now, "key": key,
                                         "message": message}) + "\n")
        except OSError as exc:
            logger.warning("escalation log unwritable: %s", exc)

    def _save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"sent": self.sent}))
        except OSError:
            pass

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            self.sent = dict(json.loads(self.state_path.read_text()).get("sent") or {})
        except (OSError, json.JSONDecodeError, AttributeError):
            self.sent = {}

    def report(self) -> Dict[str, Any]:
        delivered = [item for item in self.deliveries if item.sent]
        return {
            "schema": ALERT_SCHEMA_VERSION,
            "status": ("OK" if delivered or not self.deliveries else "DEGRADED"),
            "detail": ("" if delivered or not self.deliveries else
                       "faults were escalated and none was delivered; the "
                       "paper trail is on disk and nobody has been told"),
            "open_faults": sorted(self.sent),
            "attempts": len(self.deliveries),
            "delivered": len(delivered),
            "last": self.deliveries[-1].to_dict() if self.deliveries else None,
        }
