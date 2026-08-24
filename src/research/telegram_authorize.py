"""Create the persistent read-only Telegram session used by the shadow collector."""

import asyncio
import os
from pathlib import Path

from telethon import TelegramClient


async def authorize():
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    if not api_id or not api_hash:
        raise SystemExit("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")
    session_dir = Path("data/telegram")
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "collector"
    client = TelegramClient(str(session_path), int(api_id), api_hash, receive_updates=False)
    await client.start()
    me = await client.get_me()
    await client.disconnect()
    session_file = session_path.with_suffix(".session")
    if session_file.exists():
        session_file.chmod(0o600)
    print(f"Telegram session authorized for account id {me.id}; no credential was displayed")


if __name__ == "__main__":
    asyncio.run(authorize())
