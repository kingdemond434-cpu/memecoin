#!/usr/bin/env python3
"""Interactively store data credentials, accepting bare keys or dashboard URLs."""

import getpass
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def extract_provider_key(raw, provider):
    value = str(raw or "").strip().strip('"').strip("'")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "ws", "wss"}:
        if provider == "helius":
            return (parse_qs(parsed.query).get("api-key") or [""])[0].strip()
        parts = [part for part in parsed.path.split("/") if part]
        if provider == "alchemy" and len(parts) >= 2 and parts[-2] == "v2":
            return parts[-1].strip()
    return value


ENV_PATH = Path.home() / ".config" / "memecoin-shadow" / "env"
PUBLICNODE_YELLOWSTONE_URL = "https://solana-yellowstone-grpc.publicnode.com"
SECRET_FIELDS = [
    ("HELIUS_API_KEY", "Helius API key or RPC URL"),
    ("ALCHEMY_KEY", "Alchemy API key or Solana RPC URL"),
    ("GITHUB_TOKEN", "GitHub token"),
    ("TELEGRAM_API_ID", "Telegram API ID"),
    ("TELEGRAM_API_HASH", "Telegram API hash"),
    ("YOUTUBE_API_KEY", "YouTube API key"),
    ("YELLOWSTONE_GRPC_TOKEN", "PublicNode Yellowstone token"),
]
OPTIONAL_FIELDS = [
    "TELEGRAM_CHANNELS", "YELLOWSTONE_GRPC_URL", "X_BEARER_TOKEN",
    "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
]


def load_existing():
    values = {}
    if not ENV_PATH.exists():
        return values
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        try:
            values[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            values[key] = value.strip().strip('"')
    return values


def main():
    ENV_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    values = load_existing()
    print("Enter one value at each hidden prompt. Press Enter to preserve a value.")
    for key, label in SECRET_FIELDS:
        suffix = " [already set]" if values.get(key) else ""
        entered = getpass.getpass(f"{label}{suffix}: ").strip()
        if entered:
            values[key] = entered
    values["HELIUS_API_KEY"] = extract_provider_key(values.get("HELIUS_API_KEY", ""), "helius")
    values["ALCHEMY_KEY"] = extract_provider_key(values.get("ALCHEMY_KEY", ""), "alchemy")
    if values.get("YELLOWSTONE_GRPC_TOKEN"):
        values["YELLOWSTONE_GRPC_URL"] = PUBLICNODE_YELLOWSTONE_URL
    suffix = " [already set]" if values.get("TELEGRAM_CHANNELS") else ""
    channels = input(f"Public Telegram channels, comma-separated (optional){suffix}: ").strip()
    if channels:
        values["TELEGRAM_CHANNELS"] = channels
    ordered = [key for key, _ in SECRET_FIELDS] + OPTIONAL_FIELDS
    body = "\n".join(f"{key}={json.dumps(values.get(key, ''))}" for key in ordered) + "\n"
    temporary = ENV_PATH.with_suffix(".tmp")
    temporary.write_text(body, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, ENV_PATH)
    os.chmod(ENV_PATH, 0o600)
    print("Saved securely. Endpoint URLs were normalized; no credential values were displayed.")


if __name__ == "__main__":
    main()
