"""Resolve domain-published handles to platform-stable account identifiers.

``verify_entities.py`` proves that an organisation's own domain published a
profile link.  A handle is still mutable, however, so it is deliberately kept
under ``metadata.published_handles`` and confers no authenticity authority.
This second pass asks the platform for the stable identifier and only then
adds it to ``accounts``.

Supported without private or scraped data:

* Telegram, through the already-authorised read-only Telethon session;
* YouTube channel IDs, through the operator's YouTube Data API key;
* X numeric user IDs, when an official bearer token exists;
* Bluesky DIDs, through its public identity resolver;
* GitHub numeric account IDs, through its public REST endpoint.

Nothing is guessed.  An unavailable provider leaves the handle unresolved,
and an ID claimed by two entities is rejected for both rather than assigned by
first-wins order.

    python tools/resolve_entity_ids.py \
      --input config/entities.verified.yaml --in-place
"""

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

TIMEOUT = 15
USER_AGENT = "memecoin-entity-id-resolver/1.0 (+public identity resolution)"


def _json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.getcode() != 200:
            raise RuntimeError(f"HTTP {response.getcode()}")
        return json.loads(response.read(1_000_000).decode("utf-8"))


def resolve_public(platform: str, handle: str) -> Tuple[Optional[str], str]:
    """Resolve one non-Telegram handle without ever exposing credentials."""
    try:
        if platform == "github":
            token = os.getenv("GITHUB_TOKEN", "").strip()
            headers = ({"Authorization": f"Bearer {token}"} if token else {})
            payload = _json(
                f"https://api.github.com/users/{urllib.parse.quote(handle)}", headers)
            value = payload.get("id")
        elif platform == "bluesky":
            payload = _json(
                "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?"
                + urllib.parse.urlencode({"handle": handle}))
            value = payload.get("did")
        elif platform == "youtube":
            if handle.startswith("UC") and len(handle) == 24:
                return handle, "already a stable channel id"
            key = os.getenv("YOUTUBE_API_KEY", "").strip()
            if not key:
                return None, "YOUTUBE_API_KEY missing"
            base = "https://www.googleapis.com/youtube/v3/channels?"
            payload = _json(base + urllib.parse.urlencode(
                {"part": "id", "forHandle": handle.lstrip("@"), "key": key}))
            items = payload.get("items") or []
            value = items[0].get("id") if len(items) == 1 else None
        elif platform == "x":
            token = os.getenv("X_BEARER_TOKEN", "").strip()
            if not token:
                return None, "X_BEARER_TOKEN missing"
            payload = _json(
                "https://api.x.com/2/users/by/username/"
                + urllib.parse.quote(handle.lstrip("@")),
                {"Authorization": f"Bearer {token}"},
            )
            value = (payload.get("data") or {}).get("id")
        else:
            return None, "no stable-id resolver for this platform"
        return (str(value), "OK") if value else (None, "provider returned no unique id")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def resolve_telegram(handles: List[str], session: Path) -> Dict[str, Tuple[Optional[str], str]]:
    """Resolve public Telegram names through the existing authorised session."""
    results = {handle: (None, "Telegram credentials/session unavailable") for handle in handles}
    if not handles:
        return results
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    if not api_id or not api_hash or not session.with_suffix(".session").exists():
        return results
    try:
        from telethon import TelegramClient
    except ImportError:
        return {handle: (None, "Telethon dependency unavailable") for handle in handles}

    client = TelegramClient(str(session), int(api_id), api_hash, receive_updates=False)
    try:
        await client.connect()
        if not await client.is_user_authorized():
            return {handle: (None, "Telegram session is not authorised") for handle in handles}
        for handle in handles:
            try:
                entity = await client.get_entity(handle)
                value = getattr(entity, "id", None)
                results[handle] = ((str(value), "OK") if value is not None
                                   else (None, "provider returned no id"))
            except Exception as exc:  # Telethon has provider-version-specific errors.
                results[handle] = (None, f"{type(exc).__name__}: {exc}")
    finally:
        await client.disconnect()
    return results


async def resolve_document(document: Dict[str, Any], session: Path) -> Dict[str, Any]:
    entities = list(document.get("entities") or [])
    telegram_handles = sorted({
        str(handle)
        for entity in entities
        for handle in ((entity.get("metadata") or {}).get("published_handles") or {}).get(
            "telegram", [])
    })
    telegram = await resolve_telegram(telegram_handles, session)
    report: Dict[str, Any] = {"resolved": 0, "unresolved": [], "conflicts": []}
    proposed: List[Tuple[Dict[str, Any], str, str, str]] = []

    for entity in entities:
        published = ((entity.get("metadata") or {}).get("published_handles") or {})
        for platform, handles in published.items():
            for handle in handles or []:
                value, detail = (telegram.get(str(handle), (None, "not resolved"))
                                 if platform == "telegram"
                                 else resolve_public(str(platform), str(handle)))
                if value:
                    proposed.append((entity, str(platform), str(value), str(handle)))
                else:
                    report["unresolved"].append({
                        "entity_id": entity.get("entity_id"), "platform": platform,
                        "handle": handle, "detail": detail,
                    })

    owners: Dict[Tuple[str, str], List[str]] = {}
    for entity, platform, value, _ in proposed:
        owners.setdefault((platform, value), []).append(str(entity.get("entity_id")))
    conflicts = {key: values for key, values in owners.items() if len(set(values)) > 1}

    for entity, platform, value, handle in proposed:
        if (platform, value) in conflicts:
            report["conflicts"].append({
                "platform": platform, "account_id": value,
                "entities": sorted(set(conflicts[(platform, value)])),
            })
            continue
        accounts = entity.setdefault("accounts", {})
        bucket = accounts.setdefault(platform, [])
        bucket = [str(item) for item in (bucket or [])]
        if value not in bucket:
            bucket.append(value)
            report["resolved"] += 1
        accounts[platform] = sorted(set(bucket))
        resolutions = entity.setdefault("metadata", {}).setdefault("resolved_handles", {})
        resolutions.setdefault(platform, {})[handle] = value

    # De-duplicate a conflict observed once per proposed row.
    report["conflicts"] = list({
        (item["platform"], item["account_id"]): item
        for item in report["conflicts"]
    }.values())
    report["entities"] = len(entities)
    report["status"] = "OK" if report["resolved"] else "DATA_BLOCKED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="config/entities.verified.yaml")
    parser.add_argument("--out", default="")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--telegram-session", default="data/telegram/collector")
    args = parser.parse_args()
    source = Path(args.input)
    document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    report = asyncio.run(resolve_document(document, Path(args.telegram_session)))

    destination = source if args.in_place else (Path(args.out) if args.out else None)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
