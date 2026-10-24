"""Automatic, bounded admission of public onion links into the existing mesh."""
from __future__ import annotations

import hashlib
import os
import re
from urllib.parse import urlsplit

from src.collectors.onion import PROXY_ENV, onion_url
from src.collectors.transports import TransportError

COMPONENT = "public_onion_discovery_v1"
PATTERN = re.compile(r"(?<![a-z0-9])(?:https?://)?[a-z2-7]{56}\.onion(?:/[^\s<>\"']*)?", re.I)


def candidate_url(text):
    text = text.rstrip(".,);]")
    return onion_url(text if text.lower().startswith(("http://", "https://")) else "http://" + text)


def candidate_urls(event):
    metadata = getattr(event, "metadata", {}) or {}
    material = " ".join([getattr(event, "text", ""), *getattr(event, "urls", ()),
                         *metadata.get("discovered_onion_links", [])])
    result = []
    for match in PATTERN.findall(material):
        try:
            url = candidate_url(match)
        except TransportError:
            continue
        if url not in result:
            result.append(url)
        if len(result) == 128:
            break
    return result


def _register(desk, entry):
    from src.collectors.registry import SourceDeclaration, build_sources
    from src.collectors.transports import build_transports
    source_id = entry["source_id"]
    if source_id in desk.transports:
        return True
    declaration = SourceDeclaration(source_id, "onion", tier=3,
        poll_interval_seconds=120, poll_timeout_seconds=45,
        options={"url": entry["url"], "source_ref": entry["source_ref"],
                 "max_pages": 128, "max_depth": 6, "pages_per_poll": 1})
    transports, report, _ = build_transports([declaration], desk.http_client)
    if source_id not in transports:
        return False
    sources, _ = build_sources([declaration], transports)
    if not sources:
        return False
    desk.transports.update(transports)
    desk.source_fetchers.update(transports)
    desk.source_mesh.add(sources[0])
    return True


def restore(desk):
    engine = getattr(desk, "unknown_engine", None)
    if not engine or not os.getenv(PROXY_ENV) or not getattr(desk, "global_config", {}).get("onion_auto_discovery", False):
        return 0
    state = engine.framework.get_research_component(COMPONENT) or {}
    restored = 0
    maximum = max(1, min(32, int(getattr(desk, "global_config", {}).get("onion_max_sources", 16))))
    for entry in list(state.get("admitted", {}).values())[:maximum]:
        restored += int(_register(desk, entry))
    return restored


def _admit_pending(desk, state):
    if not os.getenv(PROXY_ENV):
        return False
    maximum = max(1, min(32, int(getattr(desk, "global_config", {}).get("onion_max_sources", 16))))
    active = sum(getattr(transport, "kind", None) == "onion" for transport in desk.transports.values())
    changed = False
    for identity, entry in list(state["pending"].items())[:2]:
        if active >= maximum:
            break
        if _register(desk, entry):
            state["admitted"][identity] = state["pending"].pop(identity)
            active += 1
            changed = True
    return changed


def drain(desk):
    engine = getattr(desk, "unknown_engine", None)
    if not engine or not getattr(desk, "global_config", {}).get("onion_auto_discovery", False):
        return
    state = engine.framework.get_research_component(COMPONENT)
    if state and _admit_pending(desk, state):
        engine.framework.checkpoint_research_component(COMPONENT, state)


def discover(desk, event, canonical_job_id):
    engine = getattr(desk, "unknown_engine", None)
    if not engine or not canonical_job_id or not getattr(desk, "global_config", {}).get("onion_auto_discovery", False):
        return
    state = engine.framework.get_research_component(COMPONENT) or {"admitted": {}, "pending": {}, "capacity_debt": 0}
    admitted, pending = state["admitted"], state["pending"]
    changed = False
    for url in candidate_urls(event):
        host = urlsplit(url).hostname
        identity = hashlib.sha256(host.encode()).hexdigest()
        if identity in admitted or identity in pending:
            continue
        if len(pending) >= 512:
            state["capacity_debt"] += 1
            changed = True
            continue
        pending[identity] = {"source_id": "onion:discovered-" + identity[:20], "url": url,
            "source_ref": "frontier_job:" + canonical_job_id,
            "discovered_at": event.observed_at, "identity_status": "UNVERIFIED_PUBLIC_LINK"}
        changed = True
    changed = _admit_pending(desk, state) or changed
    if changed:
        engine.framework.checkpoint_research_component(COMPONENT, state)
