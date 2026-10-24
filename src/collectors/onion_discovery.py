"""Continuous, provenance-bound admission into a disk-backed public frontier."""
from __future__ import annotations

import os
import re
from pathlib import Path

from src.collectors.onion import PROXY_ENV, onion_url
from src.collectors.transports import TransportError

# Legacy archive remains readable for migration and historical capacity debt.
COMPONENT = "public_onion_discovery_v1"
PATTERN = re.compile(r"(?<![a-z0-9])(?:https?://)?[a-z2-7]{56}\.onion(?:/[^\s<>\"']*)?", re.I)


def candidate_url(text):
    text = text.rstrip(".,);]")
    return onion_url(text if text.lower().startswith(("http://", "https://")) else "http://" + text)


def candidate_urls(event):
    metadata = getattr(event, "metadata", {}) or {}
    material = " ".join([getattr(event, "text", ""), *getattr(event, "urls", ()),
                         metadata.get('url') or '', *metadata.get("discovered_onion_links", [])])
    result = {}
    for match in PATTERN.finditer(material):
        try:
            result[candidate_url(match.group())] = None
        except TransportError:
            continue
    return list(result)


def _frontier(desk):
    config = getattr(desk, "global_config", {})
    if not getattr(desk, "unknown_engine", None) or not config.get("onion_auto_discovery", False):
        return None
    current = getattr(desk, "onion_frontier", None)
    if current is None:
        from src.collectors.onion_frontier import OnionFrontierTransport
        current = OnionFrontierTransport(config.get("onion_frontier_path") or
            Path(config.get("ops_state_dir", "data/state")) / "public_onion_frontier.sqlite")
        current.migrate(desk.unknown_engine.framework.get_research_component(COMPONENT) or {},
                        config.get('onion_legacy_state_dir', 'data/research/onion'))
        desk.onion_frontier = current
    if os.getenv(PROXY_ENV) and current.source_id not in desk.transports:
        from src.collectors.adapters import onion_source
        source = onion_source(current.source_id, current)
        source.poll_interval_seconds = 2
        source.poll_timeout_seconds = 45
        desk.transports[current.source_id] = current
        desk.source_fetchers[current.source_id] = current
        desk.source_mesh.add(source)
    return current


def restore(desk):
    frontier = _frontier(desk)
    return frontier.report()["discovered_hosts"] if frontier and os.getenv(PROXY_ENV) else 0


def drain(desk):
    _frontier(desk)


def discover(desk, event, canonical_job_id):
    if not canonical_job_id:
        return
    frontier = _frontier(desk)
    if frontier is None:
        return
    if event.source_id == frontier.source_id:
        # Child admission is committed with canonical acknowledgement by the
        # transport, retaining the exact parent URL and crawl depth.
        return
    frontier.enqueue(candidate_urls(event), "frontier_job:" + canonical_job_id, event.observed_at)
