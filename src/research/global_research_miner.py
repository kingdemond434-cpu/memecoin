"""Hourly public-source research miner.

Discovery is deliberately separated from promotion: new public mechanisms are
registered as hypotheses, then marked DATA_BLOCKED until chronological PIT tests
provide evidence. Source popularity never grants trading authority.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import numpy as np

from src.strategies.champion_challenger import ChampionChallengerFramework, HypothesisSpec

logger = logging.getLogger(__name__)


@dataclass
class ResearchLead:
    source_type: str
    title: str
    url: str
    summary: str
    language: str
    discovered_at: float = field(default_factory=time.time)
    mechanism: Optional[str] = None
    candidate_features: List[str] = field(default_factory=list)
    status: str = "DISCOVERED"
    license_spdx: Optional[str] = None
    stars: int = 0
    updated_at: str = ""
    source_quality: float = 0.0


class GlobalResearchMiner:
    RSS_FEEDS = [
        ("https://solana.com/rss.xml", "en", "solana"),
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "en", "coindesk"),
        ("https://blog.chainalysis.com/feed/", "en", "chainalysis"),
        ("https://www.theblock.co/rss.xml", "en", "theblock"),
        ("https://www.chaincatcher.com/rss/clist", "zh-cn", "chaincatcher"),
        ("https://rss.odaily.news/rss/newsflash", "zh-cn", "odaily"),
        ("https://rss.odaily.news/rss/post", "zh-cn", "odaily"),
        ("https://www.panewslab.com/rss.xml?lang=zh&type=NORMAL%2CNEWS", "zh-cn", "panews"),
        ("https://www.panewslab.com/rss.xml?lang=ja&type=NORMAL%2CNEWS", "ja", "panews"),
        ("https://www.panewslab.com/rss.xml?lang=ko&type=NORMAL%2CNEWS", "ko", "panews"),
    ]
    QUERIES = [
        ("solana pump fun sniper trading", "en"),
        ("raydium solana mev execution", "en"),
        ("solana wallet copy trading", "en"),
        ("pump fun same slot bundle detector", "en"),
        ("solana token rug pull detector", "en"),
        ("solana bonding curve trading", "en"),
        ("kelly criterion trading bot", "en"),
        ("optimal stopping trailing stop trading", "en"),
        ("hawkes order flow trading", "en"),
        ("ソラナ ミームコイン ボット", "ja"),
        ("솔라나 밈코인 봇", "ko"),
        ("солана мемкоин бот", "ru"),
        ("روبوت سولانا ميم", "ar"),
        ("bot memecoin solana", "pt"),
        ("bot memecoin solana", "es"),
    ]

    FEATURE_PATTERNS = {
        "wallet_copy_policy": (["copy", "wallet", "smart money", "聪明钱", "钱包", "跟单"], ["wallet_lead_time", "wallet_independence", "copy_crowding"]),
        "bundle_detection": (["bundle", "jito", "mev", "捆绑", "套利"], ["bundle_concentration", "same_slot_buyers", "independent_funding"]),
        "curve_velocity": (["bonding curve", "pump", "velocity", "联合曲线", "绑定曲线", "发射"], ["buy_velocity", "buy_acceleration", "curve_progress"]),
        "liquidity_execution": (["liquidity", "slippage", "route", "流动性", "滑点", "路由"], ["liquidity_usd", "price_impact", "route_availability"]),
        "social_propagation": (["telegram", "twitter", "social", "narrative", "电报", "推特", "叙事"], ["social_velocity", "cross_platform", "source_lead_time"]),
        "risk_constrained_kelly": (["kelly", "geometric growth", "log utility"], ["tail_probabilities", "drawdown_budget", "calibration_width"]),
        "optimal_stopping": (["optimal stopping", "trailing stop", "take profit", "止盈", "止损", "追踪止损"], ["high_water_mark", "continuation_probability", "hazard_rate"]),
        "survival_hazard": (["survival", "hazard", "rug pull", "风险率", "砸盘", "貔貅盘"], ["liquidity_change", "sell_acceleration", "route_degradation"]),
        "order_flow_point_process": (["hawkes", "order flow", "self exciting", "订单流", "买盘", "卖盘"], ["buy_intensity", "sell_intensity", "intensity_acceleration"]),
        "probability_calibration": (["conformal", "calibration", "brier"], ["calibration_error", "interval_width", "regime_coverage"]),
        "public_coordination": (["same slot", "shared funder", "bundle detector"], ["same_slot_buyers", "shared_funder_fraction", "buy_size_similarity"]),
    }

    def __init__(self, framework: ChampionChallengerFramework, interval_seconds: int = 3600,
                 ledger_path: str = "data/research/public_leads.jsonl", *,
                 unknown_engine: Any = None, chain_scout: Any = None,
                 evm_observers: Any = (), native_event_sink: Any = None,
                 observation_configuration: Any = (), native_probes: Any = (),
                 native_probe_sink: Any = None, native_probe_configuration: Any = ()):
        self.framework = framework
        self.interval_seconds = interval_seconds
        self.leads: List[ResearchLead] = []
        self._seen_urls: Set[str] = set()
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.data_status: Dict[str, str] = {}
        self._github_token = ""
        self.ledger_path = Path(ledger_path)
        self._query_cursor = 0
        self.unknown_engine = unknown_engine
        self.chain_scout = chain_scout
        self.evm_observers = list(evm_observers)
        self.native_event_sink = native_event_sink
        self.observation_configuration = list(observation_configuration)
        self._evm_cursor = 0
        self.native_probes = list(native_probes)
        self.native_probe_sink = native_probe_sink
        self.native_probe_configuration = list(native_probe_configuration)
        self._native_probe_cursor = 0
        self.observation_status: Dict[str, Any] = {"status": "NOT_RUN", "observation_only": True}
        self.conversion_status = {"passes": 0, "processed": 0, "recent_results": []}

    async def start(self):
        await self._load_ledger()
        self._github_token = os.getenv("GITHUB_TOKEN", "").strip()
        headers = {"User-Agent": "memecoin-quant-public-research/1.0"}
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
        )
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None

    async def _loop(self):
        next_collection = 0.0
        while self._running:
            try:
                if time.monotonic() >= next_collection:
                    await self.run_once()
                    next_collection = time.monotonic() + self.interval_seconds
                else:
                    await self._convert_pending()
            except Exception as exc:
                logger.error("Global research miner failed: %s", exc)
            await asyncio.sleep(min(10, self.interval_seconds))

    async def _convert_pending(self):
        if self.unknown_engine is not None:
            run = getattr(self.unknown_engine, "run_async_pass", None)
            if run is not None:
                result = await run()
            else:
                result = self.unknown_engine.run_pass()
            self.conversion_status["passes"] += 1
            if isinstance(result, dict):
                self.conversion_status["processed"] += result.get("processed", 0)
                self.conversion_status["recent_results"] = (self.conversion_status["recent_results"] + result.get("results", []))[-64:]
            return result

    async def run_once(self):
        # Deterministic public/native collection is independent of proposal
        # providers and uses this existing research pass, never a new task.
        await self._run_observation_pass()
        await self._convert_pending()
        # Anonymous GitHub search permits only 10 requests/minute. Rotate eight
        # queries per hourly cycle so a bad/expired token never exhausts the
        # fallback budget while all languages are still covered over time.
        count = min(8, len(self.QUERIES))
        selected = [self.QUERIES[(self._query_cursor + i) % len(self.QUERIES)] for i in range(count)]
        self._query_cursor = (self._query_cursor + count) % len(self.QUERIES)
        for query, language in selected:
            await self._mine_github(query, language)
            await self._convert_pending()
        await self._mine_arxiv()
        for url, language, source in self.RSS_FEEDS:
            await self._mine_rss(url, language, source)
            await self._convert_pending()
        if self.unknown_engine is not None:
            await self.unknown_engine.run_proposal_pass()

    async def _run_observation_pass(self):
        result: Dict[str, Any] = {"observation_only": True, "execution_eligible": False,
                                  "configuration": list(self.observation_configuration)}
        if self.chain_scout is None:
            result["chain_scout"] = {"status": "DATA_BLOCKED", "reason": "offline or no shared HTTP client"}
        else:
            try:
                result["chain_scout"] = await self.chain_scout.run_once()
            except Exception as exc:
                result["chain_scout"] = {"status": "DATA_BLOCKED", "failure_type": type(exc).__name__}
        observations = []
        # More configured chains remain registered for later passes. A
        # bounded rotation changes coverage without creating a scheduler.
        count = min(4, len(self.evm_observers))
        for offset in range(count):
            adapter = self.evm_observers[(self._evm_cursor + offset) % len(self.evm_observers)]
            if self.native_event_sink is None:
                observations.append({"status": "DATA_BLOCKED", "reason": "no canonical native-event sink"})
                continue
            try:
                observations.append(await adapter.poll_once(self.native_event_sink))
            except Exception as exc:
                observations.append({"status": "DATA_BLOCKED", "failure_type": type(exc).__name__})
        if self.evm_observers:
            self._evm_cursor = (self._evm_cursor + count) % len(self.evm_observers)
        result["evm"] = observations or [{"status": "NOT_CONFIGURED", "observations": None}]
        result["evm_deferred_this_pass"] = max(0, len(self.evm_observers) - count)
        native = []
        for offset in range(min(2, len(self.native_probes))):
            adapter = self.native_probes[(self._native_probe_cursor + offset) % len(self.native_probes)]
            try:
                receipt = await adapter.probe()
                if self.native_probe_sink is None:
                    raise ValueError("canonical native probe sink unavailable")
                reference = self.native_probe_sink(receipt)
                if not reference:
                    raise ValueError("canonical native probe receipt not retained")
                native.append({"chain": adapter.chain, "status": receipt["status"], "evidence_ref": reference,
                    "execution_eligible": False})
            except Exception as exc:
                native.append({"chain": adapter.chain, "status": "DATA_BLOCKED", "failure_type": type(exc).__name__})
        if self.native_probes:
            self._native_probe_cursor = (self._native_probe_cursor + 2) % len(self.native_probes)
        result["native_quote_probes"] = native
        result["native_quote_configuration"] = self.native_probe_configuration
        self.observation_status = result
        return result

    async def _mine_github(self, query: str, language: str):
        try:
            params = {"q": query, "sort": "updated", "order": "desc", "per_page": 10}
            headers = {"Authorization": f"Bearer {self._github_token}"} if self._github_token else {}
            fallback_used = False
            async with self._session.get("https://api.github.com/search/repositories", params=params,
                                         headers=headers) as resp:
                if resp.status in (401, 403) and self._github_token:
                    async with self._session.get("https://api.github.com/search/repositories",
                                                 params=params) as fallback:
                        if fallback.status != 200:
                            self.data_status["github"] = f"DATA_BLOCKED: HTTP {fallback.status}"
                            return
                        payload = await fallback.json()
                    fallback_used = True
                elif resp.status != 200:
                    self.data_status["github"] = f"DATA_BLOCKED: HTTP {resp.status}"
                    return
                else:
                    payload = await resp.json()
            for item in payload.get("items", []):
                await self._register_lead(ResearchLead(
                    source_type="github",
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    summary=item.get("description") or "",
                    language=language,
                    license_spdx=(item.get("license") or {}).get("spdx_id"),
                    stars=int(item.get("stargazers_count", 0) or 0),
                    updated_at=item.get("pushed_at", ""),
                    source_quality=min(1.0, 0.25 + np.log1p(int(item.get("stargazers_count", 0) or 0)) / 10),
                ))
            self.data_status["github"] = "OK_FALLBACK_UNAUTHENTICATED" if fallback_used else "OK"
        except Exception as exc:
            self.data_status["github"] = f"DATA_BLOCKED: {exc}"

    async def _mine_arxiv(self):
        try:
            async with self._session.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": 'all:"Solana" OR all:"meme coin"', "start": 0, "max_results": 20, "sortBy": "submittedDate"},
            ) as resp:
                if resp.status != 200:
                    self.data_status["arxiv"] = f"DATA_BLOCKED: HTTP {resp.status}"
                    return
                raw = await resp.text()
            root = ET.fromstring(raw)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", namespace):
                await self._register_lead(ResearchLead(
                    source_type="paper",
                    title=(entry.findtext("atom:title", "", namespace) or "").strip(),
                    url=(entry.findtext("atom:id", "", namespace) or "").strip(),
                    summary=(entry.findtext("atom:summary", "", namespace) or "").strip(),
                    language="en",
                    license_spdx="ARXIV_ABSTRACT_ONLY",
                    source_quality=0.75,
                ))
            self.data_status["arxiv"] = "OK"
        except Exception as exc:
            self.data_status["arxiv"] = f"DATA_BLOCKED: {exc}"

    async def _mine_rss(self, url: str, language: str, source: str):
        """Ingest publisher-provided RSS summaries as research-only leads."""
        status_key = f"rss:{source}:{language}"
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    self.data_status[status_key] = f"DATA_BLOCKED: HTTP {resp.status}"
                    return
                raw = await resp.text()
            root = ET.fromstring(raw)
            for item in root.findall(".//item")[:100]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or item.findtext("guid") or "").strip()
                summary = (item.findtext("description") or "").strip()
                if not title or not link:
                    continue
                await self._register_lead(ResearchLead(
                    source_type="publisher_rss", title=title, url=link,
                    summary=summary[:4_000], language=language,
                    license_spdx="RSS_SUMMARY_ONLY", source_quality=0.70,
                ))
            self.data_status[status_key] = "OK"
        except (aiohttp.ClientError, asyncio.TimeoutError, ET.ParseError, ValueError) as exc:
            self.data_status[status_key] = f"DATA_BLOCKED: {exc}"

    async def _load_ledger(self):
        if not self.ledger_path.exists():
            return
        try:
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    await self._register_lead(ResearchLead(**json.loads(line)), persist=False)
            self.data_status["research_ledger"] = "OK"
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            self.data_status["research_ledger"] = f"DATA_BLOCKED: {exc}"

    async def _register_lead(self, lead: ResearchLead, *, persist: bool = True):
        if not lead.url or lead.url in self._seen_urls:
            return
        self._seen_urls.add(lead.url)
        text = f"{lead.title} {lead.summary}".lower()
        for mechanism, (keywords, features) in self.FEATURE_PATTERNS.items():
            if any(keyword in text for keyword in keywords):
                lead.mechanism = mechanism
                lead.candidate_features = list(features)
                break
        if self.unknown_engine is not None:
            self.unknown_engine.ingest({
                "source_id": lead.source_type, "source_url": lead.url,
                "rights": "PUBLIC", "event_at": lead.discovered_at,
                "available_at": lead.discovered_at, "observed_at": lead.discovered_at,
                "region_language": lead.language, "title": lead.title,
                "summary": lead.summary, "mechanism": lead.mechanism,
                "candidate_features": list(lead.candidate_features),
                "research_partition": "training",
                "license_spdx": lead.license_spdx, "origin": "PUBLIC_SOURCE_OBSERVATION"})
        self.leads.append(lead)
        if len(self.leads) > 10_000:
            self.leads = self.leads[-5_000:]
        if not lead.mechanism:
            lead.status = "DATA_BLOCKED"
            self._persist_lead(lead, persist)
            return
        if lead.source_type == "github" and not lead.license_spdx:
            lead.status = "DATA_BLOCKED_LICENSE"
            self._persist_lead(lead, persist)
            return
        hypothesis_id = hashlib.sha256(f"{lead.url}:{lead.mechanism}".encode()).hexdigest()[:20]
        hypothesis = HypothesisSpec(
            hypothesis_id=hypothesis_id,
            mechanism=lead.mechanism,
            target="net_elogw",
            features=lead.candidate_features,
            feature_hash=hashlib.sha256("|".join(sorted(lead.candidate_features)).encode()).hexdigest()[:16],
            model_type="chronological_oos_candidate",
            model_params={},
            training_window="expanding_point_in_time",
            threshold=0.0,
            sizing_rule={"authority": "none_until_promoted"},
            exit_rule={"authority": "none_until_promoted"},
            execution_policy={"mode": "shadow"},
            fakeability={feature: 0.5 for feature in lead.candidate_features},
            cost_model={"fees_bps": 30, "slippage": "observed"},
            falsifier="chronological OOS net E[log W] <= 0 after costs",
            kill_thesis="forward shadow decay or adversarial spoofability",
            source_provenance=lead.url,
            trial_family=lead.mechanism,
            created_at=lead.discovered_at,
        )
        result = self.framework.submit_hypothesis(hypothesis)
        if result == "accepted":
            self.framework.mark_data_blocked(hypothesis_id, "awaiting PIT feature coverage and chronological OOS evidence")
        lead.status = "DATA_BLOCKED"
        self._persist_lead(lead, persist)

    def _persist_lead(self, lead: ResearchLead, persist: bool):
        if not persist:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(lead), separators=(",", ":")) + "\n")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "leads": len(self.leads),
            "mechanisms": len([lead for lead in self.leads if lead.mechanism]),
            "languages": sorted({lead.language for lead in self.leads}),
            "data_status": dict(self.data_status),
            "multi_chain_observation": dict(self.observation_status),
            "conversion": dict(self.conversion_status),
            "license_blocked": sum(lead.status == "DATA_BLOCKED_LICENSE" for lead in self.leads),
            "unknown_opportunity_engine": (self.unknown_engine.report()
                                           if self.unknown_engine is not None
                                           else {"status": "NOT_CONFIGURED", "authority": "none"}),
            "top_sources": [
                {"title": lead.title, "url": lead.url, "mechanism": lead.mechanism,
                 "quality": lead.source_quality, "license": lead.license_spdx, "status": lead.status}
                for lead in sorted(self.leads, key=lambda item: item.source_quality, reverse=True)[:20]
            ],
        }
