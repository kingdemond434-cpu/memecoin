import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import hashlib
from pathlib import Path

import aiohttp
import numpy as np

try:
    from telethon import TelegramClient, events
except ImportError:  # Dependency absence remains an explicit data blocker.
    TelegramClient = None
    events = None

logger = logging.getLogger(__name__)


class SocialPlatform(Enum):
    X = "x"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    BILIBILI = "bilibili"
    ZHIHU = "zhihu"
    DISCORD = "discord"
    GITHUB = "github"


@dataclass
class SocialAccount:
    platform: SocialPlatform
    handle: str
    account_id: str
    display_name: str
    followers: int = 0
    following: int = 0
    created_at: Optional[float] = None
    verified: bool = False
    
    call_history: List[Dict] = field(default_factory=list)
    total_calls: int = 0
    successful_calls: int = 0
    avg_roi: float = 0.0
    median_entry_delay: float = 0.0
    rug_exposure: float = 0.0
    credibility_score: float = 0.5
    narrative_specialties: Set[str] = field(default_factory=set)
    language: str = "en"
    region: str = "global"
    last_active: float = field(default_factory=time.time)
    linked_wallets: Set[str] = field(default_factory=set)


@dataclass
class SocialMention:
    platform: SocialPlatform
    account: SocialAccount
    token: str
    content: str
    timestamp: float
    engagement: Dict[str, int] = field(default_factory=dict)
    url: str = ""
    token_first_mention: bool = False
    mention_index: int = 0
    chain_activity_before: bool = False
    chain_activity_after: bool = False
    causality_score: float = 0.0


@dataclass
class NarrativeCluster:
    narrative_id: str
    keywords: List[str]
    tokens: Set[str] = field(default_factory=set)
    accounts: Set[str] = field(default_factory=set)
    start_time: float = field(default_factory=time.time)
    peak_time: Optional[float] = None
    velocity: float = 0.0
    coherence: float = 0.0
    cross_platform: bool = False


class SocialIntelligenceEngine:
    YOUTUBE_QUERIES = ("solana memecoin", "pump fun solana", "solana rug pull")

    def __init__(
        self,
        chain_config,
        rpc,
        genealogy,
        wallet_intel,
        api_keys: Dict[str, str],
        recalc_interval_hours: int = 1
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.api_keys = api_keys
        self.recalc_interval = recalc_interval_hours * 3600
        
        self.accounts: Dict[str, SocialAccount] = {}
        self.mentions: deque = deque(maxlen=100000)
        self.narratives: Dict[str, NarrativeCluster] = {}
        self.token_mentions: Dict[str, List[SocialMention]] = defaultdict(list)
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._hunter_task: Optional[asyncio.Task] = None
        self._watcher_task: Optional[asyncio.Task] = None
        self._narrative_task: Optional[asyncio.Task] = None
        self._recalc_task: Optional[asyncio.Task] = None
        self._telegram_client = None
        self._telegram_handles: Set[str] = set()
        self._telegram_event_handler = None
        self._telegram_poll_successes: Set[str] = set()
        self._telegram_poll_failures: Dict[str, str] = {}
        self._account_fetched_at: Dict[str, float] = {}
        self._seen_social_items: Set[str] = set()
        self._reddit_access_token = ""
        self._reddit_token_expires_at = 0.0
        
        self._known_contracts: Set[str] = set()
        self._mention_callbacks: List[Callable] = []
        self.data_status: Dict[str, str] = {}

    async def start(self, initial_accounts: List[Dict] = None):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50)
        )
        self._running = True
        await self._setup_telegram()
        
        if initial_accounts:
            for acc_data in initial_accounts:
                await self._add_account(acc_data)
        
        self._hunter_task = asyncio.create_task(self._hunter_loop())
        self._watcher_task = asyncio.create_task(self._watcher_loop())
        self._narrative_task = asyncio.create_task(self._narrative_loop())
        self._recalc_task = asyncio.create_task(self._recalc_loop())
        
        await self._initial_discovery()

    async def stop(self):
        self._running = False
        for task in [self._hunter_task, self._watcher_task, self._narrative_task, self._recalc_task]:
            if task:
                task.cancel()
        if self._session:
            await self._session.close()
        if self._telegram_client:
            await self._telegram_client.disconnect()
            self._telegram_client = None

    def on_mention(self, callback: Callable):
        self._mention_callbacks.append(callback)

    async def _initial_discovery(self):
        await self._discover_accounts_from_successful_wallets()
        await self._discover_accounts_from_recent_launches()
        await self._discover_youtube_sources()
        await self._recalculate_account_credibility()

    async def _hunter_loop(self):
        while self._running:
            try:
                await self._discover_accounts_from_successful_wallets()
                await self._discover_accounts_from_recent_launches()
                await self._discover_accounts_from_foreign_sources()
                await self._discover_accounts_from_competitors()
                await self._discover_youtube_sources()
                await self._recalculate_account_credibility()
            except Exception as e:
                logger.error(f"Social hunter error: {e}")
            await asyncio.sleep(self.recalc_interval)

    async def _watcher_loop(self):
        while self._running:
            try:
                await self._watch_known_accounts()
                await self._watch_token_mentions()
            except Exception as e:
                logger.error(f"Social watcher error: {e}")
            await asyncio.sleep(5)

    async def _narrative_loop(self):
        while self._running:
            try:
                await self._update_narratives()
            except Exception as e:
                logger.error(f"Narrative loop error: {e}")
            await asyncio.sleep(60)

    async def _recalc_loop(self):
        while self._running:
            try:
                await self._recalculate_account_credibility()
                await self._update_token_mention_causality()
            except Exception as e:
                logger.error(f"Social recalc error: {e}")
            await asyncio.sleep(300)

    async def _discover_accounts_from_successful_wallets(self):
        smart_wallets = self.wallet_intel.get_top_wallets(limit=100)
        for ws in smart_wallets:
            await self._find_linked_social(ws.wallet)

    async def _setup_telegram(self):
        api_id = str(self.api_keys.get("telegram_api_id", "")).strip()
        api_hash = str(self.api_keys.get("telegram_api_hash", "")).strip()
        if not api_id or not api_hash:
            self.data_status["telegram"] = "DATA_BLOCKED: TELEGRAM_API_ID/API_HASH missing"
            return
        if TelegramClient is None:
            self.data_status["telegram"] = "DATA_BLOCKED: Telethon dependency unavailable"
            return
        try:
            session_dir = Path("data/telegram")
            session_dir.mkdir(parents=True, exist_ok=True)
            self._telegram_client = TelegramClient(
                str(session_dir / "collector"), int(api_id), api_hash,
                receive_updates=True,
            )
            await self._telegram_client.connect()
            if not await self._telegram_client.is_user_authorized():
                self.data_status["telegram"] = "DATA_BLOCKED: interactive Telegram authorization required"
                await self._telegram_client.disconnect()
                self._telegram_client = None
                return
            channels = [
                value.strip().lstrip("@").replace("https://t.me/", "")
                for value in str(self.api_keys.get("telegram_channels", "")).split(",")
                if value.strip()
            ]
            for channel in channels:
                await self._add_account({
                    "platform": "telegram", "handle": channel, "account_id": channel,
                })
            self._telegram_handles = {channel.casefold() for channel in channels}
            if events is not None:
                self._telegram_event_handler = self._handle_telegram_event
                self._telegram_client.add_event_handler(
                    self._telegram_event_handler, events.NewMessage(incoming=True)
                )
            if not channels:
                self.data_status["telegram"] = "DATA_BLOCKED: TELEGRAM_CHANNELS empty"
            elif self._telegram_event_handler is not None:
                self.data_status["telegram"] = "OK_PUSH"
            else:
                self.data_status["telegram"] = "OK_POLLING"
        except Exception as exc:
            self.data_status["telegram"] = f"DATA_BLOCKED: {exc}"
            if self._telegram_client:
                await self._telegram_client.disconnect()
                self._telegram_client = None

    async def _discover_youtube_sources(self):
        api_key = str(self.api_keys.get("youtube", "")).strip()
        if not api_key or not self._session:
            self.data_status["youtube"] = "DATA_BLOCKED: YOUTUBE_API_KEY missing"
            return
        published_after = datetime.fromtimestamp(time.time() - 86400, timezone.utc).isoformat(timespec="seconds")
        try:
            items = []
            for query in self.YOUTUBE_QUERIES:
                async with self._session.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet", "type": "video", "order": "date", "maxResults": 10,
                        "publishedAfter": published_after, "q": query, "key": api_key,
                    },
                ) as resp:
                    if resp.status != 200:
                        self.data_status["youtube"] = f"DATA_BLOCKED: HTTP {resp.status}"
                        return
                    items.extend((await resp.json()).get("items", []))
            video_ids = [item.get("id", {}).get("videoId") for item in items if item.get("id", {}).get("videoId")]
            statistics = {}
            if video_ids:
                async with self._session.get(
                    "https://www.googleapis.com/youtube/v3/videos",
                    params={"part": "statistics", "id": ",".join(video_ids[:50]), "key": api_key},
                ) as resp:
                    if resp.status == 200:
                        statistics = {item.get("id"): item.get("statistics", {})
                                      for item in (await resp.json()).get("items", [])}
            for item in items:
                video_id = item.get("id", {}).get("videoId")
                snippet = item.get("snippet", {})
                channel_id = snippet.get("channelId")
                if not video_id or not channel_id:
                    continue
                dedupe = f"youtube:{video_id}"
                if dedupe in self._seen_social_items:
                    continue
                self._seen_social_items.add(dedupe)
                await self._add_account({
                    "platform": "youtube", "handle": channel_id, "account_id": channel_id,
                    "display_name": snippet.get("channelTitle", channel_id),
                })
                await self._process_youtube_video(
                    self.accounts[f"youtube:{channel_id}"], item, statistics.get(video_id, {}),
                )
            self.data_status["youtube"] = "OK"
        except Exception as exc:
            self.data_status["youtube"] = f"DATA_BLOCKED: {exc}"

    async def _process_youtube_video(self, account: SocialAccount, item: Dict, statistics: Dict):
        video_id = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})
        content = f"{snippet.get('title', '')}\n{snippet.get('description', '')}"
        timestamp = self._parse_twitter_time(snippet.get("publishedAt", ""))
        for token in self._extract_contracts(content):
            await self._process_mention(SocialMention(
                platform=SocialPlatform.YOUTUBE, account=account, token=token,
                content=content[:500], timestamp=timestamp,
                engagement={
                    "views": int(statistics.get("viewCount", 0) or 0),
                    "likes": int(statistics.get("likeCount", 0) or 0),
                    "comments": int(statistics.get("commentCount", 0) or 0),
                },
                url=f"https://www.youtube.com/watch?v={video_id}",
            ))

    async def _find_linked_social(self, wallet: str):
        # There is no verified Helius wallet-to-social endpoint. Treat this as
        # unavailable instead of fabricating identity links from display names.
        self.data_status["wallet_social_links"] = "DATA_BLOCKED: no verified public identity-link source"

    async def _discover_accounts_from_recent_launches(self):
        self.data_status["recent_launches"] = "OK: program_stream"

    async def _scan_token_social(self, token: str):
        if token in self._known_contracts:
            return
        self._known_contracts.add(token)
        
        for platform in [SocialPlatform.X, SocialPlatform.TELEGRAM, SocialPlatform.REDDIT]:
            await self._search_platform_for_token(platform, token)

    async def scan_token(self, token: str):
        """Search configured public sources for a stream-validated launch."""
        await self._scan_token_social(token)

    async def _search_platform_for_token(self, platform: SocialPlatform, token: str):
        if platform == SocialPlatform.X:
            await self._search_x_query(token)
        elif platform == SocialPlatform.REDDIT:
            await self._search_reddit(token)
        elif platform == SocialPlatform.TELEGRAM:
            self.data_status["telegram_search"] = "DATA_BLOCKED: Telegram API has no global public-channel search; configured channels are monitored"

    async def _search_x_query(self, query: str):
        bearer = self.api_keys.get("x_bearer", "")
        if not bearer:
            self.data_status["x"] = "DATA_BLOCKED: X_BEARER_TOKEN missing"
            return
        try:
            async with self._session.get(
                "https://api.twitter.com/2/tweets/search/recent",
                headers={"Authorization": f"Bearer {bearer}"},
                params={
                    "query": f"{query} -is:retweet",
                    "max_results": 25,
                    "tweet.fields": "created_at,public_metrics,lang,author_id",
                    "expansions": "author_id",
                    "user.fields": "username,name,public_metrics,created_at,verified",
                },
            ) as resp:
                if resp.status != 200:
                    self.data_status["x"] = f"DATA_BLOCKED: HTTP {resp.status}"
                    return
                payload = await resp.json()
            users = {item["id"]: item for item in payload.get("includes", {}).get("users", [])}
            for tweet in payload.get("data", []):
                user = users.get(tweet.get("author_id"), {})
                handle = user.get("username")
                if not handle:
                    continue
                await self._add_account({
                    "platform": "x", "handle": handle, "account_id": user.get("id", handle),
                    "display_name": user.get("name", handle), "followers": user.get("public_metrics", {}).get("followers_count", 0),
                    "verified": user.get("verified", False), "language": tweet.get("lang", "unknown"),
                })
                await self._process_x_tweet(self.accounts[f"x:{handle}"], tweet)
            self.data_status["x"] = "OK"
        except Exception as exc:
            self.data_status["x"] = f"DATA_BLOCKED: {exc}"

    async def _search_reddit(self, query: str):
        headers = await self._reddit_headers()
        if not headers:
            return
        try:
            async with self._session.get(
                "https://oauth.reddit.com/search",
                params={"q": query, "sort": "new", "limit": 25, "t": "week"},
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    self.data_status["reddit"] = f"DATA_BLOCKED: HTTP {resp.status}"
                    return
                payload = await resp.json()
            for child in payload.get("data", {}).get("children", []):
                post = child.get("data", {})
                handle = post.get("author")
                if not handle or handle == "[deleted]":
                    continue
                await self._add_account({"platform": "reddit", "handle": handle, "account_id": handle})
                await self._process_reddit_post(self.accounts[f"reddit:{handle}"], post)
            self.data_status["reddit"] = "OK"
        except Exception as exc:
            self.data_status["reddit"] = f"DATA_BLOCKED: {exc}"

    async def _reddit_headers(self) -> Optional[Dict[str, str]]:
        client_id = str(self.api_keys.get("reddit", "")).strip()
        client_secret = str(self.api_keys.get("reddit_secret", "")).strip()
        if not client_id or not client_secret:
            self.data_status["reddit"] = "DATA_BLOCKED: REDDIT_CLIENT_ID/SECRET missing or approval pending"
            return None
        if self._reddit_access_token and time.time() < self._reddit_token_expires_at - 60:
            return {
                "Authorization": f"bearer {self._reddit_access_token}",
                "User-Agent": "memecoin-shadow/1.0 read-only research collector",
            }
        try:
            async with self._session.post(
                "https://www.reddit.com/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=aiohttp.BasicAuth(client_id, client_secret),
                headers={"User-Agent": "memecoin-shadow/1.0 read-only research collector"},
            ) as resp:
                payload = await resp.json(content_type=None)
                if resp.status != 200 or not payload.get("access_token"):
                    self.data_status["reddit"] = f"DATA_BLOCKED: OAuth HTTP {resp.status}"
                    return None
            self._reddit_access_token = str(payload["access_token"])
            self._reddit_token_expires_at = time.time() + int(payload.get("expires_in", 3600))
            return {
                "Authorization": f"bearer {self._reddit_access_token}",
                "User-Agent": "memecoin-shadow/1.0 read-only research collector",
            }
        except Exception as exc:
            self.data_status["reddit"] = f"DATA_BLOCKED: OAuth {exc}"
            return None

    async def _discover_accounts_from_foreign_sources(self):
        terms = ["solana memecoin", "ソラナ ミームコイン", "솔라나 밈코인", "солана мемкоин", "عملة سولانا ميم"]
        for term in terms:
            await self._search_x_query(term)

    async def _discover_accounts_from_competitors(self):
        known_bots = [
            "solsniper", "phantom", "bonkbot", "unibot", "maestro",
            "trojan", "bullx", "banana", "shuriken"
        ]
        for bot in known_bots:
            await self._scan_competitor_followers(bot)

    async def _scan_competitor_followers(self, bot_name: str):
        bearer = self.api_keys.get("x_bearer", "")
        if not bearer:
            self.data_status["competitor_followers"] = "DATA_BLOCKED: X_BEARER_TOKEN missing"
            return
        headers = {"Authorization": f"Bearer {bearer}"}
        try:
            async with self._session.get(f"https://api.twitter.com/2/users/by/username/{bot_name}", headers=headers) as resp:
                if resp.status != 200:
                    return
                user_id = (await resp.json()).get("data", {}).get("id")
            if not user_id:
                return
            async with self._session.get(
                f"https://api.twitter.com/2/users/{user_id}/followers",
                headers=headers,
                params={"max_results": 100, "user.fields": "public_metrics,verified"},
            ) as resp:
                if resp.status != 200:
                    return
                payload = await resp.json()
            for user in payload.get("data", []):
                await self._add_account({
                    "platform": "x", "handle": user.get("username"), "account_id": user.get("id"),
                    "display_name": user.get("name", user.get("username")),
                    "followers": user.get("public_metrics", {}).get("followers_count", 0),
                    "verified": user.get("verified", False),
                })
            self.data_status["competitor_followers"] = "OK"
        except Exception as exc:
            self.data_status["competitor_followers"] = f"DATA_BLOCKED: {exc}"

    async def _add_account(self, data: Dict):
        if not data.get("platform") or not data.get("handle"):
            self.data_status["account_ingest"] = "DATA_BLOCKED: source account missing platform or handle"
            return
        key = f"{data['platform']}:{data['handle']}"
        if key in self.accounts:
            return
        
        platform = SocialPlatform(data["platform"]) if isinstance(data["platform"], str) else data["platform"]
        account = SocialAccount(
            platform=platform,
            handle=data["handle"],
            account_id=data.get("account_id", key),
            display_name=data.get("display_name", data["handle"]),
            language=data.get("language", "en"),
            region=data.get("region", "global"),
            followers=int(data.get("followers", 0) or 0),
            verified=bool(data.get("verified", False)),
        )
        if data.get("source_wallet"):
            account.linked_wallets.add(data["source_wallet"])
            self.wallet_intel.register_social_wallet(data["source_wallet"])
        self.accounts[key] = account

    async def _watch_known_accounts(self):
        top_accounts = sorted(
            [a for a in self.accounts.values() if a.credibility_score > 0.4],
            key=lambda x: x.credibility_score,
            reverse=True
        )[:200]
        
        for account in top_accounts:
            await self._fetch_recent_posts(account)

    async def _fetch_recent_posts(self, account: SocialAccount):
        key = f"{account.platform.value}:{account.handle}"
        if time.time() - self._account_fetched_at.get(key, 0) < 60:
            return
        self._account_fetched_at[key] = time.time()
        if account.platform == SocialPlatform.X:
            await self._fetch_x_posts(account)
        elif account.platform == SocialPlatform.TELEGRAM:
            await self._fetch_telegram_posts(account)
        elif account.platform == SocialPlatform.REDDIT:
            await self._fetch_reddit_posts(account)

    async def _fetch_x_posts(self, account: SocialAccount):
        try:
            bearer = self.api_keys.get("x_bearer", "")
            if not bearer:
                return
            
            async with self._session.get(
                f"https://api.twitter.com/2/users/by/username/{account.handle}",
                headers={"Authorization": f"Bearer {bearer}"}
            ) as resp:
                if resp.status != 200:
                    return
                user_data = await resp.json()
                user_id = user_data.get("data", {}).get("id")
                if not user_id:
                    return
            
            async with self._session.get(
                f"https://api.twitter.com/2/users/{user_id}/tweets",
                headers={"Authorization": f"Bearer {bearer}"},
                params={"max_results": 50, "tweet.fields": "created_at,public_metrics,context_annotations"}
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                
                for tweet in data.get("data", []):
                    await self._process_x_tweet(account, tweet)
                    
        except Exception as e:
            logger.debug(f"X fetch error for {account.handle}: {e}")

    async def _process_x_tweet(self, account: SocialAccount, tweet: Dict):
        content = tweet.get("text", "")
        tokens = self._extract_contracts(content)
        
        for token in tokens:
            mention = SocialMention(
                platform=SocialPlatform.X,
                account=account,
                token=token,
                content=content[:500],
                timestamp=self._parse_twitter_time(tweet.get("created_at", "")),
                engagement={
                    "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                    "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                    "replies": tweet.get("public_metrics", {}).get("reply_count", 0)
                },
                url=f"https://x.com/{account.handle}/status/{tweet.get('id')}"
            )
            
            await self._process_mention(mention)

    def _extract_contracts(self, text: str) -> List[str]:
        import re
        solana_pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        eth_pattern = r'\b0x[a-fA-F0-9]{40}\b'
        
        contracts = []
        contracts.extend(re.findall(solana_pattern, text))
        contracts.extend(re.findall(eth_pattern, text))
        return contracts

    def _parse_twitter_time(self, time_str: str) -> float:
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return time.time()

    async def _fetch_telegram_posts(self, account: SocialAccount):
        if not self._telegram_client:
            self.data_status["telegram"] = "DATA_BLOCKED: authorized Telegram session unavailable"
            return
        try:
            target = int(account.handle) if account.handle.lstrip("-").isdigit() else account.handle
            async for message in self._telegram_client.iter_messages(target, limit=100):
                await self._process_telegram_message(account, message)
            self._telegram_poll_successes.add(account.handle)
            self._telegram_poll_failures.pop(account.handle, None)
            self._update_telegram_poll_status()
        except Exception as exc:
            self._telegram_poll_failures[account.handle] = str(exc)
            self._update_telegram_poll_status()

    def _update_telegram_poll_status(self):
        """An invalid optional channel must not mask healthy Telegram ingestion."""
        failed = len(self._telegram_poll_failures)
        if failed and self._telegram_poll_successes:
            self.data_status["telegram"] = f"OK_PARTIAL: {failed} configured channels unavailable"
        elif failed:
            self.data_status["telegram"] = f"DATA_BLOCKED: {failed} configured channels unavailable"
        else:
            self.data_status["telegram"] = "OK"

    async def _handle_telegram_event(self, event):
        """Process configured bot/channel messages from Telegram's push stream."""
        try:
            chat = await event.get_chat()
            handle = str(getattr(chat, "username", "") or event.chat_id or "").lstrip("@")
            if not handle or handle.casefold() not in self._telegram_handles:
                return
            key = f"telegram:{handle}"
            account = self.accounts.get(key)
            if account is None:
                await self._add_account({
                    "platform": "telegram", "handle": handle, "account_id": str(event.chat_id),
                    "display_name": str(getattr(chat, "title", "") or getattr(chat, "first_name", "") or handle),
                })
                account = self.accounts.get(key)
            if account is not None:
                await self._process_telegram_message(account, event.message)
                self._telegram_poll_successes.add(handle)
                self._telegram_poll_failures.pop(handle, None)
                self._update_telegram_poll_status()
        except Exception as exc:
            logger.warning("Telegram push ingest failed; polling backfill remains active: %s", exc)
            self.data_status["telegram_push"] = f"DEGRADED: {exc}"

    async def _process_telegram_message(self, account: SocialAccount, message):
        message_id = getattr(message, "id", None)
        if message_id is None:
            return
        dedupe = f"telegram:{account.handle}:{message_id}"
        if dedupe in self._seen_social_items:
            return
        self._seen_social_items.add(dedupe)
        content = getattr(message, "message", "") or ""
        timestamp = getattr(message, "date", None)
        timestamp_value = timestamp.timestamp() if timestamp else time.time()
        for token_address in self._extract_contracts(content):
            await self._process_mention(SocialMention(
                platform=SocialPlatform.TELEGRAM, account=account, token=token_address,
                content=content[:500], timestamp=timestamp_value,
                engagement={
                    "views": int(getattr(message, "views", 0) or 0),
                    "forwards": int(getattr(message, "forwards", 0) or 0),
                    "replies": int(getattr(getattr(message, "replies", None), "replies", 0) or 0),
                },
                url=f"https://t.me/{account.handle}/{message_id}",
            ))

    async def _fetch_reddit_posts(self, account: SocialAccount):
        headers = await self._reddit_headers()
        if not headers:
            return
        try:
            async with self._session.get(
                f"https://oauth.reddit.com/user/{account.handle}/submitted",
                params={"limit": 50}, headers=headers,
            ) as resp:
                if resp.status != 200:
                    return
                payload = await resp.json()
            for child in payload.get("data", {}).get("children", []):
                await self._process_reddit_post(account, child.get("data", {}))
        except Exception as exc:
            self.data_status["reddit"] = f"DATA_BLOCKED: {exc}"

    async def _process_reddit_post(self, account: SocialAccount, post: Dict):
        content = f"{post.get('title', '')}\n{post.get('selftext', '')}"
        for token in self._extract_contracts(content):
            await self._process_mention(SocialMention(
                platform=SocialPlatform.REDDIT,
                account=account,
                token=token,
                content=content[:500],
                timestamp=float(post.get("created_utc", time.time())),
                engagement={"score": int(post.get("score", 0) or 0), "comments": int(post.get("num_comments", 0) or 0)},
                url=f"https://reddit.com{post.get('permalink', '')}",
            ))

    async def _watch_token_mentions(self):
        for token, mentions in list(self.token_mentions.items()):
            recent = [m for m in mentions if time.time() - m.timestamp < 3600]
            if len(recent) >= 3:
                await self._analyze_token_social_momentum(token, recent)

    async def _analyze_token_social_momentum(self, token: str, mentions: List[SocialMention]):
        velocities = []
        for i in range(1, len(mentions)):
            dt = mentions[i].timestamp - mentions[i-1].timestamp
            if dt > 0:
                velocities.append(1 / dt)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            acceleration = np.mean(np.diff(velocities)) if len(velocities) > 1 else 0
            
            account_scores = [m.account.credibility_score for m in mentions]
            avg_credibility = np.mean(account_scores) if account_scores else 0
            
            chain_before = sum(1 for m in mentions if m.chain_activity_before)
            chain_after = sum(1 for m in mentions if m.chain_activity_after)
            
            signal = {
                "token": token,
                "mention_count": len(mentions),
                "avg_velocity": avg_velocity,
                "acceleration": acceleration,
                "avg_credibility": avg_credibility,
                "chain_before_pct": chain_before / len(mentions) if mentions else 0,
                "chain_after_pct": chain_after / len(mentions) if mentions else 0,
                "platforms": list(set(m.platform.value for m in mentions)),
                "unique_accounts": len(set(m.account.handle for m in mentions)),
                "timestamp": time.time()
            }
            
            for callback in self._mention_callbacks:
                try:
                    await callback(signal)
                except Exception as e:
                    logger.error(f"Mention callback error: {e}")

    async def _process_mention(self, mention: SocialMention):
        token = mention.token
        self.token_mentions[token].append(mention)
        self.mentions.append(mention)
        mention.account.total_calls += 1
        mention.account.call_history.append({"token": token, "timestamp": mention.timestamp, "url": mention.url})
        
        is_first = len(self.token_mentions[token]) == 1
        mention.token_first_mention = is_first
        mention.mention_index = len(self.token_mentions[token]) - 1
        
        await self._check_chain_causality(mention)
        
        for callback in self._mention_callbacks:
            try:
                await callback({
                    "type": "new_mention",
                    "token": token,
                    "platform": mention.platform.value,
                    "account": mention.account.handle,
                    "credibility": mention.account.credibility_score,
                    "first_mention": is_first,
                    "content": mention.content[:200],
                    "timestamp": mention.timestamp
                })
            except Exception as e:
                logger.error(f"Mention callback error: {e}")

    async def _check_chain_causality(self, mention: SocialMention):
        token = mention.token
        mention_time = mention.timestamp
        
        try:
            largest = await self.rpc.request("getTokenLargestAccounts", [token, {"commitment": "confirmed"}])
            accounts = [item.get("address") for item in (largest or {}).get("value", []) if item.get("address")]
            parsed = await self.rpc.request(
                "getMultipleAccounts", [accounts, {"encoding": "jsonParsed", "commitment": "confirmed"}],
            ) if accounts else None
            for item in (parsed or {}).get("value", []):
                owner = (((item or {}).get("data") or {}).get("parsed") or {}).get("info", {}).get("owner")
                if owner and await self._get_wallet_txs_before(owner, mention_time, limit=10):
                    mention.chain_activity_before = True
                    break
        except Exception as exc:
            self.data_status["chain_causality"] = f"DATA_BLOCKED: {exc}"

    async def _get_wallet_txs_before(self, wallet: str, before_time: float, limit: int = 10) -> List:
        helius = self.api_keys.get("helius", "")
        if not helius:
            self.data_status["chain_causality"] = "DATA_BLOCKED: HELIUS_API_KEY missing"
            return []
        try:
            async with self._session.get(
                f"https://api.helius.xyz/v0/addresses/{wallet}/transactions",
                params={"api-key": helius, "limit": min(100, max(limit * 5, 20))},
            ) as resp:
                if resp.status != 200:
                    return []
                transactions = await resp.json()
            return [item for item in transactions if float(item.get("timestamp", 0) or 0) < before_time][:limit]
        except Exception:
            return []

    def record_token_outcome(self, token: str, outcome: Dict[str, Any]):
        multiple = float(outcome.get("max_multiple", 0) or 0)
        rugged = bool(outcome.get("rugged", False))
        for mention in self.token_mentions.get(token, []):
            account = mention.account
            account.successful_calls += int(multiple >= 2 and not rugged)
            n = max(account.total_calls, 1)
            account.avg_roi = (account.avg_roi * (n - 1) + (multiple - 1)) / n
            account.rug_exposure = (account.rug_exposure * (n - 1) + int(rugged)) / n

    async def _update_narratives(self):
        recent_mentions = [m for m in self.mentions if time.time() - m.timestamp < 3600]
        
        keyword_counts = defaultdict(int)
        for m in recent_mentions:
            words = self._extract_keywords(m.content)
            for w in words:
                keyword_counts[w] += 1
        
        trending = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for keyword, count in trending:
            if count < 3:
                continue
            
            narrative_id = hashlib.md5(keyword.encode()).hexdigest()[:12]
            if narrative_id not in self.narratives:
                self.narratives[narrative_id] = NarrativeCluster(
                    narrative_id=narrative_id,
                    keywords=[keyword],
                    start_time=time.time()
                )
            
            narrative = self.narratives[narrative_id]
            narrative.tokens.update(m.token for m in recent_mentions if keyword in m.content)
            narrative.accounts.update(m.account.handle for m in recent_mentions if keyword in m.content)
            
            if len(narrative.tokens) >= 3:
                narrative.cross_platform = len(set(m.platform for m in recent_mentions if keyword in m.content)) > 1

    def _extract_keywords(self, text: str) -> List[str]:
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", "did", "let", "put", "say", "she", "too", "use", "will", "with", "this", "that", "have", "from", "they", "know", "want", "been", "good", "much", "some", "time", "very", "when", "come", "here", "just", "like", "long", "make", "many", "over", "such", "take", "than", "them", "well", "were"}
        return [w for w in words if w not in stopwords and len(w) > 3]

    async def _recalculate_account_credibility(self):
        for account in self.accounts.values():
            if account.total_calls < 5:
                continue
            
            credibility = 0.0
            
            success_rate = account.successful_calls / max(account.total_calls, 1)
            credibility += success_rate * 0.3
            
            credibility += min(account.avg_roi / 5, 1) * 0.2
            
            credibility += max(0, 1 - account.median_entry_delay / 300) * 0.15
            
            credibility += (1 - account.rug_exposure) * 0.15
            
            sample_factor = min(1.0, account.total_calls / 50)
            credibility += sample_factor * 0.1
            
            independence = self._calculate_account_independence(account)
            credibility += independence * 0.1
            
            account.credibility_score = max(0, min(1, credibility))
            
            if account.credibility_score > 0.7:
                account.verified = True

    def _calculate_account_independence(self, account: SocialAccount) -> float:
        if not account.call_history:
            return 1.0
        
        unique_tokens = set(c.get("token") for c in account.call_history if c.get("token"))
        total_calls = len(account.call_history)
        
        if total_calls == 0:
            return 1.0
        
        return len(unique_tokens) / total_calls

    async def _update_token_mention_causality(self):
        for token, mentions in self.token_mentions.items():
            for mention in mentions:
                if mention.causality_score > 0:
                    continue
                
                chain_before = mention.chain_activity_before
                chain_after = mention.chain_activity_after
                
                if chain_before and not chain_after:
                    mention.causality_score = 0.8
                elif not chain_before and chain_after:
                    mention.causality_score = 0.3
                elif chain_before and chain_after:
                    mention.causality_score = 0.5
                else:
                    mention.causality_score = 0.1

    def get_token_social_signal(self, token: str, *, as_of: Optional[float] = None) -> Dict[str, Any]:
        """Return a point-in-time signal without admitting future mentions."""
        cutoff = float(as_of if as_of is not None else time.time())
        mentions = self.token_mentions.get(token, [])
        if not mentions:
            return {"signal": 0, "confidence": 0, "reason": "no_mentions"}
        
        recent = sorted(
            [m for m in mentions if cutoff - 1800 < m.timestamp <= cutoff],
            key=lambda mention: mention.timestamp,
        )
        if not recent:
            return {"signal": 0, "confidence": 0, "reason": "stale_mentions"}
        
        velocities = []
        for i in range(1, len(recent)):
            dt = recent[i].timestamp - recent[i-1].timestamp
            if dt > 0:
                velocities.append(1 / dt)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        acceleration = np.mean(np.diff(velocities)) if len(velocities) > 1 else 0
        
        credibility_scores = [m.account.credibility_score for m in recent]
        avg_cred = np.mean(credibility_scores) if credibility_scores else 0
        
        first_mention_time = min(m.timestamp for m in recent)
        chain_before = sum(1 for m in recent if m.chain_activity_before)
        
        signal = avg_cred * 0.4 + min(avg_velocity * 10, 1) * 0.3 + max(0, acceleration) * 0.2 + (chain_before / len(recent)) * 0.1
        
        return {
            "signal": min(1, signal),
            "confidence": min(1, len(recent) / 10),
            "mention_count": len(recent),
            "avg_velocity": avg_velocity,
            "acceleration": acceleration,
            "avg_credibility": avg_cred,
            "chain_before_pct": chain_before / len(recent) if recent else 0,
            "platforms": list(set(m.platform.value for m in recent)),
            "first_mention_delay": cutoff - first_mention_time,
            "cross_platform": len(set(m.platform for m in recent)) > 1
        }

    def get_account_profile(self, platform: SocialPlatform, handle: str) -> Optional[SocialAccount]:
        return self.accounts.get(f"{platform.value}:{handle}")

    def get_top_accounts(self, platform: Optional[SocialPlatform] = None, limit: int = 20) -> List[SocialAccount]:
        accounts = list(self.accounts.values())
        if platform:
            accounts = [a for a in accounts if a.platform == platform]
        accounts = [a for a in accounts if a.total_calls >= 5]
        accounts.sort(key=lambda x: x.credibility_score, reverse=True)
        return accounts[:limit]

    def get_narrative_clusters(self, min_tokens: int = 3) -> List[NarrativeCluster]:
        return [n for n in self.narratives.values() if len(n.tokens) >= min_tokens]

    def get_stats(self) -> Dict:
        return {
            "tracked_accounts": len(self.accounts),
            "verified_accounts": sum(1 for a in self.accounts.values() if a.credibility_score > 0.7),
            "total_mentions": len(self.mentions),
            "active_narratives": len(self.get_narrative_clusters()),
            "tokens_with_mentions": len(self.token_mentions),
            "top_10_accounts": [
                {"handle": a.handle, "platform": a.platform.value, "cred": round(a.credibility_score, 3), "calls": a.total_calls}
                for a in self.get_top_accounts(limit=10)
            ],
            "data_status": dict(self.data_status),
        }
