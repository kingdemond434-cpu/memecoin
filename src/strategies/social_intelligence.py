import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import hashlib

import aiohttp
import numpy as np

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
        
        self._known_contracts: Set[str] = set()
        self._mention_callbacks: List[Callable] = []

    async def start(self, initial_accounts: List[Dict] = None):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50)
        )
        self._running = True
        
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

    def on_mention(self, callback: Callable):
        self._mention_callbacks.append(callback)

    async def _initial_discovery(self):
        await self._discover_accounts_from_successful_wallets()
        await self._discover_accounts_from_recent_launches()
        await self._recalculate_account_credibility()

    async def _hunter_loop(self):
        while self._running:
            try:
                await self._discover_accounts_from_successful_wallets()
                await self._discover_accounts_from_recent_launches()
                await self._discover_accounts_from_foreign_sources()
                await self._discover_accounts_from_competitors()
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

    async def _find_linked_social(self, wallet: str):
        try:
            async with self._session.get(
                f"https://api.helius.xyz/v0/addresses/{wallet}/social",
                params={"api-key": self.api_keys.get("helius", "")}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for link in data.get("social_links", []):
                        await self._add_account({
                            "platform": link.get("platform"),
                            "handle": link.get("username"),
                            "account_id": link.get("user_id"),
                            "source": "wallet_link",
                            "source_wallet": wallet
                        })
        except Exception:
            pass

    async def _discover_accounts_from_recent_launches(self):
        try:
            async with self._session.get(
                "https://api.helius.xyz/v0/tokens/mintlist",
                params={"api-key": self.api_keys.get("helius", ""), "limit": 50}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for token in data:
                        await self._scan_token_social(token.get("mint"))
        except Exception as e:
            logger.debug(f"Launch social discovery error: {e}")

    async def _scan_token_social(self, token: str):
        if token in self._known_contracts:
            return
        self._known_contracts.add(token)
        
        for platform in [SocialPlatform.X, SocialPlatform.TELEGRAM, SocialPlatform.REDDIT]:
            await self._search_platform_for_token(platform, token)

    async def _search_platform_for_token(self, platform: SocialPlatform, token: str):
        pass

    async def _discover_accounts_from_foreign_sources(self):
        pass

    async def _discover_accounts_from_competitors(self):
        known_bots = [
            "solsniper", "phantom", "bonkbot", "unibot", "maestro",
            "trojan", "bullx", "banana", "shuriken"
        ]
        for bot in known_bots:
            await self._scan_competitor_followers(bot)

    async def _scan_competitor_followers(self, bot_name: str):
        pass

    async def _add_account(self, data: Dict):
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
            region=data.get("region", "global")
        )
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
        pass

    async def _fetch_reddit_posts(self, account: SocialAccount):
        pass

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
            async with self._session.get(
                f"https://api.helius.xyz/v0/token-accounts",
                params={"api-key": self.api_keys.get("helius", ""), "mint": token, "limit": 100}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for acc in data.get("token_accounts", []):
                        owner = acc.get("owner")
                        if owner:
                            txs = await self._get_wallet_txs_before(owner, mention_time, limit=10)
                            if txs:
                                mention.chain_activity_before = True
                                break
        except Exception:
            pass

    async def _get_wallet_txs_before(self, wallet: str, before_time: float, limit: int = 10) -> List:
        return []

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
                account.is_verified = True

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

    def get_token_social_signal(self, token: str) -> Dict[str, Any]:
        mentions = self.token_mentions.get(token, [])
        if not mentions:
            return {"signal": 0, "confidence": 0, "reason": "no_mentions"}
        
        recent = [m for m in mentions if time.time() - m.timestamp < 1800]
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
            "first_mention_delay": time.time() - first_mention_time,
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
            ]
        }