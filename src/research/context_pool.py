"""The miners that need nothing from the desk, built inside a child process.

Which miners can leave this interpreter is not a preference, it is a fact
about their inputs. The chain miners read desk state -- watched tokens,
tracked wallets, contended accounts, known deployers -- through callables
bound to the desk object. Those cannot cross a process boundary, and a
snapshot of them would go stale the moment it was sent.

The web, world and venue miners need none of that. They mine the public
universe: the token list, new pools, exchange tickers, regional venues,
news, Telegram previews. And they are precisely the expensive ones. From
the desk's own report: jupiter_tokens returns 3,174 records a pass,
dexscreener_pairs 468, venue_tickers 401, regional_venues 331. That is
where the multi-megabyte JSON is, which is where the GIL contention is.

So the split falls exactly where it should: the heaviest parsers are the
ones with no desk dependency, and they are the ones that move.

The child builds its own HttpClient and its own substitution registry --
they are not sent. An aiohttp session and a loop-bound registry belong to
the loop that made them, and shipping one across a process boundary is a
subtler version of the cross-loop bug that already cost this desk a night.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


class ContextMinerPool:
    """A DataMinerPool carrying only the desk-independent miners."""

    def __init__(self, config: Dict[str, Any]):
        from src.collectors.transports import HttpClient
        from src.research.data_miners import DataMinerPool
        from src.research.source_catalogue import default_registry

        self.config = dict(config)
        self.http = HttpClient(
            timeout_s=float(config.get("http_timeout_s", 10.0)))
        self.registry = default_registry()
        self.pool = DataMinerPool(
            concurrency=int(config.get("concurrency", 4)),
            on_records=self._forward)
        self.registered: Dict[str, bool] = {}
        # Rebound by the parent's child entry point before start().
        self.on_records = lambda miner_id, records: None
        self._register()

    def _forward(self, miner_id: str, records: List[Dict[str, Any]]) -> None:
        self.on_records(miner_id, records)

    def _register(self) -> None:
        terms = tuple(self.config.get("search_terms") or ("pump.fun", "solana"))
        youtube = str(self.config.get("youtube_key", "") or "")
        github = str(self.config.get("github_token", "") or "")

        from src.research.web_miners import register_web_miners

        self.registered.update(register_web_miners(
            self.pool, http=self.http,
            search_terms=lambda: terms,
            youtube_key=lambda: youtube,
            github_token=lambda: github))

        # The venue and breadth miners -- new pools, exchange tickers,
        # regional venues, market regime, supply control. These are the
        # heavy ones: from the desk's own report, venue_tickers returns 401
        # records a pass and regional_venues 331, all of it JSON parsed in
        # this interpreter. Moving them is most of the point.
        #
        # `watched_tokens` and `tracked_wallets` are empty here on purpose,
        # not by oversight. Those two feed the wallet and token passes that
        # read desk state, and a snapshot of them sent at startup would go
        # stale immediately -- so those passes stay with the desk, and the
        # miners that need nothing come here. rpc is None for the same
        # reason: an RPC manager is loop-bound and desk-bound, and shipping
        # one across a process boundary is a subtler version of the
        # cross-loop bug that already cost this desk a night.
        from src.research.regional_miners import register_regional_miners

        self.registered.update(register_regional_miners(
            self.pool, http=self.http, rpc=None, registry=self.registry,
            watched_tokens=lambda: (), tracked_wallets=lambda: ()))

        # register_regional_miners also declares the two wallet passes, and
        # in this process they have no RPC and no wallets to read -- so they
        # would sit permanently IDLE in the child while the desk's report
        # counted them as registered. A miner that can never produce is worse
        # than an absent one: it looks like coverage. Dropped by name, and
        # the desk keeps its own copies, which do have both.
        self.dropped = [miner_id for miner_id in list(self.registered)
                        if miner_id.startswith("chain:")]
        for miner_id in self.dropped:
            self.registered.pop(miner_id, None)
            self.pool._specs.pop(miner_id, None)
            self.pool._callables.pop(miner_id, None)
            self.pool._health.pop(miner_id, None)
            self.pool._next_due.pop(miner_id, None)
        logger.info("CONTEXT POOL registered %d desk-independent miners; "
                    "left %d chain miner(s) with the desk, which has the "
                    "state they read", len(self.registered), len(self.dropped))

    async def start(self) -> None:
        await self.pool.start()

    async def stop(self) -> None:
        await self.pool.stop()
        try:
            await self.http.close()
        except Exception:  # pragma: no cover - teardown only
            pass


def build_context_pool(config: Dict[str, Any]) -> ContextMinerPool:
    """Factory the child imports BY PATH. Must stay module level."""
    return ContextMinerPool(config)
