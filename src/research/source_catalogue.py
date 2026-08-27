"""The endpoint catalogue: every public interface this desk knows how to ask.

Separated from the rotator in ``source_substitution.py`` because the two
change for different reasons. The rotator is machinery and should almost never
change; this is a list, and a list of public endpoints changes every time one
of them moves a path or starts refusing datacentre addresses. Keeping them
apart means adding the ninth Korean venue is an edit here and nothing else.

The catalogue is organised by DOMAIN -- the question, not the vendor -- and
every domain carries several regions on purpose. Two things follow from that
which are worth stating, because they are the reason the breadth is not
decoration:

**Different regions are awake at different hours.** A desk whose entire market
context comes from two US aggregators is running on a thin, correlated signal
during the Asian session, which is exactly when a large share of memecoin
attention originates. Korean, Japanese, Chinese-language, Indian, Turkish,
Indonesian and Thai venues are not redundancy for the US ones; they are the
only view of flow that starts there.

**Correlated failure is the failure that matters.** Three endpoints owned by
the same company are one endpoint. Rungs are ordered so that the substitute
for an aggregator is a DIFFERENT operator wherever one exists, rather than
another path on the same host, because a ladder whose rungs all fail together
is a ladder that has never been tested by the only outage that counts.

Every URL here is a documented public interface that needs no account, with
two exceptions marked by ``requires_env`` where we hold the key ourselves.
None of it reaches a private venue or anything behind an access control.

None of these has been probed from your node. Run
``python tools/verify_substitution.py`` there: it reports which rungs answer
from that address and which do not, and a rung that never answers is named
permanently rather than quietly padding a coverage count.
"""

from __future__ import annotations

from typing import Dict, Sequence

from src.research.source_substitution import Endpoint, SubstitutionRegistry

CATALOGUE_SCHEMA_VERSION = "v1"


#: Lists of tradable Solana tokens. The universe a copycat corpus and a name
#: search are drawn from.
TOKEN_UNIVERSE: Sequence[Endpoint] = (
    Endpoint("jupiter_tokens", "https://tokens.jup.ag/tokens?tags=verified",
             region="global", shape="jup_token_list",
             detail="Jupiter's verified set; the canonical routable universe"),
    Endpoint("jupiter_all", "https://token.jup.ag/all", region="global",
             shape="jup_token_list", detail="the unfiltered list, same operator"),
    Endpoint("coingecko_solana_list",
             "https://api.coingecko.com/api/v3/coins/list?include_platform=true",
             region="global", shape="coingecko_list",
             detail="different operator; slower to list, harder to spoof"),
    Endpoint("coinpaprika_coins", "https://api.coinpaprika.com/v1/coins",
             region="eu", shape="paprika_list",
             detail="European operator; independent of the US aggregators"),
    Endpoint("coincap_assets", "https://api.coincap.io/v2/assets?limit=2000",
             region="global", shape="coincap_list"),
)


#: Pools created in the last few minutes. The discovery domain: this is where
#: a launch that did not come through our own program stream shows up.
NEW_POOLS: Sequence[Endpoint] = (
    Endpoint("geckoterminal_new",
             "https://api.geckoterminal.com/api/v2/networks/solana/new_pools?page=1",
             region="global", shape="geckoterminal_pools",
             detail="newest Solana pools, keyless and documented"),
    Endpoint("dexscreener_profiles",
             "https://api.dexscreener.com/token-profiles/latest/v1",
             region="global", shape="dexscreener_profiles",
             detail="tokens whose team just published a profile"),
    Endpoint("dexscreener_boosts",
             "https://api.dexscreener.com/token-boosts/latest/v1",
             region="global", shape="dexscreener_profiles",
             detail="paid promotion, which is itself a spend signal"),
    Endpoint("raydium_pools",
             "https://api-v3.raydium.io/pools/info/list"
             "?poolType=all&poolSortField=default&sortType=desc&pageSize=100&page=1",
             region="global", shape="raydium_pools",
             detail="the venue's own list; independent of every aggregator"),
    Endpoint("geckoterminal_trending",
             "https://api.geckoterminal.com/api/v2/networks/solana/trending_pools",
             region="global", shape="geckoterminal_pools",
             detail="last rung: trending is not new, but it is never empty"),
)


#: Everything the market thinks one mint is worth right now.
TOKEN_PRICE: Sequence[Endpoint] = (
    Endpoint("jupiter_price", "https://api.jup.ag/price/v2?ids={mints}",
             region="global", shape="jup_price",
             detail="routed price; the number an order would actually get"),
    Endpoint("jupiter_price_v6", "https://price.jup.ag/v6/price?ids={mints}",
             region="global", shape="jup_price", detail="older path, same operator"),
    Endpoint("dexscreener_tokens",
             "https://api.dexscreener.com/latest/dex/tokens/{mints}",
             region="global", shape="dexscreener_pairs"),
    Endpoint("llama_prices",
             "https://coins.llama.fi/prices/current/{llama_ids}",
             region="global", shape="llama_prices",
             detail="DefiLlama; a fourth operator with its own oracle set"),
    Endpoint("geckoterminal_price",
             "https://api.geckoterminal.com/api/v2/simple/networks/solana/"
             "token_price/{mints}", region="global", shape="geckoterminal_price"),
    Endpoint("raydium_mint_price", "https://api-v3.raydium.io/mint/price?mints={mints}",
             region="global", shape="raydium_price"),
)


#: Whether the supply we priced is the supply that will exist. A rug is a
#: supply event before it is a price event.
SUPPLY_CONTROL: Sequence[Endpoint] = (
    Endpoint("rugcheck", "https://api.rugcheck.xyz/v1/tokens/{mint}/report",
             region="global", shape="rugcheck",
             detail="mint and freeze authority, LP lock, holder concentration"),
    Endpoint("rugcheck_summary",
             "https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary",
             region="global", shape="rugcheck_summary"),
    Endpoint("dexscreener_supply",
             "https://api.dexscreener.com/latest/dex/tokens/{mint}",
             region="global", shape="dexscreener_pairs",
             detail="proxy: liquidity and FDV move when supply control does"),
)


#: The wider market's state while a launch happens. Regime, not trigger.
MARKET_CONTEXT: Sequence[Endpoint] = (
    Endpoint("coingecko_global", "https://api.coingecko.com/api/v3/global",
             region="global", shape="coingecko_global"),
    Endpoint("coinpaprika_global", "https://api.coinpaprika.com/v1/global",
             region="eu", shape="paprika_global"),
    Endpoint("coinlore_global", "https://api.coinlore.net/api/global/",
             region="global", shape="coinlore_global"),
    Endpoint("llama_dex_solana", "https://api.llama.fi/overview/dexs/solana",
             region="global", shape="llama_overview",
             detail="Solana DEX volume; the denominator for 'is anyone trading'"),
    Endpoint("fear_greed", "https://api.alternative.me/fng/?limit=2",
             region="global", shape="fear_greed",
             detail="crowd risk appetite; daily, a regime input only"),
)


#: Venue tickers, by region. Not for pricing a memecoin -- none of these list
#: one on day zero -- but for reading which part of the world is risk-on
#: right now, and for catching the moment a launch graduates onto a venue.
VENUE_TICKERS: Sequence[Endpoint] = (
    Endpoint("binance", "https://api.binance.com/api/v3/ticker/24hr",
             region="global", shape="binance_ticker"),
    Endpoint("okx", "https://www.okx.com/api/v5/market/tickers?instType=SPOT",
             region="asia", shape="okx_ticker"),
    Endpoint("bybit", "https://api.bybit.com/v5/market/tickers?category=spot",
             region="asia", shape="bybit_ticker"),
    Endpoint("gate", "https://api.gateio.ws/api/v4/spot/tickers",
             region="asia", shape="gate_ticker"),
    Endpoint("mexc", "https://api.mexc.com/api/v3/ticker/24hr",
             region="asia", shape="binance_ticker"),
    Endpoint("bitget", "https://api.bitget.com/api/v2/spot/market/tickers",
             region="asia", shape="bitget_ticker"),
    Endpoint("kucoin", "https://api.kucoin.com/api/v1/market/allTickers",
             region="asia", shape="kucoin_ticker"),
    Endpoint("htx", "https://api.huobi.pro/market/tickers",
             region="asia", shape="htx_ticker"),
)


#: Korea, Japan and the rest of Asia-Pacific in their own currencies. The
#: kimchi premium is a real, measurable, tradeable divergence and it is
#: invisible in USD pairs; a Korean listing pop is invisible everywhere else
#: until it has already happened.
REGIONAL_VENUES: Sequence[Endpoint] = (
    Endpoint("upbit_markets", "https://api.upbit.com/v1/market/all?isDetails=false",
             region="kr", shape="upbit_markets",
             detail="Korea's dominant venue; a new KRW market is a step change"),
    Endpoint("bithumb", "https://api.bithumb.com/public/ticker/ALL_KRW",
             region="kr", shape="bithumb_ticker"),
    Endpoint("coinone", "https://api.coinone.co.kr/public/v2/ticker_new/KRW",
             region="kr", shape="coinone_ticker"),
    Endpoint("bitflyer", "https://api.bitflyer.com/v1/markets",
             region="jp", shape="bitflyer_markets"),
    Endpoint("gmo_japan", "https://api.coin.z.com/public/v1/ticker",
             region="jp", shape="gmo_ticker"),
    Endpoint("bitbank", "https://public.bitbank.cc/tickers",
             region="jp", shape="bitbank_ticker"),
    Endpoint("indodax", "https://indodax.com/api/tickers",
             region="id", shape="indodax_ticker"),
    Endpoint("bitkub", "https://api.bitkub.com/api/market/ticker",
             region="th", shape="bitkub_ticker"),
    Endpoint("coindcx", "https://api.coindcx.com/exchange/ticker",
             region="in", shape="coindcx_ticker"),
    Endpoint("wazirx", "https://api.wazirx.com/api/v2/tickers",
             region="in", shape="wazirx_ticker"),
    Endpoint("btcturk", "https://api.btcturk.com/api/v2/ticker",
             region="tr", shape="btcturk_ticker"),
    Endpoint("paribu", "https://www.paribu.com/ticker", region="tr",
             shape="paribu_ticker"),
    Endpoint("bitso", "https://api.bitso.com/v3/ticker/", region="latam",
             shape="bitso_ticker"),
    Endpoint("mercado_bitcoin",
             "https://api.mercadobitcoin.net/api/v4/tickers?symbols=BTC-BRL,SOL-BRL",
             region="latam", shape="mercado_ticker"),
    Endpoint("luno", "https://api.luno.com/api/1/tickers", region="africa",
             shape="luno_ticker"),
    Endpoint("valr", "https://api.valr.com/v1/public/marketsummary",
             region="africa", shape="valr_ticker"),
)


#: Solana RPC itself. The one domain where losing every rung stops the desk
#: rather than degrading it, which is why it carries the most substitutes.
SOLANA_RPC: Sequence[Endpoint] = (
    Endpoint("helius", "https://mainnet.helius-rpc.com/?api-key={helius_key}",
             region="global", requires_env=("HELIUS_API_KEY",), shape="rpc",
             detail="paid, ours, fastest"),
    Endpoint("alchemy", "https://solana-mainnet.g.alchemy.com/v2/{alchemy_key}",
             region="global", requires_env=("ALCHEMY_API_KEY",), shape="rpc"),
    Endpoint("publicnode", "https://solana-rpc.publicnode.com", region="eu",
             shape="rpc", detail="keyless; rate limited but genuinely public"),
    Endpoint("ankr", "https://rpc.ankr.com/solana", region="global", shape="rpc"),
    Endpoint("drpc", "https://solana.drpc.org", region="global", shape="rpc"),
    Endpoint("solana_foundation", "https://api.mainnet-beta.solana.com",
             region="global", shape="rpc",
             detail="last rung: heavily throttled, but it is never gone"),
)


#: Measured public attention. A mention is a touch; these say how many people
#: went looking, which is what separates a post nobody read from a wave.
SOCIAL_ATTENTION: Sequence[Endpoint] = (
    Endpoint("reddit_new", "https://www.reddit.com/r/{sub}/new.json?limit=50",
             region="global", shape="reddit"),
    Endpoint("hn_algolia",
             "https://hn.algolia.com/api/v1/search_by_date?query={query}"
             "&tags=story&hitsPerPage=50", region="global", shape="hn"),
    Endpoint("lemmy", "https://lemmy.world/api/v3/search?q={query}&type_=Posts&limit=30",
             region="eu", shape="lemmy",
             detail="federated; not subject to Reddit's address refusals"),
    Endpoint("mastodon_search",
             "https://mastodon.social/api/v2/search?q={query}&type=statuses&limit=20",
             region="eu", shape="mastodon"),
    Endpoint("wikipedia_pageviews",
             "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "en.wikipedia/all-access/user/{article}/daily/{start}/{end}",
             region="global", shape="wikimedia",
             detail="daily only; a regime input, never a trigger"),
)


#: Public Telegram previews. ``t.me/s/<channel>`` is the page a browser shows
#: for a public channel with no account at all. It is the honest way to read
#: public Telegram: nothing here can open a private channel, because the
#: mechanism itself cannot.
TELEGRAM_PREVIEW: Sequence[Endpoint] = (
    Endpoint("tme_preview", "https://t.me/s/{channel}", region="global",
             shape="tme_html",
             detail="public channel preview; no account, no session, no key"),
)


DOMAINS: Dict[str, Sequence[Endpoint]] = {
    "token_universe": TOKEN_UNIVERSE,
    "new_pools": NEW_POOLS,
    "token_price": TOKEN_PRICE,
    "supply_control": SUPPLY_CONTROL,
    "market_context": MARKET_CONTEXT,
    "venue_tickers": VENUE_TICKERS,
    "regional_venues": REGIONAL_VENUES,
    "solana_rpc": SOLANA_RPC,
    "social_attention": SOCIAL_ATTENTION,
    "telegram_preview": TELEGRAM_PREVIEW,
}


def default_registry(**kwargs) -> SubstitutionRegistry:
    """A registry with every declared domain, ready to rotate."""
    registry = SubstitutionRegistry(**kwargs)
    for domain, endpoints in DOMAINS.items():
        registry.declare(domain, endpoints)
    return registry


def endpoint_count() -> int:
    return sum(len(endpoints) for endpoints in DOMAINS.values())


def regions() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for endpoints in DOMAINS.values():
        for endpoint in endpoints:
            counts[endpoint.region] = counts.get(endpoint.region, 0) + 1
    return dict(sorted(counts.items()))
