"""Resolving whether a token is actually the one a public figure announced.

When a globally-followed account says "coin", dozens of impostor mints exist
within seconds, most of them named exactly what a ticker-matching bot would
look for. A system that buys the first mint whose symbol matches the name in
the news is not fast -- it is the reliable buyer of every copycat, which is a
worse position than not trading the event at all.

So this resolves proof, in levels, and refuses to promote a weak proof into a
strong one:

  A  DIRECT_MINT      the official post contains the mint address itself
  B  OFFICIAL_DOMAIN  the official post links to an official domain that
                      publishes the mint
  C  CREATOR_WALLET   a wallet already known to belong to the entity created
                      or funded the token
  D  CROSS_SOURCE     several INDEPENDENT canonical sources resolve to the
                      same mint
  E  NAME_ONLY        only the name or ticker matches; proves nothing

Level E is explicitly not a licence to trade the event. It is the level at
which nearly all impostors live.

Two parsing rules carry most of the security weight, and both are places where
the obvious implementation is exploitable:

Domain matching is exact host or true subdomain, never substring. Matching
"trump.com" as a substring accepts ``trump.com.attacker.io`` and
``nottrump.com``; an attacker controls both and the second costs nothing to
register. The check compares labels from the right.

Mint extraction validates, it does not pattern-match. A base58-looking run of
characters is not an address: the decoded value has to be exactly 32 bytes.
Loose matching turns any 40-character word in a post into a candidate mint,
and a candidate mint is one step from a purchase.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

AUTHENTICITY_SCHEMA_VERSION = "v1"

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {char: index for index, char in enumerate(_BASE58_ALPHABET)}
# Solana addresses are 32 bytes, which is 32-44 base58 characters.
_BASE58_RUN = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}")
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_PUMP_PATH_MINT = re.compile(r"/(?:coin|board)?/?([1-9A-HJ-NP-Za-km-z]{32,44})")


class ProofLevel(Enum):
    """Ordered weakest to strongest; comparison is by ``rank``."""

    NONE = "none"
    NAME_ONLY = "name_only"
    CROSS_SOURCE = "cross_source"
    CREATOR_WALLET = "creator_wallet"
    OFFICIAL_DOMAIN = "official_domain"
    DIRECT_MINT = "direct_mint"

    @property
    def rank(self) -> int:
        return _PROOF_RANK[self]


_PROOF_RANK = {
    ProofLevel.NONE: 0,
    ProofLevel.NAME_ONLY: 1,
    ProofLevel.CROSS_SOURCE: 2,
    ProofLevel.CREATOR_WALLET: 3,
    ProofLevel.OFFICIAL_DOMAIN: 4,
    ProofLevel.DIRECT_MINT: 5,
}

# Below this, the event is observed and logged but never sized on. Name-only
# agreement is where essentially every impostor sits.
MIN_TRADEABLE_PROOF = ProofLevel.CROSS_SOURCE


def decode_base58(value: str) -> Optional[bytes]:
    """Decode base58, returning None on any invalid character.

    Written out rather than imported so that mint validation does not depend
    on an optional package being installed -- a validator that silently stops
    validating when a dependency is missing is worse than no validator.
    """
    number = 0
    for char in value:
        index = _BASE58_INDEX.get(char)
        if index is None:
            return None
        number = number * 58 + index
    raw = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    leading_zeros = len(value) - len(value.lstrip("1"))
    return b"\x00" * leading_zeros + raw


def looks_like_mint(value: str) -> bool:
    """True only for a string that decodes to exactly 32 bytes."""
    if not 32 <= len(value) <= 44:
        return False
    decoded = decode_base58(value)
    return decoded is not None and len(decoded) == 32


def extract_mints(text: str) -> List[str]:
    """Every validated 32-byte address in a piece of text, in order of appearance.

    Includes addresses embedded in URL paths, which is how pump links carry
    the mint.
    """
    found: List[str] = []
    seen: Set[str] = set()
    for match in _BASE58_RUN.finditer(text or ""):
        candidate = match.group(0)
        if candidate not in seen and looks_like_mint(candidate):
            seen.add(candidate)
            found.append(candidate)
    for url in _URL.findall(text or ""):
        for match in _PUMP_PATH_MINT.finditer(url):
            candidate = match.group(1)
            if candidate not in seen and looks_like_mint(candidate):
                seen.add(candidate)
                found.append(candidate)
    return found


def normalise_host(value: str) -> str:
    """Bare lowercase host from a URL or hostname, without port or credentials."""
    host = (value or "").strip().lower()
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    host = host.split(":", 1)[0]
    return host.strip(".")


def host_matches(host: str, official: str) -> bool:
    """Exact host or a true subdomain of it -- never a substring.

    ``trump.com`` must match ``www.trump.com`` and must not match
    ``trump.com.attacker.io`` or ``nottrump.com``. An attacker can register
    both of the second pair for the price of a domain, so substring matching
    here is equivalent to no matching at all.
    """
    host, official = normalise_host(host), normalise_host(official)
    if not host or not official:
        return False
    if host == official:
        return True
    return host.endswith("." + official)


def extract_hosts(text: str) -> List[str]:
    return [normalise_host(url) for url in _URL.findall(text or "")]


@dataclass
class WatchedEntity:
    """One public figure or organisation and everything that canonically is them."""

    entity_id: str
    display_name: str
    accounts: Dict[str, Set[str]] = field(default_factory=dict)
    official_domains: Set[str] = field(default_factory=set)
    known_wallets: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def owns_account(self, platform: str, account_id: str) -> bool:
        return str(account_id) in self.accounts.get(str(platform).lower(), set())

    def owns_host(self, host: str) -> bool:
        return any(host_matches(host, domain) for domain in self.official_domains)


@dataclass
class SourceSignal:
    """One message from one place, already attributed to an account.

    ``account_id`` is the platform's stable identifier, never the display
    name: display names are changeable and impersonating one is free.
    """

    platform: str
    account_id: str
    text: str
    timestamp: float
    url: str = ""
    verified_publisher: bool = False


@dataclass
class AuthenticityVerdict:
    mint: Optional[str]
    level: ProofLevel
    entity_id: Optional[str]
    supporting_sources: List[str] = field(default_factory=list)
    detail: str = ""
    rejected: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def tradeable(self) -> bool:
        return self.mint is not None and self.level.rank >= MIN_TRADEABLE_PROOF.rank


class EntityRegistry:
    """Canonical accounts, domains, wallets and aliases, indexed for lookup."""

    def __init__(self, entities: Optional[Iterable[WatchedEntity]] = None):
        self._entities: Dict[str, WatchedEntity] = {}
        self._by_account: Dict[Tuple[str, str], str] = {}
        self._by_wallet: Dict[str, str] = {}
        for entity in entities or ():
            self.add(entity)

    def add(self, entity: WatchedEntity) -> None:
        self._entities[entity.entity_id] = entity
        for platform, ids in entity.accounts.items():
            for account_id in ids:
                self._by_account[(platform.lower(), str(account_id))] = entity.entity_id
        for wallet in entity.known_wallets:
            self._by_wallet[wallet] = entity.entity_id

    def get(self, entity_id: str) -> Optional[WatchedEntity]:
        return self._entities.get(entity_id)

    def by_account(self, platform: str, account_id: str) -> Optional[WatchedEntity]:
        entity_id = self._by_account.get((str(platform).lower(), str(account_id)))
        return self._entities.get(entity_id) if entity_id else None

    def by_wallet(self, wallet: str) -> Optional[WatchedEntity]:
        entity_id = self._by_wallet.get(wallet)
        return self._entities.get(entity_id) if entity_id else None

    def by_host(self, host: str) -> Optional[WatchedEntity]:
        for entity in self._entities.values():
            if entity.owns_host(host):
                return entity
        return None

    def match_name(self, text: str) -> List[WatchedEntity]:
        """Entities whose name or alias appears in the text as a whole word.

        Whole-word only: substring matching would have every token containing
        "elon" resolve to one person, which is how a name-only match becomes
        an unlimited false-positive generator.
        """
        haystack = (text or "").lower()
        matched = []
        for entity in self._entities.values():
            names = {entity.display_name.lower()} | {alias.lower() for alias in entity.aliases}
            if any(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", haystack)
                   for name in names if name):
                matched.append(entity)
        return matched


class AuthenticityResolver:
    """Decides which mint, if any, an entity actually announced.

    The resolver never picks between competing mints on popularity, recency or
    volume. If the proof does not identify one mint, it returns the level it
    could establish and no mint, and the caller does not trade the event. That
    is the correct outcome far more often than it feels like it should be.
    """

    def __init__(self, registry: EntityRegistry, min_independent_sources: int = 2):
        self.registry = registry
        # "Independent" means distinct entities, not distinct posts: one
        # compromised or impersonated account posting three times is one
        # source, and treating it as three is exactly the attack.
        self.min_independent_sources = max(2, min_independent_sources)

    def resolve_signal(
        self,
        signal: SourceSignal,
        domain_published_mints: Optional[Dict[str, str]] = None,
    ) -> AuthenticityVerdict:
        """Strongest proof obtainable from a single message."""
        entity = self.registry.by_account(signal.platform, signal.account_id)
        rejected: List[Tuple[str, str]] = []
        if entity is None:
            named = self.registry.match_name(signal.text)
            level = ProofLevel.NAME_ONLY if named else ProofLevel.NONE
            return AuthenticityVerdict(
                mint=None, level=level,
                entity_id=named[0].entity_id if named else None,
                detail=("text names a watched entity but the account is not one of its "
                        "canonical accounts" if named else "no watched entity involved"),
            )

        mints = extract_mints(signal.text)
        if len(mints) == 1:
            return AuthenticityVerdict(
                mint=mints[0], level=ProofLevel.DIRECT_MINT, entity_id=entity.entity_id,
                supporting_sources=[f"{signal.platform}:{signal.account_id}"],
                detail="canonical account published the mint directly",
            )
        if len(mints) > 1:
            # Several addresses in one official post is ambiguous, and
            # ambiguity is the state an impostor wants to create. Refuse.
            rejected.append(("multiple_mints_in_one_post", ",".join(mints)))

        published = domain_published_mints or {}
        for host in extract_hosts(signal.text):
            if not entity.owns_host(host):
                if self.registry.by_host(host) is None:
                    rejected.append(("link_not_on_an_official_domain", host))
                continue
            for domain, mint in published.items():
                if host_matches(host, domain) and looks_like_mint(mint):
                    return AuthenticityVerdict(
                        mint=mint, level=ProofLevel.OFFICIAL_DOMAIN, entity_id=entity.entity_id,
                        supporting_sources=[f"{signal.platform}:{signal.account_id}", host],
                        detail=f"canonical account linked to official domain {host}",
                        rejected=rejected,
                    )

        return AuthenticityVerdict(
            mint=None, level=ProofLevel.NAME_ONLY, entity_id=entity.entity_id,
            supporting_sources=[f"{signal.platform}:{signal.account_id}"],
            detail="canonical account spoke but published no resolvable mint",
            rejected=rejected,
        )

    def resolve_creator(self, mint: str, creator: str, funders: Sequence[str] = ()) -> AuthenticityVerdict:
        """Chain-side proof: a wallet already known to be the entity made this token."""
        if not looks_like_mint(mint):
            return AuthenticityVerdict(mint=None, level=ProofLevel.NONE, entity_id=None,
                                       detail="mint failed 32-byte validation")
        entity = self.registry.by_wallet(creator)
        source = creator
        if entity is None:
            for funder in funders:
                entity = self.registry.by_wallet(funder)
                if entity is not None:
                    source = funder
                    break
        if entity is None:
            return AuthenticityVerdict(mint=mint, level=ProofLevel.NONE, entity_id=None,
                                       detail="creator and funders are not known entity wallets")
        return AuthenticityVerdict(
            mint=mint, level=ProofLevel.CREATOR_WALLET, entity_id=entity.entity_id,
            supporting_sources=[f"wallet:{source}"],
            detail=f"token created or funded by a known wallet of {entity.display_name}",
        )

    def combine(self, verdicts: Sequence[AuthenticityVerdict]) -> AuthenticityVerdict:
        """Fuse independent verdicts about the same event.

        The strongest single proof wins outright. Where no single proof is
        strong, several genuinely independent sources agreeing on one mint
        promote it to CROSS_SOURCE -- but only counting distinct entities, and
        only when they agree on exactly one mint. Disagreement is evidence of
        an impostor in the set, so it lowers confidence rather than being
        resolved by majority.
        """
        with_mint = [item for item in verdicts if item.mint]
        if not with_mint:
            best = max(verdicts, key=lambda item: item.level.rank, default=None)
            return best or AuthenticityVerdict(mint=None, level=ProofLevel.NONE, entity_id=None,
                                               detail="no verdicts supplied")

        strongest = max(with_mint, key=lambda item: item.level.rank)
        if strongest.level.rank > ProofLevel.CROSS_SOURCE.rank:
            return strongest

        by_mint: Dict[str, Set[str]] = {}
        for verdict in with_mint:
            key = verdict.entity_id or ";".join(verdict.supporting_sources)
            by_mint.setdefault(verdict.mint, set()).add(key)

        qualifying = {mint: sources for mint, sources in by_mint.items()
                      if len(sources) >= self.min_independent_sources}
        if len(qualifying) == 1:
            mint, sources = next(iter(qualifying.items()))
            return AuthenticityVerdict(
                mint=mint, level=ProofLevel.CROSS_SOURCE, entity_id=strongest.entity_id,
                supporting_sources=sorted(sources),
                detail=f"{len(sources)} independent sources resolve to one mint",
            )
        if len(qualifying) > 1:
            return AuthenticityVerdict(
                mint=None, level=ProofLevel.NAME_ONLY, entity_id=strongest.entity_id,
                detail="independent sources disagree about the mint; an impostor is in the set",
                rejected=[("conflicting_mints", ",".join(sorted(qualifying)))],
            )
        return AuthenticityVerdict(
            mint=None, level=ProofLevel.NAME_ONLY, entity_id=strongest.entity_id,
            detail=(f"no mint reached {self.min_independent_sources} independent sources"),
        )


def rank_copycats(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order tokens riding one narrative by independent capital, not by ticker.

    A single event can spawn hundreds of near-identical mints. The one worth
    anything is whichever is actually accumulating capital from wallets that
    are not the creator's -- which is frequently neither the first nor the
    best-named. Candidates without an observed independent-buyer measurement
    are not ranked, because ranking them at zero would order them against
    tokens that were measured.
    """
    scored = []
    for item in candidates:
        buyers = item.get("independent_buyers")
        capital = item.get("independent_capital_usd")
        if buyers is None or capital is None:
            continue
        scored.append({**item, "independent_score": float(buyers) * float(capital)})
    scored.sort(key=lambda item: item["independent_score"], reverse=True)
    return scored


def load_entities(path: str) -> List[WatchedEntity]:
    """Parse the watched-entity registry from YAML.

    A malformed entry is skipped with its id reported rather than raising, for
    the same reason the source registry does it: one bad line must not take
    the whole resolver offline. The asymmetry that matters is the other way
    round anyway -- a MISSING entity makes an official token look unverified,
    which costs a trade, while a WRONG entity makes an impersonator look
    verified, which costs the position. So the parse is strict about the
    fields that assert identity and forgiving about everything else.
    """
    import yaml  # local: keeps the resolver importable without a YAML runtime

    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("entity registry unreadable at %s: %s", path, exc)
        return []
    entities: List[WatchedEntity] = []
    for entry in raw.get("entities") or []:
        try:
            accounts = {
                str(platform).lower(): {str(account) for account in (ids or ())}
                for platform, ids in (entry.get("accounts") or {}).items()
            }
            entities.append(WatchedEntity(
                entity_id=str(entry["entity_id"]),
                display_name=str(entry["display_name"]),
                accounts=accounts,
                official_domains={normalise_host(host)
                                  for host in (entry.get("official_domains") or ())
                                  if normalise_host(host)},
                known_wallets={str(wallet) for wallet in (entry.get("known_wallets") or ())},
                aliases={str(alias) for alias in (entry.get("aliases") or ())},
                metadata=dict(entry.get("metadata") or {}),
            ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            logger.error("skipping malformed entity declaration %r: %s", entry, exc)
    return entities
