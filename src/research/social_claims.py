"""What a launch CLAIMS about itself, and whether anybody backed the claim up.

The desk could not link a wallet to a social account, and `_find_linked_social`
was right to refuse: there is no verified public wallet-to-social source, and
an edge written on a display name is an identity claim the desk cannot
support.

But that was the wrong question. A follower does not need to know which human
owns a wallet. What moves a memecoin is whether a named account with reach is
actually TALKING ABOUT THIS MINT -- and that link is between a token and an
account, not between a wallet and a person, and it is directly observable:

    the deployer DECLARES handles in the token's own metadata
    the public source stream shows whether those handles POSTED

A declaration is free and worth nothing. Anyone can put a famous handle in a
metadata field, and the most common influencer rug is exactly that. A post
from the declared account referencing the mint is evidence, because the
account controls what it publishes and the desk observed it.

So every claim carries a status rather than a truth value:

  DECLARED         the metadata says so; nobody has checked
  CORROBORATED     that account publicly referenced this mint
  UNCORROBORATED   the account had a fair window and stayed silent

The third state is the one the desk could never see before, and it is the
valuable one. A launch claiming a large account, trading actively, with that
account conspicuously silent, is the signature of an impersonation -- and
under the old arrangement it looked identical to a genuine influencer launch,
because neither produced any social evidence at all.

Nothing here claims that an account IS the deployer. Corroboration says the
account published about the token. That is all it says, and it is the only
part that transfers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from collections import deque

logger = logging.getLogger(__name__)

SOCIAL_CLAIM_SCHEMA_VERSION = "v1"

#: Metadata fields a Pump launch can declare. Read as claims, never as facts.
CLAIM_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("twitter", "twitter"), ("x", "twitter"),
    ("telegram", "telegram"), ("website", "website"),
)

#: How long a declared account gets to say something before its silence is
#: itself an observation. Long enough that a genuine account posting an hour
#: after the mint still corroborates; short enough that the answer arrives
#: while the position is open.
DEFAULT_SILENCE_WINDOW_S = 3600.0

#: Tokens tracked. Bounded, oldest first.
DEFAULT_MAX_TOKENS = 8_192

DECLARED = "DECLARED"
CORROBORATED = "CORROBORATED"
UNCORROBORATED = "UNCORROBORATED"

_HANDLE = re.compile(r"(?:^|/|@)([A-Za-z0-9_]{2,32})/?$")


def normalise_handle(value: Any) -> str:
    """`https://x.com/SomeOne`, `@SomeOne` and `someone` are one handle.

    Lowercased, because a corroboration that misses on capitalisation reports
    silence from an account that spoke.
    """
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    match = _HANDLE.search(text)
    return match.group(1).lower() if match else text.lower()


def claims_from_metadata(metadata: Any) -> List[Tuple[str, str]]:
    """(platform, handle) pairs a launch declared about itself.

    Accepts the parsed metadata JSON. A website is kept as a claim too: a
    launch declaring a domain it does not control is the same failure mode in
    a different field.
    """
    if not isinstance(metadata, dict):
        return []
    found: List[Tuple[str, str]] = []
    for field_name, platform in CLAIM_FIELDS:
        handle = normalise_handle(metadata.get(field_name))
        if handle and (platform, handle) not in found:
            found.append((platform, handle))
    return found


@dataclass
class SocialClaim:
    """One declared account, and what happened to the declaration."""

    token: str
    platform: str
    handle: str
    declared_at: float
    status: str = DECLARED
    corroborated_at: Optional[float] = None
    #: The source event that corroborated it, for audit.
    corroborating_source: str = ""
    posts_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SocialClaimLedger:
    """Declarations in, corroborations in, a status per claim out."""

    def __init__(self, path: Optional[str] = None, *,
                 silence_window_s: float = DEFAULT_SILENCE_WINDOW_S,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        self.path = Path(path) if path else None
        self.silence_window_s = float(silence_window_s)
        self.max_tokens = int(max_tokens)
        self._claims: Dict[str, List[SocialClaim]] = {}
        self._order: Deque[str] = deque()
        #: handle -> tokens that declared it, so an account claimed by nine
        #: launches this hour is visible as what it is.
        self._by_handle: Dict[str, List[str]] = {}

    # -- declaring --------------------------------------------------------

    def declare(self, token: str, metadata: Any,
                at: Optional[float] = None) -> int:
        """Record what this launch says about itself. Returns claims added."""
        if not token:
            return 0
        pairs = claims_from_metadata(metadata)
        if not pairs:
            return 0
        moment = time.time() if at is None else float(at)
        existing = self._claims.setdefault(token, [])
        if not existing:
            self._order.append(token)
            self._evict()
        known = {(claim.platform, claim.handle) for claim in existing}
        added = 0
        for platform, handle in pairs:
            if (platform, handle) in known:
                continue
            existing.append(SocialClaim(token=token, platform=platform,
                                        handle=handle, declared_at=moment))
            self._by_handle.setdefault(handle, []).append(token)
            added += 1
        return added

    def _evict(self) -> None:
        while len(self._order) > self.max_tokens:
            stale = self._order.popleft()
            for claim in self._claims.pop(stale, ()):
                tokens = self._by_handle.get(claim.handle)
                if tokens and stale in tokens:
                    tokens.remove(stale)
                if tokens is not None and not tokens:
                    self._by_handle.pop(claim.handle, None)

    # -- corroborating ----------------------------------------------------

    def corroborate(self, token: str, author_id: str,
                    at: Optional[float] = None,
                    source_id: str = "") -> bool:
        """A public post from this account referenced this mint.

        This is the whole of the evidence: the account controls what it
        publishes, and the desk observed the publication. It does NOT say the
        account owns the deployer wallet, and nothing downstream may read it
        that way.
        """
        handle = normalise_handle(author_id)
        if not handle:
            return False
        moment = time.time() if at is None else float(at)
        hit = False
        for claim in self._claims.get(token, ()):
            if claim.handle != handle:
                continue
            claim.posts_seen += 1
            if claim.status != CORROBORATED:
                claim.status = CORROBORATED
                claim.corroborated_at = moment
                claim.corroborating_source = str(source_id)
            hit = True
        return hit

    def observe_event(self, event: Any, at: Optional[float] = None) -> int:
        """Corroborate from one source event. Returns claims corroborated.

        Matches on the event's own author and the mints it names, so a post
        ABOUT a token by somebody else does not corroborate the token's claim
        about that somebody.
        """
        author = normalise_handle(getattr(event, "author_id", ""))
        if not author:
            return 0
        moment = (float(getattr(event, "source_at", 0.0) or 0.0)
                  or (time.time() if at is None else float(at)))
        source_id = str(getattr(event, "source_id", "") or "")
        return sum(1 for token in (getattr(event, "token_addresses", ()) or ())
                   if self.corroborate(str(token), author, moment, source_id))

    def settle(self, now: Optional[float] = None) -> int:
        """Close out claims whose account had a window and stayed silent."""
        moment = time.time() if now is None else float(now)
        settled = 0
        for claims in self._claims.values():
            for claim in claims:
                if claim.status != DECLARED:
                    continue
                if moment - claim.declared_at < self.silence_window_s:
                    continue
                claim.status = UNCORROBORATED
                settled += 1
        return settled

    # -- reading ----------------------------------------------------------

    def claims(self, token: str) -> List[SocialClaim]:
        return list(self._claims.get(token, ()))

    def corroborated_handles(self, token: str) -> List[str]:
        """Accounts that actually published about this mint. The only link."""
        return [claim.handle for claim in self._claims.get(token, ())
                if claim.status == CORROBORATED]

    def reading(self, token: str, now: Optional[float] = None) -> Dict[str, Any]:
        """The status of every claim this launch made about itself."""
        claims = self._claims.get(token)
        if not claims:
            return {"status": "DATA_BLOCKED", "token": token,
                    "detail": "this launch declared no social accounts"}
        moment = time.time() if now is None else float(now)
        by_status: Dict[str, int] = {}
        for claim in claims:
            state = claim.status
            if state == DECLARED and moment - claim.declared_at >= self.silence_window_s:
                state = UNCORROBORATED
            by_status[state] = by_status.get(state, 0) + 1
        corroborated = by_status.get(CORROBORATED, 0)
        silent = by_status.get(UNCORROBORATED, 0)
        # The state the desk could never see: a launch naming an account that
        # had a fair window and said nothing. Under the old arrangement this
        # looked exactly like a genuine influencer launch, because neither
        # produced any social evidence at all.
        return {
            "status": "OK",
            "token": token,
            "authority": "none",
            "declared": len(claims),
            "corroborated": corroborated,
            "silent": silent,
            "pending": by_status.get(DECLARED, 0),
            "corroborated_handles": self.corroborated_handles(token),
            # How many OTHER launches named the same accounts. A handle
            # claimed by nine mints this hour is not endorsing any of them.
            "handle_reuse": {
                claim.handle: len(set(self._by_handle.get(claim.handle, ())))
                for claim in claims},
            "impersonation_suspected": bool(silent and not corroborated),
            "claims": [claim.to_dict() for claim in claims],
            "detail": ("a declaration is free and worth nothing; only a post "
                       "from the declared account is evidence, and it says "
                       "the account published about the token, never that it "
                       "owns the wallet"),
        }

    def report(self) -> Dict[str, Any]:
        tokens = len(self._claims)
        all_claims = [claim for claims in self._claims.values() for claim in claims]
        corroborated = sum(1 for claim in all_claims if claim.status == CORROBORATED)
        silent = sum(1 for claim in all_claims if claim.status == UNCORROBORATED)
        reused = {handle: len(set(tokens_))
                  for handle, tokens_ in self._by_handle.items()
                  if len(set(tokens_)) > 1}
        return {
            "schema": SOCIAL_CLAIM_SCHEMA_VERSION,
            "status": "OK" if all_claims else "DATA_BLOCKED",
            "tokens_with_claims": tokens,
            "claims": len(all_claims),
            "corroborated": corroborated,
            "silent": silent,
            "corroboration_rate": (corroborated / len(all_claims)
                                   if all_claims else None),
            "handles_claimed_by_several_launches": dict(
                sorted(reused.items(), key=lambda item: -item[1])[:20]),
            "detail": ("" if all_claims else
                       "no launch has declared a social account yet"),
        }

    # -- persistence ------------------------------------------------------

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "schema": SOCIAL_CLAIM_SCHEMA_VERSION,
                "silence_window_s": self.silence_window_s,
                "claims": [claim.to_dict() for claims in self._claims.values()
                           for claim in claims]}), encoding="utf-8")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("social claims not saved: %s", exc)
            return False

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("social claims not loaded: %s", exc)
            return False
        if payload.get("schema") != SOCIAL_CLAIM_SCHEMA_VERSION:
            return False
        self.silence_window_s = float(
            payload.get("silence_window_s", self.silence_window_s))
        self._claims.clear()
        self._order.clear()
        self._by_handle.clear()
        for row in payload.get("claims", ()):
            claim = SocialClaim(**row)
            if not claim.token:
                continue
            if claim.token not in self._claims:
                self._claims[claim.token] = []
                self._order.append(claim.token)
            self._claims[claim.token].append(claim)
            self._by_handle.setdefault(claim.handle, []).append(claim.token)
        return True
