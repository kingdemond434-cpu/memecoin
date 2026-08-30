"""Launches that claim a famous name -- and whether the name agreed.

A large share of the biggest and the worst memecoin outcomes share one
feature: the token claims a person. A politician, a musician, an exchange
founder, a project. When the claim is real -- the figure's own public account
posts the address -- the move is violent and fast. When it is not, the same
token looks identical for the first ninety seconds and then goes to zero.
Anyone can mint a ticker with anyone's name on it, and the metadata alone
cannot tell those two cases apart.

So the desk does not ask "is this a celebrity coin". It asks three separate
questions and keeps the answers separate, because collapsing them is exactly
the error that makes celebrity coins a losing category on average:

  WHO IS BEING CLAIMED?    Matched from name, symbol, description and linked
                           socials against a declared registry of public
                           figures and projects. Cheap, immediate, and worth
                           nothing on its own.

  DID THEY CONFIRM IT?     Did a channel THIS registry already knew to belong
                           to that figure carry this exact mint address, and
                           when relative to creation? This is the whole
                           question. A claim nobody confirmed is the base
                           case, and the base case is bad.

  WHO ELSE HAS DONE THIS?  Has this deployer, or this funder cluster, minted
                           impersonations before? A repeat impersonator is
                           the most reliably identifiable actor in the market
                           and the registry remembers them.

Three disciplines hold this together.

**Corroboration is never assumed and never inferred from popularity.** A
thousand people posting an address is a thousand people repeating a claim, not
a confirmation. Only a channel that was registered as the figure's own BEFORE
this launch counts, and the window is bounded: a mention six hours later is a
reaction to the price, not an announcement.

**Uncorroborated is the default and is stated as such.** The verdict for a
launch claiming a public figure with no confirmation is UNCORROBORATED, and it
carries the population's own realised base rate for that class rather than a
guessed penalty. If uncorroborated claims turn out to pay, the ledger will say
so and the desk will act on it; nothing here hard-codes an opinion.

**Nothing here reaches a private account.** The figure registry holds public,
self-published handles -- the ones a person links from their own verified
profile. Corroboration is read from public channel previews and public web
sources. There is no credential in this module that could open a private
message, and impersonation is detected by comparing public claims against
public statements, which is the only lawful way to do it and also the only
way that generalises.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

IDENTITY_SCHEMA_VERSION = "v1"

#: How long after creation a mention from the figure's own channel still reads
#: as an announcement rather than as a reaction to a price that already moved.
ANNOUNCEMENT_WINDOW_S = 1_800.0

#: Below this similarity a name match is noise. Tuned high because the cost of
#: a false claim is a position sized on a story that is not there.
NAME_MATCH_FLOOR = 0.86

_NORMALISE = re.compile(r"[^a-z0-9]+")


class Category(Enum):
    """What kind of public figure. Different categories behave differently."""

    POLITICIAN = "politician"
    CELEBRITY = "celebrity"
    INFLUENCER = "influencer"
    FOUNDER = "founder"
    ATHLETE = "athlete"
    PROJECT = "project"
    EXCHANGE = "exchange"
    BRAND = "brand"
    EVENT = "event"


class Verdict(Enum):
    """What we can actually say about the claim."""

    #: The figure's own registered public channel carried this mint inside the
    #: announcement window. The only case that is evidence of anything.
    ANNOUNCED = "announced"
    #: A claim was made and nothing confirmed it. The base case, and the one
    #: the population base rate is about.
    UNCORROBORATED = "uncorroborated"
    #: The deployer or its funder cluster has minted claims on figures before.
    #: The most identifiable actor in the market.
    SERIAL_IMPERSONATOR = "serial_impersonator"
    #: The figure's own channel carried a denial, or already registered a
    #: different canonical mint.
    CONTRADICTED = "contradicted"
    #: No figure claimed. Most launches.
    NO_CLAIM = "no_claim"


@dataclass
class Figure:
    """One public figure, project or brand, and their self-published surfaces.

    `channels` holds handles the figure links from their own public profile.
    That provenance is the whole basis for treating a mention from one as a
    confirmation, so a handle nobody can point at a public self-published
    source for does not belong here -- it would turn a stranger's channel into
    an oracle for that person's intentions.
    """

    key: str
    display: str
    category: Category
    aliases: Tuple[str, ...] = ()
    #: Public channel handles (Telegram, and the account names used on other
    #: public surfaces) that the figure publishes as their own.
    channels: Tuple[str, ...] = ()
    #: Addresses the figure has publicly disclosed as theirs. Almost always
    #: empty, and that is correct: an undisclosed address attributed by a
    #: third party is a guess, and sizing on a guess about whose wallet this
    #: is the most expensive kind of guess available.
    wallets: Tuple[str, ...] = ()
    region: str = "global"
    language: str = "en"
    #: A mint the figure has publicly declared canonical. A second token
    #: claiming them while this exists is contradicted by construction.
    canonical_mint: str = ""
    notes: str = ""

    def terms(self) -> Tuple[str, ...]:
        return tuple({_normalise(term) for term
                      in (self.display, *self.aliases) if term})


def _normalise(text: str) -> str:
    return _NORMALISE.sub("", (text or "").lower())


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


@dataclass
class Claim:
    """A launch's claim on a figure, and where in the metadata it was made."""

    figure_key: str
    display: str
    category: Category
    #: symbol | name | description | social_link | channel_link
    evidence: str
    matched_term: str
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {"figure": self.figure_key, "display": self.display,
                "category": self.category.value, "evidence": self.evidence,
                "matched": self.matched_term, "similarity": round(self.similarity, 3)}


@dataclass
class IdentityAssessment:
    """Everything the desk can say about who a launch claims to be."""

    mint: str
    verdict: Verdict
    claims: List[Claim] = field(default_factory=list)
    corroborated_by: str = ""
    corroboration_lag_s: Optional[float] = None
    deployer: str = ""
    prior_impersonations: int = 0
    detail: str = ""

    @property
    def claimed(self) -> bool:
        return self.verdict is not Verdict.NO_CLAIM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mint": self.mint, "verdict": self.verdict.value,
            "claims": [claim.to_dict() for claim in self.claims],
            "corroborated_by": self.corroborated_by,
            "corroboration_lag_s": self.corroboration_lag_s,
            "deployer": self.deployer,
            "prior_impersonations": self.prior_impersonations,
            "detail": self.detail,
        }


@dataclass
class VerdictOutcomes:
    """Realised results per verdict class. What replaces a hard-coded opinion."""

    decisions: int = 0
    resolved: int = 0
    wins: int = 0
    total_return: float = 0.0
    best_return: float = 0.0

    def observe(self, multiple: float) -> None:
        self.resolved += 1
        self.total_return += float(multiple)
        self.best_return = max(self.best_return, float(multiple))
        if multiple > 1.0:
            self.wins += 1

    def to_dict(self) -> Dict[str, Any]:
        if not self.resolved:
            # Never a zero. An unmeasured base rate is not a bad one, and a
            # zero here would suppress the whole class before it was ever
            # given a chance to be measured.
            return {"decisions": self.decisions, "resolved": 0,
                    "hit_rate": None, "mean_multiple": None,
                    "best_multiple": None, "data_status": "DATA_BLOCKED"}
        return {
            "decisions": self.decisions, "resolved": self.resolved,
            "hit_rate": round(self.wins / self.resolved, 4),
            "mean_multiple": round(self.total_return / self.resolved, 4),
            "best_multiple": round(self.best_return, 4),
            "data_status": "OK",
        }


class IdentityWatch:
    """The figure registry, the claim matcher, and the corroboration ledger."""

    def __init__(self, *, announcement_window_s: float = ANNOUNCEMENT_WINDOW_S,
                 match_floor: float = NAME_MATCH_FLOOR,
                 max_mentions: int = 50_000):
        self.announcement_window_s = float(announcement_window_s)
        self.match_floor = float(match_floor)
        self.max_mentions = int(max_mentions)
        self.figures: Dict[str, Figure] = {}
        #: normalised term -> figure key. Built once, kept flat: matching is
        #: on the candidate path and a per-launch walk of every alias of every
        #: figure is a linear scan in the one place that cannot afford one.
        self._terms: Dict[str, str] = {}
        #: handle (lowered) -> figure key, for corroboration.
        self._channel_owner: Dict[str, str] = {}
        #: mint -> (handle, first seen at) for mentions from OWNED channels.
        self._owned_mentions: Dict[str, Tuple[str, float]] = {}
        #: deployer -> how many uncorroborated claims it has minted.
        self._impersonations: Dict[str, int] = {}
        self.outcomes: Dict[str, VerdictOutcomes] = {
            verdict.value: VerdictOutcomes() for verdict in Verdict}
        self.assessed = 0
        self.claims_found = 0

    # --- registry --------------------------------------------------------

    def register(self, figure: Figure) -> None:
        self.figures[figure.key] = figure
        for term in figure.terms():
            if len(term) < 3:
                # Two-character aliases match half the market. A figure whose
                # only distinguishing term is that short is not identifiable
                # from metadata and pretending otherwise fills the ledger with
                # claims nobody made.
                continue
            self._terms.setdefault(term, figure.key)
        for handle in figure.channels:
            self._channel_owner[handle.lower().lstrip("@")] = figure.key

    def load_yaml(self, path: str) -> int:
        """Load the declared registry. Declaration, not code, like sources."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except (OSError, ImportError) as exc:
            logger.warning("figure registry not loaded from %s: %s", path, exc)
            return 0
        except Exception as exc:
            logger.warning("figure registry at %s is malformed: %s", path, exc)
            return 0
        loaded = 0
        for row in payload.get("figures") or []:
            try:
                figure = Figure(
                    key=str(row["key"]), display=str(row.get("display", row["key"])),
                    category=Category(str(row.get("category", "celebrity"))),
                    aliases=tuple(row.get("aliases") or ()),
                    channels=tuple(row.get("channels") or ()),
                    wallets=tuple(row.get("wallets") or ()),
                    region=str(row.get("region", "global")),
                    language=str(row.get("language", "en")),
                    canonical_mint=str(row.get("canonical_mint", "")),
                    notes=str(row.get("notes", "")))
            except (KeyError, ValueError) as exc:
                logger.warning("skipping malformed figure row: %s", exc)
                continue
            self.register(figure)
            loaded += 1
        return loaded

    # --- corroboration input ---------------------------------------------

    def note_channel_message(self, handle: str, mints: Sequence[str],
                             at: Optional[float] = None) -> int:
        """A public message from a channel. Only OWNED channels can confirm.

        Every channel the desk reads is passed through here; the filter is
        ownership, and it is checked against a registry that was written
        before the launch existed. That ordering is what stops a launch from
        supplying its own corroboration by naming a channel it controls.
        """
        moment = time.time() if at is None else at
        owner = self._channel_owner.get((handle or "").lower().lstrip("@"))
        if owner is None:
            return 0
        recorded = 0
        for mint in mints:
            if not mint or mint in self._owned_mentions:
                continue
            if len(self._owned_mentions) >= self.max_mentions:
                self._owned_mentions.pop(next(iter(self._owned_mentions)), None)
            self._owned_mentions[mint] = (handle, moment)
            recorded += 1
        return recorded

    def declare_canonical(self, figure_key: str, mint: str) -> bool:
        """The figure has published a token as theirs. Everything else claiming
        them from now on is contradicted by their own statement."""
        figure = self.figures.get(figure_key)
        if figure is None or not mint:
            return False
        self.figures[figure_key] = Figure(**{**vars(figure), "canonical_mint": mint})
        return True

    # --- matching --------------------------------------------------------

    def match(self, *, symbol: str = "", name: str = "", description: str = "",
              links: Sequence[str] = ()) -> List[Claim]:
        """Which figures this launch's metadata claims, and on what evidence.

        Ordered by how hard the claim is to make accidentally. A linked
        channel that a figure owns is nearly unfakeable as a coincidence; a
        four-letter symbol matching a surname happens all day.
        """
        claims: List[Claim] = []
        seen: Set[str] = set()

        for url in links or ():
            handle = _handle_from_url(url)
            owner = self._channel_owner.get(handle.lower()) if handle else None
            if owner and owner not in seen:
                figure = self.figures[owner]
                seen.add(owner)
                claims.append(Claim(owner, figure.display, figure.category,
                                    "channel_link", handle, 1.0))

        for evidence, text in (("symbol", symbol), ("name", name),
                               ("description", description)):
            if not text:
                continue
            for figure_key, term, score in self._scan(text, evidence):
                if figure_key in seen:
                    continue
                seen.add(figure_key)
                figure = self.figures[figure_key]
                claims.append(Claim(figure_key, figure.display, figure.category,
                                    evidence, term, score))
        if claims:
            self.claims_found += 1
        return claims

    def _scan(self, text: str, evidence: str) -> List[Tuple[str, str, float]]:
        """Exact normalised containment first, near-match only on short fields.

        Fuzzy matching a description is how every launch mentioning "trump
challenge" becomes a claim; fuzzy matching a six-character symbol is how a
        deliberate one-letter misspelling is caught. So the expensive
        comparison runs only where it pays.
        """
        normalised = _normalise(text)
        if not normalised:
            return []
        hits: List[Tuple[str, str, float]] = []
        for term, figure_key in self._terms.items():
            if term in normalised:
                hits.append((figure_key, term, 1.0))
        if hits or evidence == "description":
            return hits
        for term, figure_key in self._terms.items():
            if abs(len(term) - len(normalised)) > 3:
                continue
            score = _similar(term, normalised)
            if score >= self.match_floor:
                hits.append((figure_key, term, score))
        return hits

    # --- assessment ------------------------------------------------------

    def assess(self, mint: str, *, symbol: str = "", name: str = "",
               description: str = "", links: Sequence[str] = (),
               deployer: str = "", created_at: Optional[float] = None,
               now: Optional[float] = None) -> IdentityAssessment:
        """The full verdict for one launch."""
        moment = time.time() if now is None else now
        birth = float(created_at) if created_at else moment
        self.assessed += 1
        claims = self.match(symbol=symbol, name=name, description=description,
                            links=links)
        if not claims:
            self.outcomes[Verdict.NO_CLAIM.value].decisions += 1
            return IdentityAssessment(mint=mint, verdict=Verdict.NO_CLAIM,
                                      deployer=deployer)

        mention = self._owned_mentions.get(mint)
        if mention is not None:
            handle, seen_at = mention
            owner = self._channel_owner.get(handle.lower())
            lag = max(0.0, seen_at - birth)
            if owner in {claim.figure_key for claim in claims}:
                if lag <= self.announcement_window_s:
                    return self._finalise(IdentityAssessment(
                        mint=mint, verdict=Verdict.ANNOUNCED, claims=claims,
                        corroborated_by=handle, corroboration_lag_s=round(lag, 1),
                        deployer=deployer,
                        detail=f"{handle} carried this mint {lag:.0f}s after creation"))
                # Outside the window it is a reaction, and saying so is the
                # point: a figure posting a chart six hours in did not launch
                # this, and treating it as an announcement would let the
                # market manufacture its own corroboration.
                return self._finalise(IdentityAssessment(
                    mint=mint, verdict=Verdict.UNCORROBORATED, claims=claims,
                    deployer=deployer, corroboration_lag_s=round(lag, 1),
                    detail=(f"{handle} mentioned this {lag/60:.0f} minutes after "
                            "creation -- a reaction, not an announcement")))

        for claim in claims:
            figure = self.figures.get(claim.figure_key)
            if figure is not None and figure.canonical_mint and figure.canonical_mint != mint:
                return self._finalise(IdentityAssessment(
                    mint=mint, verdict=Verdict.CONTRADICTED, claims=claims,
                    deployer=deployer,
                    detail=(f"{figure.display} has publicly declared "
                            f"{figure.canonical_mint} as theirs; this is a second "
                            "token claiming the same person")))

        priors = self._impersonations.get(deployer, 0) if deployer else 0
        if deployer:
            self._impersonations[deployer] = priors + 1
        if priors >= 2:
            return self._finalise(IdentityAssessment(
                mint=mint, verdict=Verdict.SERIAL_IMPERSONATOR, claims=claims,
                deployer=deployer, prior_impersonations=priors,
                detail=(f"this deployer has minted {priors} previous "
                        "uncorroborated claims on public figures")))
        return self._finalise(IdentityAssessment(
            mint=mint, verdict=Verdict.UNCORROBORATED, claims=claims,
            deployer=deployer, prior_impersonations=priors,
            detail="a public figure is claimed and nothing has confirmed it"))

    def _finalise(self, assessment: IdentityAssessment) -> IdentityAssessment:
        self.outcomes[assessment.verdict.value].decisions += 1
        return assessment

    def resolve(self, verdict: Verdict, realised_multiple: float) -> None:
        """Feed a realised outcome back so the class earns its own base rate."""
        self.outcomes[verdict.value].observe(realised_multiple)

    # --- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """What the registry covers and what each verdict class has been worth.

        `unmeasured_classes` is the honest half: a verdict with decisions and
        no resolutions has no base rate yet, and reading its absence as a low
        one is the error this whole module exists to avoid.
        """
        by_category: Dict[str, int] = {}
        for figure in self.figures.values():
            by_category[figure.category.value] = by_category.get(figure.category.value, 0) + 1
        outcomes = {name: bucket.to_dict() for name, bucket in self.outcomes.items()}
        unmeasured = sorted(name for name, row in outcomes.items()
                            if row["data_status"] == "DATA_BLOCKED" and row["decisions"])
        with_channels = sum(1 for figure in self.figures.values() if figure.channels)
        return {
            "schema": IDENTITY_SCHEMA_VERSION,
            "status": "OK" if with_channels else "DEGRADED",
            "detail": ("" if with_channels else
                       "no figure has a registered public channel, so no claim "
                       "can ever be corroborated; every celebrity launch will "
                       "read UNCORROBORATED regardless of the truth"),
            "figures": len(self.figures),
            "figures_with_channels": with_channels,
            "terms_indexed": len(self._terms),
            "by_category": dict(sorted(by_category.items())),
            "assessed": self.assessed,
            "claims_found": self.claims_found,
            "owned_mentions": len(self._owned_mentions),
            "serial_impersonators": sum(1 for count in self._impersonations.values()
                                        if count >= 2),
            "outcomes": outcomes,
            "unmeasured_classes": unmeasured,
        }


def _handle_from_url(url: str) -> str:
    match = re.search(r"(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z][A-Za-z0-9_]{4,31})",
                      url or "")
    if match:
        return match.group(1)
    match = re.search(r"(?:twitter\.com|x\.com)/([A-Za-z0-9_]{2,15})", url or "")
    return match.group(1) if match else ""
