"""Millions of past launches, compressed to what a decision can carry.

The extraction layer can pull years of the chain. What it cannot do is hold
them: a 4GB box running a live desk has no room for a million reconstructed
episodes, and a prior that has to be recomputed by scanning a warehouse is a
prior no T0 decision will ever consult.

So the history is distilled once, offline, into fixed-size records the desk
loads at startup and reads in a dictionary lookup. A hundred thousand
deployers at forty bytes each is four megabytes -- small enough to sit
beside a running desk, and the difference between a first-launch deployer
being unknown and being the one whose previous nineteen tokens all rugged
inside a minute.

Three shapes, because they answer three different questions:

    DEPLOYER    what has this creator done before
    FUNDER      what happens to launches this wallet funds
    COHORT      the base rate for this venue at this hour

Two disciplines make this safe to use, and neither is optional.

**Provenance survives.** Every prior is stamped RECONSTRUCTED. A model or a
size that leans on cold history has to be able to say so, because a
reconstruction flatters itself in known ways -- survivorship, latency, depth
-- and a number that arrives without its provenance cannot be discounted for
any of them.

**A distillate has a horizon.** It covers up to an instant, and it refuses
to answer about anything before that instant. Serving a deployer prior built
from a launch that had not happened yet is lookahead, and lookahead is the
one contamination that cannot be undone once it is in a training set. For
live decisions this is free -- the cold data is all in the past -- and it
costs one comparison to make it impossible rather than merely unlikely.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

COLD_DISTILLATION_SCHEMA_VERSION = "v1"

#: Stamped onto every prior this module serves. Deliberately the same word
#: the backfill layer uses, so a consumer already filtering on provenance
#: catches these too without being taught a second vocabulary.
RECONSTRUCTED = "reconstructed"

#: A deployer needs this many past launches before a rate computed from them
#: is worth anything. Below it the record is kept -- knowing a deployer has
#: launched twice before is itself information -- but the RATES are withheld.
MIN_LAUNCHES_FOR_RATE = 5

#: How many deployers to keep. The tail of one-launch creators is most of
#: the population and almost none of the information, so the artifact keeps
#: the deployers with history and lets the rest be genuinely unknown.
MAX_DEPLOYERS = 200_000

#: Multiple at or above which a launch counts as having run. Same threshold
#: the latency ledger uses, on purpose: two modules disagreeing about what
#: "worked" makes their numbers incomparable for no reason.
RAN_MULTIPLE = 2.0


def _quantise(value: Optional[float], places: int = 4) -> Optional[float]:
    """Round for the artifact. Storage, not precision, is the constraint."""
    return None if value is None else round(float(value), places)


@dataclass
class DeployerDNA:
    """One creator's history, in the space of a short line.

    COLLAPSES and RUGS are counted separately, and the separation is not
    fussiness. A reconstruction can see that a price fell to nothing; it
    cannot see who made it fall. Recording a collapse as a rug teaches a rug
    model to predict drawdowns instead of rugs, which is a different and much
    easier thing that happens to look like success on reconstructed data. So
    the collapse rate is always available and the rug rate exists only where
    something actually observed the rug.
    """

    deployer: str
    launches: int = 0
    collapses: int = 0
    rug_labelled: int = 0
    rugs: int = 0
    ran: int = 0
    peak_multiples: List[float] = field(default_factory=list)
    first_launch_at: float = 0.0
    last_launch_at: float = 0.0

    def observe(self, *, at: float, collapsed: bool,
                rugged: Optional[bool],
                peak_multiple: Optional[float]) -> None:
        self.launches += 1
        self.first_launch_at = min(self.first_launch_at or at, at)
        self.last_launch_at = max(self.last_launch_at, at)
        if collapsed:
            self.collapses += 1
        if rugged is not None:
            self.rug_labelled += 1
            if rugged:
                self.rugs += 1
        if peak_multiple is not None:
            peak = float(peak_multiple)
            self.peak_multiples.append(peak)
            if peak >= RAN_MULTIPLE:
                self.ran += 1
            # Bounded: the median of forty launches and the median of four
            # hundred differ by nothing worth four hundred floats.
            if len(self.peak_multiples) > 64:
                self.peak_multiples.sort()
                self.peak_multiples = self.peak_multiples[::2]

    @property
    def measurable(self) -> bool:
        return self.launches >= MIN_LAUNCHES_FOR_RATE

    def as_prior(self) -> Dict[str, Any]:
        """What a decision reads. Rates only where they mean something."""
        prior: Dict[str, Any] = {
            "provenance": RECONSTRUCTED,
            "launches": self.launches,
            "first_launch_at": self.first_launch_at or None,
            "last_launch_at": self.last_launch_at or None,
        }
        if not self.measurable:
            prior["status"] = "DATA_BLOCKED"
            prior["reason"] = (
                f"{self.launches} prior launch(es); a rate needs "
                f"{MIN_LAUNCHES_FOR_RATE}")
            return prior
        prior["status"] = "OK"
        prior["collapse_rate"] = _quantise(self.collapses / self.launches)
        prior["ran_rate"] = _quantise(self.ran / self.launches)
        # Only where something actually LABELLED a rug. A reconstruction sees
        # a price fall to nothing and cannot see who pushed it, so a rug rate
        # inferred from collapses would be a claim about intent nobody made.
        if self.rug_labelled >= MIN_LAUNCHES_FOR_RATE:
            prior["rug_rate"] = _quantise(self.rugs / self.rug_labelled)
            prior["rug_rate_from"] = self.rug_labelled
        else:
            prior["rug_rate"] = None
            prior["rug_rate_reason"] = (
                f"only {self.rug_labelled} of {self.launches} launches carry a "
                "rug label; a collapse is not a rug")
        prior["median_peak_multiple"] = (
            _quantise(statistics.median(self.peak_multiples))
            if self.peak_multiples else None)
        return prior

    def as_row(self) -> List[Any]:
        """The artifact's on-disk shape: a list, not a dict.

        Deliberate. A dict per deployer spends its keys again for every one
        of two hundred thousand records, and the keys are the same every
        time -- roughly two thirds of the file for no information.
        """
        return [self.deployer, self.launches, self.collapses, self.ran,
                _quantise(self.first_launch_at, 0),
                _quantise(self.last_launch_at, 0),
                [_quantise(value, 3) for value in self.peak_multiples],
                self.rug_labelled, self.rugs]

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "DeployerDNA":
        dna = cls(str(row[0]))
        dna.launches = int(row[1])
        dna.collapses = int(row[2])
        dna.ran = int(row[3])
        dna.first_launch_at = float(row[4] or 0.0)
        dna.last_launch_at = float(row[5] or 0.0)
        dna.peak_multiples = [float(value) for value in (row[6] or [])]
        dna.rug_labelled = int(row[7]) if len(row) > 7 else 0
        dna.rugs = int(row[8]) if len(row) > 8 else 0
        return dna


@dataclass
class FunderDNA:
    """What happens to launches this wallet funded. Bounded the same way."""

    funder: str
    launches: int = 0
    collapses: int = 0
    ran: int = 0

    def as_prior(self) -> Dict[str, Any]:
        if self.launches < MIN_LAUNCHES_FOR_RATE:
            return {"provenance": RECONSTRUCTED, "status": "DATA_BLOCKED",
                    "launches": self.launches}
        return {
            "provenance": RECONSTRUCTED, "status": "OK",
            "launches": self.launches,
            "collapse_rate": _quantise(self.collapses / self.launches),
            "ran_rate": _quantise(self.ran / self.launches),
        }


class ColdDistillate:
    """The artifact: everything the cold history says, in a lookup.

    Loaded once at startup and read in constant time. It answers about the
    past and refuses to answer about anything at or before its own horizon
    -- see the module docstring on why that refusal is not optional.
    """

    def __init__(self, covers_until: float = 0.0, source: str = ""):
        self.covers_until = float(covers_until)
        self.source = str(source)
        self.deployers: Dict[str, DeployerDNA] = {}
        self.funders: Dict[str, FunderDNA] = {}
        #: (venue, hour of day UTC) -> [launches, rugs, ran]
        self.cohorts: Dict[Tuple[str, int], List[int]] = {}
        self.launches_distilled = 0
        self.skipped_unresolved = 0
        self.lookahead_refusals = 0

    # --- building --------------------------------------------------------

    def observe(self, record: Dict[str, Any]) -> bool:
        """One resolved historical launch. Returns whether it contributed.

        An unresolved launch contributes NOTHING rather than contributing as
        a non-rug. Treating "we never found out" as "it survived" is how a
        rug rate built from cold data comes out lower than the truth, in the
        direction that makes every deployer look safer than they are.
        """
        outcome = record.get("outcome") or record.get("final_outcome") or {}
        if outcome.get("status") not in (None, "OK"):
            self.skipped_unresolved += 1
            return False
        rugged = outcome.get("rugged")
        collapsed = bool(outcome.get("collapsed") or rugged)
        # `peak_multiple` is what the reconstruction path writes,
        # `max_multiple` what the live builder writes. Both, so one adapter
        # is not needed between two halves of the same desk.
        peak = outcome.get("max_multiple")
        if peak is None:
            peak = outcome.get("peak_multiple")
        if rugged is None and peak is None and not outcome.get("collapsed"):
            self.skipped_unresolved += 1
            return False
        at = float(record.get("created_at", 0) or 0)
        if at <= 0:
            self.skipped_unresolved += 1
            return False
        self.launches_distilled += 1
        self.covers_until = max(self.covers_until, at)

        deployer = str(record.get("creator", "") or "")
        if deployer:
            dna = self.deployers.get(deployer)
            if dna is None:
                dna = DeployerDNA(deployer)
                self.deployers[deployer] = dna
            dna.observe(at=at, collapsed=collapsed, rugged=rugged,
                        peak_multiple=peak)

        ran = peak is not None and float(peak) >= RAN_MULTIPLE
        for transfer in record.get("funding_transfers", []) or []:
            funder = str(transfer.get("from", "") or "")
            if not funder:
                continue
            entry = self.funders.get(funder)
            if entry is None:
                entry = FunderDNA(funder)
                self.funders[funder] = entry
            entry.launches += 1
            entry.collapses += 1 if collapsed else 0
            entry.ran += 1 if ran else 0

        venue = str(record.get("venue", "") or record.get("program", "") or "unknown")
        hour = int(time.gmtime(at).tm_hour)
        cohort = self.cohorts.setdefault((venue, hour), [0, 0, 0])
        cohort[0] += 1
        cohort[1] += 1 if collapsed else 0
        cohort[2] += 1 if ran else 0
        return True

    def compact(self, max_deployers: int = MAX_DEPLOYERS) -> int:
        """Drop the single-launch tail. Returns how many were dropped.

        Most creators launch once. That fact is worth nothing as a prior --
        a deployer with one launch is indistinguishable from one with none
        for every purpose a decision has -- and it is most of the file.
        """
        if len(self.deployers) <= max_deployers:
            return 0
        ordered = sorted(self.deployers.values(),
                         key=lambda dna: (dna.launches, dna.last_launch_at),
                         reverse=True)
        keep = {dna.deployer for dna in ordered[:max_deployers]}
        dropped = [key for key in self.deployers if key not in keep]
        for key in dropped:
            del self.deployers[key]
        return len(dropped)

    # --- reading ---------------------------------------------------------

    def deployer_prior(self, deployer: str,
                       as_of: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """This creator's cold prior, or None when there is none to give.

        `as_of` is the instant the decision is being made. A distillate whose
        horizon is AFTER that instant contains launches the decision could
        not have seen, and serving it would be lookahead -- the one
        contamination that cannot be undone once it reaches a training set.
        """
        if as_of is not None and self.covers_until > float(as_of):
            self.lookahead_refusals += 1
            return None
        dna = self.deployers.get(str(deployer or ""))
        return dna.as_prior() if dna is not None else None

    def funder_prior(self, funder: str,
                     as_of: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if as_of is not None and self.covers_until > float(as_of):
            self.lookahead_refusals += 1
            return None
        entry = self.funders.get(str(funder or ""))
        return entry.as_prior() if entry is not None else None

    def cohort_prior(self, venue: str, hour: int) -> Optional[Dict[str, Any]]:
        cohort = self.cohorts.get((str(venue), int(hour)))
        if not cohort or cohort[0] < MIN_LAUNCHES_FOR_RATE:
            return None
        launches, collapses, ran = cohort
        return {
            "provenance": RECONSTRUCTED, "status": "OK",
            "launches": launches,
            "collapse_rate": _quantise(collapses / launches),
            "ran_rate": _quantise(ran / launches),
        }

    # --- persistence -----------------------------------------------------

    def save(self, path: Path) -> bool:
        payload = {
            "schema": COLD_DISTILLATION_SCHEMA_VERSION,
            "provenance": RECONSTRUCTED,
            "covers_until": self.covers_until,
            "source": self.source,
            "launches_distilled": self.launches_distilled,
            "skipped_unresolved": self.skipped_unresolved,
            "deployers": [dna.as_row() for dna in self.deployers.values()],
            "funders": [[entry.funder, entry.launches, entry.collapses, entry.ran]
                        for entry in self.funders.values()],
            "cohorts": [[venue, hour, *counts]
                        for (venue, hour), counts in self.cohorts.items()],
        }
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")),
                                 encoding="utf-8")
            temporary.replace(path)
            return True
        except OSError as exc:
            logger.warning("cold distillate not saved: %s", exc)
            return False

    @classmethod
    def load(cls, path: Path) -> Optional["ColdDistillate"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("cold distillate not loaded: %s", exc)
            return None
        if payload.get("schema") != COLD_DISTILLATION_SCHEMA_VERSION:
            # A distillate written by a different version is not a distillate
            # this code understands, and guessing at its layout would put
            # silently wrong priors in front of decisions.
            logger.warning(
                "cold distillate is schema %s, this build reads %s; ignoring",
                payload.get("schema"), COLD_DISTILLATION_SCHEMA_VERSION)
            return None
        distillate = cls(float(payload.get("covers_until", 0.0) or 0.0),
                         str(payload.get("source", "") or ""))
        distillate.launches_distilled = int(payload.get("launches_distilled", 0) or 0)
        distillate.skipped_unresolved = int(payload.get("skipped_unresolved", 0) or 0)
        for row in payload.get("deployers", []) or []:
            dna = DeployerDNA.from_row(row)
            distillate.deployers[dna.deployer] = dna
        for row in payload.get("funders", []) or []:
            entry = FunderDNA(str(row[0]), int(row[1]), int(row[2]), int(row[3]))
            distillate.funders[entry.funder] = entry
        for row in payload.get("cohorts", []) or []:
            distillate.cohorts[(str(row[0]), int(row[1]))] = [
                int(row[2]), int(row[3]), int(row[4])]
        return distillate

    def report(self) -> Dict[str, Any]:
        measurable = sum(1 for dna in self.deployers.values() if dna.measurable)
        return {
            "schema": COLD_DISTILLATION_SCHEMA_VERSION,
            "status": "OK" if self.launches_distilled else "DATA_BLOCKED",
            "provenance": RECONSTRUCTED,
            "source": self.source or None,
            "covers_until": self.covers_until or None,
            "launches_distilled": self.launches_distilled,
            "skipped_unresolved": self.skipped_unresolved,
            "deployers": len(self.deployers),
            "deployers_with_a_rate": measurable,
            "funders": len(self.funders),
            "cohorts": len(self.cohorts),
            "lookahead_refusals": self.lookahead_refusals,
            "detail": ("priors distilled from reconstructed history; stamped "
                       "so a decision leaning on them can be discounted for "
                       "the survivorship, latency and depth a reconstruction "
                       "flatters itself with"),
        }


def distil(records: Iterable[Dict[str, Any]], *, source: str = "",
           max_deployers: int = MAX_DEPLOYERS) -> ColdDistillate:
    """Run the whole distillation. Offline: this is not a hot path."""
    distillate = ColdDistillate(source=source)
    for record in records:
        distillate.observe(record)
    dropped = distillate.compact(max_deployers)
    if dropped:
        logger.info("cold distillation dropped %d single-launch deployers", dropped)
    return distillate
