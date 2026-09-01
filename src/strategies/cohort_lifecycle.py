"""What happened to the opening cohort AFTER it bought.

``BuyerDNA`` answers who entered and in what order. It stops at the fill, and
the fill is the least interesting moment: every launch has a first twenty-five
buyers, including every launch that goes to zero. The question that separates
them is what that exact cohort did next, and whether anybody was willing to
take the other side when it left.

Three readings, each answering a question the entry fingerprint cannot:

``cohort_retention`` follows the first 10 / 25 / 70 buyers as named sets
through time. A cohort still holding 90% of what it bought at sixty seconds
and a cohort down to 20% are the same fingerprint at T0 and opposite tokens
afterwards.

``post_sniper_absorption`` is the one that matters most. When the opening
cohort distributes, somebody is on the other side. If independent wallets --
not the sellers' own funders, not each other -- absorb that supply without the
price breaking, that is demand arriving to replace the people who were always
going to leave. If the same supply hits and the price collapses, the cohort
WAS the demand. Identical sell volume, opposite meanings, and only the actor
graph can tell them apart.

``late_chaser`` measures the arrival of wallets with poor historical records
while the skilled opening cohort is distributing. Retail buying what snipers
are selling is the classic distribution top, and it is exit evidence rather
than entry evidence -- so it belongs to the position, not to the candidate.

Every reading is computed from public chain observations the desk already
holds: transfers, entries, and the wallet skill/independence machinery in
``actor_graph``. Nothing here needs a private feed, and nothing here invents a
number when the observations are missing -- an unmeasured cohort reports
DATA_BLOCKED, because a retention of zero and a retention nobody watched are
opposite findings that a float cannot distinguish.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

COHORT_SCHEMA_VERSION = "v1"

#: The cohort depths worth following separately. 10 is the same-block crowd,
#: 25 is the sniper layer, 70 reaches the first wave of ordinary buyers.
COHORT_DEPTHS: Tuple[int, ...] = (10, 25, 70)

#: Retention is sampled at these ages, in seconds since the cohort's own entry.
RETENTION_MARKS: Tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 60.0)

#: Below this share of a cohort observed selling or holding, retention is not
#: measured -- it is guessed from whoever happened to be visible.
MIN_COHORT_COVERAGE = 0.5

#: An absorber whose independence from the selling cohort is below this is not
#: counted as new demand. A funder buying back its own wallets' supply is one
#: actor moving inventory between pockets, and it prints as absorption.
MIN_ABSORBER_INDEPENDENCE = 0.5

#: Wallets whose historical skill is at or below this are "late chasers" for
#: the purposes of the distribution reading.
CHASER_SKILL_CEILING = 0.35


@dataclass
class RetentionReading:
    """How much of what a cohort bought it still holds, per mark."""

    status: str
    depth: int = 0
    cohort_size: int = 0
    observed: int = 0
    #: age in seconds -> fraction of the cohort's purchased units still held.
    retained: Dict[float, float] = field(default_factory=dict)
    #: Wallets that had fully exited by the last observed mark.
    fully_exited: List[str] = field(default_factory=list)
    detail: str = ""

    @property
    def coverage(self) -> float:
        return self.observed / self.cohort_size if self.cohort_size else 0.0


@dataclass
class AbsorptionReading:
    """Who took the other side when the opening cohort sold."""

    status: str
    #: Units the opening cohort distributed over the window.
    cohort_sold_units: float = 0.0
    #: Of those, units bought by wallets independent of the selling cohort.
    independent_absorbed_units: float = 0.0
    #: Units absorbed by wallets NOT independent of the sellers. Supply that
    #: went back to the same actor is not demand and is reported separately
    #: rather than being quietly added to either side.
    related_absorbed_units: float = 0.0
    absorbers: int = 0
    independent_absorbers: int = 0
    #: Price at the end of the window over price at the start.
    price_ratio: Optional[float] = None
    detail: str = ""

    @property
    def absorption_ratio(self) -> Optional[float]:
        """Independent demand as a fraction of what the cohort distributed."""
        if self.status != "OK" or self.cohort_sold_units <= 0:
            return None
        return self.independent_absorbed_units / self.cohort_sold_units

    @property
    def verdict(self) -> str:
        """A label, only where the evidence supports one.

        ABSORBED means independent wallets took most of the supply and the
        price held. FAILED means the supply was not absorbed or the price
        broke taking it. CAPTURED means the buyers were related to the
        sellers, which looks like absorption and is not.
        """
        ratio = self.absorption_ratio
        if ratio is None:
            return "DATA_BLOCKED"
        related = self.related_absorbed_units
        absorbed_total = self.independent_absorbed_units + related
        if absorbed_total > 0 and related / absorbed_total > 0.5:
            return "CAPTURED"
        held = self.price_ratio is None or self.price_ratio >= 0.9
        if ratio >= 0.7 and held:
            return "ABSORBED"
        if ratio < 0.35 or (self.price_ratio is not None and self.price_ratio < 0.8):
            return "FAILED"
        return "PARTIAL"


@dataclass
class ChaserReading:
    """Low-skill arrivals while the skilled cohort leaves."""

    status: str
    arrivals: int = 0
    chaser_arrivals: int = 0
    chaser_notional_usd: float = 0.0
    skilled_exiting: int = 0
    detail: str = ""

    @property
    def chaser_share(self) -> Optional[float]:
        if self.status != "OK" or self.arrivals <= 0:
            return None
        return self.chaser_arrivals / self.arrivals

    @property
    def is_distribution_pattern(self) -> bool:
        """Retail arriving into skilled selling, both measured."""
        share = self.chaser_share
        return (share is not None and share >= 0.6 and self.skilled_exiting >= 2)


def opening_cohort(entries: Sequence[Any], depth: int) -> List[str]:
    """The first `depth` DISTINCT wallets to enter, in order.

    Distinct on purpose: one wallet buying eight times is one member of the
    opening cohort, and counting its fills would let a single actor fill the
    cohort by itself -- which is exactly the manufactured pattern the actor
    graph exists to see through.
    """
    seen: List[str] = []
    for entry in sorted(entries, key=lambda item: float(getattr(item, "timestamp", 0.0))):
        wallet = str(getattr(entry, "wallet", "") or "")
        if wallet and wallet not in seen:
            seen.append(wallet)
        if len(seen) >= depth:
            break
    return seen


def cohort_retention(entries: Sequence[Any], flows: Sequence[Dict[str, Any]],
                     depth: int, as_of: float) -> RetentionReading:
    """Fraction of the cohort's purchased units still held, at each mark.

    `flows` are public transfer/trade observations, each a mapping with
    `wallet`, `timestamp`, `units` (signed: positive bought, negative sold).
    A wallet with no flow after its entry is HELD, not unknown -- its entry is
    itself an observation, and absence of a sale on a chain the desk is
    streaming is evidence. A wallet the desk never saw enter is unknown, and
    those are what coverage counts.
    """
    cohort = opening_cohort(entries, depth)
    if not cohort:
        return RetentionReading(status="DATA_BLOCKED", depth=depth,
                                detail="no entries observed for this token")
    if len(cohort) < depth:
        # A launch with 25 distinct buyers has no first-70 cohort. Reporting
        # the 25 under the 70's name would emit three perfectly correlated
        # features and tell a model a seventy-wallet cohort held everything,
        # when the seventy wallets never existed.
        return RetentionReading(
            status="DATA_BLOCKED", depth=depth, cohort_size=len(cohort),
            detail=(f"only {len(cohort)} distinct buyers entered; there is no "
                    f"first-{depth} cohort to follow"))
    members = set(cohort)
    bought: Dict[str, float] = {}
    entered_at: Dict[str, float] = {}
    for entry in entries:
        wallet = str(getattr(entry, "wallet", "") or "")
        if wallet not in members:
            continue
        entered_at.setdefault(wallet, float(getattr(entry, "timestamp", 0.0)))

    # Units bought come from the flows, because an Entry carries capital, not
    # units, and dividing one by a price the desk may not have measured would
    # manufacture a denominator.
    sold: Dict[str, float] = {wallet: 0.0 for wallet in cohort}
    for flow in flows:
        wallet = str(flow.get("wallet", "") or "")
        if wallet not in members:
            continue
        units = float(flow.get("units", 0.0) or 0.0)
        stamp = float(flow.get("timestamp", 0.0) or 0.0)
        if units > 0 and stamp <= entered_at.get(wallet, stamp) + 1e-9:
            bought[wallet] = bought.get(wallet, 0.0) + units
        elif units < 0:
            sold[wallet] = sold.get(wallet, 0.0) - units

    observed = [wallet for wallet in cohort if bought.get(wallet, 0.0) > 0]
    if not observed:
        return RetentionReading(
            status="DATA_BLOCKED", depth=depth, cohort_size=len(cohort),
            detail="no purchase units observed for any cohort member")
    coverage = len(observed) / len(cohort)
    if coverage < MIN_COHORT_COVERAGE:
        return RetentionReading(
            status="DATA_BLOCKED", depth=depth, cohort_size=len(cohort),
            observed=len(observed),
            detail=(f"units observed for {len(observed)} of {len(cohort)} cohort "
                    f"members, below the {MIN_COHORT_COVERAGE:.0%} needed"))

    retained: Dict[float, float] = {}
    for mark in RETENTION_MARKS:
        total_bought = 0.0
        total_held = 0.0
        measurable = False
        for wallet in observed:
            start = entered_at.get(wallet)
            if start is None or as_of < start + mark:
                # The mark has not happened yet for this wallet. Counting it
                # as fully retained would report a fresh cohort as diamond
                # handed, which is the flattering direction of the error.
                continue
            measurable = True
            units = bought.get(wallet, 0.0)
            gone = 0.0
            for flow in flows:
                if str(flow.get("wallet", "") or "") != wallet:
                    continue
                value = float(flow.get("units", 0.0) or 0.0)
                stamp = float(flow.get("timestamp", 0.0) or 0.0)
                if value < 0 and start < stamp <= start + mark:
                    gone += -value
            total_bought += units
            total_held += max(0.0, units - gone)
        if measurable and total_bought > 0:
            retained[mark] = total_held / total_bought
    if not retained:
        return RetentionReading(
            status="DATA_BLOCKED", depth=depth, cohort_size=len(cohort),
            observed=len(observed),
            detail="the cohort is younger than the earliest retention mark")

    exited = [wallet for wallet in observed
              if sold.get(wallet, 0.0) >= bought.get(wallet, 0.0) - 1e-9]
    return RetentionReading(
        status="OK", depth=depth, cohort_size=len(cohort), observed=len(observed),
        retained=retained, fully_exited=sorted(exited),
        detail=f"{len(observed)} of {len(cohort)} cohort members measured")


def post_sniper_absorption(entries: Sequence[Any], flows: Sequence[Dict[str, Any]],
                           independence: Dict[str, float], depth: int,
                           window: Tuple[float, float],
                           marks: Optional[Sequence[Tuple[float, float]]] = None,
                           ) -> AbsorptionReading:
    """Did independent demand take the opening cohort's supply, or not.

    `independence` maps wallet -> its independence score from the selling
    cohort, as produced by ``WalletIndependence``. A buyer that is not
    independent of the sellers is inventory moving between pockets of one
    actor: it absorbs supply on the chain and creates no demand, and counting
    it would make a wash trade look like conviction. It is reported in its own
    field rather than dropped, because "the buyers were the sellers" is a
    finding.

    `marks` are (timestamp, price) observations used only to say whether the
    price survived the distribution.
    """
    start, end = float(window[0]), float(window[1])
    if end <= start:
        return AbsorptionReading(status="DATA_BLOCKED",
                                 detail="absorption window is empty")
    cohort = set(opening_cohort(entries, depth))
    if not cohort:
        return AbsorptionReading(status="DATA_BLOCKED",
                                 detail="no opening cohort observed")

    sold_units = 0.0
    absorbed_independent = 0.0
    absorbed_related = 0.0
    absorbers: Dict[str, float] = {}
    for flow in flows:
        stamp = float(flow.get("timestamp", 0.0) or 0.0)
        if not (start <= stamp <= end):
            continue
        wallet = str(flow.get("wallet", "") or "")
        units = float(flow.get("units", 0.0) or 0.0)
        if units < 0 and wallet in cohort:
            sold_units += -units
        elif units > 0 and wallet not in cohort:
            absorbers[wallet] = absorbers.get(wallet, 0.0) + units

    if sold_units <= 0:
        return AbsorptionReading(
            status="DATA_BLOCKED",
            detail="the opening cohort sold nothing in this window; "
                   "there is no distribution to absorb")

    independent_count = 0
    for wallet, units in absorbers.items():
        score = independence.get(wallet)
        if score is None:
            # Unscored is not independent. Treating an unknown relationship as
            # arm's length is the assumption that makes a Sybil look like a
            # crowd, and it is the one this whole module exists to refuse.
            absorbed_related += units
            continue
        if score >= MIN_ABSORBER_INDEPENDENCE:
            absorbed_independent += units
            independent_count += 1
        else:
            absorbed_related += units

    ratio: Optional[float] = None
    if marks:
        inside = [(stamp, price) for stamp, price in marks if start <= stamp <= end]
        if len(inside) >= 2:
            inside.sort()
            first, last = inside[0][1], inside[-1][1]
            if first > 0:
                ratio = float(last / first)

    return AbsorptionReading(
        status="OK", cohort_sold_units=sold_units,
        independent_absorbed_units=absorbed_independent,
        related_absorbed_units=absorbed_related,
        absorbers=len(absorbers), independent_absorbers=independent_count,
        price_ratio=ratio,
        detail=(f"cohort sold {sold_units:.4g} units; {independent_count} of "
                f"{len(absorbers)} absorbers independent"))


def late_chaser(arrivals: Sequence[Any], exiting_wallets: Sequence[str],
                skills: Dict[str, float], window: Tuple[float, float],
                ) -> ChaserReading:
    """Poor-record wallets arriving while skilled wallets leave.

    Exit evidence, not entry evidence. The pattern -- wallets with bad
    historical records buying in size exactly while the wallets with good ones
    distribute -- says the marginal buyer has changed, and the marginal buyer
    is what the exit is sold into.
    """
    start, end = float(window[0]), float(window[1])
    inside = [entry for entry in arrivals
              if start <= float(getattr(entry, "timestamp", 0.0)) <= end]
    if not inside:
        return ChaserReading(status="DATA_BLOCKED",
                             detail="no arrivals observed in this window")
    scored = [entry for entry in inside
              if skills.get(str(getattr(entry, "wallet", "") or "")) is not None]
    if len(scored) < max(3, len(inside) // 2):
        return ChaserReading(
            status="DATA_BLOCKED", arrivals=len(inside),
            detail=(f"only {len(scored)} of {len(inside)} arrivals have a "
                    "historical record; the rest are unknown, not unskilled"))

    chasers = [entry for entry in scored
               if skills[str(getattr(entry, "wallet", ""))] <= CHASER_SKILL_CEILING]
    notional = sum(float(getattr(entry, "capital_usd", 0.0) or 0.0) for entry in chasers)
    leaving = len({wallet for wallet in exiting_wallets
                   if skills.get(wallet, 0.0) > CHASER_SKILL_CEILING})
    return ChaserReading(
        status="OK", arrivals=len(scored), chaser_arrivals=len(chasers),
        chaser_notional_usd=notional, skilled_exiting=leaving,
        detail=(f"{len(chasers)} of {len(scored)} scored arrivals are "
                f"low-record wallets; {leaving} skilled wallets exiting"))


@dataclass
class CohortReport:
    """Every cohort reading for one token, for the decision snapshot."""

    status: str
    retention: Dict[int, RetentionReading] = field(default_factory=dict)
    absorption: Optional[AbsorptionReading] = None
    chasers: Optional[ChaserReading] = None

    def features(self) -> Dict[str, float]:
        """Only what was measured. An unmeasured cohort contributes nothing.

        Deliberately not zero-filled: a model trained on zeros for unobserved
        cohorts learns that silence predicts whatever silence coincided with,
        which is how a data pipeline becomes a signal.
        """
        out: Dict[str, float] = {}
        for depth, reading in self.retention.items():
            if reading.status != "OK":
                continue
            for mark, value in reading.retained.items():
                out[f"cohort{depth}_retained_{mark:g}s"] = float(value)
        if self.absorption is not None:
            ratio = self.absorption.absorption_ratio
            if ratio is not None:
                out["post_sniper_absorption"] = float(ratio)
            if self.absorption.price_ratio is not None:
                out["absorption_price_ratio"] = float(self.absorption.price_ratio)
        if self.chasers is not None:
            share = self.chasers.chaser_share
            if share is not None:
                out["late_chaser_share"] = float(share)
        return out

    def report(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "schema": COHORT_SCHEMA_VERSION,
            "retention": {
                str(depth): {"status": reading.status,
                             "coverage": round(reading.coverage, 3),
                             "retained": {f"{k:g}s": round(v, 4)
                                          for k, v in reading.retained.items()},
                             "fully_exited": len(reading.fully_exited),
                             "detail": reading.detail}
                for depth, reading in sorted(self.retention.items())},
            "absorption": (None if self.absorption is None else {
                "status": self.absorption.status,
                "verdict": self.absorption.verdict,
                "ratio": self.absorption.absorption_ratio,
                "independent_absorbers": self.absorption.independent_absorbers,
                "absorbers": self.absorption.absorbers,
                "price_ratio": self.absorption.price_ratio,
                "detail": self.absorption.detail}),
            "late_chasers": (None if self.chasers is None else {
                "status": self.chasers.status,
                "share": self.chasers.chaser_share,
                "distribution_pattern": self.chasers.is_distribution_pattern,
                "detail": self.chasers.detail}),
        }


def evaluate_cohorts(entries: Sequence[Any], flows: Sequence[Dict[str, Any]],
                     independence: Dict[str, float], skills: Dict[str, float],
                     as_of: float,
                     absorption_window: Optional[Tuple[float, float]] = None,
                     marks: Optional[Sequence[Tuple[float, float]]] = None,
                     ) -> CohortReport:
    """Every cohort reading the observations support, and no more."""
    retention = {depth: cohort_retention(entries, flows, depth, as_of)
                 for depth in COHORT_DEPTHS}
    absorption = None
    chasers = None
    if absorption_window is not None:
        absorption = post_sniper_absorption(
            entries, flows, independence, COHORT_DEPTHS[1], absorption_window, marks)
        exiting = [wallet for reading in retention.values()
                   for wallet in reading.fully_exited]
        chasers = late_chaser(entries, exiting, skills, absorption_window)
    measured = any(reading.status == "OK" for reading in retention.values())
    return CohortReport(
        status="OK" if measured else "DATA_BLOCKED",
        retention=retention, absorption=absorption, chasers=chasers)
