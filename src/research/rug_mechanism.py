"""What actually killed a token, named rather than lumped.

"Rugged" is one label over at least eight different events, and treating them
as one is why a rug model trained on it generalises badly. They have different
precursors, different warning times, and crucially different ESCAPABILITY:

  LP_PULL           liquidity removed in one transaction. Warning time is
                    approximately zero once it starts; the only defence is
                    having priced the possibility beforehand, from who held
                    the LP.
  MINT_DILUTION     the supply we priced was not the supply that would exist.
                    Visible BEFORE it happens -- a live mint authority is a
                    public fact -- which makes this the cheapest of all of
                    these to avoid, and the most embarrassing to be caught by.
  FREEZE            transfers disabled. Position value goes to zero with the
                    tokens still in the wallet.
  CREATOR_DUMP      the deployer sells their allocation into the book. Slower
                    than an LP pull and often survivable if the exit starts on
                    the first sign rather than the confirmation.
  SNIPER_CASCADE    the first-block buyers exit together. Not the creator at
                    all, which is why creator-only rug models miss it.
  HONEYPOT          buys land, sells do not. The position was never real.
  MIGRATION_STALL   the curve never filled and interest died. No villain, and
                    the most common outcome by far.
  SLOW_BLEED        no single event; attention decayed and the price with it.

Two rules run through the classification.

**Every verdict needs positive evidence.** A collapse with no mechanism
identified is UNCLASSIFIED, not SLOW_BLEED. Slow bleed is the residual that
absorbs everything if it is allowed to be the default, and a residual that
absorbs everything teaches a model nothing.

**Unobserved is not absent.** If the observation stream carries no supply
readings, the answer to "was this a mint dilution" is "we could not see", and
the verdict says so. A classifier that reports MIGRATION_STALL because it
never looked at the supply is worse than one that abstains, because it fills
the training set with confident mislabels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

RUG_MECHANISM_SCHEMA_VERSION = "v1"

#: A drawdown this deep from peak is what makes a token a candidate for
#: classification at all. Above it there is nothing to explain.
DEATH_DRAWDOWN = 0.80

#: LP or reserve share removed in one step for it to read as a pull rather
#: than as ordinary selling.
LP_PULL_FRACTION = 0.50

#: Share of supply the creator must sell for a creator dump.
CREATOR_DUMP_SHARE = 0.10

#: Share of the first-block cohort exiting inside the window for a cascade.
CASCADE_SHARE = 0.60
CASCADE_WINDOW_S = 120.0


class RugMechanism(Enum):
    LP_PULL = "lp_pull"
    MINT_DILUTION = "mint_dilution"
    FREEZE = "freeze"
    CREATOR_DUMP = "creator_dump"
    SNIPER_CASCADE = "sniper_cascade"
    HONEYPOT = "honeypot"
    MIGRATION_STALL = "migration_stall"
    SLOW_BLEED = "slow_bleed"
    #: Died, and we cannot say from what. An honest count of these is the
    #: measure of how much the observation stream is missing.
    UNCLASSIFIED = "unclassified"
    #: Did not die. Kept distinct from "we do not know why it died".
    SURVIVED = "survived"


@dataclass
class RugVerdict:
    """One classification, with the evidence that produced it."""

    mechanism: RugMechanism
    data_status: str = "OK"
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    #: Which mechanisms could not be ruled in or out for want of data. This is
    #: the field that turns a weak label into a usable one: a LP_PULL verdict
    #: with mint supply unobserved is a different training example from one
    #: where the supply was watched throughout and never moved.
    unobserved: List[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.data_status == "OK" and not self.unobserved

    def to_dict(self) -> Dict[str, Any]:
        return {"mechanism": self.mechanism.value, "data_status": self.data_status,
                "detail": self.detail, "evidence": dict(self.evidence),
                "unobserved": list(self.unobserved)}


def _by_type(observations: Sequence[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [row for row in observations if row.get("type") == kind]


def _drawdown_from_peak(observations: Sequence[Dict[str, Any]]) -> Optional[float]:
    """How far below its own peak the token finished, or None if unpriced."""
    multiples = [float(row["price_multiple"]) for row in observations
                 if row.get("price_multiple") is not None]
    if len(multiples) < 2:
        return None
    peak = max(multiples)
    if peak <= 0:
        return None
    return 1.0 - (multiples[-1] / peak)


def classify(observations: Sequence[Dict[str, Any]], *,
             migrated: Optional[bool] = None) -> RugVerdict:
    """Name the mechanism, or say honestly that it cannot be named.

    Ordered by specificity, not by frequency: a token can satisfy several of
    these at once -- a creator who dumps and then pulls the LP -- and the
    earlier, sharper mechanism is the one that decided the outcome and the one
    an escape policy needed to see.
    """
    rows = list(observations or ())
    if not rows:
        return RugVerdict(
            mechanism=RugMechanism.UNCLASSIFIED, data_status="DATA_BLOCKED",
            detail="no observations; nothing can be ruled in or out")

    unobserved: List[str] = []
    supply_rows = _by_type(rows, "authority") + _by_type(rows, "supply")
    lp_rows = _by_type(rows, "lp_supply")
    trade_rows = _by_type(rows, "trade")
    if not supply_rows:
        unobserved.append(RugMechanism.MINT_DILUTION.value)
    if not lp_rows:
        unobserved.append(RugMechanism.LP_PULL.value)

    drawdown = _drawdown_from_peak(rows)
    if drawdown is None:
        return RugVerdict(
            mechanism=RugMechanism.UNCLASSIFIED, data_status="DATA_BLOCKED",
            detail="fewer than two priced observations; no path to classify",
            unobserved=unobserved)
    if drawdown < DEATH_DRAWDOWN:
        return RugVerdict(mechanism=RugMechanism.SURVIVED,
                          detail=f"drawdown from peak {drawdown:.0%} is not a death",
                          evidence={"drawdown": drawdown}, unobserved=unobserved)

    # --- freeze: the position stops being moveable at all -----------------
    frozen = [row for row in rows if row.get("freeze_used") or row.get("frozen")]
    if frozen:
        return RugVerdict(
            mechanism=RugMechanism.FREEZE,
            detail="freeze authority exercised; the position cannot be sold",
            evidence={"drawdown": drawdown, "at": frozen[0].get("timestamp")},
            unobserved=unobserved)

    # --- honeypot: buys land, sells do not --------------------------------
    sells = [row for row in trade_rows if str(row.get("side", "")).lower() == "sell"]
    buys = [row for row in trade_rows if str(row.get("side", "")).lower() == "buy"]
    if len(buys) >= 10 and not sells:
        return RugVerdict(
            mechanism=RugMechanism.HONEYPOT,
            detail=f"{len(buys)} buys and no sell ever observed",
            evidence={"buys": len(buys), "sells": 0, "drawdown": drawdown},
            unobserved=unobserved)

    # --- mint dilution: the supply we priced was not the supply -----------
    supplies = [float(row["supply"]) for row in supply_rows
                if row.get("supply") is not None]
    if len(supplies) >= 2 and max(supplies) > min(supplies) * 1.01:
        return RugVerdict(
            mechanism=RugMechanism.MINT_DILUTION,
            detail=("supply grew after launch; a live mint authority was "
                    "exercised and this was visible beforehand"),
            evidence={"supply_from": min(supplies), "supply_to": max(supplies),
                      "drawdown": drawdown},
            unobserved=[name for name in unobserved
                        if name != RugMechanism.MINT_DILUTION.value])

    # --- LP pull: liquidity removed in one step ---------------------------
    lp_supplies = [float(row["lp_supply"]) for row in lp_rows
                   if row.get("lp_supply") is not None]
    if len(lp_supplies) >= 2:
        opening = max(lp_supplies)
        closing = lp_supplies[-1]
        if opening > 0 and (opening - closing) / opening >= LP_PULL_FRACTION:
            return RugVerdict(
                mechanism=RugMechanism.LP_PULL,
                detail=f"LP supply fell {(opening - closing) / opening:.0%} in the window",
                evidence={"lp_from": opening, "lp_to": closing,
                          "drawdown": drawdown},
                unobserved=[name for name in unobserved
                            if name != RugMechanism.LP_PULL.value])

    # --- creator dump ------------------------------------------------------
    creators = {str(row.get("creator", "")) for row in rows if row.get("creator")}
    if creators:
        creator = next(iter(creators))
        sold = sum(abs(float(row.get("amount", 0) or 0)) for row in sells
                   if str(row.get("wallet", "")) == creator)
        total = sum(abs(float(row.get("amount", 0) or 0)) for row in trade_rows) or 0.0
        if total > 0 and sold / total >= CREATOR_DUMP_SHARE:
            return RugVerdict(
                mechanism=RugMechanism.CREATOR_DUMP,
                detail=f"the deployer sold {sold / total:.0%} of observed volume",
                evidence={"creator_sold_share": sold / total, "drawdown": drawdown},
                unobserved=unobserved)

    # --- sniper cascade: the first cohort leaves together ------------------
    if trade_rows:
        opened = min(float(row.get("timestamp", 0) or 0) for row in trade_rows)
        first_cohort = {str(row.get("wallet", "")) for row in buys
                        if float(row.get("timestamp", 0) or 0) - opened <= 30.0
                        and row.get("wallet")}
        if len(first_cohort) >= 5:
            # (when, who), because the question is how many of the cohort
            # LEFT -- not how many sell transactions they sent. Counting
            # transactions let one wallet selling repeatedly stand in for
            # the whole cohort, so the ratio could exceed 1.0 and a cascade
            # fired on ordinary active trading. Observed 2026-08-29 in a
            # live verdict reading "24 of 5 first-block buyers exited",
            # which is not a thing that can happen. A false mechanism label
            # is worse than none: it trains the one head that currently
            # passes validation on an event that did not occur.
            exits = sorted((float(row.get("timestamp", 0) or 0),
                            str(row.get("wallet", "")))
                           for row in sells
                           if str(row.get("wallet", "")) in first_cohort)
            for index, (start, _wallet) in enumerate(exits):
                leavers = {who for when, who in exits[index:]
                           if when - start <= CASCADE_WINDOW_S}
                if len(leavers) / len(first_cohort) >= CASCADE_SHARE:
                    return RugVerdict(
                        mechanism=RugMechanism.SNIPER_CASCADE,
                        detail=(f"{len(leavers)} of {len(first_cohort)} first-block "
                                f"buyers exited within {CASCADE_WINDOW_S:.0f}s"),
                        evidence={"cohort": len(first_cohort), "exited": len(leavers),
                                  "drawdown": drawdown},
                        unobserved=unobserved)

    # --- migration stall: it simply never got there ------------------------
    if migrated is False:
        return RugVerdict(
            mechanism=RugMechanism.MIGRATION_STALL,
            detail="the curve never filled and trading stopped; no actor required",
            evidence={"drawdown": drawdown, "trades": len(trade_rows)},
            unobserved=unobserved)

    # --- residual ----------------------------------------------------------
    # SLOW_BLEED is claimed only with enough of a price path to show a decay
    # rather than a step. Without that this is UNCLASSIFIED, on purpose: a
    # residual that absorbs every unexplained death teaches a model nothing
    # and hides exactly how much the observation stream is missing.
    priced = [row for row in rows if row.get("price_multiple") is not None]
    if len(priced) >= 20:
        return RugVerdict(
            mechanism=RugMechanism.SLOW_BLEED,
            detail=(f"{len(priced)} priced points decayed {drawdown:.0%} with no "
                    "single mechanism identified"),
            evidence={"drawdown": drawdown, "priced_points": len(priced)},
            unobserved=unobserved)
    return RugVerdict(
        mechanism=RugMechanism.UNCLASSIFIED, data_status="DATA_BLOCKED",
        detail=(f"died {drawdown:.0%} from peak with {len(priced)} priced points "
                "and no mechanism evidence; not attributed to slow bleed"),
        evidence={"drawdown": drawdown, "priced_points": len(priced)},
        unobserved=unobserved)


def coverage_report(verdicts: Sequence[RugVerdict]) -> Dict[str, Any]:
    """How much of the rug label set is actually usable.

    The number that matters is the confident share. A corpus that is mostly
    UNCLASSIFIED is not a corpus of slow bleeds; it is a measurement gap, and
    reporting it as a mechanism distribution would hide that.
    """
    rows = list(verdicts or ())
    if not rows:
        return {"status": "DATA_BLOCKED", "detail": "no verdicts yet",
                "labelled": 0, "confident": 0}
    counts: Dict[str, int] = {}
    for verdict in rows:
        counts[verdict.mechanism.value] = counts.get(verdict.mechanism.value, 0) + 1
    deaths = [row for row in rows
              if row.mechanism not in (RugMechanism.SURVIVED,)]
    confident = [row for row in deaths if row.confident]
    unclassified = counts.get(RugMechanism.UNCLASSIFIED.value, 0)
    return {
        "status": "OK",
        "labelled": len(deaths),
        "confident": len(confident),
        "confident_share": (len(confident) / len(deaths)) if deaths else None,
        "unclassified": unclassified,
        "unclassified_share": (unclassified / len(deaths)) if deaths else None,
        "by_mechanism": dict(sorted(counts.items(), key=lambda item: item[1],
                                    reverse=True)),
        "detail": ("" if deaths and len(confident) / len(deaths) >= 0.5 else
                   "most deaths carry an unobserved mechanism; the label set "
                   "is a measurement gap, not a mechanism distribution"),
    }
