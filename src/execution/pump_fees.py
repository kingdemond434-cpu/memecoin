"""One canonical, versioned Pump fee function.

Every quote, capacity calculation, training label, counterfactual replay and
E[log W] evaluation has to agree about what a round trip costs. If they do not,
part of the measured edge is an artefact of which code path computed the fee,
and no amount of validation will surface it -- the model and the simulator will
simply be confidently wrong together.

The schedule is versioned rather than constant because Pump's published docs
state that on Monday, September 1 at 20:00 UTC a market-cap-dependent dynamic
fee structure becomes mandatory for Pump bonding curves and for canonical
PumpSwap pools (those whose ``pool.creator`` is the pump pool authority). A
system that hardcodes today's flat fee does not merely become slightly wrong on
that date; it keeps producing labels and counterfactuals against economics that
no longer exist, which is worse than producing none.

DATA_BLOCKED, deliberately: the tier table itself is published in
``docs/fees.png``, an image. The boundaries and per-tier basis points are not
available as text, and inventing plausible ones would put fabricated economics
underneath every label in the research lake -- exactly the failure this module
exists to prevent. So the dynamic schedule ships with its structure defined and
its tiers empty, and any quote at or after the activation instant returns
DATA_BLOCKED until an operator loads the real table via
``PUMP_FEE_TIERS_PATH``. The system fails loudly on the date instead of quietly
using stale numbers.

What IS verified from the published docs and therefore hardcoded:
  - the legacy trade fee of 100 bps
  - the activation instant, 2026-09-01 20:00:00 UTC
  - that bonding-curve fees split protocol/creator, and canonical PumpSwap
    fees split lp/protocol/creator
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FEE_SCHEDULE_VERSION = "pump-fees-v2"

# "Monday, September 1, 20:00 UTC" -- pump-public-docs/docs/FEE_PROGRAM_README.md
DYNAMIC_FEE_ACTIVATION_UTC = datetime(2026, 9, 1, 20, 0, 0, tzinfo=timezone.utc).timestamp()

# The flat pre-activation trade fee, stated as `fee_basis_points == 100 bps` in
# pump-public-docs/docs/PUMP_PROGRAM_README.md.
LEGACY_TOTAL_FEE_BPS = 100

VENUE_BONDING_CURVE = "pump_bonding_curve"
VENUE_PUMPSWAP_CANONICAL = "pumpswap_canonical"


@dataclass(frozen=True)
class FeeTier:
    """One market-cap bracket of the dynamic schedule.

    ``max_market_cap_lamports`` is the exclusive upper bound of the bracket;
    ``None`` means the final open-ended tier.
    """

    max_market_cap_lamports: Optional[int]
    protocol_fee_bps: int
    creator_fee_bps: int
    lp_fee_bps: int = 0

    @property
    def total_bps(self) -> int:
        return self.protocol_fee_bps + self.creator_fee_bps + self.lp_fee_bps


@dataclass(frozen=True)
class FeeQuote:
    """The fee on one leg, or an explicit statement that it is unknown."""

    status: str
    total_bps: int = 0
    protocol_fee_bps: int = 0
    creator_fee_bps: int = 0
    lp_fee_bps: int = 0
    schedule_version: str = ""
    tier_index: Optional[int] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def fee_lamports(self, quote_lamports: int) -> int:
        """Fee charged on a quote-leg notional. Raises if the fee is unknown.

        Callers must branch on ``ok`` first. Returning zero for an unknown fee
        would make a blocked quote look like a free trade, which is the most
        expensive possible default.
        """
        if not self.ok:
            raise ValueError(f"fee is {self.status}: {self.reason}")
        return (int(quote_lamports) * self.total_bps) // 10_000


def _blocked(reason: str, version: str) -> FeeQuote:
    return FeeQuote(status="DATA_BLOCKED", schedule_version=version, reason=reason)


@dataclass
class PumpFeeSchedule:
    """Resolves a fee for (venue, market cap, instant).

    Two eras live behind one interface so that no caller has to know which one
    applies. Callers that need the number ask for it and check ``ok``.
    """

    dynamic_tiers: Dict[str, List[FeeTier]] = field(default_factory=dict)
    activation_utc: float = DYNAMIC_FEE_ACTIVATION_UTC
    version: str = FEE_SCHEDULE_VERSION
    source: str = "builtin-legacy-only"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PumpFeeSchedule":
        """Load the dynamic tier table if an operator has supplied one.

        The table is published as an image, so it cannot be shipped here
        honestly. ``PUMP_FEE_TIERS_PATH`` points at a JSON transcription of it:

            {"pump_bonding_curve": [
                {"max_market_cap_lamports": 1000000000,
                 "protocol_fee_bps": 95, "creator_fee_bps": 5}, ...]}

        A missing or unreadable file is not an error at load time -- it becomes
        a DATA_BLOCKED at quote time, and only for instants at or after
        activation, so pre-activation operation is unaffected.
        """
        location = path or os.getenv("PUMP_FEE_TIERS_PATH", "")
        if not location:
            return cls()
        try:
            raw = json.loads(Path(location).read_text())
        except (OSError, ValueError) as exc:
            logger.warning("pump fee tiers unreadable at %s: %s", location, exc)
            return cls()
        tiers: Dict[str, List[FeeTier]] = {}
        for venue, rows in (raw or {}).items():
            parsed = [
                FeeTier(
                    max_market_cap_lamports=(int(row["max_market_cap_lamports"])
                                             if row.get("max_market_cap_lamports") is not None else None),
                    protocol_fee_bps=int(row.get("protocol_fee_bps", 0)),
                    creator_fee_bps=int(row.get("creator_fee_bps", 0)),
                    lp_fee_bps=int(row.get("lp_fee_bps", 0)),
                )
                for row in rows
            ]
            # Bounded tiers ascending, the open-ended tier last. An unsorted
            # table would silently resolve the wrong bracket.
            parsed.sort(key=lambda tier: (tier.max_market_cap_lamports is None,
                                          tier.max_market_cap_lamports or 0))
            if parsed:
                tiers[str(venue)] = parsed
        return cls(dynamic_tiers=tiers, source=location or "builtin-legacy-only")

    def is_dynamic(self, at_utc: float) -> bool:
        return float(at_utc) >= self.activation_utc

    def quote(
        self,
        venue: str = VENUE_BONDING_CURVE,
        market_cap_lamports: Optional[int] = None,
        at_utc: Optional[float] = None,
    ) -> FeeQuote:
        """The fee applying to one trade leg at ``at_utc``.

        ``at_utc`` is required rather than defaulting to now, because a
        historical replay that silently used today's schedule would relabel
        every past episode with fees that were not charged at the time.
        """
        if at_utc is None:
            return _blocked("no_timestamp_supplied", self.version)
        if not self.is_dynamic(at_utc):
            return FeeQuote(
                status="OK", total_bps=LEGACY_TOTAL_FEE_BPS,
                protocol_fee_bps=LEGACY_TOTAL_FEE_BPS,
                schedule_version=f"{self.version}:legacy_flat_{LEGACY_TOTAL_FEE_BPS}bps",
            )

        tiers = self.dynamic_tiers.get(venue)
        if not tiers:
            return _blocked(
                f"dynamic schedule active for {venue} but no tier table loaded; "
                "pump publishes the tiers as an image (docs/fees.png), set "
                "PUMP_FEE_TIERS_PATH to a transcription",
                self.version,
            )
        if market_cap_lamports is None or market_cap_lamports < 0:
            return _blocked("market cap not observed; dynamic fee depends on it",
                            self.version)

        for index, tier in enumerate(tiers):
            if tier.max_market_cap_lamports is None or market_cap_lamports < tier.max_market_cap_lamports:
                return FeeQuote(
                    status="OK", total_bps=tier.total_bps,
                    protocol_fee_bps=tier.protocol_fee_bps,
                    creator_fee_bps=tier.creator_fee_bps,
                    lp_fee_bps=tier.lp_fee_bps,
                    schedule_version=f"{self.version}:dynamic",
                    tier_index=index,
                )
        return _blocked("market cap above every loaded tier and no open-ended tier exists",
                        self.version)

    def round_trip_bps(
        self,
        venue: str = VENUE_BONDING_CURVE,
        entry_market_cap_lamports: Optional[int] = None,
        exit_market_cap_lamports: Optional[int] = None,
        entry_utc: Optional[float] = None,
        exit_utc: Optional[float] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Total protocol cost of getting in and back out.

        Entry and exit are priced separately and at their own market caps: a
        position that goes up crosses tiers, so charging the exit at the entry
        tier understates the cost of exactly the trades that made money.
        """
        entry = self.quote(venue, entry_market_cap_lamports, entry_utc)
        exit_leg = self.quote(venue, exit_market_cap_lamports, exit_utc)
        detail = {"entry": entry, "exit": exit_leg}
        if not entry.ok:
            return "DATA_BLOCKED", 0, detail
        if not exit_leg.ok:
            return "DATA_BLOCKED", 0, detail
        return "OK", entry.total_bps + exit_leg.total_bps, detail


DEFAULT_SCHEDULE = PumpFeeSchedule.load()
