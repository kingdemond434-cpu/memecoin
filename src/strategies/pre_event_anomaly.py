"""Wallets that are reliably early to things that later become public.

The tempting framing is "insider detection", and it is the wrong one twice
over. It asserts private knowledge from public data, which the evidence
cannot support; and it invites the desk to go looking for the private
information itself, which is a line this system does not cross. Everything
here is on-chain behaviour and publicly posted timestamps, and the claim it
makes is deliberately weaker and more useful:

    this wallet buys before the public signal arrives, more often than
    chance, across enough launches that chance is not a live explanation.

That is a statement about TIMING, which is measurable, rather than about
knowledge, which is not. It is also the only version of the claim the desk
can act on: a wallet that is consistently thirty seconds ahead of the first
public mention is worth watching whether it is an insider, a better scraper,
or the person who wrote the bot that posts the mention.

The measurement has one trap, and it is fatal if missed. The desk's own
source coverage is uneven -- some launches are named by five channels
within a second and some are never named at all -- so a wallet that happens
to trade the launches nobody covers would look prescient on a naive count.
So a launch contributes NOTHING unless the desk actually observed a public
mention of it, and the lead is measured against the FIRST such mention. The
denominator is launches where the comparison was possible, never launches
the wallet touched.

The null is per-wallet, not global: how often would this wallet lead the
public signal if its entry times were unrelated to it? Compared against its
own entry-time distribution, so an early-buying bot that leads everything
because it buys everything at T+200ms scores nothing.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategies.surprisal import binomial_surprisal

logger = logging.getLogger(__name__)

PRE_EVENT_SCHEMA_VERSION = "v1"

#: Comparable launches a wallet needs before its lead rate is reported.
#: Comparable means the desk saw a public mention AND the wallet entered;
#: anything else is not evidence either way.
MIN_COMPARABLE = 25

#: Seconds a wallet must be ahead of the first public mention for the entry
#: to count as leading. Not zero: the desk's own observation of a mention is
#: itself delayed, and a 200ms "lead" is measuring our scraper, not them.
LEAD_MARGIN_S = 2.0

#: Surprisal in nats above which the lead rate is called anomalous. Same
#: bar the ring detector uses, for the same reason: thousands of wallets are
#: tested at once and the threshold has to survive that.
SURPRISAL_THRESHOLD = 12.0

#: Wallets tracked. Bounded like every other per-wallet map here.
MAX_WALLETS = 20_000


@dataclass
class WalletLead:
    """How often one wallet is ahead of the public signal, and by how much."""

    wallet: str
    comparable: int = 0
    led: int = 0
    leads_s: List[float] = field(default_factory=list)
    #: Entry ages on launches where NO public mention was observed. Kept
    #: because they are what the null rate is built from -- how early this
    #: wallet buys in general, independent of anybody talking.
    entry_ages_s: List[float] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def lead_rate(self) -> Optional[float]:
        return self.led / self.comparable if self.comparable else None

    @property
    def median_lead_s(self) -> Optional[float]:
        return float(statistics.median(self.leads_s)) if self.leads_s else None

    def as_dict(self, surprisal: float, null_rate: Optional[float]) -> Dict[str, Any]:
        return {
            "wallet": self.wallet,
            "comparable_launches": self.comparable,
            "led": self.led,
            "lead_rate": (round(self.lead_rate, 4)
                          if self.lead_rate is not None else None),
            "null_rate": round(null_rate, 4) if null_rate is not None else None,
            "median_lead_s": (round(self.median_lead_s, 2)
                              if self.median_lead_s is not None else None),
            "surprisal_nats": round(surprisal, 2),
            "status": "OK" if self.comparable >= MIN_COMPARABLE else "DATA_BLOCKED",
        }


class PreEventAnomaly:
    """Which wallets are reliably ahead of the first public mention."""

    def __init__(self, *, min_comparable: int = MIN_COMPARABLE,
                 lead_margin_s: float = LEAD_MARGIN_S,
                 surprisal_threshold: float = SURPRISAL_THRESHOLD,
                 max_wallets: int = MAX_WALLETS):
        self.min_comparable = int(min_comparable)
        self.lead_margin_s = float(lead_margin_s)
        self.surprisal_threshold = float(surprisal_threshold)
        self.max_wallets = int(max_wallets)
        self.wallets: Dict[str, WalletLead] = {}
        #: token -> earliest observed public mention, relative to launch.
        self._first_mention_s: Dict[str, float] = {}
        self.launches_with_a_mention = 0
        self.launches_without = 0
        self.evicted = 0

    # --- observation -----------------------------------------------------

    def note_public_mention(self, token: str, seconds_after_launch: float) -> None:
        """The first time the desk saw anybody publicly name this launch.

        Kept as the EARLIEST, because a launch mentioned at t+4s and again
        at t+90s was public at t+4s, and comparing a wallet against the
        later mention would manufacture leads.
        """
        key = str(token or "")
        if not key:
            return
        value = float(seconds_after_launch)
        previous = self._first_mention_s.get(key)
        if previous is None:
            self._first_mention_s[key] = value
            self.launches_with_a_mention += 1
        elif value < previous:
            self._first_mention_s[key] = value

    def observe_entry(self, wallet: str, token: str,
                      seconds_after_launch: float,
                      at: Optional[float] = None) -> None:
        """One wallet's entry into one launch, timed from the launch.

        A launch with no observed public mention contributes to the wallet's
        NULL rate and to nothing else -- the desk's source coverage is
        uneven, and a wallet that happens to trade the launches nobody
        covers would look prescient on a naive count.
        """
        key = str(wallet or "")
        token_key = str(token or "")
        if not key or not token_key:
            return
        moment = float(at or time.time())
        record = self.wallets.get(key)
        if record is None:
            record = WalletLead(wallet=key, first_seen=moment)
            self.wallets[key] = record
            self._evict()
        record.last_seen = moment
        age = float(seconds_after_launch)
        record.entry_ages_s.append(age)
        if len(record.entry_ages_s) > 512:
            record.entry_ages_s = record.entry_ages_s[-512:]
        mention = self._first_mention_s.get(token_key)
        if mention is None:
            # Not comparable. Counted nowhere near the numerator.
            return
        record.comparable += 1
        lead = mention - age
        if lead >= self.lead_margin_s:
            record.led += 1
            record.leads_s.append(lead)
            if len(record.leads_s) > 512:
                record.leads_s = record.leads_s[-512:]

    def _evict(self) -> None:
        if len(self.wallets) <= self.max_wallets:
            return
        keep = set(sorted(self.wallets, key=lambda w: self.wallets[w].comparable,
                          reverse=True)[:self.max_wallets])
        for wallet in [w for w in self.wallets if w not in keep]:
            del self.wallets[wallet]
            self.evicted += 1

    # --- scoring ---------------------------------------------------------

    def _null_rate(self, record: WalletLead) -> Optional[float]:
        """How often this wallet would lead if its timing meant nothing.

        Built from its OWN entry-age distribution against the observed
        distribution of first mentions -- so a bot that buys everything at
        T+200ms has a null rate near one and scores no surprise for leading
        everything, which is the whole point.
        """
        if not record.entry_ages_s or not self._first_mention_s:
            return None
        mentions = list(self._first_mention_s.values())
        if not mentions:
            return None
        # P(a random entry of this wallet leads a random mention by the
        # margin). Computed by sampling both empirical distributions against
        # each other, which needs no distributional assumption.
        ages = record.entry_ages_s[-256:]
        sample = mentions[-256:]
        wins = sum(1 for age in ages for mention in sample
                   if mention - age >= self.lead_margin_s)
        total = len(ages) * len(sample)
        return wins / total if total else None

    def score(self, wallet: str) -> Dict[str, Any]:
        """How anomalous this wallet's earliness is, or why it cannot be said."""
        record = self.wallets.get(str(wallet or ""))
        if record is None:
            return {"status": "DATA_BLOCKED", "reason": "wallet not observed"}
        if record.comparable < self.min_comparable:
            return {
                "status": "DATA_BLOCKED",
                "reason": (f"{record.comparable} comparable launch(es), "
                           f"{self.min_comparable} needed -- comparable means "
                           "the desk saw a public mention AND this wallet "
                           "entered"),
                "comparable_launches": record.comparable,
            }
        null_rate = self._null_rate(record)
        if null_rate is None:
            return {"status": "DATA_BLOCKED",
                    "reason": "no null rate could be built for this wallet"}
        surprisal = binomial_surprisal(record.led, record.comparable, null_rate)
        payload = record.as_dict(surprisal, null_rate)
        payload["anomalous"] = surprisal >= self.surprisal_threshold
        payload["means"] = (
            "this wallet enters before the first PUBLIC mention more often "
            "than its own timing predicts. That is a statement about timing, "
            "not about knowledge: it is equally consistent with an insider, "
            "a better scraper, and the author of the bot that posts the "
            "mention -- and the desk does not need to know which")
        payload["provenance"] = "PUBLIC_TIMING_ONLY"
        return payload

    def anomalous_wallets(self) -> List[Dict[str, Any]]:
        out = []
        for wallet in self.wallets:
            scored = self.score(wallet)
            if scored.get("anomalous"):
                out.append(scored)
        out.sort(key=lambda row: row.get("surprisal_nats", 0.0), reverse=True)
        return out

    def report(self) -> Dict[str, Any]:
        comparable = [record for record in self.wallets.values()
                      if record.comparable >= self.min_comparable]
        anomalous = self.anomalous_wallets()
        return {
            "schema": PRE_EVENT_SCHEMA_VERSION,
            "status": "OK" if comparable else "DATA_BLOCKED",
            "wallets_observed": len(self.wallets),
            "wallets_comparable": len(comparable),
            "min_comparable": self.min_comparable,
            "launches_with_a_public_mention": self.launches_with_a_mention,
            "lead_margin_s": self.lead_margin_s,
            "anomalous": len(anomalous),
            "top": anomalous[:5],
            "evicted": self.evicted,
            "coverage_caveat": (
                "only launches the desk OBSERVED a public mention of are "
                "comparable; source coverage is uneven, and counting the "
                "launches nobody covered would make a wallet that trades "
                "them look prescient"),
            "detail": ("wallets that enter before the first public mention "
                       "more often than their own entry timing predicts -- "
                       "measured from on-chain times and publicly posted "
                       "timestamps, never from private information"),
        }
