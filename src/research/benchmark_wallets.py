"""Mining public wallets for what they did BEFORE the outcome was obvious.

A wallet with $11M of realised profit is not a strategy. It is an outcome,
and the number is the least transferable thing about it: it was earned with
that wallet's capital, at that wallet's latency, in that wallet's market, and
copying it delivers none of those. Encoding

    wallet -> smart

is how a follower ends up as exit liquidity for the wallet it admires.

So nothing here stores a verdict about a wallet. It stores that wallet's
DECISIONS -- one row per launch it entered, with the state of the world as of
its entry -- and answers the only question that transfers:

    E[Δ log W | we follow this wallet, after OUR delay, at OUR size,
                through OUR fills, net of OUR costs]

Which is routinely negative for a genuinely profitable wallet. A wallet whose
edge is being first cannot be followed at all: by the time its fill is public,
the information is in the price, and the follower buys the move the wallet
just made. That is a measurement, not a pessimistic assumption, and this
module exists to make it rather than to assume either way.

Three deliberate refusals:

**Headline PnL is never an input.** It is recorded as an unverified claim with
its source, because two public trackers routinely disagree by 20% on the same
address, and a number two sources cannot agree on is not a measurement. It is
never used in scoring; a wallet earns weight here by its decisions surviving
follow simulation, or not at all.

**Follow returns are computed at several delays.** +50, +100, +250, +500 and
+1000 ms, because "would copying this have worked" has a different answer at
each, and the honest version of the question names the delay.

**Survivorship is stated, not hidden.** Wallets get famous by winning. A
corpus of celebrated addresses is a sample selected on the outcome, and any
statistic from it is biased upwards -- so the corpus carries its own selection
warning into every report it produces, and the matched-control machinery
exists to give it a denominator.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

BENCHMARK_SCHEMA_VERSION = "v1"

#: The delays at which following is simulated, in seconds.
FOLLOW_DELAYS_S: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00)

#: Below this many resolved decisions a wallet has no measurable behaviour.
#: A follow-return computed from four launches is a story about four launches.
MIN_DECISIONS_FOR_VERDICT = 30

#: Wallets that got famous by winning are a sample selected on the outcome.
SELECTION_WARNING = (
    "these addresses were selected because they were publicly reported as "
    "profitable; any statistic computed over them is biased upward by that "
    "selection and is not evidence about wallets in general")


@dataclass
class BenchmarkWallet:
    """A public address worth studying, and what is merely CLAIMED about it."""

    address: str
    label: str = ""
    #: Why this address is in the corpus at all.
    rationale: str = ""
    #: Unverified third-party figures, kept only so they are never mistaken
    #: for measurements. Maps source name -> claimed realised PnL in USD.
    claimed_pnl_usd: Dict[str, float] = field(default_factory=dict)
    claim_note: str = ""

    @property
    def claims_disagree(self) -> bool:
        """True when public trackers do not agree about this address.

        Common, and worth surfacing: it is the cheapest available proof that
        headline PnL is not a measurement.
        """
        values = [value for value in self.claimed_pnl_usd.values() if value > 0]
        if len(values) < 2:
            return False
        return (max(values) - min(values)) / max(values) > 0.1


@dataclass
class WalletDecision:
    """One launch this wallet entered, frozen as of ITS entry.

    `state` is the point-in-time feature snapshot -- what was observable when
    the wallet acted, never anything from afterwards. `outcome` is filled in
    later, once resolved, and nothing that lands in `outcome` may ever be
    copied back into `state`.
    """

    wallet: str
    token: str
    entered_at: float
    #: Launch age in seconds at the wallet's entry.
    launch_age_s: Optional[float] = None
    #: The wallet's position in the entry order.
    buyer_rank: Optional[int] = None
    entry_size_sol: Optional[float] = None
    state: Dict[str, Any] = field(default_factory=dict)
    #: Price the wallet itself got, and prices at each follow delay.
    entry_price: Optional[float] = None
    price_at_delay: Dict[float, float] = field(default_factory=dict)
    #: Resolution, written once.
    exited_at: Optional[float] = None
    exit_price: Optional[float] = None
    peak_price: Optional[float] = None

    @property
    def resolved(self) -> bool:
        return self.exit_price is not None and self.entry_price is not None

    def wallet_multiple(self) -> Optional[float]:
        if not self.resolved or not self.entry_price:
            return None
        return float(self.exit_price / self.entry_price)

    def follow_multiple(self, delay_s: float) -> Optional[float]:
        """What a follower entering `delay_s` later would have realised.

        Same exit as the wallet -- this isolates the cost of being late to the
        ENTRY, which is the thing being measured. A follower with a different
        exit policy is a different question and gets a different study.
        """
        if not self.resolved:
            return None
        entry = self.price_at_delay.get(delay_s)
        if entry is None or entry <= 0:
            return None
        return float(self.exit_price / entry)


@dataclass
class FollowVerdict:
    status: str
    wallet: str = ""
    decisions: int = 0
    resolved: int = 0
    #: delay -> mean log return of following, net of costs.
    mean_log_return: Dict[float, float] = field(default_factory=dict)
    #: delay -> share of follows that beat holding nothing.
    win_rate: Dict[float, float] = field(default_factory=dict)
    #: The wallet's own mean log return, for contrast. Not our return.
    wallet_mean_log_return: Optional[float] = None
    detail: str = ""

    @property
    def followable(self) -> Optional[bool]:
        """Is there ANY delay at which copying this wallet has positive E[log W]."""
        if self.status != "OK" or not self.mean_log_return:
            return None
        return max(self.mean_log_return.values()) > 0.0

    @property
    def best_delay(self) -> Optional[float]:
        if self.status != "OK" or not self.mean_log_return:
            return None
        return max(self.mean_log_return.items(), key=lambda item: item[1])[0]

    @property
    def edge_decay(self) -> Optional[float]:
        """How much of the edge is lost between the fastest and slowest delay.

        The number that says whether a wallet's edge IS its speed. A wallet
        whose follow return collapses from +0.4 to -0.1 over 950ms was never
        sharing information; it was arriving early.
        """
        if self.status != "OK" or len(self.mean_log_return) < 2:
            return None
        fastest = min(self.mean_log_return)
        slowest = max(self.mean_log_return)
        return float(self.mean_log_return[fastest] - self.mean_log_return[slowest])


class BenchmarkCorpus:
    """Decisions of publicly-reported wallets, and what following them costs.

    Append-only and persisted, so the corpus survives a restart -- the whole
    value of it is that it accumulates.
    """

    def __init__(self, path: Optional[str] = None,
                 cost_per_round_trip: float = 0.02):
        self.path = Path(path) if path else None
        # Round-trip cost as a fraction: fees, priority, and slippage. Applied
        # to every simulated follow, because an edge measured gross is an edge
        # that does not exist.
        self.cost_per_round_trip = float(cost_per_round_trip)
        self.wallets: Dict[str, BenchmarkWallet] = {}
        self._decisions: Dict[str, List[WalletDecision]] = {}

    def register(self, wallet: BenchmarkWallet) -> None:
        self.wallets[wallet.address] = wallet
        self._decisions.setdefault(wallet.address, [])
        if wallet.claims_disagree:
            logger.info(
                "BENCHMARK %s: public trackers disagree on realised PnL (%s); "
                "headline figures are recorded as claims and never scored",
                wallet.label or wallet.address[:8],
                ", ".join(f"{k}={v:,.0f}" for k, v in sorted(wallet.claimed_pnl_usd.items())))

    def record(self, decision: WalletDecision) -> None:
        self._decisions.setdefault(decision.wallet, []).append(decision)

    def decisions(self, wallet: str) -> List[WalletDecision]:
        return list(self._decisions.get(wallet, ()))

    @property
    def size(self) -> int:
        return sum(len(rows) for rows in self._decisions.values())

    def follow_verdict(self, wallet: str) -> FollowVerdict:
        """Would copying this wallet have made money, at each delay."""
        rows = self._decisions.get(wallet, [])
        resolved = [row for row in rows if row.resolved]
        if len(resolved) < MIN_DECISIONS_FOR_VERDICT:
            return FollowVerdict(
                status="DATA_BLOCKED", wallet=wallet, decisions=len(rows),
                resolved=len(resolved),
                detail=(f"{len(resolved)} resolved decisions, below the "
                        f"{MIN_DECISIONS_FOR_VERDICT} needed for a verdict"))

        cost = math.log(max(1e-9, 1.0 - self.cost_per_round_trip))
        means: Dict[float, float] = {}
        wins: Dict[float, float] = {}
        for delay in FOLLOW_DELAYS_S:
            returns = []
            for row in resolved:
                multiple = row.follow_multiple(delay)
                if multiple is None or multiple <= 0:
                    continue
                returns.append(math.log(multiple) + cost)
            if len(returns) < MIN_DECISIONS_FOR_VERDICT:
                continue
            means[delay] = float(sum(returns) / len(returns))
            wins[delay] = float(sum(1 for value in returns if value > 0) / len(returns))
        if not means:
            return FollowVerdict(
                status="DATA_BLOCKED", wallet=wallet, decisions=len(rows),
                resolved=len(resolved),
                detail="no delay had enough priced follows to measure")

        own = [math.log(row.wallet_multiple()) for row in resolved
               if row.wallet_multiple() and row.wallet_multiple() > 0]
        return FollowVerdict(
            status="OK", wallet=wallet, decisions=len(rows), resolved=len(resolved),
            mean_log_return=means, win_rate=wins,
            wallet_mean_log_return=(float(sum(own) / len(own)) if own else None),
            detail=f"{len(resolved)} resolved decisions across {len(means)} delays")

    def report(self) -> Dict[str, Any]:
        verdicts = {address: self.follow_verdict(address) for address in self.wallets}
        measured = [v for v in verdicts.values() if v.status == "OK"]
        followable = [v for v in measured if v.followable]
        return {
            "status": "OK" if measured else "DATA_BLOCKED",
            "schema": BENCHMARK_SCHEMA_VERSION,
            "selection_warning": SELECTION_WARNING,
            "wallets": len(self.wallets),
            "decisions": self.size,
            "measured": len(measured),
            "followable": len(followable),
            "by_wallet": {
                address: {
                    "label": self.wallets[address].label,
                    "status": verdict.status,
                    "resolved": verdict.resolved,
                    "followable": verdict.followable,
                    "best_delay_s": verdict.best_delay,
                    "edge_decay": verdict.edge_decay,
                    "mean_log_return": {f"{k:g}s": round(v, 5)
                                        for k, v in verdict.mean_log_return.items()},
                    "wallet_own_mean_log_return": verdict.wallet_mean_log_return,
                    "claims_disagree": self.wallets[address].claims_disagree,
                    "detail": verdict.detail}
                for address, verdict in sorted(verdicts.items())},
        }

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": BENCHMARK_SCHEMA_VERSION,
                "wallets": [
                    {"address": w.address, "label": w.label, "rationale": w.rationale,
                     "claimed_pnl_usd": w.claimed_pnl_usd, "claim_note": w.claim_note}
                    for w in self.wallets.values()],
                "decisions": [
                    {"wallet": d.wallet, "token": d.token, "entered_at": d.entered_at,
                     "launch_age_s": d.launch_age_s, "buyer_rank": d.buyer_rank,
                     "entry_size_sol": d.entry_size_sol, "state": d.state,
                     "entry_price": d.entry_price,
                     "price_at_delay": {str(k): v for k, v in d.price_at_delay.items()},
                     "exited_at": d.exited_at, "exit_price": d.exit_price,
                     "peak_price": d.peak_price}
                    for rows in self._decisions.values() for d in rows],
            }
            self.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            return True
        except OSError as exc:  # pragma: no cover - disk only
            logger.warning("benchmark corpus save failed: %s", exc)
            return False

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # pragma: no cover - disk only
            logger.warning("benchmark corpus load failed: %s", exc)
            return False
        for row in payload.get("wallets", []):
            self.register(BenchmarkWallet(
                address=str(row.get("address", "")), label=str(row.get("label", "")),
                rationale=str(row.get("rationale", "")),
                claimed_pnl_usd={str(k): float(v) for k, v
                                 in (row.get("claimed_pnl_usd") or {}).items()},
                claim_note=str(row.get("claim_note", ""))))
        for row in payload.get("decisions", []):
            self.record(WalletDecision(
                wallet=str(row.get("wallet", "")), token=str(row.get("token", "")),
                entered_at=float(row.get("entered_at", 0.0)),
                launch_age_s=row.get("launch_age_s"), buyer_rank=row.get("buyer_rank"),
                entry_size_sol=row.get("entry_size_sol"),
                state=dict(row.get("state") or {}), entry_price=row.get("entry_price"),
                price_at_delay={float(k): float(v) for k, v
                                in (row.get("price_at_delay") or {}).items()},
                exited_at=row.get("exited_at"), exit_price=row.get("exit_price"),
                peak_price=row.get("peak_price")))
        return True


def load_roster(path: str, corpus: Optional[BenchmarkCorpus] = None) -> BenchmarkCorpus:
    """Read the seed roster of publicly-reported wallets.

    Missing file is not an error: the roster is a research convenience, and a
    desk that refuses to start because a list of other people's addresses is
    absent has its priorities wrong.
    """
    target = corpus or BenchmarkCorpus()
    location = Path(path)
    if not location.exists():
        logger.info("BENCHMARK no roster at %s; corpus starts from discovery only",
                    path)
        return target
    try:
        import yaml

        payload = yaml.safe_load(location.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - config only
        logger.warning("BENCHMARK roster unreadable (%s); continuing without it", exc)
        return target
    for row in payload.get("wallets", []) or []:
        address = str(row.get("address", "") or "").strip()
        if not address:
            continue
        target.register(BenchmarkWallet(
            address=address, label=str(row.get("label", "") or ""),
            rationale=str(row.get("rationale", "") or ""),
            claimed_pnl_usd={str(k): float(v) for k, v
                             in (row.get("claimed_pnl_usd") or {}).items()},
            claim_note=str(row.get("claim_note", "") or "")))
    logger.info("BENCHMARK roster loaded: %d wallets to reconstruct", len(target.wallets))
    return target


@dataclass
class DiscoveryCriteria:
    """What makes a wallet worth adding to the corpus, from OUR observations.

    Deliberately not "made a lot of money". A wallet that turned $300 into
    $600k on one launch has one observation and nothing to learn from; a
    wallet that made a tenth as much across four hundred launches has a
    behaviour. The thresholds encode that preference explicitly.
    """

    #: Launches the wallet must have entered under our own observation.
    min_decisions: int = 40
    #: Its own mean log return must clear this. Modest on purpose: the corpus
    #: is for studying behaviour, and a filter tuned to spectacular winners
    #: rebuilds the survivorship bias the roster already suffers from.
    min_mean_log_return: float = 0.05
    #: Fraction of its entries that must be early enough to be a sniper at
    #: all. A wallet that always buys hour-old tokens is a different animal.
    max_median_launch_age_s: float = 600.0


def discover_candidates(observations: Dict[str, Sequence[WalletDecision]],
                        criteria: Optional[DiscoveryCriteria] = None,
                        ) -> List[Tuple[str, Dict[str, Any]]]:
    """Wallets from our OWN stream that behave like the roster's, with evidence.

    This is the unbiased half of the corpus. The roster is a list of addresses
    other people published because they won; this finds wallets in the desk's
    own observation window, where the desk saw every wallet that entered --
    winners and losers alike -- so a statistic computed over what comes out of
    here has a denominator the roster can never have.

    Returns (address, evidence) pairs rather than registering anything: what
    to do with a candidate is the caller's decision, and a discovery function
    that silently mutates a corpus is one that cannot be tested honestly.
    """
    rules = criteria or DiscoveryCriteria()
    found: List[Tuple[str, Dict[str, Any]]] = []
    for wallet, rows in sorted(observations.items()):
        resolved = [row for row in rows if row.resolved]
        if len(resolved) < rules.min_decisions:
            continue
        returns = [math.log(row.wallet_multiple()) for row in resolved
                   if row.wallet_multiple() and row.wallet_multiple() > 0]
        if len(returns) < rules.min_decisions:
            continue
        mean = sum(returns) / len(returns)
        if mean < rules.min_mean_log_return:
            continue
        ages = sorted(float(row.launch_age_s) for row in resolved
                      if row.launch_age_s is not None)
        if not ages:
            continue
        median_age = ages[len(ages) // 2]
        if median_age > rules.max_median_launch_age_s:
            continue
        found.append((wallet, {
            "decisions": len(resolved),
            "mean_log_return": round(float(mean), 5),
            "median_launch_age_s": round(float(median_age), 2),
            # Stated so nobody reads the list as a ranking of skill.
            "basis": ("observed in this desk's own stream, where losing "
                      "wallets were equally visible"),
        }))
    return found
