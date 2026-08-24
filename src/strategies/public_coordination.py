"""Inference of coordinated launch activity from public-chain evidence only."""

import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

import numpy as np


@dataclass
class CoordinationEvidence:
    kind: str
    strength: float
    confidence: float
    timestamp: float
    wallets: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


class PublicCoordinationMiner:
    """Find same-slot, shared-funder and repeated-creator patterns.

    The output is an inference, never a claim about a person's legal status or
    access to non-public information. Sparse evidence is explicitly blocked.
    """

    def __init__(self, genealogy: Any, wallet_intel: Any):
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.trades: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10_000))
        self.funding: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=2_000))
        self.creators: Dict[str, str] = {}
        self.token_evidence: Dict[str, List[CoordinationEvidence]] = defaultdict(list)

    def record_trade(self, token: str, event: Dict[str, Any]):
        if not token or event.get("side") not in {"buy", "sell"}:
            return
        item = dict(event)
        item.setdefault("timestamp", time.time())
        self.trades[token].append(item)
        self._analyze(token)

    def record_funding(self, token: str, wallet: str, funder: str, amount_sol: float,
                       timestamp: Optional[float] = None):
        if not token or not wallet or not funder:
            return
        self.funding[token].append({"wallet": wallet, "funder": funder, "amount_sol": amount_sol,
                                    "timestamp": timestamp or time.time()})
        self._analyze(token)

    def record_creator(self, token: str, creator: str):
        if token and creator:
            self.creators[token] = creator
            self._analyze(token)

    def _analyze(self, token: str):
        now = time.time()
        buys = [item for item in self.trades[token]
                if item.get("side") == "buy" and now - float(item.get("timestamp", now)) <= 300]
        evidence: List[CoordinationEvidence] = []
        slots: Dict[int, Set[str]] = defaultdict(set)
        for item in buys:
            if item.get("slot") is not None and item.get("wallet"):
                slots[int(item["slot"])].add(item["wallet"])
        largest_same_slot = max((wallets for wallets in slots.values()), key=len, default=set())
        if len(largest_same_slot) >= 3:
            evidence.append(CoordinationEvidence(
                "same_slot_buy_cluster", min(len(largest_same_slot) / 10, 1), 0.90, now,
                sorted(largest_same_slot), {"wallet_count": len(largest_same_slot)},
            ))

        funder_counts = Counter(item["funder"] for item in self.funding[token])
        if funder_counts:
            funder, count = funder_counts.most_common(1)[0]
            wallets = sorted({item["wallet"] for item in self.funding[token] if item["funder"] == funder})
            if count >= 2:
                evidence.append(CoordinationEvidence(
                    "shared_funder", min(count / 6, 1), 0.95, now, wallets,
                    {"funder": funder, "funded_wallet_count": count},
                ))

        creator = self.creators.get(token)
        if creator:
            profile = self.genealogy.get_deployer_profile(creator)
            prior = len(profile.tokens_created) if profile else 0
            if prior:
                confidence = min(0.95, 0.45 + np.log1p(prior) / 10)
                evidence.append(CoordinationEvidence(
                    "repeat_creator", min(prior / 10, 1), float(confidence), now, [creator],
                    {"prior_launches": prior, "rug_rate": profile.rug_rate, "success_rate": profile.success_rate},
                ))

        amounts = [float(item.get("amount", 0) or 0) for item in buys if float(item.get("amount", 0) or 0) > 0]
        if len(amounts) >= 5:
            coefficient = float(np.std(amounts) / max(np.mean(amounts), 1e-12))
            if coefficient <= 0.05:
                evidence.append(CoordinationEvidence(
                    "near_identical_buy_sizes", 1 - coefficient / 0.05, 0.70, now,
                    sorted({item.get("wallet", "") for item in buys if item.get("wallet")}),
                    {"coefficient_of_variation": coefficient},
                ))
        self.token_evidence[token] = evidence

    def get_features(self, token: str) -> Dict[str, Any]:
        trades = list(self.trades.get(token, ()))
        if len(trades) < 3:
            return {"status": "DATA_BLOCKED", "reason": "fewer_than_three_public_trade_observations"}
        evidence = self.token_evidence.get(token, [])
        coordination = 1.0
        confidence_survival = 1.0
        coordinated_wallets: Set[str] = set()
        for item in evidence:
            coordination *= 1 - float(np.clip(item.strength * item.confidence, 0, 0.95))
            confidence_survival *= 1 - float(np.clip(item.confidence, 0, 0.95))
            coordinated_wallets.update(item.wallets)
        coordination_score = 1 - coordination
        confidence = 1 - confidence_survival if evidence else min(1.0, len(trades) / 20)
        buyers = {item.get("wallet") for item in trades if item.get("side") == "buy" and item.get("wallet")}
        return {
            "status": "OK", "coordination_score": coordination_score,
            "organic_ratio": max(0.0, 1 - coordination_score), "confidence": confidence,
            "coordinated_wallets": sorted(coordinated_wallets),
            "coordinated_buyer_fraction": len(buyers & coordinated_wallets) / max(len(buyers), 1),
            "evidence": [item.__dict__ for item in evidence], "observations": len(trades),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {"tracked_tokens": len(self.trades),
                "tokens_with_coordination_evidence": sum(bool(items) for items in self.token_evidence.values()),
                "public_chain_only": True}
