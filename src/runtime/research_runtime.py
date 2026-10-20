"""The desk-facing seam for research state and shadow-only tail proposals."""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from src.strategies.asymmetric_tail import AsymmetricTailSleeve, TailSleevePolicy
from src.strategies.opportunity_allocator import Opportunity


class ResearchRuntime:
    async def _close_research_worker_resources(self) -> None:
        await self.http_client.close_current_loop()
        await self.solana_rpc.close_current_loop()

    def _retain_mined_batch(self, miner_id: str, records: list[dict], now: float) -> str:
        return self.unknown_engine._record("mined_batch", {
            "miner_id": miner_id, "records": records,
            "scope": "raw observations; timestamps and rights require consumer validation"}, now)

    def _retain_mined_disposition(self, ref: str, status: str, detail: str, now: float) -> None:
        receipt = self.unknown_engine._record("mined_batch_disposition", {
            "batch_ref": ref, "status": status, "detail": detail}, now)
        if status != "DELIVERED":
            self.research_frontier.enqueue("MINER_BATCH_REPLAY", ref,
                evidence_refs=[ref, receipt], exploration_tags=["new_source_classes"],
                disposition="DATA_REQUEST", now=now)

    def _ensure_asymmetric_tail(self) -> None:
        if getattr(self, "asymmetric_tail", None) is not None:
            return
        if getattr(self, "equity_status", "DATA_BLOCKED") != "OK":
            return
        equity = float(getattr(self, "wallet_equity_usd", 0))
        if equity <= 0:
            return
        framework = getattr(self, "champion_challenger", None)
        if framework is None:
            return
        saved = framework.get_research_component("ASYMMETRIC_TAIL")
        # The original risk envelope never renews because the paper wallet's
        # balance changed or the process restarted.
        policy = TailSleevePolicy(**(saved["policy"] if saved else
            self.global_config.get("asymmetric_tail_policy", {})))
        sleeve = AsymmetricTailSleeve(saved["equity_usd"] if saved else equity,
                                      policy=policy, allocator=self.opportunity_allocator)
        if saved:
            sleeve.restore(saved)
        self.asymmetric_tail = sleeve
        self._tail_pending = {}
        if not saved:
            framework.checkpoint_research_component("ASYMMETRIC_TAIL", sleeve.state())

    def _tail_shadow_candidate(self, candidate: Any, opportunity: Opportunity,
                               veto: dict) -> dict:
        sleeve = getattr(self, "asymmetric_tail", None)
        if sleeve is None:
            return {"status": "DATA_BLOCKED", "reason": "tail budget has no observed equity"}
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        template = self.exit_readiness.template_for(candidate.address)
        metadata.update(chain=candidate.chain,
                        actor_id=metadata.get("actor_cluster_id") or candidate.deployer or None,
                        narrative_id=metadata.get("narrative_id"),
                        safety_clear=veto.get("clear") is True and not veto.get("unmeasured"),
                        exit_ready=template is not None)
        proposal = sleeve.propose([replace(opportunity, metadata=metadata)], now=time.time())
        for item in proposal.proposed:
            self._tail_pending[item.token] = item
        result = proposal.report()
        self.trade_evidence.record("asymmetric_tail_shadow_proposal", result)
        return result

    def _tail_shadow_fill(self, token: str, *, success: bool, simulated: bool) -> None:
        pending = getattr(self, "_tail_pending", {}).pop(token, None)
        if pending is None or not success or not simulated:
            return
        admitted = self.asymmetric_tail.record_shadow_entry(pending, now=time.time())
        if not admitted:
            self.trade_evidence.record("asymmetric_tail_shadow_fill_rejected", {
                "token": token, "reason": "shadow_budget_or_candidate_state_changed",
                "shadow_only": True})
            return
        self.champion_challenger.checkpoint_research_component(
            "ASYMMETRIC_TAIL", self.asymmetric_tail.state())
        self.trade_evidence.record("asymmetric_tail_shadow_fill", {
            "token": token, "capital_usd": pending.capital_usd,
            "scope": "bounded subposition of canonical simulated fill", "shadow_only": True})

    def _tail_shadow_exit(self, token: str, *, fraction: float,
                          proceeds_usd: float, allocated_cost_usd: float) -> None:
        sleeve = getattr(self, "asymmetric_tail", None)
        if sleeve is None or token not in sleeve.positions or allocated_cost_usd <= 0:
            return
        cost = sleeve.positions[token].remaining_cost_usd * fraction
        proceeds = proceeds_usd * min(1.0, cost / allocated_cost_usd)
        sleeve.record_shadow_exit(token, fraction=fraction, proceeds_usd=proceeds, now=time.time())
        self.champion_challenger.checkpoint_research_component("ASYMMETRIC_TAIL", sleeve.state())
        self.trade_evidence.record("asymmetric_tail_shadow_exit", {
            "token": token, "fraction": fraction, "proceeds_usd": proceeds,
            "shadow_only": True})
