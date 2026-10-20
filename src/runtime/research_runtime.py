"""The desk-facing seam for research state and shadow-only tail proposals."""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from src.strategies.asymmetric_tail import AsymmetricTailSleeve, TailSleevePolicy
from src.strategies.opportunity_allocator import Opportunity


class ResearchRuntime:
    def _retain_native_probe(self, result: dict) -> str:
        """Retain real native probe receipts and route successful typed facts."""
        now = time.time()
        ref = self.unknown_engine._record("research_native_evm_probe", result, now)
        if result.get("status") == "NATIVE_QUOTE_PARITY_VERIFIED":
            points = [{"variable": name, "value": result[field], "unit": unit,
                "chain": result["chain"], "entity_id": result["pair"],
                "event_at": result["event_at"], "available_at": result["observed_at"],
                "observed_at": result["observed_at"], "source_id": "native:uniswap_v2",
                "source_class": "canonical_native_observation", "rights": "PUBLIC", "partition": "forward",
                "row_id": ":".join((result["block_hash"], result["pair"], result["token_in"],
                    result["token_out"], str(result["input_amount"]))),
                "native_probe_ref": ref, "point_in_time": True}
                for name, field, unit in (("native_v2_buy_output", "quoted_buy_output", "token_raw"),
                    ("native_v2_sell_output", "quoted_sell_output", "token_raw"))]
            self.unknown_engine.enqueue_work("DATA_OBSERVATION", {"points": points},
                evidence_refs=[ref], axes={"chain": result["chain"], "protocol": "uniswap_v2"}, now=now)
        return ref

    def _setup_research_consumers(self) -> None:
        from src.research.population_hunter import PopulationHunter
        from src.research.mechanism_labs import MechanismLaboratory
        from src.research.research_validation import ResearchValidationService
        from src.research.research_workflows import ResearchWorkflows
        validation = ResearchValidationService(self.champion_challenger,
            gauntlet_handler=self.unknown_engine.gauntlet_handler,
            forward_evidence=self.forward_evidence)
        validation.configure_evidence(self.trade_evidence)
        validation.configure_integrations(self.chain_registry,
            {adapter.chain_name: adapter for adapter in self.evm_observers})
        workflows = ResearchWorkflows(self.unknown_engine,
            population=PopulationHunter(self.champion_challenger, actor_store=self.actor_store),
            laboratory=MechanismLaboratory(self.champion_challenger), validation=validation, seed_labs=True,
            lab_contexts=self.global_config.get("research_lab_contexts"))
        self.unknown_engine.configure_workflows(workflows)
        for miner_id, fetch in self.data_miners._callables.items():
            spec = self.data_miners._specs[miner_id]
            async def acquire_registered(name=miner_id):
                now = time.time()
                health = self.data_miners._health[name]
                if (self.data_miners._next_due.get(name, 0) > now or health.backoff_until > now
                        or self.data_miners._specs[name].missing_credentials()):
                    raise RuntimeError("registered source cadence, backoff or credentials currently blocks acquisition")
                result = await self.data_miners._run_one(name, now)
                if not result.ok:
                    raise RuntimeError("registered acquisition failed: " + result.status)
                return result.records
            workflows.data.register_source(miner_id, acquire_registered,
                source_classes=(spec.enriches.value, "registered_desk_source",
                    *( ("public native events", "canonical_native_observation") if spec.endpoint.startswith("rpc:") else () )),
                rights="AUTHORIZED" if spec.requires_env else "PUBLIC",
                rights_ref=workflows.configuration_ref if spec.requires_env else None,
                chain="solana" if spec.endpoint.startswith("rpc:") else None)

    async def _close_research_worker_resources(self) -> None:
        await self.http_client.close_current_loop()
        await self.solana_rpc.close_current_loop()

    def _retain_mined_batch(self, miner_id: str, records: list[dict], now: float) -> str:
        payload = {
            "miner_id": miner_id, "records": records,
            "observed_at": now, "cursor": 0,
            "rights": "PUBLIC", "scope": "raw observations; timestamps require consumer validation"}
        workflows = getattr(self.unknown_engine, "workflows", None)
        if workflows is not None:
            spec = self.data_miners._specs.get(miner_id)
            if spec is not None and spec.endpoint.startswith("rpc:"):
                payload["chain"] = "solana"  # these registered miners use self.solana_rpc
            if spec is not None and spec.requires_env:
                payload.update(rights="AUTHORIZED", rights_ref=workflows.configuration_ref)
        ref = self.unknown_engine._record("mined_batch", payload, now)
        if workflows is not None:
            self.research_frontier.enqueue("MINER_BATCH_REPLAY", ref,
                evidence_refs=[ref], auction=workflows.auction("MINER_BATCH_REPLAY"),
                exploration_tags=["new_source_classes"], now=now)
        return ref

    def _retain_mined_disposition(self, ref: str, status: str, detail: str, now: float) -> None:
        receipt = self.unknown_engine._record("mined_batch_disposition", {
            "batch_ref": ref, "status": status, "detail": detail}, now)
        if status != "DELIVERED" and getattr(self.unknown_engine, "workflows", None) is None:
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

    async def _native_forward_tick(self, token: str, observed_at: float) -> dict:
        """Paper research follows the existing bounded market observation loop.

        Local native quote calls only. Generic router marks never enter this
        producer, and an absent actual simulation-fee receipt stays unknown.
        """
        from src.research.native_forward import NativeForwardPaper, NativeSimulationFeeAdapter
        from src.research.candidate_replay import ReplayBlocked
        workflows = getattr(getattr(self, "unknown_engine", None), "workflows", None)
        validation = getattr(workflows, "validation", None)
        builder = getattr(self, "dataset_builder", None)
        episode = getattr(builder, "active_episodes", {}).get(token)
        if validation is None or episode is None:
            return {"status": "DATA_BLOCKED", "reason": "registered validation consumer/active episode unavailable"}
        producer = getattr(self, "native_forward_paper", None)
        if producer is None or producer.validation is not validation:
            producer = self.native_forward_paper = NativeForwardPaper(validation)
        raw = builder._episode_to_dict(episode)
        # These are actual capture times, not claimed availability at the
        # scheduled snapshot cutoff. The exact interpreter may refuse a lag.
        for snapshot in raw.get("snapshots", {}).values():
            event_at = snapshot.get("timestamp")
            lag = snapshot.get("capture_lag_s")
            if type(event_at) in (float, int) and type(lag) in (float, int) and lag >= 0:
                snapshot.setdefault("event_at", event_at)
                snapshot.setdefault("available_at", event_at + lag)
                snapshot.setdefault("received_at", event_at + lag)
        curve = getattr(self, "_latest_curve_state", {}).get(token)
        pool = getattr(self, "_latest_pool_state", {}).get(token)
        state = curve if curve is not None and curve.tradeable else pool
        stamp, source_ref, fee_bps = None, None, None
        if state is pool and pool is not None:
            stamp = pool.updated_at
            source_ref = f"decoded-native-pool:{pool.pool}:{pool.slot}:{pool.source}"
        elif curve is not None:
            # Only actual decoded trade reserves that still match the quoted
            # state can establish freshness. A timer cannot refresh a reserve.
            matching = [row for row in raw.get("market_observations", [])
                        if row.get("signature")
                        and row.get("virtual_sol_reserves") == curve.virtual_sol_reserves
                        and row.get("virtual_token_reserves") == curve.virtual_token_reserves]
            if matching:
                latest = max(matching, key=lambda row: row.get("timestamp", 0))
                stamp = latest.get("timestamp")
                source_ref = "decoded-native-trade:" + latest["signature"]
            costs = self._cost_model(token, observed_at)
            if costs.get("status") == "OK" and costs.get("assumed") is False:
                fee_bps = round(costs["entry_cost"] * 10000)
        fee_rows = {}
        adapter = NativeSimulationFeeAdapter(getattr(self, "execution_engine", None), self.trade_evidence)
        for intent in producer.cost_requests(token, raw, now=observed_at):
            try:
                quote, _ = producer._quote(state, intent["amount"], buying=intent["phase"] == "entry",
                                           fee_bps=fee_bps, now=observed_at)
                if quote.data_status != "OK" or quote.output_amount <= 0:
                    raise ReplayBlocked("actual native exact-size quote unavailable before fee acquisition")
                # Native preparation + read-only RPC simulation; no signer and
                # no transaction submission exists in this registered adapter.
                row = await adapter.acquire(token, intent["amount"], buying=intent["phase"] == "entry", now=observed_at)
                fee_rows[f"{intent['phase']}:{intent['amount']}"] = row
            except (ReplayBlocked, ValueError, TypeError, KeyError, ConnectionError, TimeoutError) as exc:
                self.trade_evidence.record("research_native_simulation_blocked", {
                    "token": token, "reason": str(exc) if isinstance(exc, ReplayBlocked) else type(exc).__name__,
                    "scope": "READ_ONLY_NATIVE_SIMULATION"})
        decision_at = max(observed_at, time.time())
        result = producer.tick(token, raw, state, state_observed_at=stamp,
            source_ref=source_ref, equity_usd=(getattr(self, "wallet_equity_usd", None)
                if getattr(self, "equity_status", None) == "OK" else None),
            native_to_usd=(getattr(self, "sol_price_usd", None)
                if getattr(self, "sol_price_age_s", float("inf")) <= 30 else None), now=decision_at,
            fee_bps=fee_bps, execution_costs=fee_rows)
        # Requests join the existing queue, which already owns backoff and named
        # wake conditions; this seam creates no scheduler.
        for child in result.get("children", []):
            self.unknown_engine.enqueue_work(child["kind"], child["payload"],
                evidence_refs=[], now=observed_at)
        return result
