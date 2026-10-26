"""Bounded paper forward experiments driven by the existing market observer.

Uses the canonical native curve/pool quoting functions against fresh observed
state. It never builds/signs/submits a transaction and never uses aggregator or
future prices. Missing fee/depth/PIT observations remain named data requests.
"""
from __future__ import annotations

import copy
import asyncio
import base64
import math
import time
from dataclasses import asdict

from src.chains.pump_curve import BondingCurveState, quote_buy, quote_sell
from src.chains.pumpswap_curve import PumpSwapPoolState
from src.chains.pumpswap_curve import quote_buy as pool_buy, quote_sell as pool_sell
from src.research.candidate_replay import ReplayBlocked, _digest, compile_candidate, compile_exit
from src.research.gauntlet_feed import EpisodeView
from src.research.lifecycle_replay import Mark
from src.research.research_validation import ResearchValidationService, _number

COMPONENT = "native_forward_paper_v1"
LAMPORTS = 1_000_000_000


class NativeSimulationFeeAdapter:
    """Acquire public RPC simulation/fee facts for exact unsigned native intent.

    Reuses the executor's reviewed instruction preparation and TxKernel. Empty
    signatures are serialization placeholders with sigVerify=False; this class
    never reads the signer and has no submission operation.
    """
    def __init__(self, engine, ledger):
        self.engine, self.ledger = engine, ledger

    async def acquire(self, token, amount, *, buying, now):
        from solders.compute_budget import set_compute_unit_limit
        from solders.hash import Hash
        from solders.instruction import AccountMeta, Instruction
        from solders.message import MessageV0, to_bytes_versioned
        from solders.pubkey import Pubkey
        from solders.signature import Signature
        from solders.transaction import VersionedTransaction
        from src.chains.pump_route import WSOL_MINT
        engine = self.engine
        if engine is None or type(amount) is not int or amount <= 0:
            raise ReplayBlocked("native simulation compiler/positive exact amount unavailable")
        native = engine.prepare_native_route(WSOL_MINT if buying else token,
            token if buying else WSOL_MINT, amount, 300)
        if native is None or not native.ok or native.venue not in {"pump_curve", "pumpswap"}:
            raise ReplayBlocked("reviewed native instruction not currently constructible")
        builder = engine.tx_builder
        payer = Pubkey.from_string(builder.public_key)
        response = await asyncio.wait_for(engine.rpc.request("getLatestBlockhash",
            [{"commitment": "confirmed"}]), timeout=3)
        blockhash = ((response or {}).get("value") or {}).get("blockhash")
        if not blockhash:
            raise ReplayBlocked("actual recent native blockhash unavailable")
        blockhash = Hash.from_string(blockhash)
        instructions = [set_compute_unit_limit(engine.native_compute_unit_limit), Instruction(
            Pubkey.from_string(native.program_id), bytes(native.data),
            [AccountMeta(Pubkey.from_string(meta.pubkey), meta.is_signer, meta.is_writable)
             for meta in native.accounts])]
        raw = builder.tx_kernel.compile_message(bytes(payer), instructions, bytes(blockhash),
            lambda: bytes(to_bytes_versioned(MessageV0.try_compile(payer, instructions, [], blockhash))))
        message = MessageV0.from_bytes(raw[1:] if raw[:1] == b"\x80" else raw)
        unsigned = VersionedTransaction.populate(message,
            [Signature.default()] * message.header.num_required_signatures)
        simulation, fee = await asyncio.gather(
            asyncio.wait_for(engine.rpc.request("simulateTransaction", [base64.b64encode(bytes(unsigned)).decode(),
                {"encoding": "base64", "sigVerify": False, "replaceRecentBlockhash": True,
                 "commitment": "confirmed"}]), timeout=3),
            asyncio.wait_for(engine.rpc.request("getFeeForMessage", [base64.b64encode(raw).decode(),
                {"commitment": "confirmed"}]), timeout=3))
        value = (simulation or {}).get("value")
        lamports = (fee or {}).get("value")
        if (not isinstance(value, dict) or "err" not in value or value["err"] is not None
                or not _number(value.get("unitsConsumed"), positive=True)
                or type(lamports) is not int or lamports < 0):
            raise ReplayBlocked("native unsigned simulation/observed message fee incomplete or failed")
        now = max(now, time.time())
        payload = {"origin": "canonical_native_simulation", "token": token,
            "phase": "entry" if buying else "exit", "input_amount": amount,
            "route": "pumpswap_native" if native.venue == "pumpswap" else "pump_curve",
            "message_hash": _digest(list(raw)), "network_fee_native": lamports / LAMPORTS,
            "failed_cost_native": 0., "observed_at": now,
            "failed_cost_semantics": "successful unsigned simulation; no failed transaction submitted",
            "simulation": value, "fee": fee, "scope": "UNSIGNED_NATIVE_SIMULATION_ONLY"}
        ref = self.ledger.record_reference("research_native_simulation_cost", payload, timestamp=now)
        if ref is None:
            raise ReplayBlocked("native simulation cost evidence append failed")
        return {**payload, "evidence_ref": ref}


class NativeForwardPaper:
    def __init__(self, validation: ResearchValidationService, *, max_experiments=8, max_state_age=30, component=COMPONENT):
        self.validation = validation
        self.framework = validation.framework
        self.ledger = validation.evidence_ledger
        self.max_experiments = max(1, min(32, int(max_experiments)))
        self.max_state_age = max(1., min(60., float(max_state_age)))
        self.component = component
        self.state = self.framework.get_research_component(component) or {
            "positions": {}, "completed": {}, "blocked": {}, "cursor": 0}

    def _save(self):
        self.framework.checkpoint_research_component(self.component, self.state)

    def _compiled(self, spec):
        return compile_candidate(spec.exact_candidate)

    def _blocked(self, variable, reason, candidate_id=None):
        result = self.validation._blocked("native_forward_" + variable, reason, {"candidate_id": candidate_id})
        counts = self.state.setdefault('input_refusals', {})
        counts[variable] = counts.get(variable, 0) + 1
        self.state['last_tick_status'] = 'BLOCKED'
        self.state['last_input_wait'] = variable
        self._save()
        return result

    def _record(self, event_type, payload, now):
        ref = self.ledger.record_reference(event_type, payload, timestamp=now)
        if ref is None:
            raise ReplayBlocked("canonical native paper evidence append failed")
        return ref

    def _quote(self, native_state, amount, *, buying, fee_bps, now):
        if isinstance(native_state, BondingCurveState):
            if native_state.real_token_reserves <= 0 or native_state.real_sol_reserves <= 0:
                raise ReplayBlocked("curve real inventory/depth unmeasured; reconstructed virtual reserves are insufficient")
            if type(fee_bps) is not int or not 0 <= fee_bps < 10000:
                raise ReplayBlocked("measured current native curve fee schedule unavailable")
            return ((quote_buy if buying else quote_sell)(native_state, amount, fee_bps), "pump_curve")
        if isinstance(native_state, PumpSwapPoolState):
            return ((pool_buy if buying else pool_sell)(native_state, amount, now=now,
                    max_age_s=self.max_state_age), "pumpswap_native")
        raise ReplayBlocked("actual decoded native curve/pool state required")

    def _cost(self, costs, *, now, token, phase, amount):
        if isinstance(costs, dict) and "evidence_ref" not in costs:
            costs = costs.get(f"{phase}:{amount}")
        if not isinstance(costs, dict) or not costs.get("evidence_ref"):
            raise ReplayBlocked("actual native simulation network/failed-cost evidence unavailable")
        for key in ("network_fee_native", "failed_cost_native", "observed_at"):
            if not _number(costs.get(key)) or costs[key] < 0:
                raise ReplayBlocked("measured simulation network and failed costs required")
        if not now - self.max_state_age <= costs["observed_at"] <= now:
            raise ReplayBlocked("native simulation cost receipt is stale/future")
        # The trusted adapter provides cost receipt references, not configuration
        # defaults. Its canonical ledger binding is checked before admitting it.
        ref = costs["evidence_ref"]
        row = self.validation.read_evidence(ref)
        body = row.get("payload", {})
        if (row.get("event_type") != "research_native_simulation_cost"
                or body.get("origin") != "canonical_native_simulation"
                or body.get("token") != token or body.get("phase") != phase
                or body.get("input_amount") != amount
                or any(body.get(key) != costs[key] for key in
                       ("network_fee_native", "failed_cost_native", "observed_at"))):
            raise ReplayBlocked("canonical native simulation cost receipt mismatch")
        return costs["network_fee_native"], costs["failed_cost_native"]

    def cost_requests(self, token, episode, *, now, native_state=None, fee_bps=None):
        """Bounded exact intents for the registered asynchronous fee adapter."""
        intents = {}
        for eid, experiment in self.validation.state["experiments"].items():
            if experiment["registered_at"] > episode.get("created_at", 0):
                continue
            key = _digest({"experiment": eid, "token": token})
            if key in self.state["completed"]:
                continue
            try:
                spec = self.validation._candidate(experiment)
                position = self.state["positions"].get(key)
                if position:
                    phase, amount = "exit", position["tokens"]
                    if native_state is not None:
                        quote,_=self._quote(native_state,amount,buying=False,fee_bps=fee_bps,now=now)
                        if quote.data_status!='OK' or quote.output_amount<=0:
                            continue
                        multiple=(quote.output_amount+quote.fee_amount)/LAMPORTS/position['entry_cost_native']
                        mark=Mark(now,multiple,position['entry_cost_native'],True)
                        rule=compile_exit(spec.exact_candidate,entry_time=position['entry_at'])
                        if rule([mark],1.) is None:
                            continue  # Local mark only; do not spend RPC quota while simply holding.
                else:
                    program = self._compiled(spec)
                    target = episode["created_at"] + program.mechanism.decision_offset_s
                    if (not target <= now <= target + spec.exact_candidate["max_decision_latency_ms"] / 1000
                            or program.mechanism.rule(EpisodeView(episode)) is not True):
                        continue
                    phase, amount = "entry", int(program.size_native * LAMPORTS)
                intents[f"{phase}:{amount}"] = {"token": token, "phase": phase, "amount": amount}
                if len(intents) >= 2:
                    break
            except (ReplayBlocked, TypeError, ValueError, KeyError):
                continue
        return list(intents.values())

    def tick(self, token, episode, native_state, *, state_observed_at, source_ref,
             equity_usd, native_to_usd, now, fee_bps=None, execution_costs=None):
        self.state['observed_ticks'] = self.state.get('observed_ticks', 0) + 1
        self.state['last_observed_at'] = now
        if self.ledger is None:
            return self._blocked("ledger", "canonical evidence ledger unavailable")
        if not isinstance(episode, dict) or episode.get("token") != token:
            return self._blocked("episode", "active untouched native episode required")
        eligible = [(eid, experiment) for eid, experiment in self.validation.state["experiments"].items()
                    if experiment["registered_at"] <= episode.get("created_at", 0)]
        if not eligible:
            self.state['last_tick_status'] = 'NO_REGISTERED_FRESH_COHORT'
            self._save()
            return {"disposition": "DEEPENING_TASK", "detail": {"status": "NO_REGISTERED_FRESH_COHORT"}, "children": []}
        if (not _number(now, positive=True) or not _number(state_observed_at, positive=True)
                or not now - self.max_state_age <= state_observed_at <= now or not source_ref):
            return self._blocked("native_state", "fresh decoded native state provenance required")
        if not _number(equity_usd, positive=True) or not _number(native_to_usd, positive=True):
            return self._blocked("equity_conversion", "observed equity and native conversion required")
        # Existing positions get first opportunity to exit. A bounded rotation
        # services the remaining experiments without another task/scheduler.
        eligible.sort(key=lambda row: (_digest({"experiment": row[0], "token": token})
                                       not in self.state["positions"], row[0]))
        start = self.state["cursor"] % len(eligible)
        selected = (eligible[start:] + eligible[:start])[:self.max_experiments]
        self.state["cursor"] = (start + len(selected)) % len(eligible)
        results, children = [], []
        for eid, experiment in selected:
            key = _digest({"experiment": eid, "token": token})
            if key in self.state["completed"]:
                continue
            try:
                recovered = experiment["outcomes"].get("native-paper-outcome:" + key)
                if recovered:
                    self.state["completed"][key] = {"status": "NATIVE_PAPER_CLOSED",
                                                     "evidence_refs": recovered["receipt"]["evidence_refs"]}
                    self.state["positions"].pop(key, None)
                    continue
                spec = self.validation._candidate(experiment)
                program = self._compiled(spec)
                position = self.state["positions"].get(key)
                if position is not None and position.get("cohort_pending"):
                    consumed = self.validation.consume("FORWARD_SHADOW", position["cohort_payload"],
                                                       [position["cohort_ref"]], now)
                    if consumed["disposition"] == "BLOCKED":
                        raise ReplayBlocked(consumed["detail"]["reason"])
                    position["cohort_pending"] = False
                    self._save()
                if position is None:
                    target = episode["created_at"] + program.mechanism.decision_offset_s
                    if not target <= now <= target + spec.exact_candidate["max_decision_latency_ms"] / 1000:
                        raise ReplayBlocked("exact candidate decision horizon elapsed/not yet reached")
                    selection = program.mechanism.rule(EpisodeView(episode))
                    if selection is None:
                        raise ReplayBlocked("exact point-in-time program inputs unmeasured at decision cutoff")
                    if selection is False:
                        self.state["completed"][key] = {"status": "DECLINED_EXACT_PROGRAM", "observed_at": now}
                        continue
                    amount = int(program.size_native * LAMPORTS)
                    network, failed = self._cost(execution_costs, now=now, token=token, phase="entry", amount=amount)
                    quote, route = self._quote(native_state, int(program.size_native * LAMPORTS),
                                               buying=True, fee_bps=fee_bps, now=now)
                    if quote.data_status != "OK" or quote.output_amount <= 0:
                        raise ReplayBlocked("native entry quote unfilled: " + quote.reason)
                    # Ensure an executable exit can be quoted at the chosen size.
                    probe, _ = self._quote(native_state, quote.output_amount,
                                           buying=False, fee_bps=fee_bps, now=now)
                    if probe.data_status != "OK" or probe.output_amount <= 0:
                        raise ReplayBlocked("native exit depth unavailable at exact entry size")
                    cid = "native-paper:" + key
                    evidence = self._record("research_forward_native_state", {
                        "token": token, "route": route, "native_state": asdict(native_state),
                        "state_observed_at": state_observed_at, "source_ref": source_ref,
                        "episode_hash": _digest(episode), "observed_at": now}, now)
                    cohort = {"action": "cohort", "experiment_id": eid, "candidate_id": spec.hypothesis_id,
                        "spec_hash": spec.frozen_hash, "cohort_id": cid, "event_at": episode["created_at"],
                        "available_at": now, "decision_at": now, "observed_at": now, "token": token}
                    quote_ref = self._record("research_forward_native_quote", {
                        "token": token, "phase": "entry", "route": route, "quote": asdict(quote),
                        "state_ref": evidence, "observed_at": now}, now)
                    principal = (quote.input_amount - quote.fee_amount) / LAMPORTS
                    costs = quote.fee_amount / LAMPORTS + network + failed
                    cost_row = (execution_costs if "evidence_ref" in execution_costs else execution_costs[f"entry:{amount}"])
                    entry = self._fill(spec, cid, "entry", route, principal, costs, quote_ref,
                                       cost_row["evidence_ref"], now, token=token)
                    position = {"candidate_id": spec.hypothesis_id, "spec_hash": spec.frozen_hash,
                        "cohort_id": cid, "entry_at": now, "entry_cost_native": principal,
                        "entry_costs_native": costs, "tokens": quote.output_amount, "entry_ref": entry,
                        "route": route, "equity_usd": equity_usd, "marks": [],
                        "cohort_pending": True, "cohort_payload": cohort, "cohort_ref": evidence}
                    self.state["positions"][key] = position
                    self._save()
                    consumed = self.validation.consume("FORWARD_SHADOW", cohort, [evidence], now)
                    if consumed["disposition"] == "BLOCKED":
                        raise ReplayBlocked(consumed["detail"]["reason"])
                    position["cohort_pending"] = False
                    self._save()
                    results.append({"status": "NATIVE_PAPER_OPENED", "candidate_id": spec.hypothesis_id})
                    continue
                if position["spec_hash"] != spec.frozen_hash:
                    raise ReplayBlocked("open paper position cannot inherit mutated candidate")
                if now <= position["entry_at"]:
                    continue
                quote, route = self._quote(native_state, position["tokens"], buying=False, fee_bps=fee_bps, now=now)
                if quote.data_status != "OK" or quote.output_amount <= 0 or route != position["route"]:
                    raise ReplayBlocked("native forward exit currently unquotable or route changed")
                multiple = (quote.output_amount + quote.fee_amount) / LAMPORTS / position["entry_cost_native"]
                mark = Mark(now, multiple, position["entry_cost_native"], True)
                position["marks"].append(asdict(mark))
                position["marks"] = position["marks"][-256:]
                rule = compile_exit(spec.exact_candidate, entry_time=position["entry_at"])
                chosen = rule([mark], 1.)
                if chosen is None:
                    continue
                network, failed = self._cost(execution_costs, now=now, token=token, phase="exit", amount=position["tokens"])
                state_ref = self._record("research_forward_native_state", {
                    "token": token, "route": route, "native_state": asdict(native_state),
                    "state_observed_at": state_observed_at, "source_ref": source_ref,
                    "observed_at": now}, now)
                quote_ref = self._record("research_forward_native_quote", {
                    "token": token, "phase": "exit", "route": route, "quote": asdict(quote),
                    "state_ref": state_ref, "observed_at": now}, now)
                proceeds = (quote.output_amount + quote.fee_amount) / LAMPORTS
                exit_cost = quote.fee_amount / LAMPORTS + network + failed
                cost_row = (execution_costs if "evidence_ref" in execution_costs else execution_costs[f"exit:{position['tokens']}"])
                exit_ref = self._fill(spec, position["cohort_id"], "exit", route, proceeds, exit_cost,
                                      quote_ref, cost_row["evidence_ref"], now, token=token)
                execution = {"evidence_kind": "native_simulation", "native": True, "filled": True,
                    "route": route, "entry_evidence_ref": position["entry_ref"], "exit_evidence_ref": exit_ref,
                    "entry_cost_native": position["entry_cost_native"], "exit_proceeds_native": proceeds,
                    "fees_native": position["entry_costs_native"] + exit_cost,
                    "failed_cost_native": 0., "native_to_usd": native_to_usd}
                payload = {"action": "outcome", "experiment_id": eid, "candidate_id": spec.hypothesis_id,
                    "spec_hash": spec.frozen_hash, "cohort_id": position["cohort_id"],
                    "event_id": "native-paper-outcome:" + key, "partition": "forward",
                    "scope": "observed_forward_shadow", "token": token, "entry_at": position["entry_at"],
                    "exit_at": now, "observed_at": now, "execution": execution, "all_costs_included": True,
                    "realized_pnl_usd": (proceeds - position["entry_cost_native"]
                        - execution["fees_native"]) * native_to_usd,
                    "equity_at_decision_usd": position["equity_usd"]}
                result = self.validation.consume("FORWARD_SHADOW", payload, [position["entry_ref"], exit_ref], now)
                if result["disposition"] == "BLOCKED":
                    raise ReplayBlocked(result["detail"]["reason"])
                self.state["completed"][key] = {"status": "NATIVE_PAPER_CLOSED", "event_id": payload["event_id"],
                                                 "observed_at": now, "evidence_refs": [position["entry_ref"], exit_ref]}
                del self.state["positions"][key]
                results.append(result["detail"])
            except (ReplayBlocked, ValueError, TypeError, KeyError) as exc:
                blocked = self._blocked("inputs", str(exc), experiment["candidate_id"])
                self.state["blocked"][key] = blocked["detail"]
                children.extend(blocked["children"])
                results.append(blocked["detail"])
        self.state['last_tick_status'] = 'BLOCKED' if children else 'NATIVE_PAPER_PASS'
        self._save()
        return {"disposition": "BLOCKED" if children else "DEEPENING_TASK",
                "detail": {"status": "NATIVE_PAPER_PASS", "results": results,
                           "real_fill": False, "open_positions": len(self.state["positions"])},
                "children": children}

    def _fill(self, spec, cid, phase, route, size, costs, quote_ref, simulation_ref, now, *, token):
        return self._record("research_forward_native_execution", {
            "origin": "canonical_native_execution", "phase": phase, "candidate_id": spec.hypothesis_id,
            "spec_hash": spec.frozen_hash, "cohort_id": cid, "token": token, "evidence_kind": "native_simulation",
            "observed_at": now, "available_at": now, "route": route,
            "size_native": size, "costs_native": costs, "filled": True,
            "quote_ref": quote_ref, "simulation_ref": simulation_ref,
            "paper": True, "real_fill": False}, now)

    def report(self):
        return {"open_positions": len(self.state["positions"]), "completed": len(self.state["completed"]),
                "blocked": len(self.state["blocked"]), "scope": "NATIVE_QUOTE_PAPER_ONLY", "real_fill": False,
                'observed_ticks':self.state.get('observed_ticks',0),
                'last_observed_at':self.state.get('last_observed_at'),
                'last_tick_status':self.state.get('last_tick_status','NOT_OBSERVED'),
                'last_input_wait':self.state.get('last_input_wait'),
                'input_refusals':dict(self.state.get('input_refusals',{}))}
