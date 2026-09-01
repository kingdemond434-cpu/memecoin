"""The desk's self-description: readiness, reports, and the HTTP surface.

Split out of a 5,500-line `main.py`. The audit's objection to that file was
not aesthetic -- one module holding the trading path, the reporting surface
and the HTTP server means a change to any of them risks all of them, and a
merge conflict in one is a merge conflict in every one.

This is the largest cohesive slice that cannot affect a trade: twenty-five
methods that read state and format it, plus the aiohttp endpoints that serve
the result. Nothing here decides anything.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from aiohttp import web
from src.chains.yellowstone_grpc import (
    NATIVE_FASTPATH_STATUS, PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream, YellowstoneClient,
    create_combined_subscription,
)
from src.runtime.serialisation import jsonable as _jsonable
from src.collectors.transports import (
    HttpClient, build_transports, start_transports, stop_transports, transport_report,
)
from src.strategies.actor_graph import (
    BuyerDNA, Entry, IndependenceReport, SwarmPredictor, WalletIndependence,
    aggregate_smart_flow, build_fingerprint,
)
from src.chains.idl import report as idl_report
import logging
from src.runtime.serialisation import jsonable as _jsonable

logger = logging.getLogger(__name__)


class ReportingSurface:
    """Everything the desk says about itself, and the HTTP that serves it.

    Moved out of `main.py` as a mixin rather than a collaborator, and that
    choice is deliberate. A collaborator would need the twenty-odd subsystems
    these methods read passed to it or reached through a back-reference, and
    rewriting that plumbing is a behaviour-changing refactor wearing the
    clothes of a tidy-up. A mixin is a pure move: every `self.` here resolves
    exactly as it did, so a regression can only come from the move itself and
    the test suite covers that.

    What it buys is ownership. Reporting is a quarter of the desk's methods
    and none of them can affect a decision, so separating them means a change
    to the status page cannot touch the trading path, and a merge conflict in
    one is not a merge conflict in the other.

    Two things did change, because leaving them would have been a bug rather
    than a move. `__file__` now resolves inside `src/runtime/`, so the
    dashboard asset path is computed from the package root instead of from
    this module's parent. And the dashboard cache lives on this class rather
    than on the desk, because a mixin assigning to `SomeOtherClass.attr` is
    a reach across a boundary that the move exists to draw."""

    #: Read once and held: the dashboard is a static asset and re-reading it
    #: on every request turns a page refresh into disk IO on the trading box.
    _dashboard_cache = None

    def pool_route_report(self) -> Dict[str, Any]:
        """Whether graduation actually keeps native execution.

        A route that exists, is wired, and answers DATA_BLOCKED on every
        migrated coin is indistinguishable from one that was never wired --
        except that it looks finished. This is what tells the two apart.
        """
        tracked = len(self._latest_pool_state)
        decoded = len(self._pool_accounts)
        priced = sum(1 for state in self._latest_pool_state.values()
                     if state.blocked_reason() is None)
        executable = sum(1 for token, state in self._latest_pool_state.items()
                         if state.blocked_reason() is None and token in self._pool_accounts)
        return {
            "status": "OK" if executable else "DATA_BLOCKED",
            "pools_tracked": tracked, "accounts_decoded": decoded,
            "pools_priceable": priced, "pools_executable": executable,
            "executable_share": (executable / tracked) if tracked else None,
            "reasons": dict(collections.Counter(
                state.blocked_reason() or "ok"
                for state in self._latest_pool_state.values())),
        }

    def follow_report(self) -> Dict[str, Any]:
        """Whether wallet value is being measured or merely modelled."""
        return {
            "open_follows": sum(len(items) for items in self._follow_candidates.values()),
            "tokens_with_follows": len(self._follow_candidates),
            "resolved": self._follow_resolved,
            "rejected": self._follow_unresolved,
            "horizon_seconds": float(self.global_config.get("follow_horizon_seconds", 300.0)),
            "reference_sol": float(self.global_config.get("follow_reference_sol", 0.5)),
            "model": self.wallet_intel.wallet_value.report(),
        }

    def source_intelligence(self, token: str) -> Dict[str, Any]:
        """What public sources have said about this token, and how early.

        Reports the first observation and its lag, because who was first and
        how stale their information already was when it reached us is the
        whole signal. A token nobody mentioned is DATA_BLOCKED, not silent
        agreement that it is uninteresting.
        """
        observations = self._source_events.get(token) or []
        if not observations:
            return {"status": "DATA_BLOCKED",
                    "detail": "no public source has named this token",
                    "mesh_sources": len(self.source_mesh.sources)}
        first = observations[0]
        return {
            "status": "OK",
            "observations": len(observations),
            "first_source": first.source_id,
            "first_source_class": first.source_class.value,
            "first_observation_lag_s": first.observation_lag,
            "repeaters": self.source_mesh.repeaters_of(first.content_hash),
            "languages": sorted({event.language for event in observations
                                 if event.language}),
        }

    def actor_intelligence(self, token: str, as_of: Optional[float] = None) -> Dict[str, Any]:
        """First25 DNA, actor-adjusted flow and forward swarm probability.

        Built from the same entries the independence graph consumes, so the
        three cannot disagree about who bought and in what order. Every field
        is DATA_BLOCKED rather than defaulted: a launch with no scored buyers
        must not read as a launch whose buyers scored zero.
        """
        as_of = time.time() if as_of is None else as_of
        entries = self._actor_entries.get(token) or []
        report = self.independence_report
        intelligence: Dict[str, Any] = {
            "observed_buyers": len(entries),
            "independence_status": report.status,
        }

        if not entries:
            intelligence["status"] = "DATA_BLOCKED"
            intelligence["detail"] = "no scored buyer observed for this token yet"
            return intelligence

        # Funding ancestry over the same ordered buyers. Traced once and used
        # three ways -- to compress smart flow, to feed First25, and to report
        # how many actors those wallets actually are.
        ordered = []
        for entry in sorted(entries, key=lambda item: item.timestamp):
            if entry.wallet not in ordered:
                ordered.append(entry.wallet)
        creator = str((self.genealogy.tokens.get(token).deployer
                       if getattr(self.genealogy, "tokens", {}).get(token) else "") or "")
        ancestry = self.funder_ancestry.analyse(ordered[:self.buyer_dna.depth])
        funding_features = self.funder_ancestry.buyer_features(
            ordered[:self.buyer_dna.depth], creator)
        intelligence["funder_ancestry"] = ancestry.to_dict()

        fingerprint = build_fingerprint(token, entries, report,
                                        depth=self.buyer_dna.depth,
                                        funding_features=funding_features)
        match = self.buyer_dna.match(fingerprint)
        intelligence["buyer_dna"] = {
            "status": match.status, "label": match.label,
            "confidence": match.confidence, "detail": match.detail,
            "depth": fingerprint.depth,
        }

        flow = aggregate_smart_flow(entries, report, ancestry=ancestry)
        intelligence["smart_flow"] = {
            "status": flow.status, "evidence": flow.evidence,
            "naive_evidence": flow.naive_evidence, "discount": flow.discount,
            "measured_wallets": flow.measured_wallets,
            "unmeasured_wallets": flow.unmeasured_wallets,
            "ancestry_compressed": flow.ancestry_compressed,
        }

        swarm = self.swarm_predictor.evaluate(entries, report, as_of)
        intelligence["swarm"] = {
            "status": swarm.status, "evidence": swarm.evidence,
            "probability": swarm.probability,
            "independent_skilled": swarm.independent_skilled_so_far,
        }
        intelligence["status"] = "OK"
        return intelligence

    def mark_report(self) -> Dict[str, Any]:
        """Whether marking is actually local, and whether it is right.

        Two different questions. A desk marking 100% locally and drifting 40%
        from the router is fast and wrong, which is worse than slow.
        """
        total = self._marks_local + self._marks_router
        return {
            "marks_local": self._marks_local,
            "marks_via_router": self._marks_router,
            "local_share": (self._marks_local / total) if total else None,
            "cross_checks": self._mark_checks,
            "cross_checks_blocked": self._mark_checks_blocked,
            "cross_checks_diverged": self._mark_checks_diverged,
            "mean_drift": (self._mark_drift_total / self._mark_checks
                           if self._mark_checks else None),
            "divergence_tolerance": float(
                self.global_config.get("mark_cross_check_tolerance", 0.10)),
            "recent_divergences": list(self._mark_divergences),
            "status": "OK" if self._marks_local else "DATA_BLOCKED",
        }

    def stream_event_report(self) -> Dict[str, Any]:
        """What the chain stream has delivered, by event type.

        A connected stream that carries trades but no creations produces an
        empty launch census while every other panel looks healthy, and nothing
        else in the desk distinguishes that from a quiet market.
        """
        counts = dict(sorted(self._stream_events.items()))
        creations = counts.get("token_created", 0)
        total = sum(value for key, value in counts.items() if ":" not in key)
        return {
            "status": ("OK" if creations else
                       "DEGRADED" if total else "DATA_BLOCKED"),
            "detail": (
                "" if creations else
                f"{total} events delivered and not one token_created; the "
                "launch census cannot fill and every rate over it is "
                "undefined" if total else
                "no chain event has been delivered at all"),
            "total": total,
            "token_created": creations,
            "by_type": counts,
        }

    def signer_report(self) -> Dict[str, Any]:
        """Where the private key actually lives, and what the signer has done.

        Reported unconditionally, because "no signer configured" and "signer
        holding the key in this process" are different states an operator must
        be able to tell apart at a glance -- and the second one is the default,
        which is exactly why it has to be visible rather than inferred from
        the absence of the first.
        """
        engine = self.execution_engine
        signer = getattr(getattr(engine, "tx_builder", None), "signer", None)
        if signer is None:
            return {
                "status": "DATA_BLOCKED",
                "mode": "none",
                "isolated": False,
                "detail": ("no execution engine is built yet; nothing can "
                           "sign, which is correct before setup completes"),
                "signed": 0, "refused": 0,
            }
        try:
            report = dict(signer.report())
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "mode": "unknown",
                    "isolated": False, "signed": 0, "refused": 0,
                    "detail": f"signer report failed: {exc}"}
        report.setdefault("status", "OK" if report.get("isolated") else "DEGRADED")
        # A local signer is not an error, and it is not fine either. It is a
        # deliberate configuration the operator should see stated.
        report.setdefault("live_submission_locked",
                          os.getenv("ALLOW_LIVE_TRADING", "").lower()
                          != "yes-i-understand")
        return report

    def execution_conditions_report(self) -> Dict[str, Any]:
        """What the chain costs and whether it is keeping up.

        Reported separately from the miner health because an operator asking
        "why did that bid miss" wants the conditions, not the poll status of
        the thing that measured them. DATA_BLOCKED here is honest: it means
        every bid is being made against an unknown congestion bucket.
        """
        congestion = self._measured_congestion()
        measured = bool(self._network_health or self._priority_fees)
        return {
            "status": "OK" if measured else "DATA_BLOCKED",
            "detail": ("" if measured else
                       "no execution-conditions pass has completed; bids are "
                       "made against an unknown congestion bucket"),
            "congestion": congestion,
            "slot_time_ratio": self._network_health.get("slot_time_ratio"),
            "tps": self._network_health.get("tps"),
            "fee_p50_lamports": self._priority_fees.get("fee_p50_lamports"),
            "fee_p90_lamports": self._priority_fees.get("fee_p90_lamports"),
            "contested_share": self._priority_fees.get("contested_share"),
            "accounts_sampled": self._priority_fees.get("accounts_sampled"),
            "unattributed_wallet_readings": len(self._wallet_readings),
        }

    def identity_report(self) -> Dict[str, Any]:
        """The figure registry, plus the launches currently claiming someone."""
        report = dict(self.identity_watch.report())
        claims = list(self._identity_claims.items())[-25:]
        report["recent"] = [
            {"mint": mint, **assessment.to_dict()} for mint, assessment in claims]
        return report

    def discovery_report(self) -> Dict[str, Any]:
        """How complete our own census denominator actually is.

        `missed` counts pools an outside operator reported that our own stream
        never did. It is the only honest measurement of decoder coverage
        available, because every other view of it is taken from inside the
        decoder -- and a decoder with a gap reports a quiet market, not a gap.
        """
        seen = len(self._discovered_pools)
        missed = self._discovery_misses
        if not seen:
            return {"status": "DATA_BLOCKED", "external_pools_seen": 0,
                    "missed_by_our_stream": 0, "coverage": None,
                    "detail": "no external pool discovery yet this run"}
        coverage = 1.0 - (missed / seen)
        return {
            "status": "OK" if coverage >= 0.9 else "DEGRADED",
            "external_pools_seen": seen,
            "missed_by_our_stream": missed,
            "coverage": round(coverage, 4),
            "detail": ("" if coverage >= 0.9 else
                       f"{missed} of {seen} pools seen by outside operators "
                       "never reached our census; that is a decoder or "
                       "program-coverage gap, not a quiet market"),
        }

    def credential_report(self) -> Dict[str, Any]:
        """Which credentials are present, by NAME.

        No value is read, logged or returned. What an operator needs to know
        is whether the desk can see the key they set -- an env file loaded by
        the wrong unit, or a variable set in a shell the service never
        inherited, both look exactly like a missing key from the outside and
        this is what tells them apart.
        """
        present = [name for name, _ in self.CREDENTIALS if os.getenv(name)]
        absent = [(name, unlocks) for name, unlocks in self.CREDENTIALS
                  if not os.getenv(name)]
        session = Path("data/telegram/collector.session")
        telegram_ready = bool(os.getenv("TELEGRAM_API_ID")
                              and os.getenv("TELEGRAM_API_HASH")
                              and session.exists())
        return {
            "present": present,
            "absent": [{"name": name, "unlocks": unlocks} for name, unlocks in absent],
            "telegram": {
                "keys_present": bool(os.getenv("TELEGRAM_API_ID")
                                     and os.getenv("TELEGRAM_API_HASH")),
                "channels_listed": len([item for item in
                                        os.getenv("TELEGRAM_CHANNELS", "").split(",")
                                        if item.strip()]),
                # Telethon asks for a phone number when it finds no session,
                # and under systemd there is no stdin to ask on. So the keys
                # being set is not the same as Telegram being ready.
                "session_authorised": session.exists(),
                "ready": telegram_ready,
                "authorise_with": (
                    "" if telegram_ready
                    else ".venv/bin/python -m src.research.telegram_authorize"),
            },
            # Which of the always-on miners each key actually feeds, and how
            # often that miner runs. "The key is set" and "something is using
            # it" are different facts, and only the second one produces data.
            "miners": {
                "chain_stream": {
                    "keys": ["YELLOWSTONE_GRPC_URL", "YELLOWSTONE_GRPC_TOKEN"],
                    "cadence": "push",
                    "active": bool(os.getenv("YELLOWSTONE_GRPC_URL")),
                    "detail": "gRPC program stream; falls back to RPC polling",
                },
                "rpc": {
                    "keys": ["ALCHEMY_KEY", "HELIUS_API_KEY"],
                    "cadence": "on demand",
                    "active": bool(os.getenv("ALCHEMY_KEY") or os.getenv("HELIUS_API_KEY")),
                    "detail": "account reads, wallet history, liquidity probes",
                },
                "social_watcher": {
                    "keys": ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "YOUTUBE_API_KEY"],
                    "cadence": "5s",
                    "active": bool(os.getenv("TELEGRAM_API_ID")
                                   or os.getenv("YOUTUBE_API_KEY")),
                    "detail": "watched accounts and token mentions",
                },
                "source_mesh": {
                    "keys": ["TELEGRAM_API_ID", "TELEGRAM_CHANNELS"],
                    "cadence": "per source, 1s to 30m",
                    "active": telegram_ready,
                    "detail": "MTProto push for Telegram; needs the authorised session",
                },
                "global_research": {
                    "keys": ["GITHUB_TOKEN"],
                    "cadence": "hourly",
                    "active": True,
                    "detail": "public research mining; the token raises the quota",
                },
            },
            "live_trading_acknowledged": (
                os.getenv("ALLOW_LIVE_TRADING", "").lower() == "yes-i-understand"),
        }

    def ignition_census(self) -> Dict[str, Any]:
        """How many tracked narratives are in each lifecycle state."""
        counts: Dict[str, int] = {}
        igniting: List[str] = []
        for token in list(self._source_events)[:500]:
            try:
                reading = self._read_ignition(token)
            except Exception:
                continue
            counts[reading.state.value] = counts.get(reading.state.value, 0) + 1
            if reading.igniting and len(igniting) < 20:
                igniting.append(token)
        return {
            "status": "OK" if counts else "DATA_BLOCKED",
            "detail": ("" if counts else
                       "no token has a source touch yet; every narrative is chain-only"),
            "states": dict(sorted(counts.items())), "igniting": igniting,
            "kol_reach_threshold": self.ignition.kol_reach,
        }

    def readiness(self) -> Dict[str, Any]:
        return {
            "mode": "DRY_RUN" if self.dry_run else "LIVE",
            "live_submission_locked": os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand",
            "offline": self.offline, "rpc": self.chain_registry.get_all_stats() if self.chain_registry else {},
            "yellowstone": self.yellowstone.get_status() if self.yellowstone else {"status": "NOT_STARTED"},
            "rpc_program_stream": self.rpc_program_stream.get_status() if self.rpc_program_stream else None,
            "prediction": "OK" if self.predictor and self.predictor._is_trained else "DATA_BLOCKED",
            "age_bands": self.predictor.report() if self.predictor else {"status": "DATA_BLOCKED"},
            "exit_policy": {"status": self.exit_policy_status, "detail": self.exit_policy_detail,
                            "policy": asdict(self.exit_policy)},
            "equity": {"status": self.equity_status, "wallet_equity_usd": self.wallet_equity_usd,
                       "sol_price_usd": self.sol_price_usd},
            "execution": {"dry_run": self.execution_engine.dry_run if self.execution_engine else True},
            "native_fastpath": NATIVE_FASTPATH_STATUS,
            "native_route": (self.execution_engine.native_route_report()
                             if self.execution_engine else {"status": "DATA_BLOCKED"}),
            "pumpswap_route": self.pumpswap_route.report(),
            "pumpswap_execution": self.pool_route_report(),
            "idl": idl_report(),
            "action_policy": {"trained": self.action_policy.is_trained,
                              "min_edge": self.action_policy.min_edge},
            # Whether the canonical decision path is on Rust yet, and the
            # measured agreement that put it there. A kernel that exists and
            # is never called is the same defect as one never written.
            "t0_kernel": self.t0_kernel.report(),
            # An empty registry is not "nothing is a copycat". It is "we
            # cannot tell", and a status page that stays silent about it lets
            # an operator read silence as safety.
            # Presence only, never a value. An env file loaded by the wrong
            # unit and a missing key look identical from outside.
            "credentials": self.credential_report(),
            # What is being mined, what is dark, and why. A miner that has
            # never returned a record reads differently from one that is
            # failing, and the two need different fixes.
            "data_miners": self.data_miners.report(),
            # Which operator is answering each question right now, and what
            # is dark. A domain running on its third rung is fine and is
            # reported as such; a domain with no rung left is a question the
            # desk asks continuously and currently cannot answer.
            # Where the milliseconds go. The only thing on this page that
            # can tell you whether the next hour belongs to code or to money.
            "latency": self.latency.report(),
            # Every decision and what it was judged against, ignored launches
            # included. `ignore_share` is the line that matters: a corpus
            # where IGNORE is a minority is still only recording trades.
            "decision_corpus": self.counterfactual_corpus.report(),
            # What the opening cohort did after its fill, per held token.
            "cohorts": {
                token: report.report()
                for token, report in sorted(
                    getattr(self, "cohort_reports", {}).items())},
            # Coordination an exchange withdrawal was used to hide. Reports
            # its own denominator: without measured emission rates there are
            # no clusters, and that is a finding rather than a clean slate.
            "temporal_funding": {
                "status": ("OK" if getattr(self, "temporal_clusters", None)
                           else "DATA_BLOCKED"),
                "clusters": len([c for c in getattr(self, "temporal_clusters", ())
                                 if getattr(c, "status", "") == "OK"]),
                "wallets_discounted": len(getattr(self, "temporal_discounts", {})),
                "sources_rated": len(getattr(self, "exchange_rates", {})),
                "detail": ("no exchange emission rates measured yet; a cluster "
                           "scored without one would flag every busy exchange"
                           if not getattr(self, "exchange_rates", {}) else ""),
            },
            # Public wallets under reconstruction, scored by what FOLLOWING
            # them would return -- never by their headline PnL.
            "benchmark_wallets": (
                self.benchmark_corpus.report()
                if getattr(self, "benchmark_corpus", None) is not None
                else {"status": "DATA_BLOCKED"}),
            # The chain receive path in Rust, and whether it has yet
            # matched the Python client it runs beside.
            "t0_risk": (
                self.t0_risk.report()
                if getattr(self, "t0_risk", None) is not None
                else {"status": "MISSING"}),
            "sol_price_age_s": round(float(getattr(self, "sol_price_age_s", 0.0)), 1),
            "native_ingress": (
                self.native_ingress.report()
                if getattr(self, "native_ingress", None) is not None
                else {"status": "OFF"}),
            # The heavy miners in their own interpreter. OFF is a real and
            # common state: it is a behaviour change on a two-vCPU box and
            # should be turned on against a measured p99.
            "context_offload": (
                self.context_offload.report()
                if getattr(self, "context_offload", None) is not None
                else {"status": "OFF", "isolation": "thread",
                      "detail": ("desk-independent miners still share this "
                                 "interpreter; set offload_context_miners "
                                 "to move them")}),
            # Launch venues, and which of them this node has PROVEN it can
            # decode. A declared program that never matches shows up here as
            # a venue stuck at zero rather than as silent absence.
            "launchpads": (self.launchpads.report()
                           if getattr(self, "launchpads", None) is not None
                           else {"status": "DATA_BLOCKED"}),
            # Which inbound feed arrives first, and which one sees everything.
            "feed_race": (self.feed_race.report()
                          if getattr(self, "feed_race", None) is not None
                          else {"status": "DATA_BLOCKED"}),
            # Proof the exit was ready before it was needed.
            "exit_readiness": (self.exit_readiness.report()
                               if getattr(self, "exit_readiness", None) is not None
                               else {"status": "DATA_BLOCKED"}),
            # MFE:MAE per entry state -- what win rate cannot see.
            "excursions": (self.excursions.report()
                           if getattr(self, "excursions", None) is not None
                           else {"status": "DATA_BLOCKED"}),
            # Independent landing MECHANISMS, and which ones actually land.
            # Seven Jito regions is one mechanism; a router reporting one
            # mechanism has the redundancy of having none.
            "landing_router": (self.execution_engine.landing_router.report()
                               if getattr(self.execution_engine, "landing_router", None)
                               is not None else
                               {"status": "DATA_BLOCKED",
                                "detail": "no execution engine yet"}),
            "miner_offload": (self.miner_offload.report()
                              if self.miner_offload is not None
                              else {"status": "OFF",
                                    "detail": "miners are running on the main loop"}),
            "substitution": self.substitution.report(),
            "telegram_channels": (self.channel_book.report()
                                  if self.channel_book is not None
                                  else {"status": "DATA_BLOCKED",
                                        "detail": "channel book not built yet"}),
            "identity_watch": self.identity_report(),
            "discovery": self.discovery_report(),
            "execution_conditions": self.execution_conditions_report(),
            "stream_events": self.stream_event_report(),
            # Which implementation compiles and assembles a transaction, and
            # on what evidence. Byte parity, so there is no tolerance to tune.
            "tx_kernel": (self.execution_engine.tx_builder.tx_kernel.report()
                          if getattr(getattr(self.execution_engine, "tx_builder", None),
                                     "tx_kernel", None) is not None
                          else {"status": "DATA_BLOCKED",
                                "detail": "no transaction builder yet"}),
            "dry_build": (self.execution_engine.dry_build_report()
                          if self.execution_engine is not None
                          else {"status": "DATA_BLOCKED",
                                "detail": "no execution engine yet"}),
            "pump_decoder": (self.pump_monitor.decoder_report()
                             if self.pump_monitor is not None
                             else {"status": "DATA_BLOCKED",
                                   "detail": "no Pump monitor is wired"}),
            "launch_census": self.launch_census.report(),
            "screen_policy": self.screen_policy.report(),
            "memory": self.memory.report(),
            "signer": self.signer_report(),
            "fact_ladder": self.facts.report(),
            "calibration": self.calibration.report(),
            "entity_registry": self.entity_registry.report(),
            # What following the wallets we watch has actually returned, at
            # fills we could have got. A watch list nobody has scored is a
            # list of wallets, not intelligence.
            "wallet_follow": self.follow_report(),
            # Local marking versus the router, and whether the two agree.
            "marking": self.mark_report(),
            # Whether exits are actually served from the prepared ladder. One
            # built for every position and used for none is pure cost.
            "staged_exits": self.staged_exits.report(),
            # The narrative lifecycle across tracked tokens. Reported as a
            # census rather than per token: what an operator needs is whether
            # anything is igniting, not a row per launch.
            "ignition": self.ignition_census(),
            "source_mesh": {**self.source_mesh.health(),
                            "registry": self.source_registry_report.to_dict(),
                            # What is wired, what answered, and what could not
                            # be built and why. A declaration with no transport
                            # is a coverage hole; one with a transport that has
                            # never returned a record is a different hole, and
                            # the two need telling apart.
                            "transports": {**self.transport_report.to_dict(),
                                           **transport_report(self.transports)},
                            "transport_start_failures": dict(
                                getattr(self, "transport_start_failures", {}))},
            "actor_graph": {"independence_status": self.independence_report.status,
                            "measured_pairs": self.independence_report.observed_pairs,
                            "scored_wallets": len(self.independence_report.scores),
                            # How many actors the tracked wallets actually are.
                            "funder_ancestry": (self.ancestry_report.to_dict()
                                                if self.ancestry_report is not None
                                                else {"status": "DATA_BLOCKED"})},
            "reentry": self.reentry_book.report(),
            # Distance to the next promotion stage, as ratios. A gate that
            # says FAIL cannot distinguish a week away from a year away, and
            # that difference decides whether to keep running or change
            # something.
            "forward_evidence": self.forward_evidence.report(),
            "regime": self.current_regime,
            # A queue silently shedding work looks exactly like a quiet market,
            # so both drop counters are surfaced rather than only logged.
            "event_loop": {
                # Folded in here because it was a SECOND "event_loop" key in
                # this same dict literal and Python kept the later one, so
                # the loop implementation has never once reached /status --
                # the one field that says whether uvloop is actually in use.
                "implementation": os.getenv("MEMECOIN_EVENT_LOOP", "unmeasured"),
                "redecision_queued": self._redecide.qsize(),
                "redecision_drops": self._redecision_drops,
                "candidate_drops": self._candidate_drops,
                "candidate_pipelines": len(self._candidate_pipelines),
                "redecision_workers": len(self._redecision_tasks),
            },
            # Whether the objective actually owns the decisions. A fallback
            # that quietly becomes the main path is the failure this catches.
            "action_authority": {
                "priced_holds": self._priced_holds,
                "unpriced_cycles": self._unpriced_cycles,
                "suppressed_monster_banks": self._suppressed_monster_banks,
            },
            "exit_latency": self.landing_latency.estimate().report(),
            "decision_contribution": self.contribution_ledger.report(),
            "wallet_coverage": (self.wallet_intel.coverage_report()
                                if self.wallet_intel else {"status": "DATA_BLOCKED"}),
            # Which declared modules actually reached a decision. A rate that
            # falls to zero between two audit packs means a component was
            # disconnected, and no test will say so.
            "intelligence_coverage": {"entry": self.entry_coverage.report(),
                                      "position": self.position_coverage.report()},
            "authenticity": {"watched_entities": len(self._watched_entities)},
            "hot_state": self.hot_state.report(),
            "mega_event_reserve": self.mega_event_reserve_state,
            "portfolio": self.elogw_engine.get_portfolio_state() if self.elogw_engine else {},
            "rug_hazard": self.rug_hazard.get_stats() if self.rug_hazard else {},
            "dataset": self.dataset_builder.get_stats() if self.dataset_builder else {},
            "research": self.global_research.get_stats() if self.global_research else {},
            "social": self.social_intel.get_stats() if self.social_intel else {},
            "public_coordination": self.public_coordination.get_stats() if self.public_coordination else {},
            "champions": self.champion_challenger.get_stats() if self.champion_challenger else {},
        }

    async def _health_loop(self):
        while self._running:
            # Checked far more often than health is logged: an allocation
            # spike that kills the process does not wait for the minute mark.
            for _ in range(6):
                if not self._running:
                    break
                try:
                    self.memory.observe()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("memory governor read failed: %s", exc)
                await asyncio.sleep(10)
            snapshot = _jsonable(self.readiness())
            logger.info("HEALTH %s", json.dumps(snapshot, separators=(",", ":")))
            self._persist_readiness(snapshot)

    def _record_ops_event(self, stream: str, payload: Dict[str, Any]) -> None:
        """Append one operational telemetry row for the monitor and audit pack.

        Deliberately separate from the research lake. The lake is optimised for
        point-in-time correctness and completeness; this is optimised for a
        monitor being able to answer "what is the recent failure rate" in one
        cheap pass without loading episodes. Failures are swallowed for the
        same reason readiness persistence is: telemetry must never be able to
        halt the desk it describes.
        """
        # Trade outcomes are the promotion ledger's only input, and this is
        # the one place every outcome passes through -- entered or declined.
        if stream == "trade_outcomes":
            token = str(payload.get("token", "") or "")
            if token and not payload.get("rejection_reason"):
                # Screens are recorded by _record_blocked_decision; anything
                # arriving here without one reached the decision path.
                self.launch_census.decide(
                    token, str(payload.get("action", "") or ""))
                if payload.get("entered"):
                    self.launch_census.enter(token)
            self._record_forward_evidence(payload)
            # Every stated probability, scored against what happened. The
            # harness measures; this is what gives it something to measure.
            try:
                self._record_calibration(payload)
            except (TypeError, ValueError) as exc:
                logger.debug("calibration record failed: %s", exc)
        try:
            root = Path(self.global_config.get("ops_state_dir", "data/state"))
            root.mkdir(parents=True, exist_ok=True)
            row = {"timestamp": time.time(), **payload}
            with (root / f"{stream}.jsonl").open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError as exc:
            logger.debug("ops telemetry write failed for %s: %s", stream, exc)

    def _persist_readiness(self, snapshot: Dict[str, Any]) -> None:
        """Write the snapshot the out-of-process monitor reads.

        Logging health is not the same as exposing it. A monitor that has to
        parse the log stream cannot tell "the desk is fine and quiet" from "the
        desk stopped writing", whereas the mtime of this file answers that
        directly and is the first thing the monitor checks.

        Written to a temporary file and renamed, so a reader never catches a
        half-written snapshot and concludes the node is broken. Failures here
        are logged and swallowed: a monitor that cannot be updated must never
        be able to take down the desk it monitors.
        """
        path = Path(self.global_config.get("readiness_path", "data/state/readiness.json"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(snapshot, default=str))
            tmp.replace(path)
        except OSError as exc:
            logger.warning("could not persist readiness snapshot to %s: %s", path, exc)

    async def _flush_endpoint(self, _request):
        """Persist the ledgers now, and say what was written.

        Exists so a stale-persistence fault has a remedy that is not a
        restart. A restart at that moment discards everything since the last
        successful save, which is precisely the loss the check exists to
        prevent.
        """
        try:
            self._flush_ledgers()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "detail": f"{type(exc).__name__}: {exc}"},
                status=500)
        root = Path(self.global_config.get("ops_state_dir", "data/state"))
        written = {}
        for name in ("forward_evidence.json", "launch_census.json",
                     "calibration.json"):
            path = root / name
            written[name] = (round(time.time() - path.stat().st_mtime, 1)
                             if path.exists() else None)
        return web.json_response({"ok": True, "age_seconds": written})

    async def _release_sources_endpoint(self, _request):
        """Lift every source quarantine now, and report what was released.

        The remedy for a domain that has gone completely dark. Every rung
        being stood down at the same moment is almost always one shared cause
        -- this address rate limited across the board, DNS wobbling, an
        outbound proxy blip -- and waiting out four independent penalties for
        a cause that has already passed is unmeasured data nobody needed to
        lose. If the cause has NOT passed, the rungs fail again on their next
        pass and re-quarantine themselves, so this is safe to call often.
        """
        try:
            released = self.substitution.release()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "detail": f"{type(exc).__name__}: {exc}"},
                status=500)
        logger.info("SOURCE RELEASE lifted %d quarantine(s)", len(released))
        return web.json_response({"ok": True, "released": released,
                                  "dark": self.substitution.dark_domains()})

    async def _verify_channels_endpoint(self, _request):
        """Run the Telegram verification pass now rather than at its next slot.

        Verification is hourly because a handle that is public now will still
        be public in an hour. The exception is a desk that has NO verified
        channel yet, where waiting an hour to try again is an hour of the
        fastest public signal there is going unread.
        """
        book = getattr(self, "channel_book", None)
        if book is None:
            return web.json_response(
                {"ok": False, "detail": "channel book not built"}, status=503)
        try:
            from src.research.telegram_miners import verification_miner
            results = await verification_miner(self.http_client, book)()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "detail": f"{type(exc).__name__}: {exc}"},
                status=500)
        verified = [row["handle"] for row in results if row.get("verified")]
        return web.json_response({"ok": True, "checked": len(results),
                                  "verified": verified,
                                  "book": book.report()})

    async def _dashboard_endpoint(self, _request):
        """Serve the operator terminal from the desk itself.

        Bound to the same loopback interface as /status, and carrying the same
        exposure: this page renders the desk's whole interior, so it must not
        reach a public interface any more than /status may.
        """
        if ReportingSurface._dashboard_cache is None:
            # From the package root, not this module's parent: this file now
            # lives inside `runtime/`, so the old relative path would resolve
            # to `runtime/runtime/assets` and serve a 404 that looks like a
            # missing asset rather than a bad refactor.
            path = Path(__file__).resolve().parents[1] / "runtime" / "assets" / "dashboard.html"
            try:
                ReportingSurface._dashboard_cache = path.read_text(encoding="utf-8")
            except OSError as exc:
                return web.Response(
                    status=500, content_type="text/plain",
                    text=(f"dashboard asset missing at {path}: {exc}\n"
                          "This ships with the repository; a missing file means "
                          "an incomplete install rather than a runtime fault."))
        return web.Response(text=ReportingSurface._dashboard_cache,
                            content_type="text/html")

    async def _setup_health_server(self):
        # Called from initialize() so the desk is observable while it boots,
        # and still called from start() for any caller that only runs the
        # latter. Binding twice would raise on the second attempt, so the
        # already-bound case is the no-op it should be.
        if self._web_runner is not None:
            return
        app = web.Application()
        app.router.add_get("/health", self._health_endpoint)
        app.router.add_get("/metrics", self._metrics_endpoint)
        app.router.add_get("/status", self._status_endpoint)
        # Force the evidence ledgers to disk. The supervisor calls this when
        # persistence goes stale, because the alternative -- restarting -- is
        # the one action that would LOSE the very data the check is worried
        # about. Loopback only, like everything else here.
        app.router.add_post("/flush", self._flush_endpoint)
        app.router.add_post("/release-sources", self._release_sources_endpoint)
        app.router.add_post("/verify-channels", self._verify_channels_endpoint)
        # The desk's own terminal. Same origin as /status, so it polls the
        # live desk directly instead of asking anyone to paste JSON around.
        app.router.add_get("/", self._dashboard_endpoint)
        app.router.add_get("/dashboard", self._dashboard_endpoint)
        self._web_runner = web.AppRunner(app)
        await self._web_runner.setup()
        # Loopback by default. /status serves the desk's whole interior --
        # open positions, watched wallets, model reports, the wallet the
        # keypair belongs to -- and binding that to every interface publishes
        # it to whatever else can reach the box. An operator who wants it
        # remote sets HEALTH_HOST deliberately and puts something in front.
        host = os.getenv("HEALTH_HOST", "127.0.0.1")
        await web.TCPSite(self._web_runner, host,
                          int(os.getenv("HEALTH_PORT", "8080"))).start()
        logger.info("health server on %s:%s", host, os.getenv("HEALTH_PORT", "8080"))

    async def _health_endpoint(self, request):
        # The port is bound before the subsystems exist, so a probe can
        # arrive during startup. Answering it with an exception from a
        # half-built desk is worse than answering it honestly.
        starting = getattr(self, "_starting_phase", "")
        if starting:
            return web.json_response({
                "status": "starting", "phase": starting,
                "uptime_seconds": time.time() - self.start_time})
        return web.json_response({"status": "healthy" if self._running else "stopping", "dry_run": self.dry_run,
                                  "uptime_seconds": time.time() - self.start_time,
                                  "live_submission_locked": os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand"})

    async def _metrics_endpoint(self, request):
        return web.json_response(_jsonable({"portfolio": self.elogw_engine.get_portfolio_state(),
                                            "total_pnl": self.total_pnl, "trade_count": self.trade_count,
                                            "successful_exits": self.successful_exits}))

    async def _status_endpoint(self, request):
        starting = getattr(self, "_starting_phase", "")
        if starting:
            # 503, not 200: a caller polling for readiness must be able to
            # tell "not finished booting" from "booted, and this is the
            # state", and a 200 with a partial body cannot say that.
            return web.json_response(
                {"status": "starting", "phase": starting,
                 "uptime_seconds": time.time() - self.start_time},
                status=503)
        return web.json_response(_jsonable(self.readiness()))

    async def _close_health_server(self):
        if self._web_runner:
            await self._web_runner.cleanup()
            self._web_runner = None
