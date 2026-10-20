"""Discover ecosystem/protocol identities before Solana-only market filters.

The scout reuses the shared HttpClient and catalogue/substitution interfaces.
It starts no background work. Raw public API responses are acknowledged by the
existing canonical evidence sink before bounded extraction. Aggregator labels
are discovery observations, never native-chain, factory-address or execution
verification. All chain opportunity metrics remain unknown until measured.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import quote

from src.research.frontier import CHAIN_METRICS
from src.research.source_catalogue import NEW_POOLS
from src.research.source_substitution import Endpoint, SubstitutionRegistry


API_DOCUMENTATION = "https://docs.dexscreener.com/api/reference"
PAIR_ENDPOINT = Endpoint(
    "dexscreener_chain_token_pairs",
    "https://api.dexscreener.com/token-pairs/v1/{chain_id}/{token_address}",
    shape="dexscreener_pairs", detail="official chain-qualified token pair lookup")
_PROFILES_DOMAIN = "unknown_opportunity_profiles"
_PAIRS_DOMAIN = "unknown_opportunity_chain_pairs"


class ChainOpportunityScout:
    """One bounded pass; ``sink(record)`` returns a canonical evidence ref.

    Records have kind source_response, chain or protocol. Only source_response
    contains raw_body. Derived rows refer to it and its row index; no new raw
    evidence store or private transport is created. A raw over-budget response
    remains available for later conversion through its returned canonical ref.
    """

    def __init__(self, http: Any, sink: Callable[[Dict[str, Any]], Any], *,
                 registry: Optional[SubstitutionRegistry] = None,
                 max_rows: int = 256, max_pair_requests: int = 4,
                 max_body_bytes: int = 2_000_000, min_interval_seconds: float = 60):
        if not 1 <= max_rows <= 10000 or not 0 <= max_pair_requests <= 16:
            raise ValueError("scout row/request budgets must be bounded")
        if not 1024 <= max_body_bytes <= 20_000_000 or min_interval_seconds < 60:
            raise ValueError("scout body/cadence budget is outside the public API envelope")
        self.http, self.sink = http, sink
        self.registry = registry or SubstitutionRegistry()
        self.max_rows, self.max_pair_requests = int(max_rows), int(max_pair_requests)
        self.max_body_bytes = int(max_body_bytes)
        self.min_interval_seconds = float(min_interval_seconds)
        profiles = [endpoint for endpoint in NEW_POOLS
                    if endpoint.shape == "dexscreener_profiles"]
        self.registry.declare(_PROFILES_DOMAIN, profiles)
        self.registry.declare(_PAIRS_DOMAIN, [PAIR_ENDPOINT])
        self._next_due = 0.0
        self._last: Dict[str, Any] = {"status": "NOT_RUN", "chains": [], "protocols": []}

    def report(self) -> Dict[str, Any]:
        return {**self._last, "observation_only": True, "execution_eligible": False,
                "api_documentation": API_DOCUMENTATION}

    async def _publish(self, record: Dict[str, Any]) -> str:
        result = self.sink(record)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str) or not result.strip():
            raise ValueError("canonical scout sink did not acknowledge durable evidence")
        return result

    async def _response(self, domain: str, report: Dict[str, Any], **params: str):
        endpoint = self.registry.current(domain)
        if endpoint is None:
            report["blocked"].append({"domain": domain, "reason": "no eligible public endpoint"})
            return None
        url = endpoint.format(**params)
        try:
            status, body, headers = await self.http.get(url)
        except Exception as exc:
            self.registry.note_failure(domain, endpoint.name, type(exc).__name__)
            report["blocked"].append({"domain": domain, "reason": type(exc).__name__})
            return None
        received_at = time.time()
        report["http_requests"] += 1
        if not isinstance(body, str):
            report["blocked"].append({"domain": domain, "reason": "nontext public response"})
            return None
        # Preserve original bytes/text before any parser or per-pass row bound.
        # The shared HttpClient already materializes its response; the budget
        # bounds JSON parsing/extraction rather than pretending to stream it.
        reference = await self._publish({
            "kind": "source_response", "identity": hashlib.sha256(
                (url + "\n" + body).encode()).hexdigest(), "source_url": url,
            "source_at": None, "observed_at": received_at, "http_status": int(status),
            "response_shape": endpoint.shape,
            "documentation_ref": API_DOCUMENTATION, "raw_body": body,
            "source_class": "OFFICIAL_PUBLIC_API", "execution_eligible": False,
            "selection_bias": "latest profiles/promotions and their pairs; not a chain population census",
        })
        report["source_response_refs"].append(reference)
        if status == 429:
            try:
                retry = min(3600, max(60, float(headers.get("Retry-After", 60))))
            except (ValueError, TypeError):
                retry = 60
            self._next_due = max(self._next_due, received_at + retry)
            report["blocked"].append({"domain": domain, "reason": "RATE_LIMITED", "evidence_ref": reference})
            return None
        if status >= 400:
            self.registry.note_failure(domain, endpoint.name, f"HTTP {int(status)}")
            report["blocked"].append({"domain": domain, "reason": f"HTTP {int(status)}", "evidence_ref": reference})
            return None
        if len(body.encode("utf-8")) > self.max_body_bytes:
            report["deferred"].append({"evidence_ref": reference, "reason": "NOT_THIS_PASS_BODY_BUDGET"})
            return None
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            report["blocked"].append({"domain": domain, "reason": "INVALID_JSON", "evidence_ref": reference})
            return None
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
            rows = payload["pairs"]
        elif isinstance(payload, dict) and payload.get("chainId"):
            rows = [payload]
        else:
            report["blocked"].append({"domain": domain, "reason": "UNKNOWN_RESPONSE_SCHEMA", "evidence_ref": reference})
            return None
        self.registry.note_success(domain, endpoint.name)
        if len(rows) > self.max_rows:
            report["deferred"].append({"evidence_ref": reference, "reason": "NOT_THIS_PASS_ROW_BUDGET",
                                       "rows": len(rows)})
            return None
        return rows, reference, received_at

    @staticmethod
    def _identity(value: Any) -> Optional[str]:
        return value.strip() if isinstance(value, str) and 0 < len(value.strip()) <= 256 else None

    async def _identities(self, rows, reference, received_at, chains, protocols, report):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                report["invalid_rows"] += 1
                continue
            chain = self._identity(row.get("chainId"))
            if chain is None:
                report["invalid_rows"] += 1
                continue
            common = {"chain_id": chain, "chain_family": None, "source_response_ref": reference,
                      "source_row_index": index, "source_at": None, "observed_at": received_at,
                      "documentation_ref": API_DOCUMENTATION, "execution_eligible": False,
                      "identity_status": "AGGREGATOR_LABEL_NATIVE_ID_UNVERIFIED"}
            if chain not in chains:
                await self._publish({"kind": "chain", "identity": chain, **common,
                                     "metrics": dict.fromkeys(CHAIN_METRICS)})
                chains.add(chain)
            protocol = self._identity(row.get("dexId"))
            if protocol is not None and (chain, protocol) not in protocols:
                await self._publish({"kind": "protocol", "identity": protocol, **common,
                                     "version": "UNKNOWN", "contract": {
                                         "status": "DISCOVERED", "addresses": None,
                                         "verified_source": None, "abi_ref": None, "idl_ref": None,
                                         "execution_support": None}})
                protocols.add((chain, protocol))

    async def run_once(self) -> Dict[str, Any]:
        at = time.time()
        if at < self._next_due:
            return {**self.report(), "pass_status": "NOT_THIS_PASS_CADENCE", "next_due": self._next_due}
        self._next_due = at + self.min_interval_seconds
        report: Dict[str, Any] = {
            "status": "OBSERVED", "observed_at": at, "chains": [], "protocols": [],
            "http_requests": 0, "source_response_refs": [], "blocked": [], "deferred": [], "invalid_rows": 0,
        }
        chains, protocols = set(), set()
        try:
            profiles = await self._response(_PROFILES_DOMAIN, report)
            if profiles is not None:
                rows, reference, received_at = profiles
                # All identities are observed before pair budget or Solana
                # filtering. Native families/IDs are never inferred from names.
                await self._identities(rows, reference, received_at, chains, protocols, report)
                targets = []
                seen = set()
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    chain, token = self._identity(row.get("chainId")), self._identity(row.get("tokenAddress"))
                    if chain and token and (chain, token) not in seen:
                        targets.append((chain, token))
                        seen.add((chain, token))
                # Spend the small pair budget across observed ecosystems
                # before revisiting a dominant chain's next promoted token.
                # This is coverage ordering, not popularity/return ranking.
                groups: Dict[str, list] = {}
                for chain, token in targets:
                    groups.setdefault(chain, []).append((chain, token))
                planned = []
                while any(groups.values()):
                    for group in groups.values():
                        if group:
                            planned.append(group.pop(0))
                for chain, token in planned[:self.max_pair_requests]:
                    pairs = await self._response(_PAIRS_DOMAIN, report,
                                                 chain_id=quote(chain, safe=""), token_address=quote(token, safe=""))
                    if pairs is not None:
                        await self._identities(*pairs, chains, protocols, report)
                    elif report["blocked"] and report["blocked"][-1].get("reason") == "RATE_LIMITED":
                        break
                if len(targets) > self.max_pair_requests:
                    report["deferred"].append({"evidence_ref": reference,
                                               "reason": "NOT_THIS_PASS_PAIR_REQUEST_BUDGET",
                                               "remaining_targets": len(targets) - self.max_pair_requests})
        except Exception as exc:
            # Sink failure is not provider failure. Keep raw references and
            # avoid leaking credential-bearing exception text in reports.
            report["blocked"].append({"reason": "CANONICAL_SINK_OR_LOCAL_FAILURE", "type": type(exc).__name__})
        report["chains"] = sorted(chains)
        report["protocols"] = [{"chain_id": chain, "protocol_id": protocol} for chain, protocol in sorted(protocols)]
        if report["blocked"]:
            report["status"] = "PARTIAL" if chains or protocols else "DATA_BLOCKED"
        elif report["deferred"]:
            report["status"] = "OBSERVED_WITH_PERSISTED_DEBT"
        self._last = report
        return self.report()
