"""The safety view a decision can have at T0, which is the one that costs nothing.

`RugDetector.analyze` is three to five sequential JSON-RPC round trips --
`getAccountInfo` on the mint, `getTokenLargestAccounts`, owner enrichment,
a sell-route probe -- and it sat directly in front of the T0 decision, with
`await` in the middle of the hot path. On a remote endpoint that is one to
five hundred milliseconds during which the launch this desk is trying to
snipe is being bought by everyone who did not wait. The desk is optimising
signer IPC in microseconds and native decode in nanoseconds behind a
blocking network call worth six orders of magnitude more than all of it.

The obvious fix -- skip the checks -- is the wrong one. Unmeasured safety is
not safe, and the screen already knows that: a DATA_BLOCKED report degrades
the position to 35% of size rather than vetoing it. So a T0 view that
reported everything blocked would be honest and would also throw away most
of the size on most launches, which is a real cost paid every time.

What makes this tractable is that the interesting questions at T0 are not
about the TOKEN. They are about the PROGRAM that created it. Whether a
pump.fun mint has a live mint authority is not a property the deployer
chose; it is a property of the instruction pump.fun ran. That is a claim
about a program, it is the same claim for every launch, and it is checkable
-- so it is checked, continuously, against the full reports that land
moments later, and it is used at T0 only for as long as it keeps holding.

    observed at T0, free      the create instruction's own program, the
                              streamed curve, the launch transaction's
                              accounts, the deployer's history
    claimed from invariant    only while the ledger has seen enough full
                              reports agree and NONE disagree
    everything else           DATA_BLOCKED, which the screen prices

One violation withdraws a claim permanently for that program. This is the
same bargain the curve invariant makes (`pump_curve_invariant_holds`) and
the same one the native ingress makes: an assumption is allowed to be load
bearing exactly as long as it is being measured, and the measurement is the
thing that has to keep running.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

T0_RISK_SCHEMA_VERSION = "v1"

#: How many agreeing full reports a program-level claim needs before the T0
#: view will lean on it. Not a round number for its own sake: at roughly one
#: launch a second on pump.fun this is a few minutes of evidence, and it is
#: large enough that a claim surviving it is not surviving by luck.
MIN_OBSERVATIONS = 200

#: Violations tolerated. Zero, deliberately. These are claims about what a
#: program DOES, not statistical regularities; one counterexample means the
#: claim was wrong, or the program changed, and either way it stops being
#: usable. A rate-based threshold would let a program upgrade that starts
#: leaving mint authority live look like noise for hours.
MAX_VIOLATIONS = 0

#: The claims a launch program can earn. Each maps to a field of the full
#: report, so the ledger is always comparing like with like.
INVARIANT_MINT_AUTHORITY = "mint_authority_absent"
INVARIANT_FREEZE_AUTHORITY = "freeze_authority_absent"
INVARIANT_TOKEN_PROGRAM = "token_program_standard"

INVARIANTS = (INVARIANT_MINT_AUTHORITY, INVARIANT_FREEZE_AUTHORITY,
              INVARIANT_TOKEN_PROGRAM)


@dataclass
class InvariantState:
    observations: int = 0
    violations: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    #: Set the first time a violation is seen and never cleared. A claim that
    #: has once been false does not become true again because the next
    #: thousand launches agreed -- the program's behaviour is what changed,
    #: and the desk has to be told, not quietly re-reassured.
    withdrawn_at: float = 0.0
    withdrawn_example: str = ""

    @property
    def holds(self) -> bool:
        return (not self.withdrawn_at
                and self.violations <= MAX_VIOLATIONS
                and self.observations >= MIN_OBSERVATIONS)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "violations": self.violations,
            "holds": self.holds,
            "withdrawn_at": self.withdrawn_at or None,
            "withdrawn_example": self.withdrawn_example or None,
            "first_seen": self.first_seen or None,
            "last_seen": self.last_seen or None,
        }


class LaunchInvariantLedger:
    """What each launch program has been OBSERVED to guarantee.

    Fed by the full risk reports that arrive after the decision, so the
    evidence is free: the desk was going to compute those reports anyway,
    and this only reads them. Persisted, because two hundred observations
    thrown away on every restart is two hundred observations that never
    accumulate -- the same failure that kept the follow ledger empty.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._state: Dict[Tuple[str, str], InvariantState] = {}
        if self.path:
            self.load()

    def _slot(self, program: str, invariant: str) -> InvariantState:
        key = (str(program or "unknown"), str(invariant))
        state = self._state.get(key)
        if state is None:
            state = InvariantState()
            self._state[key] = state
        return state

    def observe(self, program: str, invariant: str, held: bool,
                example: str = "") -> None:
        """One full report's verdict on one claim about one program."""
        state = self._slot(program, invariant)
        now = time.time()
        state.first_seen = state.first_seen or now
        state.last_seen = now
        state.observations += 1
        if held:
            return
        state.violations += 1
        if not state.withdrawn_at:
            state.withdrawn_at = now
            state.withdrawn_example = str(example or "")
            logger.error(
                "LAUNCH INVARIANT WITHDRAWN: %s no longer guarantees %s "
                "(first counterexample %s after %d agreeing observations). "
                "T0 decisions on this program now treat it as unmeasured.",
                program, invariant, example or "unnamed", state.observations - 1)

    def observe_report(self, program: str, report: Any) -> int:
        """Read every claim this report can settle. Returns how many it settled.

        A report that could not measure something settles nothing about it:
        a blocked check is not a counterexample, and counting it as one would
        withdraw every claim the first time an RPC endpoint rate-limited us.
        """
        if report is None or not program:
            return 0
        blocked = set(getattr(report, "blocked_checks", ()) or ())
        checks = getattr(report, "checks", {}) or {}
        mint = checks.get("mint") if isinstance(checks, dict) else None
        if not isinstance(mint, dict) or "mint_account" in blocked:
            return 0
        token = str(getattr(report, "token_address", "") or "")
        settled = 0
        if "mint_authority_present" in mint:
            self.observe(program, INVARIANT_MINT_AUTHORITY,
                         not bool(mint["mint_authority_present"]), token)
            settled += 1
        if "freeze_authority_present" in mint:
            self.observe(program, INVARIANT_FREEZE_AUTHORITY,
                         not bool(mint["freeze_authority_present"]), token)
            settled += 1
        token_program = str(getattr(report, "token_program", "") or "")
        if token_program:
            self.observe(program, INVARIANT_TOKEN_PROGRAM,
                         not (getattr(report, "token_extensions", None) or []),
                         token)
            settled += 1
        return settled

    def holds(self, program: str, invariant: str) -> bool:
        return self._slot(program, invariant).holds

    def state(self, program: str, invariant: str) -> InvariantState:
        return self._slot(program, invariant)

    def report(self) -> Dict[str, Any]:
        programs: Dict[str, Dict[str, Any]] = {}
        for (program, invariant), state in self._state.items():
            programs.setdefault(program, {})[invariant] = state.as_dict()
        return {
            "schema": T0_RISK_SCHEMA_VERSION,
            "min_observations": MIN_OBSERVATIONS,
            "programs": programs,
            "detail": ("a claim is used at T0 only while it has this many "
                       "agreeing full reports and no counterexample; one "
                       "counterexample withdraws it permanently"),
        }

    # --- persistence -----------------------------------------------------

    def save(self) -> bool:
        if not self.path:
            return False
        payload = {
            "schema": T0_RISK_SCHEMA_VERSION,
            "entries": [
                {"program": program, "invariant": invariant,
                 "observations": state.observations,
                 "violations": state.violations,
                 "first_seen": state.first_seen, "last_seen": state.last_seen,
                 "withdrawn_at": state.withdrawn_at,
                 "withdrawn_example": state.withdrawn_example}
                for (program, invariant), state in self._state.items()],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self.path)
            return True
        except OSError as exc:  # pragma: no cover - disk only
            logger.warning("launch invariant ledger not saved: %s", exc)
            return False

    def load(self) -> int:
        if not self.path or not self.path.exists():
            return 0
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("launch invariant ledger not loaded: %s", exc)
            return 0
        loaded = 0
        for row in payload.get("entries", []) or []:
            state = self._slot(row.get("program", ""), row.get("invariant", ""))
            state.observations = int(row.get("observations", 0) or 0)
            state.violations = int(row.get("violations", 0) or 0)
            state.first_seen = float(row.get("first_seen", 0.0) or 0.0)
            state.last_seen = float(row.get("last_seen", 0.0) or 0.0)
            state.withdrawn_at = float(row.get("withdrawn_at", 0.0) or 0.0)
            state.withdrawn_example = str(row.get("withdrawn_example", "") or "")
            loaded += 1
        return loaded


@dataclass
class T0Risk:
    """A risk view built without touching the network.

    Shaped to be read exactly like the full report the screen already
    consumes -- same `data_status`, same `blocked_checks`, same
    `risk_level` -- so nothing downstream has to learn a second vocabulary
    and no code path can silently treat a T0 view as a completed audit.
    """

    token_address: str
    chain: str = "solana"
    program: str = ""
    risk_level: Any = None
    score: float = 0.0
    checks: Dict[str, Any] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    can_mint: bool = False
    can_freeze: bool = False
    token_program: str = ""
    token_extensions: list = field(default_factory=list)
    sell_route_feasible: Optional[bool] = None
    holder_count: Optional[int] = None
    top_10_pct: float = 0.0
    top_20_pct: Optional[float] = None
    deployer_balance_pct: Optional[float] = None
    insider_pct: Optional[float] = None
    bundler_pct: Optional[float] = None
    fresh_wallet_pct: Optional[float] = None
    whale_pct: Optional[float] = None
    connected_cluster_pct: Optional[float] = None
    liquidity_usd: Optional[float] = None
    liquidity_locked: bool = False
    ownership_renounced: bool = False
    buy_tax: float = 0.0
    sell_tax: float = 0.0
    transfer_tax: float = 0.0
    is_proxy: bool = False
    max_tx_limit: Optional[int] = None
    max_wallet_limit: Optional[int] = None
    extension_risk: float = 0.0
    data_status: str = "DATA_BLOCKED"
    blocked_checks: list = field(default_factory=list)
    #: True when this is the local view rather than the completed audit. The
    #: point of naming it is that nothing downstream should ever have to
    #: guess, and a field that exists cannot be forgotten the way an
    #: undocumented convention can.
    provisional: bool = True


class T0RiskView:
    """Builds the local view. No I/O, no awaits, no allocation worth naming."""

    def __init__(self, ledger: LaunchInvariantLedger,
                 curve_state_provider: Any = None,
                 risk_level_enum: Any = None):
        self.ledger = ledger
        self.curve_state_provider = curve_state_provider
        self._risk_level = risk_level_enum
        self.built = 0
        self.claims_used = 0

    def _level(self, name: str) -> Any:
        if self._risk_level is None:
            return name
        try:
            return self._risk_level(name)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return name

    def assess(self, token: str, program: str, *,
               chain: str = "solana",
               deployer: str = "",
               launch_metadata: Optional[Dict[str, Any]] = None) -> T0Risk:
        """The safety view available for free, right now."""
        self.built += 1
        checks: Dict[str, Any] = {}
        warnings: list = []
        blocked: list = []
        can_mint = True
        can_freeze = True
        extensions_known = False

        for invariant, field_name in (
                (INVARIANT_MINT_AUTHORITY, "mint_authority"),
                (INVARIANT_FREEZE_AUTHORITY, "freeze_authority"),
                (INVARIANT_TOKEN_PROGRAM, "token_program")):
            state = self.ledger.state(program, invariant)
            if state.holds:
                self.claims_used += 1
                checks[field_name] = {
                    "status": "OK", "source": "program_invariant",
                    "program": program,
                    "observations": state.observations,
                    "violations": state.violations,
                }
                if invariant == INVARIANT_MINT_AUTHORITY:
                    can_mint = False
                elif invariant == INVARIANT_FREEZE_AUTHORITY:
                    can_freeze = False
                else:
                    extensions_known = True
            else:
                checks[field_name] = {
                    "status": "DATA_BLOCKED",
                    "reason": ("no program invariant established: "
                               f"{state.observations} observation(s), "
                               f"{state.violations} violation(s)"
                               + (" (withdrawn)" if state.withdrawn_at else "")),
                }
                blocked.append(field_name)

        # The curve, which the stream already delivered. A token still on its
        # bonding curve has a sell route by construction: the curve is the
        # counterparty, and it cannot refuse.
        state = None
        if self.curve_state_provider is not None:
            try:
                state = self.curve_state_provider(token)
            except Exception:  # pragma: no cover - provider is the desk's
                state = None
        sell_route = None
        if state is not None and getattr(state, "virtual_sol_reserves", 0) > 0:
            sell_route = not bool(getattr(state, "complete", False))
            checks["sell_route"] = {
                "status": "OK", "source": "streamed_curve",
                "on_curve": sell_route,
            }
        else:
            checks["sell_route"] = {
                "status": "DATA_BLOCKED",
                "reason": "no streamed curve state for this mint yet"}
            blocked.append("sell_route")

        # Everything that needs an account read stays unmeasured, by name.
        # Naming them is the point: `blocked_checks` is what the screen
        # prices and what the audit pack reads, and a silently short list
        # would understate how little this view actually knows.
        for name in ("holders", "top_holders", "insider_concentration",
                     "deployer_balance"):
            checks[name] = {"status": "DATA_BLOCKED",
                            "reason": "needs an account read; not done at T0"}
            blocked.append(name)

        if can_mint:
            warnings.append("mint authority unverified at T0")
        if can_freeze:
            warnings.append("freeze authority unverified at T0")

        # Score is deliberately NOT a number invented here. The screen reads
        # `data_status` and `risk_level`; a fabricated score would flow into
        # the dataset as though it had been measured.
        return T0Risk(
            token_address=token, chain=chain, program=program,
            risk_level=self._level("medium"), score=0.0,
            checks=checks, warnings=warnings,
            can_mint=can_mint, can_freeze=can_freeze,
            token_extensions=[] if extensions_known else [],
            sell_route_feasible=sell_route,
            data_status="OK" if not blocked else "DATA_BLOCKED",
            blocked_checks=blocked,
            provisional=True,
        )

    def report(self) -> Dict[str, Any]:
        return {
            "schema": T0_RISK_SCHEMA_VERSION,
            "views_built": self.built,
            "invariant_claims_used": self.claims_used,
            "claims_per_view": (round(self.claims_used / self.built, 2)
                                if self.built else None),
            "invariants": self.ledger.report(),
            "detail": ("the safety view a decision can have without a network "
                       "round trip; the full report lands afterwards and "
                       "drives a redecision"),
        }
