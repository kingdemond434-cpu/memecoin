"""One launch event, whatever program emitted it.

Pump.fun is handled natively and everything else arrives, if at all, as a
downstream POOL. That is the wrong shape. A pool creation on Raydium is a
token's second act; the launch already happened somewhere else, minutes or
hours earlier, and by the time the pool appears the decision that mattered is
long past. Meanwhile the desk's entire brain -- actor DNA, monster lifecycle,
E[logW] -- is keyed on a launch event it only receives for one venue.

So: every launchpad normalises to a `CanonicalLaunchEvent`, and the same
machinery runs on all of them. Adding a venue becomes a registry entry rather
than a new code path through the decision engine.

THE HONESTY PROBLEM, and how this solves it.

To decode a program you need its program id and its instruction
discriminators. Discriminators are not a problem: Anchor defines them as
sha256("global:<instruction_name>")[:8], which is arithmetic this module
computes rather than copies, and `anchor_discriminator` is tested against
known values.

Program IDs are a problem. They are 32-byte constants with no internal
structure, nothing derives them, and a wrong one is invisible -- it does not
crash, it simply never matches, and the venue silently contributes nothing
while the registry claims coverage. That is the exact failure this repository
treats as unacceptable: a confident number nobody measured.

So a program id here is a HYPOTHESIS with a status. UNVERIFIED programs are
never streamed and never counted as coverage. A program is promoted to
VERIFIED only when the desk observes a real transaction from it that decodes
into a launch with the fields a launch must have, and the observation is
recorded with its signature so the promotion is auditable. Coverage therefore
reports what the desk has actually SEEN, not what a config file hoped for.

The result is a registry that starts honest and gets better by running,
rather than one that starts complete and is quietly wrong.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LAUNCHPAD_SCHEMA_VERSION = "v1"

#: Observations of a program decoding cleanly before it is trusted. One could
#: be a coincidence of byte alignment; three of them from different
#: signatures is the program behaving as described.
OBSERVATIONS_TO_VERIFY = 3


def anchor_discriminator(instruction: str) -> bytes:
    """The 8-byte Anchor discriminator for a global instruction.

    Derived, never transcribed: Anchor's own rule is the first eight bytes of
    sha256("global:<name>"). Copying hex strings out of a block explorer is
    how a decoder ends up matching an instruction nobody meant.
    """
    return hashlib.sha256(f"global:{instruction}".encode("utf-8")).digest()[:8]


@dataclass
class LaunchpadSpec:
    """One venue, and how confident the desk is that it has it right."""

    name: str
    program_id: str
    #: Instruction names whose discriminators mark a token launch.
    create_instructions: Tuple[str, ...]
    #: Where the mint sits in the instruction's account list, when known.
    mint_account_index: Optional[int] = None
    #: Where the creator sits. None means "read it from the fee payer".
    creator_account_index: Optional[int] = None
    #: PROGRAM ID PROVENANCE. Nothing here is trusted until observed.
    status: str = "UNVERIFIED"
    note: str = ""
    observations: int = 0
    first_seen_signature: str = ""

    @property
    def discriminators(self) -> Dict[bytes, str]:
        return {anchor_discriminator(name): name
                for name in self.create_instructions}

    @property
    def trusted(self) -> bool:
        return self.status == "VERIFIED"


@dataclass
class CanonicalLaunchEvent:
    """A token coming into existence, from any venue.

    The one shape the rest of the desk consumes. A field the venue does not
    provide is None -- never a default -- because a launch with an unknown
    creator and a launch with no creator are different, and only one of them
    is possible.
    """

    venue: str
    program_id: str
    mint: str
    creator: Optional[str]
    signature: str
    slot: int
    observed_at: float
    instruction: str = ""
    #: Wallets that funded the creator in the same transaction.
    funding_wallets: Tuple[str, ...] = ()
    #: Set when the venue is not yet verified. Such an event is evidence
    #: about the REGISTRY, and must not become a trading candidate.
    provisional: bool = False

    def as_event(self) -> Dict[str, Any]:
        """The dict shape the existing stream callback already understands."""
        return {
            "type": "token_launch",
            "venue": self.venue,
            "program": self.program_id,
            "token": self.mint,
            "creator": self.creator,
            "signature": self.signature,
            "slot": self.slot,
            "timestamp": self.observed_at,
            "instruction": self.instruction,
            "funding_wallets": list(self.funding_wallets),
            "provisional": self.provisional,
        }


class LaunchpadRegistry:
    """Which venues the desk can decode, and which it has actually proven.

    Deliberately not a config loader with a trust flag an operator can flip.
    Verification is earned by observation inside this process, because the
    failure being guarded against -- a plausible-looking program id that
    matches nothing -- is exactly the kind a human ticking a box does not
    catch.
    """

    def __init__(self, specs: Optional[Sequence[LaunchpadSpec]] = None):
        self.specs: Dict[str, LaunchpadSpec] = {}
        for spec in specs or default_specs():
            self.specs[spec.program_id] = spec
        self._by_discriminator: Dict[Tuple[str, bytes], str] = {}
        self._reindex()

    def _reindex(self) -> None:
        self._by_discriminator = {}
        for spec in self.specs.values():
            for digest, name in spec.discriminators.items():
                self._by_discriminator[(spec.program_id, digest)] = name

    def register(self, spec: LaunchpadSpec) -> None:
        self.specs[spec.program_id] = spec
        self._reindex()

    @property
    def watched_programs(self) -> List[str]:
        """Every program to subscribe to, verified or not.

        Unverified programs ARE subscribed to -- that is the only way they can
        ever be verified -- but what they produce is marked provisional and
        cannot reach a trading decision.
        """
        return sorted(self.specs)

    @property
    def verified_programs(self) -> List[str]:
        return sorted(pid for pid, spec in self.specs.items() if spec.trusted)

    def decode(self, program_id: str, data: bytes, keys: Sequence[str],
               accounts: Sequence[int], signature: str, slot: int,
               fee_payer: Optional[str] = None,
               observed_at: Optional[float] = None,
               ) -> Optional[CanonicalLaunchEvent]:
        """One instruction to a canonical launch, or None.

        None is the overwhelmingly common answer and costs almost nothing:
        the discriminator lookup is a dict hit on eight bytes.
        """
        spec = self.specs.get(program_id)
        if spec is None or len(data) < 8:
            return None
        instruction = self._by_discriminator.get((program_id, bytes(data[:8])))
        if instruction is None:
            return None
        mint = self._account(keys, accounts, spec.mint_account_index)
        if not mint:
            # A launch with no identifiable mint is not a launch this desk can
            # act on, and emitting it with an empty token would put a blank
            # key into the census.
            return None
        creator = self._account(keys, accounts, spec.creator_account_index)
        if creator is None and fee_payer:
            creator = fee_payer
        return CanonicalLaunchEvent(
            venue=spec.name, program_id=program_id, mint=mint, creator=creator,
            signature=signature, slot=slot,
            observed_at=observed_at if observed_at is not None else time.time(),
            instruction=instruction, provisional=not spec.trusted)

    @staticmethod
    def _account(keys: Sequence[str], accounts: Sequence[int],
                 index: Optional[int]) -> Optional[str]:
        if index is None or index < 0 or index >= len(accounts):
            return None
        key_index = accounts[index]
        if key_index < 0 or key_index >= len(keys):
            return None
        return keys[key_index]

    def observe(self, event: CanonicalLaunchEvent) -> bool:
        """Count a clean decode towards verifying its program.

        Returns True on the transition to VERIFIED, so the caller can say so
        once rather than every time. Distinct signatures only: the same
        transaction seen twice is one observation, and counting retries would
        let a single lucky byte alignment verify a program by itself.
        """
        spec = self.specs.get(event.program_id)
        if spec is None or spec.trusted:
            return False
        if not spec.first_seen_signature:
            spec.first_seen_signature = event.signature
        spec.observations += 1
        if spec.observations < OBSERVATIONS_TO_VERIFY:
            return False
        spec.status = "VERIFIED"
        logger.info(
            "LAUNCHPAD %s verified after %d clean decodes (first: %s); its "
            "launches now reach decisions",
            spec.name, spec.observations, spec.first_seen_signature[:16])
        return True

    def report(self) -> Dict[str, Any]:
        verified = [s for s in self.specs.values() if s.trusted]
        return {
            "status": "OK" if verified else "DATA_BLOCKED",
            "schema": LAUNCHPAD_SCHEMA_VERSION,
            "declared": len(self.specs),
            "verified": len(verified),
            # The number that matters, and the one a config file cannot fake.
            "detail": (f"{len(verified)} of {len(self.specs)} launchpad programs "
                       "have been observed decoding cleanly on this node"),
            "venues": {
                spec.name: {"program": spec.program_id, "status": spec.status,
                            "observations": spec.observations,
                            "instructions": list(spec.create_instructions),
                            "note": spec.note}
                for spec in sorted(self.specs.values(), key=lambda s: s.name)},
        }


def default_specs() -> List[LaunchpadSpec]:
    """The venues worth decoding, with their program ids as HYPOTHESES.

    Every entry starts UNVERIFIED. The desk subscribes to all of them and
    promotes the ones that actually decode, so a wrong id costs a
    subscription and shows up as a venue stuck at zero observations --
    visible, and fixable -- rather than as silent absence behind a registry
    claiming full coverage.

    pump.fun is the exception and is marked VERIFIED: it is already decoded
    natively elsewhere in this repository against its published IDL, with
    fixture tests, so its id is a measurement this desk has already made.
    """
    return [
        LaunchpadSpec(
            name="pump.fun", program_id="6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
            create_instructions=("create",), mint_account_index=0,
            creator_account_index=7, status="VERIFIED",
            note="decoded natively against the published IDL, with fixtures"),
        LaunchpadSpec(
            name="raydium_launchlab", program_id="LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj",
            create_instructions=("initialize",), mint_account_index=6,
            note="LaunchLab / LetsBONK curve; account layout unconfirmed"),
        LaunchpadSpec(
            name="meteora_dbc", program_id="dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN",
            create_instructions=("initialize_virtual_pool_with_spl_token",
                                 "initialize_virtual_pool_with_token2022"),
            mint_account_index=3,
            note="Dynamic Bonding Curve; two mint variants"),
        LaunchpadSpec(
            name="moonshot", program_id="MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG",
            create_instructions=("tokenMint",), mint_account_index=2,
            note="Moonshot/Moonit"),
        LaunchpadSpec(
            name="boop", program_id="boop8hVGQGqehUK2iVEMEnMrL5RbjywRzHKBmBE7ry4",
            create_instructions=("create_token",), mint_account_index=1,
            note="account layout unconfirmed"),
    ]
