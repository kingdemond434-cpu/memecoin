"""Finding launchpads by watching, rather than by being told about them.

The registry declares five venues. New ones appear continuously -- Bags,
Believe, Jupiter Studio and whatever launches next month -- and the obvious
fix is to add their program ids to the list.

I will not do that from memory, and the reason is the same one this
codebase applies everywhere else. A program id is an opaque 32-byte
constant: unlike an Anchor discriminator, which is `sha256("global:<name>")`
and therefore checkable arithmetic, there is nothing about a transcribed
base58 string that can be verified except by watching it. Getting one
character wrong produces a venue that silently never fires, and a registry
entry that never fires is indistinguishable from a venue that has no
launches -- which is precisely the failure the UNVERIFIED status was
invented to prevent, reintroduced by the act of trying to fix it.

So the desk finds them instead. `_note_launch_venue` used to see a launch
from a program it did not recognise and return, silently -- the desk was
being shown new venues and throwing them away. Every such launch is now an
observation, and a program that keeps producing launches becomes a
CANDIDATE with a program id that is a MEASUREMENT: this node watched that
address create these mints in these transactions, and can name them.

A candidate is never automatically tradeable. It is evidence that the
registry is incomplete, addressed to an operator, carrying exactly what
they need to confirm it -- the id, the discriminators seen, the distinct
mints, and a signature to look up. Promotion into the registry stays a
human decision, because the failure mode of getting this wrong
automatically is trading a program nobody has read.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

LAUNCHPAD_DISCOVERY_SCHEMA_VERSION = "v1"

#: Distinct mints a program must be seen creating before it is proposed.
#: Distinct, not events: one transaction redelivered by three racing feeds
#: is one observation, the same rule verification uses.
MIN_MINTS_TO_PROPOSE = 25

#: How many programs to track. The chain has thousands and almost none of
#: them are launchpads; the ones that matter produce mints repeatedly, so
#: the tail of single-sighting programs is dropped rather than stored.
MAX_TRACKED = 512

#: How many example signatures to keep per program. Enough for an operator
#: to look the venue up and read the instruction; not a log.
EXAMPLES = 5


@dataclass
class UnknownVenue:
    """A program this desk has watched create mints, and cannot name."""

    program_id: str
    mints: Set[str] = field(default_factory=set)
    discriminators: Set[str] = field(default_factory=set)
    instructions: Set[str] = field(default_factory=set)
    signatures: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    #: Set once an operator has been told. Stops the log repeating hourly
    #: for a venue whose proposal is already sitting in front of someone.
    announced: bool = False

    @property
    def proposable(self) -> bool:
        return len(self.mints) >= MIN_MINTS_TO_PROPOSE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "distinct_mints": len(self.mints),
            "needed": MIN_MINTS_TO_PROPOSE,
            "proposable": self.proposable,
            "instructions": sorted(self.instructions)[:8],
            "discriminators": sorted(self.discriminators)[:8],
            "example_signatures": list(self.signatures),
            "first_seen": self.first_seen or None,
            "last_seen": self.last_seen or None,
        }


class LaunchpadDiscovery:
    """Counts launches from programs the registry does not know."""

    def __init__(self, known: Sequence[str] = (),
                 min_mints: int = MIN_MINTS_TO_PROPOSE,
                 max_tracked: int = MAX_TRACKED):
        self.known: Set[str] = {str(item) for item in known if item}
        self.min_mints = int(min_mints)
        self.max_tracked = int(max_tracked)
        self.venues: Dict[str, UnknownVenue] = {}
        self.observations = 0
        self.ignored_known = 0
        self.evicted = 0

    def note_known(self, program_id: str) -> None:
        """A program the registry now declares. Stops tracking it as unknown.

        Called when a venue is added, so a program that was a candidate
        yesterday does not keep appearing as one after it was adopted.
        """
        program = str(program_id or "")
        if not program:
            return
        self.known.add(program)
        if self.venues.pop(program, None) is not None:
            logger.info("LAUNCHPAD %s is now declared; no longer a candidate",
                        program)

    def observe(self, program_id: str, mint: str, *,
                signature: str = "", instruction: str = "",
                discriminator: str = "", at: Optional[float] = None) -> bool:
        """One launch from a program. Returns True on a NEW proposal.

        A launch with no mint is not evidence of a launchpad -- the whole
        claim being accumulated is "this program creates tokens", and an
        observation that cannot name the token it created supports nothing.
        """
        program = str(program_id or "")
        token = str(mint or "")
        if not program or not token:
            return False
        if program in self.known:
            self.ignored_known += 1
            return False
        self.observations += 1
        moment = float(at or time.time())
        venue = self.venues.get(program)
        if venue is None:
            venue = UnknownVenue(program_id=program, first_seen=moment)
            self.venues[program] = venue
            self._evict()
        was_proposable = venue.proposable
        venue.mints.add(token)
        venue.last_seen = moment
        if instruction:
            venue.instructions.add(str(instruction))
        if discriminator:
            venue.discriminators.add(str(discriminator))
        if signature and len(venue.signatures) < EXAMPLES:
            venue.signatures.append(str(signature))
        if venue.proposable and not was_proposable:
            logger.warning(
                "LAUNCHPAD CANDIDATE %s has created %d distinct mints and is "
                "not in the registry. Instructions seen: %s. Example: %s. "
                "It is NOT tradeable until an operator confirms it -- the id "
                "here is something this node measured, not something read "
                "off a page.",
                program, len(venue.mints),
                ", ".join(sorted(venue.instructions)) or "unnamed",
                venue.signatures[0] if venue.signatures else "none recorded")
            venue.announced = True
            return True
        return False

    def _evict(self) -> None:
        """Drop the single-sighting tail. The chain has thousands of programs.

        Evicting by mint count and then by staleness, so a program seen once
        an hour ago goes before one seen twenty times an hour ago -- the
        second is the shape a launchpad has.
        """
        while len(self.venues) > self.max_tracked:
            worst = min(self.venues.values(),
                        key=lambda venue: (len(venue.mints), venue.last_seen))
            self.venues.pop(worst.program_id, None)
            self.evicted += 1

    def proposals(self) -> List[UnknownVenue]:
        """Everything worth an operator's attention, strongest first."""
        return sorted((venue for venue in self.venues.values() if venue.proposable),
                      key=lambda venue: len(venue.mints), reverse=True)

    def report(self) -> Dict[str, Any]:
        proposals = self.proposals()
        return {
            "schema": LAUNCHPAD_DISCOVERY_SCHEMA_VERSION,
            # A proposal is a finding, so it is not "OK" -- the registry is
            # demonstrably incomplete and somebody should look.
            "status": "ATTENTION" if proposals else "OK",
            "observations": self.observations,
            "tracked_programs": len(self.venues),
            "known_programs": len(self.known),
            "evicted": self.evicted,
            "proposals": [venue.as_dict() for venue in proposals[:10]],
            "nearly_there": [
                venue.as_dict() for venue in
                sorted((v for v in self.venues.values() if not v.proposable),
                       key=lambda v: len(v.mints), reverse=True)[:5]],
            "detail": ("programs this node has WATCHED create mints and that "
                       "the registry does not declare; a candidate is never "
                       "tradeable until an operator confirms it, because the "
                       "failure mode of adopting one automatically is trading "
                       "a program nobody has read"),
        }
