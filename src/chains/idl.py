"""Instruction account lists, generated from the vendored IDLs.

The first version of the native route transcribed its account lists from the
prose tables in Pump's docs. Those tables are readable, and on three flags they
disagree with the program: they present `fee_recipient` and
`buyback_fee_recipient` as non-writable, `global_volume_accumulator` as
writable, and `sharing_config` as a Pump PDA when it is actually derived under
the Pump Fees program. A transaction built from the prose declares the wrong
mutability on two accounts and derives a fourth from the wrong program. It does
not fail in a way that points at the cause; it just fails.

So nothing here is transcribed. The IDL is what the program was compiled
against, and the account lists, their signer and writable flags, their PDA
seeds and the program each PDA is derived under are all read out of it. The
consequence is that a hand-editing mistake in an account list is not possible:
there is no hand-written account list.

Three programs are covered, each with its address published in its own IDL:

    pump       6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
    pump_amm   pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA
    pump_fees  pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ

Resolution is deliberately strict. Every account is either a PDA the IDL knows
how to derive, or a value the caller supplies by name. An account that is
neither raises rather than defaulting, because the failure mode of a silently
substituted account is a transaction that touches something nobody chose.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

IDL_SCHEMA_VERSION = "v1"
IDL_DIR = Path(__file__).resolve().parents[2] / "idl"

PUMP_IDL = "pump"
PUMP_AMM_IDL = "pump_amm"
PUMP_FEES_IDL = "pump_fees"

try:  # pragma: no cover - exercised by the availability test
    from solders.pubkey import Pubkey
    IDL_STATUS = "OK"
except ImportError as exc:  # pragma: no cover - defensive
    Pubkey = None  # type: ignore[assignment]
    IDL_STATUS = f"DATA_BLOCKED: solders unavailable ({exc})"


class IdlError(RuntimeError):
    """An account could not be resolved. Never recovered from by substitution."""


@dataclass(frozen=True)
class AccountMeta:
    pubkey: str
    is_signer: bool = False
    is_writable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"pubkey": self.pubkey, "is_signer": self.is_signer,
                "is_writable": self.is_writable}


@lru_cache(maxsize=8)
def load_idl(name: str) -> Dict[str, Any]:
    """The vendored IDL, byte-identical to the published one.

    Vendored rather than fetched: an entry path that depends on a network call
    is one that can be slow or unavailable at exactly the wrong moment, and a
    transaction must be built from the copy that was reviewed rather than from
    whatever the network returns today.
    """
    path = IDL_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def program_id(name: str) -> str:
    """The program's own published address, read from its IDL."""
    address = load_idl(name).get("address")
    if not address:
        raise IdlError(f"{name} IDL declares no program address")
    return str(address)


@lru_cache(maxsize=64)
def instruction(idl_name: str, instruction_name: str) -> Dict[str, Any]:
    for entry in load_idl(idl_name).get("instructions", ()):
        if entry.get("name") == instruction_name:
            return entry
    raise IdlError(f"{idl_name} has no instruction {instruction_name!r}")


def discriminator(idl_name: str, instruction_name: str) -> bytes:
    """The IDL's discriminator, checked against Anchor's derivation.

    Anchor computes it as `sha256("global:<name>")[:8]`, so the two must
    agree. They are compared rather than one being trusted: a mismatch means
    either the IDL is for a different program or the name is wrong, and both
    produce an instruction the program will reject.
    """
    published = bytes(instruction(idl_name, instruction_name)["discriminator"])
    derived = hashlib.sha256(f"global:{instruction_name}".encode()).digest()[:8]
    if published != derived:
        raise IdlError(
            f"{idl_name}.{instruction_name} discriminator {published.hex()} does not "
            f"match Anchor's derivation {derived.hex()}")
    return published


def account_names(idl_name: str, instruction_name: str) -> List[str]:
    return [entry["name"] for entry in instruction(idl_name, instruction_name)["accounts"]]


def _find_program_address(seeds: Sequence[bytes], program: str) -> str:
    address, _bump = Pubkey.find_program_address(list(seeds), Pubkey.from_string(program))
    return str(address)


def _seed_bytes(seed: Mapping[str, Any], supplied: Mapping[str, str],
                resolved: Mapping[str, str]) -> bytes:
    kind = seed.get("kind")
    if kind == "const":
        return bytes(seed["value"])
    if kind in ("account", "arg"):
        path = str(seed.get("path", ""))
        # `bonding_curve.creator` and friends: the IDL names a field on another
        # account, which only the caller can know. Supplied under the dotted
        # path so a caller cannot accidentally satisfy it with the account
        # itself.
        value = supplied.get(path) or resolved.get(path)
        if value is None:
            raise IdlError(f"seed {path!r} was not supplied")
        return bytes(Pubkey.from_string(value))
    raise IdlError(f"unsupported seed kind {kind!r}")


def _pda_program(pda: Mapping[str, Any], default: str,
                 supplied: Mapping[str, str], resolved: Mapping[str, str]) -> str:
    program = pda.get("program")
    if program is None:
        return default
    if program.get("kind") == "const":
        return str(Pubkey(bytes(program["value"])))
    if program.get("kind") == "account":
        path = str(program.get("path", ""))
        value = supplied.get(path) or resolved.get(path)
        if value is None:
            raise IdlError(f"PDA program account {path!r} was not supplied")
        return value
    raise IdlError(f"unsupported PDA program kind {program.get('kind')!r}")


def build_accounts(idl_name: str, instruction_name: str,
                   supplied: Mapping[str, str]) -> List[AccountMeta]:
    """Resolve every account for one instruction, returned in the IDL's order.

    Resolution is dependency-driven rather than positional, because
    declaration order and dependency order are not the same thing:
    PumpSwap declares `coin_creator_vault_ata` at position 18 and the
    `coin_creator_vault_authority` its seeds require at position 19. A
    single forward pass fails on that, and the obvious workaround -- having
    the caller derive the authority and pass it in -- puts the same
    derivation in two places, which is how the two come to disagree.

    So the passes repeat while anything new resolves, and stop the moment a
    pass adds nothing. What is still unresolved after that is genuinely
    unresolvable: either the caller has to supply it, or the seeds form a
    cycle. Both raise, because a silently substituted account touches
    something nobody chose.
    """
    if IDL_STATUS != "OK":
        raise IdlError(IDL_STATUS)
    entry = instruction(idl_name, instruction_name)
    default_program = program_id(idl_name)
    accounts = entry["accounts"]
    resolved: Dict[str, str] = {}
    pending: List[Dict[str, Any]] = []

    for account in accounts:
        name = account["name"]
        if name in supplied:
            resolved[name] = supplied[name]
        elif "address" in account:
            # The IDL pins some accounts to a fixed address. Those are not the
            # caller's to choose.
            resolved[name] = str(account["address"])
        elif "pda" in account:
            pending.append(account)
        else:
            raise IdlError(
                f"{idl_name}.{instruction_name} account {name!r} is neither a PDA nor "
                "supplied; refusing to substitute a default")

    while pending:
        progressed = False
        deferred: List[Dict[str, Any]] = []
        for account in pending:
            pda = account["pda"]
            try:
                program = _pda_program(pda, default_program, supplied, resolved)
                seeds = [_seed_bytes(seed, supplied, resolved) for seed in pda["seeds"]]
            except IdlError:
                deferred.append(account)
                continue
            resolved[account["name"]] = _find_program_address(seeds, program)
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(item["name"] for item in deferred))
            raise IdlError(
                f"{idl_name}.{instruction_name} cannot resolve {unresolved}; "
                "a seed it depends on was never supplied")
        pending = deferred

    return [AccountMeta(pubkey=resolved[account["name"]],
                        is_signer=bool(account.get("signer", False)),
                        is_writable=bool(account.get("writable", False)))
            for account in accounts]


def unresolvable(idl_name: str, instruction_name: str,
                 supplied: Mapping[str, str]) -> List[str]:
    """Account names the caller still has to supply. Cheap pre-flight."""
    missing: List[str] = []
    for account in instruction(idl_name, instruction_name)["accounts"]:
        name = account["name"]
        if name in supplied or "address" in account or "pda" in account:
            continue
        missing.append(name)
    return missing


def encode_u64_args(idl_name: str, instruction_name: str,
                    values: Sequence[int]) -> bytes:
    """`<discriminator><u64 LE>...`, with the arg count checked against the IDL.

    The count check is what stops a caller from passing two arguments to an
    instruction that takes three -- which encodes without complaint and is
    then read as garbage by the program.
    """
    args = instruction(idl_name, instruction_name).get("args", [])
    u64_args = [arg for arg in args if arg.get("type") == "u64"]
    if len(values) != len(u64_args):
        raise IdlError(
            f"{idl_name}.{instruction_name} takes {len(u64_args)} u64 args, got {len(values)}")
    data = bytearray(discriminator(idl_name, instruction_name))
    for value in values:
        if not isinstance(value, int) or value < 0 or value >= 2 ** 64:
            raise IdlError(f"argument {value!r} does not fit in u64")
        data.extend(int(value).to_bytes(8, "little"))
    return bytes(data)


@dataclass
class IdlReport:
    status: str
    programs: Dict[str, str] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": IDL_SCHEMA_VERSION, "status": self.status,
                "programs": self.programs, "detail": self.detail}


def report() -> Dict[str, Any]:
    if IDL_STATUS != "OK":
        return IdlReport(status="DATA_BLOCKED", detail=IDL_STATUS).to_dict()
    try:
        programs = {name: program_id(name)
                    for name in (PUMP_IDL, PUMP_AMM_IDL, PUMP_FEES_IDL)}
    except (OSError, IdlError, ValueError) as exc:
        return IdlReport(status="DATA_BLOCKED", detail=str(exc)).to_dict()
    return IdlReport(status="OK", programs=programs).to_dict()
