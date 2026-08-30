"""Turning desk objects into something a status page can carry.

Moved out of `main.py` verbatim so the reporting surface can live in its own
module without importing the desk back: a mixin that imported its own host
would be a circular import, and relocating one nine-line helper is a cheaper
fix than restructuring the reporting code around the cycle.

Deliberately unchanged in behaviour, including the order of the checks. An
Enum that is also a dataclass, or a dataclass TYPE rather than an instance,
resolves exactly as it did before the move -- a refactor that quietly
"improves" a helper is a behaviour change wearing a tidy commit message.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def jsonable(value: Any) -> Any:
    """A JSON-safe view of a dataclass, enum, mapping or sequence.

    Total rather than strict: anything unrecognised is returned as-is instead
    of raising, because a status endpoint that fails on one unexpected field
    tells an operator nothing about the other forty.
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value
