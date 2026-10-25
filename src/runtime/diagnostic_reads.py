"""Share repeat reads only within one synchronous diagnostic projection."""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import threading

_reads = ContextVar('diagnostic_reads', default=None)


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


@contextmanager
def diagnostic_reads():
    token = _reads.set(({}, _owner()))
    try:
        yield
    finally:
        _reads.reset(token)


def diagnostic_value(key, read):
    state = _reads.get()
    if state is None or state[1] != _owner():
        return read()
    cache = state[0]
    if key not in cache:
        cache[key] = read()
    return cache[key]
