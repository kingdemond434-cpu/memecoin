"""One trainer at a time, across processes.

The desk trains in-process on its own long clock. The systemd timer also
trains, every fifteen minutes, in a separate unit. Both paths invoke the same
trainers over the same corpus, and neither knew the other existed.

On a 4 GB box that is not a tidiness problem. A shadow-trainer pass holds the
whole labelled corpus in memory; two of them at once is roughly twice that,
plus the always-on collector, and the kernel resolves the shortfall by killing
whichever process it likes least -- which may be the desk, whose forward
evidence cannot be backfilled. Four OOM kills is what that looks like from the
outside, and it looks identical to "the trainer is too big", which is why the
memory cap kept being raised at the wrong thing.

So training takes a lock. It is a real `flock` on a real file, because that is
the only mutual exclusion two unrelated processes share: it is held by the
kernel for as long as the file descriptor is open, and released automatically
when the holder exits -- including when it is killed, which a lock file
containing a PID would not survive.

Whoever loses does not wait. It returns SKIPPED with the holder named, and the
next tick tries again. A trainer that blocks holding a systemd `oneshot` unit
open is a trainer that eventually trips `TimeoutStartSec` and reports a failure
that did not happen.
"""

from __future__ import annotations

import errno
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

try:  # pragma: no cover - platform shape
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

try:  # pragma: no cover - platform shape
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_LOCK_NAME = "training.lock"


class TrainingBusy(RuntimeError):
    """Another process is training. Not an error -- a reason to skip."""


def lock_path(root: Path) -> Path:
    return Path(root) / DEFAULT_LOCK_NAME


@contextmanager
def training_lock(root: Path, *, owner: str = "",
                  ) -> Iterator[Optional[int]]:
    """Hold the exclusive training lock, or raise TrainingBusy immediately.

    Yields the file descriptor. POSIX uses flock; Windows uses a nonblocking
    byte-range lock. Both are released by the kernel after a crash. An
    unsupported platform refuses to train rather than silently overlapping.
    """
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = lock_path(directory)

    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        try:
            _acquire(handle)
            acquired = True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            held = _read_holder(handle)
            raise TrainingBusy(
                f"another process is already training ({held or 'unknown'}); "
                "skipping rather than running a second pass over the same "
                "corpus") from exc
        # Stamp who holds it. Advisory only -- the flock is the mechanism, and
        # this text exists so a human reading the file learns something.
        offset = 0 if fcntl is not None else 1
        os.ftruncate(handle, offset)
        os.lseek(handle, offset, os.SEEK_SET)
        os.write(handle, f"{owner or 'training'} pid={os.getpid()} "
                         f"at={time.time():.0f}\n".encode())
        os.fsync(handle)
        yield handle
    finally:
        try:
            if acquired:
                _release(handle)
        except OSError:
            pass
        os.close(handle)


def _read_holder(handle: int) -> str:
    try:
        os.lseek(handle, 0 if fcntl is not None else 1, os.SEEK_SET)
        return os.read(handle, 256).decode("utf-8", "replace").strip()
    except OSError:
        return ""


def holder(root: Path) -> Tuple[bool, str]:
    """Is anyone training right now, and what does the file say about them?

    Probes by trying to take the lock and releasing it immediately, so the
    answer reflects the kernel's view rather than a stale PID in a file.
    """
    path = lock_path(Path(root))
    if not path.exists():
        return False, ""
    try:
        handle = os.open(path, os.O_RDWR)
    except OSError as exc:
        return False, f"lock unreadable: {exc}"
    try:
        try:
            _acquire(handle)
        except OSError:
            return True, _read_holder(handle)
        _release(handle)
        return False, ""
    finally:
        os.close(handle)


def _acquire(handle: int) -> None:
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif msvcrt is not None:
        # Reserve one byte; holder text starts after it so another process can
        # still read diagnostics while the mandatory Windows lock is held.
        if os.fstat(handle).st_size == 0:
            os.write(handle, b"\0")
        os.lseek(handle, 0, os.SEEK_SET)
        msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - unsupported operating system
        raise RuntimeError("no supported cross-process training lock")


def _release(handle: int) -> None:
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(handle, 0, os.SEEK_SET)
        msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
