"""Bounded read-only digest check; caller supplies trusted artifact provenance."""
import hashlib
import math
from pathlib import Path
import re
import stat
import time

from ops.native_runtime_preflight import PRIVATE


def verify_native_digest(*, stage, relative_path, expected_sha256, expected_size,
                         continue_check, max_bytes=16*1024*1024,
                         bytes_per_second=256*1024, deadline_seconds=60):
    """Compare one private .pyd against a trusted tested-artifact expectation.

    Does not discover a trusted digest from the installed file, load code, or
    prove ABI/dependencies. Caller serializes maintenance in a trusted directory.
    Pacing bounds logical reads, not physical IO or a blocking filesystem call.
    """
    if (not isinstance(expected_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or type(expected_size) is not int or expected_size <= 0
            or type(max_bytes) is not int or not 1 <= max_bytes <= 16*1024*1024
            or expected_size > max_bytes
            or type(bytes_per_second) is not int or not 1 <= bytes_per_second <= 1024*1024
            or type(deadline_seconds) not in (int, float)
            or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 60):
        raise ValueError('Invalid native artifact expectation or read budget')
    root = Path(stage).absolute()/PRIVATE
    relative = Path(relative_path)
    if relative.is_absolute() or relative.drive or ':' in str(relative) or '..' in relative.parts or relative.suffix.lower() != '.pyd':
        raise ValueError('Private relative native extension required')
    path = root/relative
    started = time.monotonic()

    def admitted():
        if time.monotonic()-started >= deadline_seconds or continue_check() is not True:
            raise RuntimeError('Native artifact resource admission unavailable')

    def regular_chain():
        for item in [root, *list(path.parents)[:len(relative.parts)-1], path]:
            info = item.lstat()
            if item.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Native artifact reparse path rejected')
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise ValueError('Native artifact size or type mismatch')
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)

    admitted()
    before = regular_chain()
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        opened = stream.fileno()
        import os
        info = os.fstat(opened)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != before:
            raise ValueError('Native artifact changed before read')
        while count < expected_size:
            admitted()
            block = stream.read(min(65536, expected_size-count))
            if not block:
                raise ValueError('Native artifact truncated')
            digest.update(block)
            count += len(block)
            delay = count/bytes_per_second-(time.monotonic()-started)
            while delay > 0:
                admitted()
                time.sleep(min(delay, .1))
                delay = count/bytes_per_second-(time.monotonic()-started)
        if stream.read(1):
            raise ValueError('Native artifact grew')
    admitted()
    if regular_chain() != before or digest.hexdigest() != expected_sha256:
        raise ValueError('Native artifact changed or digest mismatch')
    return dict(status='PRIVATE_NATIVE_ARTIFACT_DIGEST_MATCH', bytes_read=count,
                native_binary_digest_verified=True, trusted_expectation_required=True,
                native_abi_tested=False, deployment_authorized=False,
                services_changed=False, physical_io_capped=False)

