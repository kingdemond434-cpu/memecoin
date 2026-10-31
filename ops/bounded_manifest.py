"""Small private-manifest transaction; no task registration or process launch.

The caller must independently verify release/backup evidence and snapshot the
bounded task. Retained originals contain private actions and must stay in the
same ACL-restricted directory. This is not a complete deployment orchestrator.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

from ops.bounded_release import plan_role_action

MAX_BYTES = 65536


def _read(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or getattr(info, 'st_file_attributes', 0) & 0x400
            or info.st_size > MAX_BYTES):
        raise ValueError('Regular bounded manifest required')
    with path.open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('Manifest too large')
    return data


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _security(path):
    if os.name != 'nt':
        return stat.S_IMODE(path.stat().st_mode)
    from ctypes import wintypes
    api = ctypes.WinDLL('advapi32', use_last_error=True).GetFileSecurityW
    api.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                    wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.restype = wintypes.BOOL
    size = wintypes.DWORD()
    api(str(path), 7, None, 0, ctypes.byref(size))  # owner, group, DACL only
    if not size.value:
        raise OSError('Manifest security unavailable')
    buffer = ctypes.create_string_buffer(size.value)
    if not api(str(path), 7, buffer, size.value, ctypes.byref(size)):
        raise OSError('Manifest security unavailable')
    return buffer.raw



def _protected_security(data):
    if os.name != 'nt':
        return True
    from ctypes import wintypes
    api = ctypes.WinDLL('advapi32', use_last_error=True).GetSecurityDescriptorControl
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                    ctypes.POINTER(wintypes.DWORD)]
    api.restype = wintypes.BOOL
    control, revision = wintypes.WORD(), wintypes.DWORD()
    buffer = ctypes.create_string_buffer(data)
    if not api(buffer, ctypes.byref(control), ctypes.byref(revision)):
        raise OSError('Manifest security unavailable')
    return bool(control.value & 0x1000)  # SE_DACL_PROTECTED


def _stage_security(path, data):
    if os.name != 'nt':
        os.chmod(path, data)
        return
    from ctypes import wintypes
    dll = ctypes.WinDLL('advapi32', use_last_error=True)
    get_dacl = dll.GetSecurityDescriptorDacl
    get_dacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    get_dacl.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(data)
    present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
    if not get_dacl(buffer, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)) or not present.value or not dacl.value:
        raise OSError('Explicit staged manifest DACL required')
    set_info = dll.SetNamedSecurityInfoW
    set_info.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    set_info.restype = wintypes.DWORD
    if set_info(str(path), 1, 0x80000004, None, None, dacl, None) != 0:
        raise OSError('Cannot preserve staged manifest security')



def _replace(target, staged, backup):
    if os.name == 'nt':
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True).ReplaceFileW
        api.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                        wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
        api.restype = wintypes.BOOL
        # Flags zero: never ignore ACL/metadata merge errors. ReplaceFile retains
        # the destination security descriptor and atomically keeps its original.
        if not api(str(target), str(staged), str(backup), 0, None, None):
            raise OSError('Private manifest replacement failed')
    else:
        # CI fixtures only; live deployment requires the Windows CPU wrapper.
        with backup.open('xb') as out:
            os.chmod(backup, stat.S_IMODE(target.stat().st_mode))
            out.write(_read(target)); out.flush(); os.fsync(out.fileno())
        os.replace(staged, target)


def update_role_action(path, *, expected_sha256, role, action, snapshot_task,
                       tested_head, release_head, backup_verified):
    """Serialize cooperating writers and reject changed bytes/task before commit.

    No retry or automatic rollback after an uncertain OS failure. The lock stays
    on disk on failure so a subsequent worker must inspect the target, retained
    original and exact failure phase rather than overwriting uncertain state.
    """
    path = Path(path).absolute()
    lock = path.with_name(path.name + '.release-lock')
    with lock.open('xb') as stream:
        stream.write(b'Inspect release evidence before removing this lock.\n')
        stream.flush(); os.fsync(stream.fileno())
    staged = path.with_name(path.name + '.' + uuid.uuid4().hex + '.staged')
    backup = path.with_name(path.name + '.' + uuid.uuid4().hex + '.original')
    committed = False
    try:
        original = _read(path)
        if _digest(original) != expected_sha256:
            raise ValueError('Manifest changed before release')
        security = _security(path)
        before = snapshot_task()
        candidate = plan_role_action(
            json.loads(original.decode('utf-8-sig')), role, action,
            tested_head=tested_head, release_head=release_head,
            backup_verified=backup_verified, task_before=before,
            task_after=snapshot_task())
        if candidate == json.loads(original.decode('utf-8-sig')):
            if (_read(path) != original or _security(path) != security
                    or snapshot_task() != before):
                raise ValueError('Release preconditions changed')
            committed = True
            return {'status': 'ROLE_ACTION_UNCHANGED',
                    'before_sha256': _digest(original), 'after_sha256': _digest(original),
                    'original_path': None, 'task_changed': False,
                    'services_restarted': False, 'kernel_acceptance_verified': False}
        if not _protected_security(security):
            raise ValueError('Protected manifest DACL required; no live ACL changes permitted')
        encoded = (json.dumps(candidate, indent=2) + '\n').encode('utf-8')
        if len(encoded) > MAX_BYTES:
            raise ValueError('Candidate too large')
        with staged.open('xb') as stream:
            _stage_security(staged, security)
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        if (_security(staged) != security or _read(path) != original or _security(path) != security
                or snapshot_task() != before):
            raise ValueError('Release preconditions changed')
        _replace(path, staged, backup)
        if (_read(path) != encoded or _read(backup) != original
                or _security(path) != security or _security(backup) != security
                or snapshot_task() != before):
            raise RuntimeError('Release requires recovery inspection')
        committed = True
        return {'status': 'ROLE_ACTION_UPDATED', 'before_sha256': _digest(original),
                'after_sha256': _digest(encoded), 'original_path': str(backup),
                'task_changed': False, 'services_restarted': False,
                'kernel_acceptance_verified': False}
    finally:
        if staged.exists():
            staged.unlink()  # only this invocation's unique staged file
        if committed:
            lock.unlink()
