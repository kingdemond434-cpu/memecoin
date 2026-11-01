import hashlib
import json
import os
import stat

import pytest
from ops.bounded_manifest import update_role_action, _security, _stage_security
from ops.bounded_release import RATES

def inputs():
    return ({"version": 1, "assembly_sha256": "a" * 64, "roles": {k: {"cpu_rate": v, "task": k, "execute": "old", "arguments": "private", "working_directory": "old"} for k, v in RATES.items()}}, {"tested_head": "a" * 40, "release_head": "a" * 40, "task_before": {"wrapper": "fixed", "priority": 5}})


def fixture(tmp_path):
    manifest, kw = inputs()
    path = tmp_path / 'roles.json'
    path.write_text(json.dumps(manifest))
    if os.name != 'nt':
        path.chmod(0o600)
    else:
        _stage_security(path, _security(path))  # isolated fixture only
    original = path.read_bytes()
    args = dict(expected_sha256=hashlib.sha256(original).hexdigest(), role='spot',
                action=dict(execute='new', arguments='private-new', working_directory='new'),
                snapshot_task=lambda: kw['task_before'], tested_head=kw['tested_head'],
                release_head=kw['release_head'], backup_verified=True)
    return path, original, args


def test_atomic_update_preserves_security_original_and_task(tmp_path):
    path, original, args = fixture(tmp_path)
    security = _security(path)
    result = update_role_action(path, **args)
    assert result['status'] == 'ROLE_ACTION_UPDATED'
    assert _security(path) == security
    from pathlib import Path
    assert Path(result['original_path']).read_bytes() == original
    assert json.loads(path.read_bytes())['roles']['spot']['arguments'] == 'private-new'
    assert not result['kernel_acceptance_verified']
    assert not list(tmp_path.glob('*.release-lock'))
    assert not list(tmp_path.glob('*.staged'))


def test_stale_hash_keeps_original_and_blocks_retry(tmp_path):
    path, original, args = fixture(tmp_path)
    args['expected_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='changed'):
        update_role_action(path, **args)
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        update_role_action(path, **args)


def test_task_changes_before_replace_never_commit(tmp_path):
    path, original, args = fixture(tmp_path)
    calls = iter([{'task': 'bounded'}, {'task': 'bounded'}, {'task': 'changed'}])
    args['snapshot_task'] = lambda: next(calls)
    with pytest.raises(ValueError):
        update_role_action(path, **args)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.original'))


def test_os_failure_keeps_original_and_inspection_lock(tmp_path, monkeypatch):
    path, original, args = fixture(tmp_path)
    def fail(*a):
        raise OSError('fixture failure')
    monkeypatch.setattr('ops.bounded_manifest._replace', fail)
    with pytest.raises(OSError):
        update_role_action(path, **args)
    assert path.read_bytes() == original
    assert list(tmp_path.glob('*.release-lock'))
    assert not list(tmp_path.glob('*.staged'))


def test_racing_manifest_change_not_overwritten(tmp_path):
    path, original, args = fixture(tmp_path)
    count = 0
    def task():
        nonlocal count
        count += 1
        if count == 2:
            path.write_bytes(original + b' ')
        return {'task': 'bounded'}
    args['snapshot_task'] = task
    with pytest.raises(ValueError):
        update_role_action(path, **args)
    assert path.read_bytes() == original + b' '


def test_missing_backup_never_replaces(tmp_path):
    path, original, args = fixture(tmp_path)
    args['backup_verified'] = False
    with pytest.raises(ValueError):
        update_role_action(path, **args)
    assert path.read_bytes() == original


def test_retained_original_can_restore_action_through_same_gates(tmp_path):
    path, original, args = fixture(tmp_path)
    update_role_action(path, **args)
    old = json.loads(original)['roles']['spot']
    args['action'] = {k: old[k] for k in ('execute', 'arguments', 'working_directory')}
    args['expected_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    update_role_action(path, **args)
    assert json.loads(path.read_bytes()) == json.loads(original)


def test_unprotected_manifest_rejected_before_replace(tmp_path, monkeypatch):
    path, original, args = fixture(tmp_path)
    monkeypatch.setattr('ops.bounded_manifest._protected_security', lambda data: False)
    with pytest.raises(ValueError, match='Protected'):
        update_role_action(path, **args)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.original'))


def test_task_changed_after_commit_keeps_recovery_original_and_lock(tmp_path):
    path, original, args = fixture(tmp_path)
    calls = iter([{'fixed': True}] * 3 + [{'changed': True}])
    args['snapshot_task'] = lambda: next(calls)
    with pytest.raises(RuntimeError, match='inspection'):
        update_role_action(path, **args)
    assert len(list(tmp_path.glob('*.original'))) == 1
    assert next(tmp_path.glob('*.original')).read_bytes() == original
    assert list(tmp_path.glob('*.release-lock'))


def test_unchanged_action_never_replaces_unprotected_manifest(tmp_path, monkeypatch):
    path, original, args = fixture(tmp_path)
    args['action'] = {k: 'old' if k != 'arguments' else 'private'
                      for k in ('execute', 'arguments', 'working_directory')}
    monkeypatch.setattr('ops.bounded_manifest._protected_security', lambda data: False)
    def forbidden(*a):
        raise AssertionError('No replacement allowed')
    monkeypatch.setattr('ops.bounded_manifest._replace', forbidden)
    result = update_role_action(path, **args)
    assert result['status'] == 'ROLE_ACTION_UNCHANGED'
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.original'))
    assert not list(tmp_path.glob('*.release-lock'))
