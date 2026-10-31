import json
import pytest
from ops import worker_resource_guard as guard


@pytest.fixture
def resource_fixture(tmp_path):
    path = tmp_path / 'roles.json'
    path.write_text(json.dumps({'version': 1, 'roles': {'development': {'cpu_rate': 200},
        'spot': {'cpu_rate': 500}, 'prediction': {'cpu_rate': 100},
        'capture': {'cpu_rate': 150}, 'terminal': {'cpu_rate': 50}}}))
    env = {name: '1' for name in guard.THREAD_VARIABLES}
    env['TOKENIZERS_PARALLELISM'] = 'false'
    return path, env


def ancestor():
    return dict(flags=5, rate=200, current_process_member=True,
                priority_class=16384, limit_flags=0x2020)


def test_nested_job_requires_explicit_ancestor_membership(resource_fixture):
    path, env = resource_fixture
    result = guard.verify(path, env=env, query=lambda: {'flags': 0, 'rate': 0},
                          ancestor_query=ancestor)
    assert result['status'] == 'VERIFIED'
    assert result['cpu_percent'] == 2
    assert result['kernel_evidence_source'] == 'NAMED_ANCESTOR_JOB_WITH_CURRENT_PROCESS_MEMBERSHIP'


@pytest.mark.parametrize('key,value', [('flags', 1), ('rate', 201),
    ('current_process_member', False), ('current_process_member', 1),
    ('priority_class', 32), ('limit_flags', 0x20), ('limit_flags', 0x2000)])
def test_named_rate_alone_or_wrong_limits_never_suffice(resource_fixture, key, value):
    path, env = resource_fixture
    row = ancestor(); row[key] = value
    with pytest.raises(RuntimeError):
        guard.verify(path, env=env, query=lambda: {'flags': 0, 'rate': 0},
                     ancestor_query=lambda: row)


def test_named_job_denial_propagates_without_launch_or_fallback(resource_fixture):
    path, env = resource_fixture
    def denied():
        raise PermissionError('fixture denial')
    with pytest.raises(PermissionError):
        guard.verify(path, env=env, query=lambda: {'flags': 0, 'rate': 0},
                     ancestor_query=denied)


def test_injected_bad_immediate_query_cannot_consult_real_host(resource_fixture, monkeypatch):
    path, env = resource_fixture
    monkeypatch.setattr(guard, 'development_job_cpu', lambda: pytest.fail('real host lookup'))
    with pytest.raises(RuntimeError):
        guard.verify(path, env=env, query=lambda: {'flags': 0, 'rate': 0})


def test_direct_cap_does_not_need_named_job(resource_fixture):
    path, env = resource_fixture
    def forbidden():
        pytest.fail('named job lookup not needed')
    result = guard.verify(path, env=env, query=lambda: {'flags': 5, 'rate': 200},
                          ancestor_query=forbidden)
    assert result['kernel_evidence_source'] == 'CURRENT_JOB'


def test_ancestor_does_not_relax_thread_requirements(resource_fixture):
    path, env = resource_fixture
    env['OMP_NUM_THREADS'] = '2'
    with pytest.raises(RuntimeError, match='Single-thread'):
        guard.verify(path, env=env, query=lambda: {'flags': 0, 'rate': 0},
                     ancestor_query=ancestor)
