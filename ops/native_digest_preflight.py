"""Bind trusted artifact bytes to the existing private runtime probe."""
import subprocess
from pathlib import Path
from ops.native_artifact_digest import verify_native_digest
from ops.native_runtime_preflight import PRIVATE, probe_private_runtime
from ops.worker_resource_guard import verify


def probe_digest_bound_runtime(*, stage, relative_path, expected_sha256,
                               expected_size, continue_check,
                               run=subprocess.run, resource_check=verify):
    """Read-only gate; trusted expected digest must come from tested provenance.

    Caller serializes maintenance in a trusted stage. Two bounded digest reads
    bracket the child probe; this is not protection against a malicious writer
    swapping code during import, nor proof of interpreter/dependency provenance.
    """
    cpu=resource_check()
    if cpu.get('status') != 'VERIFIED' or cpu.get('cpu_rate') != 200:
        raise RuntimeError('Development CPU proof required before artifact reads')
    args=dict(stage=stage,relative_path=relative_path,expected_sha256=expected_sha256,
              expected_size=expected_size,continue_check=continue_check)
    before=verify_native_digest(**args)
    expected=Path(stage).absolute()/PRIVATE/relative_path
    runtime=probe_private_runtime(stage=stage,run=run,resource_check=resource_check,
                                  expected_extension=expected)
    after=verify_native_digest(**args)
    return dict(status='PRIVATE_RUNTIME_DIGEST_AND_LOCATION_VERIFIED',
                artifact_before=before,artifact_after=after,runtime=runtime,
                native_binary_digest_verified=True,native_abi_tested=False,
                deployment_authorized=False,services_changed=False,
                trusted_serialized_stage_required=True,
                interpreter_and_dependency_digests_verified=False)
