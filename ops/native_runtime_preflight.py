"""Read-only private native-runtime location and CPU inheritance preflight."""
import json
import os
from pathlib import Path
import stat
import subprocess

from ops.worker_resource_guard import THREAD_VARIABLES, verify

PRIVATE = 'runtime-native-45125a6'


def inspect_runtime(envelope, stage):
    root = (Path(stage)/PRIVATE).resolve()
    def same(value, expected):
        return isinstance(value, str) and Path(value).resolve() == expected
    extensions = envelope.get('extensions', [])
    cpu = envelope.get('cpu', {})
    if (not same(envelope.get('executable'), root/'Scripts/python.exe')
            or not same(envelope.get('prefix'), root)
            or type(envelope.get('isolated')) is not int or envelope['isolated'] != 1
            or envelope.get('dont_write_bytecode') is not True
            or cpu.get('status') != 'VERIFIED' or cpu.get('cpu_rate') != 200
            or cpu.get('kernel_flags') != 5
            or type(cpu.get('native_threads')) is not int or cpu['native_threads'] != 1
            or cpu.get('total_configured_percent') != 10
            or not isinstance(extensions, list) or len(extensions) != 1):
        raise ValueError('Private native runtime identity or CPU inheritance mismatch')
    extension = extensions[0]
    if not isinstance(extension, str):
        raise ValueError('Private native extension location required')
    path = Path(extension).resolve()
    if path.suffix.lower() != '.pyd' or root not in path.parents:
        raise ValueError('Native extension resolves outside private runtime')
    return {'status': 'PRIVATE_NATIVE_RUNTIME_LOCATION_AND_CPU_VERIFIED',
            'cpu_percent': 2, 'native_threads': 1,
            'isolated_child': True, 'bytecode_writes_disabled': True,
            'private_extension_location_verified': True,
            'native_binary_digest_verified': False, 'native_abi_tested': False,
            'tested_release_verified': False, 'deployment_authorized': False,
            'services_changed': False}


def probe_private_runtime(*, stage, run=subprocess.run, resource_check=verify):
    """Launch only the private Python as a direct inherited-job child.

    Caller supplies a trusted stage directory. This does not execute a role
    action, start collectors, test production state, hash DLLs or prove ABI.
    Importing the installed module is permitted only as trusted installed code.
    No provider, signer or GitHub environment is supplied to the probe.
    """
    cpu = resource_check()
    if cpu.get('status') != 'VERIFIED' or cpu.get('cpu_rate') != 200:
        raise RuntimeError('Development CPU proof required')
    root = Path(stage).resolve()/PRIVATE
    interpreter = root/'Scripts/python.exe'
    info = interpreter.lstat()
    if (not stat.S_ISREG(info.st_mode) or interpreter.is_symlink()
            or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise ValueError('Regular private interpreter required')
    env = {key:value for key,value in os.environ.items()
           if key.upper() in {'SYSTEMROOT','WINDIR','PATH','TEMP','TMP','COMSPEC','PATHEXT',
                              'PROGRAMDATA','ALLUSERSPROFILE'}
           or key in THREAD_VARIABLES or key == 'TOKENIZERS_PARALLELISM'}
    guard = str(Path(__file__).with_name('worker_resource_guard.py').resolve())
    # No action/argument/config strings from the role manifest are executed.
    script = '''import importlib.util,importlib.machinery,json,sys
spec=importlib.util.spec_from_file_location('resource_probe_guard',GUARD)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
cpu=guard.verify()
import solana_fastpath
extensions=sorted({m.__file__ for name,m in list(sys.modules.items())
 if (name=='solana_fastpath' or name.startswith('solana_fastpath.'))
 and getattr(m,'__file__',None)
 and any(m.__file__.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)})
print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,
 isolated=sys.flags.isolated,dont_write_bytecode=bool(sys.dont_write_bytecode),
 extensions=extensions,cpu=cpu)))
'''.replace('GUARD', json.dumps(guard))
    try:
        result = run([str(interpreter), '-I', '-B', '-c', script], cwd=str(root),
                     env=env, capture_output=True, text=True, timeout=60, check=True)
        if len(result.stdout) > 16384:
            raise ValueError('Probe output budget')
        return inspect_runtime(json.loads(result.stdout), stage)
    except Exception:
        raise RuntimeError('Private native runtime probe failed; private output suppressed') from None
