"""Read-only evidence acquisition for bounded paper release preflight.

No task registration, service restart, credential output or data backup occurs.
Full exported task XML is fingerprinted internally and never returned to logs.
"""
from __future__ import annotations

import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

TASKS = {'spot': 'MemecoinShadowCodex', 'prediction': 'MemecoinPredictionPaper',
         'capture': 'MemecoinProgramCapture', 'terminal': 'MemecoinPaperTerminal'}
NS = {'t': 'http://schemas.microsoft.com/windows/2004/02/mit/task'}
ROOT = 'C:/ProgramData/MemecoinResourceLimits'
SHELL = 'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def _path(value):
    return ntpath.normcase(ntpath.normpath(value))


def _tokens(value):
    # Restricted task grammar, not a general shell parser. Embedded/escaped
    # quotes and adjacent quoted/unquoted fragments are rejected, not guessed.
    tokens, end = [], 0
    for match in re.finditer(r'"[^"\r\n]*"|[^\s"\r\n]+', value):
        gap = value[end:match.start()]
        if (end and not gap) or gap.strip():
            raise ValueError('Unsupported bounded task argument grammar')
        token = match.group()
        if token.startswith('"'):
            token = token[1:-1]
            if token.endswith('\\'):
                raise ValueError('Unsupported bounded task quoting')
        tokens.append(token)
        end = match.end()
    if value[end:].strip():
        raise ValueError('Unsupported bounded task argument grammar')
    return tokens


def inspect_task_xml(xml, role, *, resource_root=ROOT, shell=SHELL):
    if role not in TASKS or not isinstance(xml, str) or len(xml) > 131072:
        raise ValueError('Bounded task definition invalid')
    if '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper():
        raise ValueError('Task declarations forbidden')
    tree = ET.fromstring(xml)
    if tree.tag != '{'+NS['t']+'}Task':
        raise ValueError('Scheduled Task namespace required')
    principals = tree.findall('t:Principals/t:Principal', NS)
    actions = tree.findall('t:Actions/*', NS)
    if len(principals) != 1 or len(actions) != 1 or actions[0].tag != '{'+NS['t']+'}Exec':
        raise ValueError('Single bounded service principal/action required')
    principal, action = principals[0], actions[0]
    text = lambda node, tag: node.findtext('t:'+tag, default='', namespaces=NS)
    user = text(principal, 'UserId').upper()
    if (user not in ('S-1-5-19', 'LOCAL SERVICE', 'NT AUTHORITY\\LOCAL SERVICE')
            or text(principal, 'LogonType') != 'ServiceAccount'
            or text(principal, 'RunLevel') != 'LeastPrivilege'
            or tree.findtext('t:Settings/t:Priority', namespaces=NS) != '7'
            or tree.findtext('t:Settings/t:MultipleInstancesPolicy', namespaces=NS) != 'IgnoreNew'):
        raise ValueError('Bounded task identity or scheduling fence mismatch')
    if (_path(text(action, 'Command')) != _path(shell)
            or _path(text(action, 'WorkingDirectory')) != _path(resource_root)):
        raise ValueError('CPU wrapper action mismatch')
    args = _tokens(text(action, 'Arguments'))
    # Match the installed wrapper action; this module never executes it or sets
    # any execution policy. Unexpected commands/options fail closed.
    expected = ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                '-File', ntpath.join(resource_root, 'start-cpu-bounded.ps1'),
                '-ManifestPath', ntpath.join(resource_root, 'roles.json'), '-Role', role]
    if len(args) != len(expected) or any(
            (_path(got) != _path(want) if i in (5, 7) else got.lower() != want.lower())
            for i, (got, want) in enumerate(zip(args, expected))):
        raise ValueError('Installed bounded wrapper arguments mismatch')
    return {'task': TASKS[role], 'role': role, 'wrapped': True,
            'principal': 'LOCAL_SERVICE', 'run_level': 'LeastPrivilege',
            'priority': 7, 'multiple_instances': 'IgnoreNew',
            'definition_sha256': hashlib.sha256(xml.encode('utf-8')).hexdigest()}


def _environment():
    env = os.environ.copy()
    for key in ('GH_TOKEN', 'GITHUB_TOKEN'):
        env.pop(key, None)
    env['GIT_TERMINAL_PROMPT'] = '0'
    return env


def snapshot_bounded_task(role, *, resource_root=ROOT, shell=SHELL, run=subprocess.run):
    if role not in TASKS:
        raise ValueError('Unsupported release role')
    # Local read-only COM avoids ScheduledTasks CIM-module warmup on a busy,
    # CPU-capped host. Connect has no remote host or alternate credentials.
    command = ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding="
               "[Text.UTF8Encoding]::new($false); $service=New-Object -ComObject Schedule.Service; "
               "$service.Connect(); [Console]::Write($service.GetFolder('\\').GetTask('"
               + TASKS[role] + "').Xml)")
    try:
        result = run([shell, '-NoProfile', '-NonInteractive', '-Command', command],
                     env=_environment(), capture_output=True, text=True,
                     encoding='utf-8', errors='strict', timeout=30, check=True)
        return inspect_task_xml(result.stdout, role, resource_root=resource_root, shell=shell)
    except Exception:
        raise RuntimeError('Bounded task export/verification failed; private output suppressed') from None


def _git(repo, *args):
    try:
        return subprocess.run(['git', '-C', str(repo), *args], env=_environment(),
                              capture_output=True, text=True, timeout=30,
                              check=True).stdout.strip()
    except Exception:
        raise RuntimeError('Release revision verification failed; private output suppressed') from None


def verify_tested_release(repo, release, report_path):
    """Check actual isolated checkout and source trees, not a caller boolean.

    Reports must come from the private trusted validation workflow, not scraped
    source material. This does not verify CI/review or state backups.
    """
    path = Path(report_path)
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ValueError('Bounded private validation report required')
    raw = path.read_bytes()
    report = json.loads(raw.decode('utf-8-sig'))
    tested = report.get('revision', '')
    counts = report.get('counts', {})
    if (not re.fullmatch('[0-9a-f]{40}', tested)
            or not isinstance(release, str) or not re.fullmatch('[0-9a-f]{40}', release)
            or report.get('status') != 'PASS' or type(report.get('exit_code')) is not int
            or report['exit_code'] != 0 or type(counts.get('passed')) is not int
            or counts['passed'] < 1 or type(counts.get('failed')) is not int
            or counts['failed'] != 0 or type(counts.get('errors')) is not int
            or counts['errors'] != 0
            or any(report.get(k) is not True for k in ('isolated_checkout_clean',
                'native_fields_preflight', 'native_source_unchanged'))
            or report.get('production_state_copied') is not False):
        raise ValueError('Exact isolated validation evidence required')
    isolated = Path(report['checkout'])
    if (_git(isolated, 'rev-parse', 'HEAD') != tested
            or _git(isolated, 'status', '--porcelain')
            or _git(repo, 'status', '--porcelain')):
        raise ValueError('Clean matching tested checkouts required')
    tested_tree = _git(isolated, 'rev-parse', 'HEAD^{tree}')
    if _git(repo, 'rev-parse', release+'^{tree}') != tested_tree:
        raise ValueError('Release tree differs from tested tree')
    return {'status': 'EXACT_TESTED_TREE_VERIFIED', 'release': release,
            'tested_head': tested, 'tested_tree': tested_tree,
            'report_sha256': hashlib.sha256(raw).hexdigest(),
            'ci_verified': False, 'backup_verified': False, 'deployed': False}
