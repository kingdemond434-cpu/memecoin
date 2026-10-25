"""Owner-declared Solana endpoints; credential values never enter errors."""
from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

HTTP_NAME = re.compile(r'SOLANA_RESEARCH_RPC_URL(?:_[1-9][0-9]*)?\Z')


def _validate(value, *, websocket=False):
    try:
        uri = urlsplit(value)
        allowed = {'wss'} if websocket else {'https'}
        local = uri.hostname in {'localhost', '127.0.0.1', '::1'}
        if local:
            allowed.add('ws' if websocket else 'http')
        if uri.scheme not in allowed or not uri.hostname or uri.fragment:
            raise ValueError
        _ = uri.port
    except (TypeError, ValueError):
        raise ValueError('Invalid owner-declared Solana RPC endpoint') from None
    return value


def extra_solana_endpoints(environ=None):
    """Resolve arbitrary numbered HTTP/WS pairs without shifting YAML pairs."""
    env = os.environ if environ is None else environ
    names = sorted(name for name in env if HTTP_NAME.fullmatch(name))
    result, seen = [], set()
    for name in names:
        http = env[name].strip()
        if not http:
            continue
        suffix = name[len('SOLANA_RESEARCH_RPC_URL'):]
        ws = env.get('SOLANA_RESEARCH_WS_URL' + suffix, '').strip()
        http = _validate(http)
        ws = _validate(ws, websocket=True) if ws else None
        if http not in seen:
            result.append((http, ws))
            seen.add(http)
    return result
