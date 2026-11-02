"""Deadline-bounded loopback health stages; no service control or remote endpoints."""
import json, math, socket, time

PORTS = frozenset((18081, 18082, 18083, 18084))


def probe(port, *, deadline_seconds=4, max_body_bytes=65536,
          clock=time.monotonic, socket_factory=socket.socket):
    """One connection, finite header/body, total cooperative socket deadline.

    Errors expose types/stages only. Stage duration includes scheduling delay and
    cannot identify server CPU, queue, provider or event-loop root cause.
    No retries, authentication, redirects, environment proxies or service changes.
    """
    if (type(port) is not int or port not in PORTS
            or type(deadline_seconds) not in (int, float)
            or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 10
            or type(max_body_bytes) is not int or not 1 <= max_body_bytes <= 65536):
        raise ValueError('Bounded loopback health policy required')
    start = clock(); stage = 'socket'; marks = {}; conn = None
    result = {'port': port, 'status': 'PROBE_FAILED', 'service_changed': False,
              'root_cause_verified': False, 'health': {}}
    def remaining():
        left = deadline_seconds - (clock() - start)
        if left <= 0: raise TimeoutError('Health deadline expired')
        return left
    def receive(size):
        conn.settimeout(remaining())
        return conn.recv(size)
    try:
        conn = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        stage = 'connect'; conn.settimeout(remaining())
        conn.connect(('127.0.0.1', port)); marks['connect'] = clock() - start
        stage = 'request'; conn.settimeout(remaining())
        conn.sendall(('GET /health HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n' % port).encode('ascii'))
        marks['request'] = clock() - start
        stage = 'headers'; buf = bytearray()
        while b'\r\n\r\n' not in buf:
            block = receive(min(4096, 16384 - len(buf)))
            if not block: raise ValueError('Incomplete health headers')
            buf.extend(block)
            if len(buf) >= 16384 and b'\r\n\r\n' not in buf:
                raise ValueError('Health header budget exceeded')
        headers, body = bytes(buf).split(b'\r\n\r\n', 1)
        lines = headers.split(b'\r\n'); parts = lines[0].split()
        if len(parts) < 2 or parts[0] not in (b'HTTP/1.0', b'HTTP/1.1'):
            raise ValueError('Health status line invalid')
        code = int(parts[1]); result['http_status'] = code
        if code != 200: raise ValueError('Health HTTP status rejected')
        length = None
        for line in lines[1:]:
            key, sep, value = line.partition(b':')
            if not sep: raise ValueError('Health header invalid')
            if key.lower() == b'transfer-encoding':
                raise ValueError('Health transfer encoding unsupported')
            if key.lower() == b'content-length':
                if length is not None or not value.strip().isdigit():
                    raise ValueError('Health content length invalid')
                length = int(value.strip())
        if length is not None and length > max_body_bytes:
            raise ValueError('Health body budget exceeded')
        marks['headers'] = clock() - start; stage = 'body'
        body = bytearray(body)
        while length is None or len(body) < length:
            if len(body) > max_body_bytes: raise ValueError('Health body budget exceeded')
            count = min(4096, max_body_bytes + 1 - len(body))
            if length is not None: count = min(count, length - len(body))
            block = receive(count)
            if not block:
                if length is not None and len(body) != length:
                    raise ValueError('Incomplete health body')
                break
            body.extend(block)
        if len(body) > max_body_bytes or (length is not None and len(body) != length):
            raise ValueError('Health body length mismatch')
        marks['body'] = clock() - start; stage = 'decode'; remaining()
        data = json.loads(body)
        if not isinstance(data, dict): raise ValueError('Health object required')
        # No arbitrary object/string content or private error text is returned.
        health = {}
        if data.get('status') in ('healthy', 'degraded', 'unhealthy'):
            health['status'] = data['status']
        for key in ('live_submission_locked', 'read_only'):
            if type(data.get(key)) is bool: health[key] = data[key]
        value = data.get('committed_program_records')
        if type(value) is int and 0 <= value <= 2**63 - 1:
            health['committed_program_records'] = value
        remaining(); result.update(status='HEALTH_RESPONSE_DECODED', health=health)
    except Exception as exc:
        result.update(failed_stage=stage, error_type=type(exc).__name__)
    finally:
        if conn is not None:
            try: conn.close()
            except Exception: pass
    result.update(elapsed_seconds=clock() - start, completed_stages_seconds=marks)
    return result
