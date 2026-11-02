import json
import pytest
from ops.loopback_health_probe import probe

class Conn:
    def __init__(self, blocks=(), fail=None):
        self.blocks=list(blocks);self.fail=fail;self.closed=False;self.timeouts=[];self.address=None
    def settimeout(self, value):
        assert 0 < value <= 10
        self.timeouts.append(value)
    def connect(self, address):
        self.address=address
        if self.fail=='connect':raise TimeoutError('private connect detail')
    def sendall(self, data):
        assert data.startswith(b'GET /health ')
        if self.fail=='request':raise OSError('private request detail')
    def recv(self, size):
        assert 0 < size <= 4096
        if self.fail=='receive':raise TimeoutError('private receive detail')
        if not self.blocks:return b''
        value=self.blocks.pop(0)
        if isinstance(value,Exception):raise value
        self.blocks.insert(0,value[size:]) if len(value)>size else None
        return value[:size]
    def close(self):self.closed=True

def call(conn, **kwargs):
    return probe(18081,socket_factory=lambda *_:conn,**kwargs)

def response(body, headers=b''):
    return b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(body)).encode()+b'\r\n'+headers+b'\r\n'+body

def test_projection_is_typed_and_private_content_discarded():
    c=Conn([response(json.dumps(dict(status='healthy',live_submission_locked=True,read_only='true',committed_program_records=23,secret='private',error='private')).encode())])
    r=call(c)
    assert r['status']=='HEALTH_RESPONSE_DECODED'
    assert r['health']==dict(status='healthy',live_submission_locked=True,committed_program_records=23)
    assert c.address==('127.0.0.1',18081) and c.closed
    assert 'private' not in json.dumps(r) and not r['root_cause_verified']

@pytest.mark.parametrize('port',[True,80,443,18085,'18081',None])
def test_only_fixed_loopback_ports(port):
    with pytest.raises(ValueError):probe(port)

@pytest.mark.parametrize('kwargs',[dict(deadline_seconds=0),dict(deadline_seconds=float('nan')),dict(deadline_seconds=11),dict(deadline_seconds=True),dict(max_body_bytes=65537),dict(max_body_bytes=0),dict(max_body_bytes=True)])
def test_policy_bounds(kwargs):
    with pytest.raises(ValueError):probe(18081,**kwargs)

@pytest.mark.parametrize('failure,stage',[('connect','connect'),('request','request'),('receive','headers')])
def test_failures_identify_stage_without_private_error(failure,stage):
    c=Conn(fail=failure);r=call(c)
    assert r['failed_stage']==stage and c.closed
    assert 'private' not in json.dumps(r)

def test_body_timeout_distinct_from_header_timeout():
    c=Conn([b'HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n',TimeoutError('private body')]);r=call(c)
    assert r['failed_stage']=='body' and r['error_type']=='TimeoutError'
    assert set(r['completed_stages_seconds'])=={'connect','request','headers'}

@pytest.mark.parametrize('wire',[
    b'HTTP/1.1 200 OK\r\nX: '+b'x'*16384,
    b'HTTP/1.1 200 OK\r\nContent-Length: 999999\r\n\r\n',
    b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n0',
    b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n',
    b'HTTP/1.1 302 Private\r\nLocation: private\r\n\r\n',
    b'HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n',
    b'not HTTP\r\n\r\n',
])
def test_headers_fail_closed(wire):
    r=call(Conn([wire]));assert r['status']=='PROBE_FAILED' and r['failed_stage']=='headers'
    assert 'private' not in json.dumps(r).lower()

@pytest.mark.parametrize('body',[b'[]',b'null',b'private invalid json'])
def test_invalid_body_not_returned(body):
    r=call(Conn([response(body)]));assert r['failed_stage']=='decode' and r['health']=={}
    assert 'private' not in json.dumps(r)

def test_short_body_is_explicit():
    r=call(Conn([b'HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n{}']))
    assert r['failed_stage']=='body' and r['error_type']=='ValueError'

def test_close_delimited_and_fragmented_response():
    r=call(Conn([b'HTTP/1.0 200 OK\r\n',b'\r\n',b'{"status":',b'"healthy"}']))
    assert r['status']=='HEALTH_RESPONSE_DECODED'

def test_body_budget_for_close_delimited_response():
    r=call(Conn([b'HTTP/1.0 200 OK\r\n\r\n123456']),max_body_bytes=5)
    assert r['failed_stage']=='body'

def test_deadline_is_total_not_reset_per_read():
    ticks=iter([0,0,0,0,0,0,0,0,5,5,5]);c=Conn([b'HTTP/1.0 200 OK\r\n\r\n',b'{}'])
    r=call(c,clock=lambda:next(ticks))
    assert r['status']=='PROBE_FAILED' and r['error_type']=='TimeoutError' and c.closed

def test_bad_field_types_cannot_smuggle_objects():
    r=call(Conn([response(b'{"status":{"secret":"private"},"live_submission_locked":1,"committed_program_records":true}')]))
    assert r['health']=={} and 'private' not in json.dumps(r)

