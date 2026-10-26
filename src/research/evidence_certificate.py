"""Owner-local canonical verification metadata, not external notarization."""
import hashlib
import json
import re

SCHEMA = 'canonical-backup-certificate-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_certificate(value):
    if not isinstance(value, dict):
        raise ValueError('canonical prefix certificate object required')
    body = {key: item for key, item in value.items() if key != 'certificate_sha256'}
    if (value.get('schema') != SCHEMA or value.get('status') != 'PASS'
            or value.get('hash_chain_verified') is not True
            or fingerprint(body) != value.get('certificate_sha256')):
        raise ValueError('invalid canonical prefix certificate')
    for key in ('canonical_bytes', 'canonical_records'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('invalid canonical prefix extent')
    for key in ('canonical_sha256', 'last_hash'):
        if not isinstance(value.get(key), str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('invalid canonical prefix digest')
