"""Bind a trusted private JUnit parity result to stable candidate fingerprints."""
import hashlib,re
import xml.etree.ElementTree as ET

CLASS='tests.test_native_ingress.TheTwoDecodersMustAgreeOnTheSameBytes'
NAMES=frozenset(('test_the_two_decoders_agree_field_for_field_on_a_create',
 'test_the_two_decoders_agree_field_for_field_on_a_trade',
 'test_a_create_decodes_to_the_same_mint_and_creator',
 'test_a_trade_decodes_to_the_same_reserves','test_a_migration_decodes',
 'test_an_unrecognised_payload_decodes_to_nothing'))


def bind_parity_manifest(*,revision,tree,before,after,junit,exit_code):
    """Caller must acquire these from actual serialized tests, not scraped data.

    This validates receipt consistency; it is not independent proof that tests
    ran, a build attestation, or authorization to deploy. Keep the raw receipt.
    """
    if any(not isinstance(x,str) or re.fullmatch('[0-9a-f]{40}',x) is None for x in (revision,tree)):
        raise ValueError('Exact source revision and tree required')
    if (not isinstance(before,dict) or before != after
            or not isinstance(before.get('sha256'),str)
            or re.fullmatch('[0-9a-f]{64}',before['sha256']) is None
            or type(before.get('size')) is not int or not 0 < before['size'] <= 16*1024*1024
            or type(exit_code) is not int or exit_code != 0
            or not isinstance(junit,bytes) or not 0 < len(junit) <= 65536
            or b'<!' in junit):
        raise ValueError('Stable candidate bytes and bounded successful test receipt required')
    try:
        root=ET.fromstring(junit)
        cases=list(root.iter('testcase'))
        if (len(cases)!=6 or {x.get('name') for x in cases} != NAMES
                or any(x.get('classname') != CLASS for x in cases)
                or any(list(root.iter(tag)) for tag in ('failure','error','skipped'))):
            raise ValueError()
    except Exception:
        raise ValueError('Exact six passing native parity cases required') from None
    return dict(status='NATIVE_PARITY_TEST_SCOPE_RECEIPT_BOUND',revision=revision,tree=tree,
                artifact_sha256=before['sha256'],artifact_size=before['size'],
                junit_sha256=hashlib.sha256(junit).hexdigest(),passed=6,scope=CLASS,
                build_provenance_verified=False,independent_execution_attestation=False,
                full_native_abi_tested=False,deployment_authorized=False)
