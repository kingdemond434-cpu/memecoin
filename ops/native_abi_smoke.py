"""Fixed offline calls across the trusted installed native Python boundary."""
import hashlib


def check_native_abi_smoke(native):
    """No ingress construction, instruction submission, accounts or network.

    Caller must establish trusted code identity/location before loading native.
    These vectors exercise only a small pure-function surface, not the full ABI.
    """
    try:
        encoded=native.b58encode(bytes(32))
        if type(encoded) is not str or encoded != '1'*32:
            raise ValueError()
        decoded=native.b58decode('1'*32)
        if not isinstance(decoded,(bytes,list)) or list(decoded) != [0]*32:
            raise ValueError()
        invalid_rejected=False
        try:
            native.b58decode('0')
        except ValueError:
            invalid_rejected=True
        if not invalid_rejected:
            raise ValueError()
        discriminator=native.anchor_discriminator('buy')
        if bytes(discriminator) != hashlib.sha256(b'global:buy').digest()[:8]:
            raise ValueError()
        probability=native.hazard_predict([0.0],0.0,[0.0])
        if type(probability) is not float or probability != .5:
            raise ValueError()
    except Exception:
        raise RuntimeError('Offline native ABI smoke failed; private errors suppressed') from None
    return dict(status='OFFLINE_NATIVE_ABI_SMOKE_PASSED',fixed_vectors=5,
                native_abi_smoke_tested=True,full_native_abi_tested=False,
                binary_provenance_verified=False,deployment_authorized=False,
                network_used=False,services_changed=False)
