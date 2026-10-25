import os
from pathlib import Path

import pytest

from src.chains.research_rpc import extra_solana_endpoints
from src.chains.rpc_manager import ChainRegistry, RPCManager


def test_numbered_pairs_deduplicate_without_a_provider_count_ceiling():
    env = {'SOLANA_RESEARCH_RPC_URL': 'https://provider.invalid/rpc?apikey=fixture',
           'SOLANA_RESEARCH_WS_URL': 'wss://provider.invalid/ws?apikey=fixture',
           'SOLANA_RESEARCH_RPC_URL_2': 'https://provider.invalid/rpc?apikey=fixture',
           'SOLANA_RESEARCH_RPC_URL_300': 'https://another.invalid/rpc'}
    result = extra_solana_endpoints(env)
    assert len(result) == 2
    assert result[0] == (env['SOLANA_RESEARCH_RPC_URL'], env['SOLANA_RESEARCH_WS_URL'])
    assert result[1] == (env['SOLANA_RESEARCH_RPC_URL_300'], None)


@pytest.mark.parametrize('url', ['file:///private', 'http://remote.invalid/?apikey=must-not-appear',
                                'https://provider.invalid/#must-not-appear'])
def test_invalid_endpoint_never_echoes_credential(url):
    with pytest.raises(ValueError) as exc:
        extra_solana_endpoints({'SOLANA_RESEARCH_RPC_URL': url})
    assert url not in str(exc.value) and 'must-not-appear' not in str(exc.value)


def test_local_validator_is_explicit_and_ws_scheme_checked():
    assert extra_solana_endpoints({'SOLANA_RESEARCH_RPC_URL': 'http://127.0.0.1:8899',
                                  'SOLANA_RESEARCH_WS_URL': 'ws://127.0.0.1:8900'})
    with pytest.raises(ValueError):
        extra_solana_endpoints({'SOLANA_RESEARCH_RPC_URL': 'https://provider.invalid',
                                'SOLANA_RESEARCH_WS_URL': 'https://provider.invalid'})


def test_registry_appends_and_redacts_research_provider(monkeypatch):
    monkeypatch.delenv('HELIUS_API_KEY', raising=False)
    for name in list(os.environ):
        if name.startswith('SOLANA_RESEARCH_'):
            monkeypatch.delenv(name)
    config = Path(__file__).resolve().parents[1]/'config/chains.yaml'
    baseline = ChainRegistry(str(config)).get_chain('solana')
    secret = 'fixture-credential-not-real'
    monkeypatch.setenv('SOLANA_RESEARCH_RPC_URL', 'https://provider.invalid/rpc?apikey='+secret)
    monkeypatch.setenv('SOLANA_RESEARCH_WS_URL', 'wss://provider.invalid/ws?apikey='+secret)
    chain = ChainRegistry(str(config)).get_chain('solana')
    assert [(ep.url, ep.ws_url) for ep in chain.rpc_endpoints[:-1]] == [(ep.url, ep.ws_url) for ep in baseline.rpc_endpoints]
    assert chain.rpc_endpoints[-1].ws_url.endswith(secret)
    rpc = RPCManager(chain)
    assert secret not in str(rpc.get_stats())
