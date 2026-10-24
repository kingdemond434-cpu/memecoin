import asyncio
from types import SimpleNamespace

from src.research.data_miners import rotating_selector
from src.research.chain_miners import account_balance_miner, deployer_history_miner
from src.runtime.ingestion import MinedRecordIngestion


def test_later_targets_are_polled_even_when_new_names_are_prepended():
    values = ['a', 'b', 'c', 'd']
    take = rotating_selector(lambda: values, 2)
    assert take() == ['a', 'b']
    values.insert(0, 'new')
    assert take() == ['c', 'd']
    assert take() == ['new', 'a']


def test_balance_and_deployer_mining_rotate_beyond_first_request():
    class RPC:
        def __init__(self): self.calls = []
        async def request(self, method, params):
            self.calls.append((method, params))
            return {'value': [{'lamports': 1, 'owner': 'system'} for _ in params[0]]} if method == 'getMultipleAccounts' else []
    async def exercise():
        rpc = RPC()
        balances = account_balance_miner(rpc, lambda: ['a', 'b', 'c'], per_pass=2)
        await balances(); await balances()
        assert rpc.calls[1][1][0] == ['c', 'a']
        rpc.calls.clear()
        deployers = deployer_history_miner(rpc, lambda: ['a', 'b', 'c'], per_pass=2)
        await deployers(); await deployers()
        assert [params[0] for _, params in rpc.calls] == ['a', 'b', 'c', 'a']
    asyncio.run(exercise())


def test_channel_links_in_later_miner_records_are_not_discarded():
    found = []
    current = SimpleNamespace(channel_book=SimpleNamespace(harvest=lambda text, source: found.append(text)))
    MinedRecordIngestion._harvest_channels(current, 'public', [{'text': 'https://t.me/channel' + str(i)} for i in range(450)])
    assert len(found) == 450 and found[-1].endswith('channel449')
