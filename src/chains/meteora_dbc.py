"""Native DBC launch discovery from the pinned official SDK IDL.

Argument/account interpretation is independent of donor execution code.
No config fee, decimals, transferability or curve liquidity is guessed.
"""
from collections import Counter
from solders.pubkey import Pubkey

from src.chains.idl import load_idl
from src.chains.launchlab import LaunchLabMonitor, _Arguments, TOKEN_PROGRAM, TOKEN_2022_PROGRAM

DBC_PROGRAM = 'dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN'
CREATE_NAMES = ('initialize_virtual_pool_with_spl_token',
                'initialize_virtual_pool_with_token2022',
                'initialize_virtual_pool_with_token2022_transfer_hook')


class MeteoraDBCMonitor(LaunchLabMonitor):
    """Reuse the reviewed successful-transaction/ALT/CPI/timing envelope."""
    program_id = DBC_PROGRAM

    def __init__(self, stream, callback):
        self.callback = callback
        idl = load_idl('meteora_dbc')
        if idl['address'] != DBC_PROGRAM:
            raise ValueError('DBC IDL program identity changed')
        self.schemas = {bytes(row['discriminator']): row for row in idl['instructions']
                        if row['name'] in CREATE_NAMES}
        self.types = {row['name']: row['type'] for row in idl['types']}
        self.matched, self.rejected = Counter(), Counter()
        stream.on('transaction', self._on_transaction)

    def decode(self, keys, accounts, data):
        schema = self.schemas.get(data[:8])
        if schema is None:
            return None
        if len(accounts) < len(schema['accounts']):
            raise ValueError('truncated DBC account list')
        named = {}
        for offset, item in enumerate(schema['accounts']):
            index = accounts[offset]
            if type(index) is not int or not 0 <= index < len(keys):
                raise ValueError('invalid DBC account index')
            value = keys[index]
            Pubkey.from_string(value)
            if item.get('address') and item['address'] != value:
                raise ValueError('DBC fixed account mismatch')
            named[item['name']] = value
        if (named['program'] != DBC_PROGRAM or named['token_quote_program'] not in {TOKEN_PROGRAM, TOKEN_2022_PROGRAM}
                or named['base_mint'] == named['quote_mint']):
            raise ValueError('DBC asset/program identity mismatch')
        reader = _Arguments(data[8:], self.types)
        args = {item['name']: reader.read(item['type']) for item in schema['args']}
        if reader.offset != len(reader.raw):
            raise ValueError('unreviewed DBC argument tail')
        metadata = args['params']
        return {'type': 'token_launch', 'chain': 'solana', 'program': DBC_PROGRAM,
                'venue': 'meteora_dbc', 'instruction': schema['name'],
                'token': named['base_mint'], 'creator': named['creator'], 'payer': named['payer'],
                'pool': named['pool'], 'quote_mint': named['quote_mint'],
                'base_vault': named['base_vault'], 'quote_vault': named['quote_vault'],
                'base_token_program': named['token_program'], 'quote_token_program': named['token_quote_program'],
                'config': named['config'], 'name': metadata['name'], 'symbol': metadata['symbol'],
                'uri': metadata['uri'], 'token_decimals': None, 'quote_decimals': None,
                'transfer_hook_program': named.get('transfer_hook_program'),
                'data_status': 'OBSERVED_NATIVE_INITIALIZATION',
                'execution_support': 'DATA_BLOCKED_NATIVE_CURVE_EXECUTION',
                'protocol_ref': 'idl/meteora_dbc.json'}

    def decoder_report(self):
        return {'status': 'WIRED', 'program': DBC_PROGRAM, 'matched': dict(self.matched),
                'rejected': dict(self.rejected), 'config_scope': 'all observed configurations',
                'execution_support': 'DATA_BLOCKED_NATIVE_CURVE_EXECUTION',
                'transfer_hooks': 'observed program identity; transferability not established'}
