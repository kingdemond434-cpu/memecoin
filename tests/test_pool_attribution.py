import base64
import struct
from types import SimpleNamespace

import pytest
from solders.pubkey import Pubkey

from src.chains.pump_route import TOKEN_PROGRAM, TOKEN_2022_PROGRAM, WSOL_MINT
from src.chains.pumpswap_route import POOL_DISCRIMINATOR, PUMPSWAP_PROGRAM, _POOL_LAYOUT
from src.chains.yellowstone_grpc import scoped_program_data
from src.research.pool_attribution import PoolAttribution, bind_trade, mint_decimals

POOL, BASE = str(Pubkey(bytes([7])*32)), str(Pubkey(bytes([8])*32))


def account():
    body = bytearray(POOL_DISCRIMINATOR)
    for name, width, kind in _POOL_LAYOUT:
        if kind == 'pubkey':
            body.extend(bytes(Pubkey.from_string(BASE if name == 'base_mint' else WSOL_MINT if name == 'quote_mint' else str(Pubkey(bytes([9])*32)))))
        else:
            body.extend(bytes(width))
    return {'owner': PUMPSWAP_PROGRAM, 'data': [base64.b64encode(body).decode(), 'base64']}


def mint(owner=TOKEN_PROGRAM):
    raw = bytearray(82)
    raw[44], raw[45] = 6, 1
    return {'owner': owner, 'data': [base64.b64encode(raw).decode(), 'base64']}


def event(index=1):
    return {'program': PUMPSWAP_PROGRAM, 'pool': POOL, 'signature': 'observed-sig', 'slot': 12,
            'type': 'token_trade', 'wallet': 'observed-wallet', 'side': 'buy', 'token': '',
            'token_amount': 1_000_000, 'actual_token_delta_raw': 1_000_000,
            'quote_amount_gross': 500_000_000, 'timestamp': 100., 'log_index': index,
            'source_commitment': 'processed'}


def rpc(owner=PUMPSWAP_PROGRAM, decimals=True):
    async def request(method, params):
        assert method == 'getMultipleAccounts' and params[1]['commitment'] == 'confirmed'
        if params[0] == [POOL]:
            value = account()
            value['owner'] = owner
        else:
            assert params[0] == [BASE]
            value = mint() if decimals else None
        return {'context': {'slot': 50}, 'value': [value]}
    return SimpleNamespace(request=request)


@pytest.mark.asyncio
async def test_unknown_pool_events_and_mapping_survive_restart_before_canonical_ack(tmp_path):
    path = tmp_path/'attribution.sqlite'
    book = PoolAttribution(path)
    await book.capture(event())
    await book.capture(event())
    await book.capture(event(2))
    assert book.report()['pending_events'] == 2
    rows = await book.fetch(rpc())
    assert book.lookup(POOL) is None
    assert len([x for x in rows if x['attribution_kind'] == 'raw_orphan']) == 2
    book.close()
    book = PoolAttribution(path)
    assert book.report()['pending_events'] == 2
    with pytest.raises(ValueError):
        book.acknowledge(rows, '')
    book.acknowledge(rows, 'canonical:original-and-mapping')
    mapping = book.lookup(POOL)
    assert mapping['base_mint'] == BASE and mapping['base_decimals'] == 6
    assert mapping['source_ref'] == 'canonical:original-and-mapping'
    book.close()
    book = PoolAttribution(path)
    recovered = await book.fetch(rpc())
    assert len(recovered) == 2 and all(x['late_attribution'] for x in recovered)
    row = recovered[0]
    assert row['event']['timestamp'] == 100.
    assert row['event']['available_at'] > row['event']['timestamp']
    assert row['event']['token'] == BASE and row['event']['actual_token_amount_ui'] == 1
    assert row['original_source_ref'] == 'canonical:original-and-mapping'
    book.acknowledge(recovered, 'canonical:late-derivation')
    assert book.report()['pending_events'] == 2  # canonical retention is not actual owner-loop delivery
    book.delivered([row['event_id'] for row in recovered])
    assert book.report()['pending_events'] == 0 and book.report()['recovered_events'] == 2
    book.acknowledge(recovered, 'canonical:repeated-ack')
    assert book.report()['recovered_events'] == 2
    book.close()


@pytest.mark.asyncio
async def test_foreign_owner_cannot_teach_pool_identity_and_canonical_failure_preserves_raw(tmp_path):
    book = PoolAttribution(tmp_path/'attribution.sqlite')
    await book.capture(event())
    rows = await book.fetch(rpc(owner=TOKEN_PROGRAM))
    assert not any(row['attribution_kind'] == 'identity' for row in rows)
    assert book.lookup(POOL) is None and book.report()['pending_events'] == 1
    assert any(row.get('reason') == 'OWNER_OR_LAYOUT_UNVERIFIED' for row in rows)
    book.acknowledge(rows, 'canonical:blocked')
    assert book.report()['mapping_gaps'] == 1
    book.close()


@pytest.mark.asyncio
async def test_missing_decimals_never_become_zero_or_stop_retrying(tmp_path):
    book = PoolAttribution(tmp_path/'attribution.sqlite')
    await book.capture(event())
    rows = await book.fetch(rpc(decimals=False))
    book.acknowledge(rows, 'canonical:mapping-only')
    bound = bind_trade(event(), book.lookup(POOL), available_at=book.lookup(POOL)['observed_at'])
    assert bound['actual_token_amount_ui'] is None and bound['price_sol_per_token'] is None
    assert book.db.execute('SELECT last_error FROM jobs').fetchone()[0] == 'MINT_DECIMALS_UNAVAILABLE'
    assert not any(row['attribution_kind'] == 'recovered_trade' for row in await book.fetch(rpc()))
    assert book.report()['pending_events'] == 1
    book.close()


@pytest.mark.asyncio
async def test_foreign_quote_observations_are_retained_without_touching_sol_state(tmp_path):
    from src.runtime.ingestion import MinedRecordIngestion
    from src.execution.jupiter_jito import USDC_MINT
    book = PoolAttribution(tmp_path/'attribution.sqlite')
    recorded = []
    desk = SimpleNamespace(pool_attribution=book, _stream_events={},
                           trade_evidence=SimpleNamespace(record=lambda kind, value: recorded.append((kind, value))))
    observed = {**event(), 'token': BASE, 'quote_mint': USDC_MINT}
    assert await MinedRecordIngestion._retain_foreign_quote_event(desk, observed)
    assert book.report()['pending_events'] == 1
    assert recorded[0][0] == 'non_sol_quote_observation'
    assert desk._stream_events['token_trade:observation_only_quote'] == 1
    assert not await MinedRecordIngestion._retain_foreign_quote_event(desk, {**observed, 'quote_mint': WSOL_MINT})
    book.close()


def test_mint_owner_layout_and_initialization_are_checked():
    assert mint_decimals(mint()) == 6
    assert mint_decimals(mint(TOKEN_2022_PROGRAM)) == 6
    assert mint_decimals(mint(PUMPSWAP_PROGRAM)) is None
    bad = mint()
    bad['data'][0] = base64.b64encode(bytes(165)).decode()
    assert mint_decimals(bad) is None


@pytest.mark.asyncio
async def test_recovery_uses_full_protocol_batch_without_dropping_the_next_pool(tmp_path):
    book = PoolAttribution(tmp_path/'attribution.sqlite')
    for index in range(1,102):
        book._capture({**event(), 'pool': str(Pubkey(bytes([index])*32))})
    calls = []
    async def request(method, params):
        assert method == 'getMultipleAccounts' and len(params[0]) <= 100
        calls.append(len(params[0]))
        values = [mint()] if params[0] == [BASE] else [account() for _ in params[0]]
        return {'context': {'slot': 50}, 'value': values}
    rows = await book.fetch(SimpleNamespace(request=request))
    assert calls == [100,1]
    assert sum(row['attribution_kind']=='identity' for row in rows) == 100
    book.acknowledge(rows, 'canonical:full-protocol-batch')
    assert book.report()['resolved_pools'] == 100
    calls.clear()
    rows = await book.fetch(SimpleNamespace(request=request))
    assert calls == [1,1]
    assert sum(row['attribution_kind']=='identity' for row in rows) == 1
    assert book.report()['pending_events'] == 101  # no owner-loop delivery ACK yet
    book.close()


@pytest.mark.asyncio
async def test_explicit_provider_batch_limit_is_learned_without_discarding_other_jobs(tmp_path):
    from src.chains.rpc_manager import RPCError
    path = tmp_path/'attribution.sqlite'
    book = PoolAttribution(path)
    for index in range(1,102):
        book._capture({**event(), 'pool': str(Pubkey(bytes([index])*32))})
    calls = []
    async def request(method, params):
        calls.append(len(params[0]))
        if len(params[0]) > 10:
            raise RPCError({'code': -32602, 'message': 'A maximum of 10 accounts is allowed.'})
        values = [mint()] if params[0] == [BASE] else [account() for _ in params[0]]
        return {'context': {'slot': 50}, 'value': values}
    rows = await book.fetch(SimpleNamespace(request=request))
    assert calls == [100,10,1]
    assert sum(row['attribution_kind']=='identity' for row in rows) == 10
    assert any(row.get('account_limit') == 10 for row in rows)
    book.acknowledge(rows, 'canonical:provider-limit-and-first-batch')
    assert book.db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 91
    book.close()
    book = PoolAttribution(path)
    calls.clear()
    await book.fetch(SimpleNamespace(request=request))
    assert calls == [10,1] and book.report()['pool_read_batch_limit'] == 10
    book.close()


@pytest.mark.asyncio
async def test_rate_refusal_does_not_become_split_retries(tmp_path):
    from src.chains.rpc_manager import RPCError
    book = PoolAttribution(tmp_path/'attribution.sqlite')
    await book.capture(event())
    calls = []
    async def request(method,params):
        calls.append(method)
        raise RPCError({'code': 429, 'message': 'rate limit; maximum of 1 accounts'})
    rows = await book.fetch(SimpleNamespace(request=request))
    assert len(calls) == 1
    assert not any(row['attribution_kind']=='provider_capacity' for row in rows)
    assert book.report()['pool_read_batch_limit'] == 100
    book.close()


def test_log_mention_is_not_emitter_identity_and_failed_scope_is_not_a_trade():
    expected, other = PUMPSWAP_PROGRAM, TOKEN_PROGRAM
    encoded = 'Program data: ' + base64.b64encode(b'trade').decode()
    logs = [f'Program {other} invoke [1]', encoded,
            f'Program {expected} invoke [2]', encoded, f'Program {expected} success',
            f'Program {other} success']
    assert list(scoped_program_data(logs, expected)) == [(3, b'trade')]
    logs[-1] = f'Program {other} failed: custom error'
    assert list(scoped_program_data(logs, expected)) == []
    assert list(scoped_program_data([encoded], expected)) == []
    assert list(scoped_program_data([f'Program {expected} invoke [1]', encoded], expected)) == []


def test_coverage_uses_current_unique_token_sample_not_repeated_miss_observations():
    from src.runtime.reporting import ReportingSurface
    known = {'a'}
    desk = SimpleNamespace(_discovered_pools={'a': 1., 'b': 1.}, _discovery_misses=500,
                           launch_census=SimpleNamespace(knows=lambda token: token in known))
    report = ReportingSurface.discovery_report(desk)
    assert report['coverage'] == .5 and report['missed_by_our_stream'] == 1
    known.add('b')
    assert ReportingSurface.discovery_report(desk)['coverage'] == 1.
    desk.launch_census = None
    assert ReportingSurface.discovery_report(desk)['coverage'] is None


def test_hot_actor_caches_record_observations_without_inventing_profitable_dna(tmp_path):
    from src.runtime.hot_state import HotState
    hot = HotState(archive_root=tmp_path/'spool')
    hot.observe_wallet_entry('buyer', now=10)
    hot.observe_wallet_entry('buyer', now=11)
    hot.observe_creator('creator', now=11)
    wallet = hot.wallets.peek('buyer')
    assert wallet.observed_entries == 2 and wallet.status == 'UNMEASURED'
    assert wallet.information_value == 0 and wallet.launches_seen == 0
    assert hot.creators.peek('creator').observed_creations == 1


def test_late_attribution_routes_only_to_research_history_not_hot_state():
    from src.runtime.ingestion import MinedRecordIngestion
    observations, histories = [], []
    desk = SimpleNamespace(dataset_builder=SimpleNamespace(record_market_observation=lambda *x: observations.append(x)),
        wallet_intel=SimpleNamespace(record_live_trade=lambda *x: histories.append(x)), _market_context={'sol': 100})
    record = {'provider': 'native_pool_attribution', 'attribution_kind': 'recovered_trade',
              'event_id': 'event', 'event': {**event(), 'token': BASE, 'available_at': 200}, 'original_source_ref': 'canonical:raw'}
    MinedRecordIngestion._ingest_mined_records(desk, 'chain:pool_attribution', [record])
    assert observations[0][1]['late_attribution'] is True
    assert histories[0][0] == BASE and desk._market_context == {'sol': 100}


@pytest.mark.asyncio
async def test_actual_owner_delivery_releases_only_retained_recovery_rows(tmp_path):
    from src.runtime.ingestion import MinedRecordIngestion
    book = PoolAttribution(tmp_path/'attribution.sqlite')
    await book.capture(event())
    book.acknowledge(await book.fetch(rpc()), 'canonical:original')
    recovered = await book.fetch(rpc())
    book.acknowledge(recovered, 'canonical:derived')
    desk = SimpleNamespace(pool_attribution=book, _market_context={},
        dataset_builder=SimpleNamespace(record_market_observation=lambda *args: None),
        wallet_intel=SimpleNamespace(record_live_trade=lambda *args: None))
    MinedRecordIngestion._ingest_mined_records(desk, 'chain:pool_attribution', recovered)
    await book.finish_delivery_acks()
    assert book.report()['pending_events'] == 0 and book.report()['recovered_events'] == 1
    book.close()


def test_identity_confirmation_honors_observation_and_registration_clocks():
    from src.research.identity_watch import IdentityWatch, Figure, Category, Verdict
    watch = IdentityWatch()
    watch.register(Figure('project', 'Project Example', Category.PROJECT, channels=('projectofficial',), channels_verified_at=90))
    watch.note_channel_message('projectofficial', ['mint'], at=101., available_at=110.)
    assert watch.assess('mint', name='Project Example', created_at=100., now=105.).verdict == Verdict.UNCORROBORATED
    assert watch.assess('mint', name='Project Example', created_at=100., now=110.).verdict == Verdict.ANNOUNCED
    # New identity knowledge must not authenticate a launch predating registration.
    assert watch.assess('mint', name='Project Example', created_at=80., now=110.).verdict == Verdict.UNCORROBORATED


def test_two_registered_figures_may_mention_the_same_mint_without_starving_each_other():
    from src.research.identity_watch import IdentityWatch, Figure, Category, Verdict
    watch = IdentityWatch()
    watch.register(Figure('one', 'First Figure', Category.PROJECT, channels=('firstofficial',)))
    watch.register(Figure('two', 'Second Figure', Category.PROJECT, channels=('secondofficial',)))
    watch.note_channel_message('firstofficial', ['mint'], at=101.)
    watch.note_channel_message('@secondofficial', ['mint'], at=102.)
    assert watch.assess('mint', name='Second Figure', created_at=100., now=110.).verdict == Verdict.ANNOUNCED


def test_owned_confirmation_after_creation_revises_only_a_later_observation():
    from src.research.identity_watch import IdentityWatch, Figure, Category, Verdict
    from src.runtime.ingestion import MinedRecordIngestion
    watch = IdentityWatch()
    watch.register(Figure('project', 'Project Example', Category.PROJECT, channels=('projectofficial',)))
    original = watch.assess('mint', name='Project Example', created_at=100., now=100.)
    rows = []
    desk = SimpleNamespace(identity_watch=watch, _identity_claims={'mint': original},
        launch_census=SimpleNamespace(_records={'mint': SimpleNamespace(detected_at=100.)}),
        dataset_builder=SimpleNamespace(record_market_observation=lambda *args: rows.append(args)))
    watch.note_channel_message('projectofficial', ['mint'], at=101., available_at=110.)
    MinedRecordIngestion._refresh_identity_mentions(desk, ['mint'], 110.)
    assert original.verdict == Verdict.UNCORROBORATED
    assert desk._identity_claims['mint'].verdict == Verdict.ANNOUNCED
    assert rows[0][1]['timestamp'] == 110. and rows[0][1]['available_at'] == 110.


def test_reviewed_domains_do_not_claim_account_ids_wallets_or_token_endorsements():
    from src.strategies.authenticity import load_entities
    from src.research.identity_watch import IdentityWatch
    entities = load_entities('config/entities.public.yaml')
    assert {entity.entity_id for entity in entities} == {'solana', 'meteora', 'raydium'}
    assert all(not entity.accounts and not entity.known_wallets and entity.verified_from for entity in entities)
    watch = IdentityWatch()
    watch.load_yaml('config/figures.yaml')
    assert watch.figures['meteora'].channels == ('meteora_dev',)
    assert watch.report()['status'] == 'PARTIAL'
    assert watch.report()['figures_without_channels'] > 0
    assert watch.figures['meteora'].channels_source_ref.startswith('https://docs.meteora.ag/')
