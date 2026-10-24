import asyncio
import json

from src.research.regional_miners import LadderFetcher, venue_ticker_miner
from src.research.source_substitution import Endpoint, SubstitutionRegistry
from src.research.telegram_miners import ChannelBook, extract_handles


def test_public_channel_admission_exceeds_former_limits_and_survives_restart(tmp_path):
    path = str(tmp_path / 'channels.json')
    book = ChannelBook(path)
    for index in range(4100):
        assert book.observe('channel' + str(index), 'public:origin', now=index + 1)
    for index in range(650):
        book.mark_verified('channel' + str(index), 1)
    assert len(book.verified()) == 650
    assert len(book.pending()) == 3450
    assert book.save()
    restored = ChannelBook(path)
    assert len(restored.channels) == 4100
    assert restored.report()['candidate_admission_limit'] is None
    assert extract_handles(' '.join('https://t.me/channel' + str(i) for i in range(80))) == ['channel' + str(i) for i in range(80)]


def test_failed_candidate_does_not_starve_new_channel_verification(tmp_path):
    book = ChannelBook(str(tmp_path / 'channels.json'))
    book.observe('offline', now=1)
    book.observe('newone', now=2)
    book.note_failure('offline', 'temporary transport outage')
    assert book.pending()[0] == 'newone'


def test_regional_venues_rotate_even_when_first_venue_is_healthy_and_keep_all_symbols():
    registry = SubstitutionRegistry()
    registry.declare('regional', [Endpoint('kr', 'https://kr.test', region='kr', shape='upbit_markets'),
                                  Endpoint('jp', 'https://jp.test', region='jp', shape='upbit_markets')])
    class Client:
        async def get(self, url, **kwargs):
            return 200, json.dumps([{'market': 'KRW-ALT' + str(i)} for i in range(700)]), {}
    fetch = venue_ticker_miner(LadderFetcher(Client(), registry), 'regional')
    async def observations():
        first, second = await fetch(), await fetch()
        assert first[0]['venue'] == 'kr' and second[0]['venue'] == 'jp'
        assert len(first) == len(second) == 701
        assert first[-1]['symbol'] == 'KRW-ALT699'
    asyncio.run(observations())
