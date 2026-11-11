"""Event-driven sniping: spikes over all world news, matched to new launches."""
import asyncio
import io
import zipfile
from collections import Counter

from src.research.world_event_heat import EventHeat, gkg_entities, normalise

SLICE = 15 * 60


def _gkg(rows):
    lines = []
    for index, (persons, orgs) in enumerate(rows):
        columns = [""] * 14
        columns[0] = f"r{index}"
        columns[11] = ";".join(persons)
        columns[13] = ";".join(orgs)
        lines.append("\t".join(columns))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("slice.gkg.csv", "\n".join(lines))
    return buffer.getvalue()


def _warm(heat, start=0.0, slices=96):
    for index in range(slices):
        heat.observe_slice(Counter({"joe biden": 40, "fc barcelona": 3}), start + index * SLICE)
    return start + slices * SLICE


def test_a_name_always_in_the_news_is_not_an_event():
    heat = EventHeat()
    at = _warm(heat)
    assert heat.observe_slice(Counter({"joe biden": 44}), at) == []


def test_a_sudden_name_spikes_and_a_launch_named_after_it_matches_with_its_lag():
    heat = EventHeat()
    at = _warm(heat)
    assert heat.observe_slice(Counter({"lamine yamal": 40, "joe biden": 41}), at) == ["lamine yamal"]
    matches = heat.match("mintA", "Yamal Magic", "YAMAL", at + 90)
    assert matches and matches[0]["entity"] == "lamine yamal" and matches[0]["lag_s"] == 90
    report = heat.report()
    assert report["launches_matched"] == 1 and report["match_lag_s_median"] == 90
    assert report["recent_matches"][0]["mint"] == "mintA"


def test_common_or_short_words_never_match_alone():
    heat = EventHeat()
    at = _warm(heat)
    heat.observe_slice(Counter({"real madrid": 30, "al": 30}), at)
    assert heat.match("m", "Real Pepe", "REAL", at + 5) == []
    assert heat.match("m", "AL token", "AL", at + 5) == []
    assert heat.match("m", "Madrid Moon", "MAD", at + 5)[0]["entity"] == "real madrid"


def test_launches_before_the_spike_or_after_its_ttl_do_not_match():
    heat = EventHeat(hot_ttl_s=3600)
    at = _warm(heat)
    heat.observe_slice(Counter({"taylor swift": 50}), at)
    assert heat.match("early", "Swift", "SWIFT", at - 60) == []
    assert heat.match("late", "Swift", "SWIFT", at + 7200) == []
    assert heat.match("ok", "Swift", "SWIFT", at + 600)


def test_a_joined_symbol_matches_a_long_surname():
    heat = EventHeat()
    at = _warm(heat)
    heat.observe_slice(Counter({"cristiano ronaldo": 30}), at)
    assert heat.match("m", "GoRonaldoGo", "GRG", at + 1)


def test_gkg_entities_count_once_per_article_and_strip_offsets():
    payload = _gkg([(["Lamine Yamal,120", "Lamine Yamal,400"], ["FC Barcelona,10"]),
                    (["Lamine Yamal,5"], [])])
    counts = gkg_entities(payload)
    assert counts["lamine yamal"] == 2 and counts["fc barcelona"] == 1


def test_normalise_folds_accents_and_punctuation():
    assert normalise("Kylian Mbappé!") == "kylian mbappe"


def test_the_gdelt_miner_feeds_every_entity_not_only_crypto_rows(monkeypatch):
    from src.research import world_miners

    payload = _gkg([(["Lamine Yamal"], [])] * 6)
    heat = EventHeat(clock=lambda: 0.0)           # the miner stamps slices with the heat's clock
    _warm(heat, start=-96 * SLICE)

    async def fake_get(client, url, timeout_s=0):
        if url == world_miners.GDELT_LASTUPDATE_URL:
            return b"1 x http://data.gdeltproject.org/gdeltv2/20261111.gkg.csv.zip\n"
        return payload

    monkeypatch.setattr(world_miners, "_get_bytes", fake_get)
    fetch = world_miners.gdelt_world_news_miner(object(), heat=heat)
    rows = asyncio.run(fetch())
    assert rows == []                                  # nothing crypto in the corpus
    assert heat.slices == 97 and "lamine yamal" in heat.hot
    asyncio.run(fetch())                               # same slice again
    assert heat.slices == 97                           # not double-counted
