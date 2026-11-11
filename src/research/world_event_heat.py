"""Which people and organisations the world suddenly talks about, and which new
launches are named after them.

Event-driven sniping (ledger row 17) needs two things the desk did not have:

1. A spike detector over ALL of worldwide mainstream news, not only rows that
   mention crypto. A coin named after a footballer is minted because of the
   match, and the match is never reported as a crypto story. The GDELT miner
   keeps its crypto filter for the corpus; this layer counts every person and
   organisation in every 15-minute GKG slice.
2. A matcher at launch time: is this new token named after something that is
   spiking right now, and how many seconds after the spike began was it
   minted? That lag is the whole edge -- the first launches after a world
   event are the ones that run, and the hundredth copy is exit liquidity.

Spikes are scored as a Poisson surprise against an exponentially weighted
per-slice mean for each entity (decayed mention sum over decayed slice count,
so absences count as zero and a fresh desk is unbiased from its first hour),
so a name that is always in the news (a head of state)
does not read as an event, and one that goes from zero to forty mentions in a
quarter hour does.

Nothing here buys anything. A match is a feature and a recorded fact about the
launch; whether it predicts a runner is for the forward ledger to measure.
Public data only (GDELT's published raw files).
"""
from __future__ import annotations

import io
import math
import re
import threading
import time
import unicodedata
import zipfile
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

GKG_PERSONS = 11
GKG_ORGANISATIONS = 13
GKG_MIN_COLUMNS = 14

#: Baseline half-life. A day of 15-minute slices: long enough that a story
#: running all afternoon is still a surprise at its start, short enough that
#: last month's news is forgotten.
BASELINE_HALF_LIFE_S = 24 * 3600.0
#: Poisson z-score at which an entity counts as spiking.
HOT_Z = 4.0
#: Minimum mentions in one slice; below this a z-score is mostly arithmetic.
HOT_MIN_COUNT = 5
#: How long an entity stays matchable after its last hot slice. Launches named
#: after an event keep arriving for hours; after that it is a copy, not a snipe.
HOT_TTL_S = 6 * 3600.0
#: Slices folded in before anything may be called hot; the first slice has
#: no baseline and every name in it would otherwise look new.
WARMUP_SLICES = 4
#: Bounded memory: least-mentioned entities are dropped beyond this.
MAX_TRACKED = 200_000
#: Name tokens shorter than this never match on their own ("al", "fc", "the").
MIN_TOKEN_LEN = 4
#: Words too common in token names to identify an event by themselves.
STOPWORDS = frozenset("""
the and for with from that this coin token meme inu doge pepe moon pump sol
solana elon king queen president party club city united real official new
""".split())

_SPLIT = re.compile(r"[^0-9a-z]+")


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return " ".join(part for part in _SPLIT.split(folded.lower()) if part)


def gkg_entities(payload: bytes) -> Counter:
    """Mentions per person/organisation across a whole GKG slice.

    Counted once per article (a name repeated in one article is one mention),
    so a single long article cannot fake a spike.
    """
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = archive.namelist()
    if not names:
        return Counter()
    counts: Counter = Counter()
    with archive.open(names[0]) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
            columns = line.rstrip("\n").split("\t")
            if len(columns) < GKG_MIN_COLUMNS:
                continue
            seen = set()
            for column in (GKG_PERSONS, GKG_ORGANISATIONS):
                for item in columns[column].split(";"):
                    entity = normalise(item.split(",")[0])
                    if entity:
                        seen.add(entity)
            counts.update(seen)
    return counts


class EventHeat:
    """Per-entity decayed baselines, current spikes, and launch matches."""

    def __init__(self, *, half_life_s: float = BASELINE_HALF_LIFE_S, hot_z: float = HOT_Z,
                 hot_min_count: int = HOT_MIN_COUNT, hot_ttl_s: float = HOT_TTL_S,
                 max_tracked: int = MAX_TRACKED, clock=time.time):
        self.half_life_s, self.hot_z, self.hot_min_count = half_life_s, hot_z, hot_min_count
        self.hot_ttl_s, self.max_tracked, self.clock = hot_ttl_s, max_tracked, clock
        self.lock = threading.Lock()
        self.baseline: Dict[str, Tuple[float, float]] = {}   # entity -> (decayed mentions, at)
        self.weight, self.weight_at = 0.0, None              # decayed slice count
        self.hot: Dict[str, Dict[str, Any]] = {}             # entity -> spike record
        self.slices = 0
        self.last_slice_at: Optional[float] = None
        self.launches_checked = 0
        self.launches_matched = 0
        self.recent_matches: deque = deque(maxlen=200)
        self.lags: deque = deque(maxlen=2000)

    # -- the news side ----------------------------------------------------

    def observe_slice(self, counts: Counter, at: Optional[float] = None) -> List[str]:
        """Fold one slice in. Returns the entities that became or stayed hot."""
        at = self.clock() if at is None else at
        spiking = []
        with self.lock:
            decay = lambda since: 0.5 ** ((at - since) / self.half_life_s)
            slices = self.weight * decay(self.weight_at) if self.weight_at is not None else 0.0
            # Scored against the mean BEFORE this slice is folded in, so a
            # spike cannot dilute its own surprise.
            for entity, count in counts.items():
                if self.slices < WARMUP_SLICES or slices <= 0:
                    break
                mentions, since = self.baseline.get(entity, (0.0, at))
                expected = mentions * decay(since) / slices
                z = (count - expected) / math.sqrt(expected + 1.0)
                if count >= self.hot_min_count and z >= self.hot_z:
                    record = self.hot.get(entity)
                    if record is None or at - record["last_hot_at"] > self.hot_ttl_s:
                        record = {"entity": entity, "first_hot_at": at, "peak_z": z, "peak_count": count}
                        self.hot[entity] = record
                    record["last_hot_at"] = at
                    record["peak_z"] = max(record["peak_z"], z)
                    record["peak_count"] = max(record["peak_count"], count)
                    spiking.append(entity)
            for entity, count in counts.items():
                mentions, since = self.baseline.get(entity, (0.0, at))
                self.baseline[entity] = (mentions * decay(since) + count, at)
            self.weight, self.weight_at = slices + 1.0, at
            self.hot = {key: value for key, value in self.hot.items()
                        if at - value["last_hot_at"] <= self.hot_ttl_s}
            if len(self.baseline) > self.max_tracked:
                keep = sorted(self.baseline.items(),
                              key=lambda item: item[1][0] * decay(item[1][1]),
                              reverse=True)[: self.max_tracked * 3 // 4]
                self.baseline = dict(keep)
            self.slices += 1
            self.last_slice_at = at
        return spiking

    def observe_gkg(self, payload: bytes, at: Optional[float] = None) -> List[str]:
        return self.observe_slice(gkg_entities(payload), at)

    # -- the launch side --------------------------------------------------

    def match(self, mint: str, name: str = "", symbol: str = "",
              launched_at: Optional[float] = None) -> List[Dict[str, Any]]:
        """Hot entities this launch is named after, with the lag since each began.

        An entity matches when every one of its distinctive words appears in
        the launch's name or symbol ("lamine yamal" -> "YAMAL" matches on the
        surname; "real madrid" needs "madrid"). Words that are too short or
        too common in token names never match alone.
        """
        launched_at = self.clock() if launched_at is None else launched_at
        words = set(normalise(f"{name} {symbol}").split())
        joined = normalise(f"{name}{symbol}").replace(" ", "")
        found = []
        with self.lock:
            self.launches_checked += 1
            for entity, record in self.hot.items():
                if launched_at - record["last_hot_at"] > self.hot_ttl_s:
                    continue
                if launched_at < record["first_hot_at"]:
                    continue        # minted before the spike: not caused by it
                distinctive = [part for part in entity.split()
                               if len(part) >= MIN_TOKEN_LEN and part not in STOPWORDS]
                if not distinctive:
                    continue
                if not (any(part in words for part in distinctive)
                        or (len(distinctive[-1]) >= 6 and distinctive[-1] in joined)):
                    continue
                found.append({"entity": entity, "lag_s": launched_at - record["first_hot_at"],
                              "peak_z": round(record["peak_z"], 2),
                              "peak_count": record["peak_count"]})
            if found:
                found.sort(key=lambda item: (-item["peak_z"], item["lag_s"]))
                self.launches_matched += 1
                self.lags.append(found[0]["lag_s"])
                self.recent_matches.append({"mint": mint, "name": str(name)[:64],
                                            "symbol": str(symbol)[:16], "at": launched_at,
                                            **found[0]})
        return found

    def report(self) -> Dict[str, Any]:
        with self.lock:
            lags = sorted(self.lags)
            now = self.clock()
            return {
                "schema": "world_event_heat/1",
                "status": ("NOT_OBSERVED" if not self.slices else
                           "STALE" if now - (self.last_slice_at or 0) > 3600 else "OK"),
                "slices": self.slices, "last_slice_at": self.last_slice_at,
                "tracked_entities": len(self.baseline),
                "hot_now": sorted(({"entity": key, "peak_z": round(value["peak_z"], 2),
                                    "first_hot_at": value["first_hot_at"]}
                                   for key, value in self.hot.items()),
                                  key=lambda item: -item["peak_z"])[:25],
                "launches_checked": self.launches_checked,
                "launches_matched": self.launches_matched,
                "match_lag_s_median": lags[len(lags) // 2] if lags else None,
                "recent_matches": list(self.recent_matches)[-20:],
                "execution_authority": False,
            }


#: The desk's one instance: the GDELT miner feeds it, the launch path reads it.
WORLD_EVENT_HEAT = EventHeat()
