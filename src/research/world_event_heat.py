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
#: Minimum expected mentions per slice for any entity. A name never seen
#: before has a zero baseline, so without a floor five mentions would score
#: z=5; with it a new name needs about ten mentions in one slice.
EXPECTED_FLOOR = 1.0
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


def gkg_entities(payload: bytes) -> Tuple[Counter, Dict[str, str]]:
    """Mentions per person/organisation across a whole GKG slice, and each kind.

    Counted once per article (a name repeated in one article is one mention),
    so a single long article cannot fake a spike.
    """
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = archive.namelist()
    counts: Counter = Counter()
    kinds: Dict[str, str] = {}
    if not names:
        return counts, kinds
    with archive.open(names[0]) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
            columns = line.rstrip("\n").split("\t")
            if len(columns) < GKG_MIN_COLUMNS:
                continue
            seen = set()
            for column, kind in ((GKG_PERSONS, "person"), (GKG_ORGANISATIONS, "organisation")):
                for item in columns[column].split(";"):
                    entity = normalise(item.split(",")[0])
                    if entity:
                        seen.add(entity)
                        kinds.setdefault(entity, kind)
            counts.update(seen)
    return counts, kinds


_SLICE_STAMP = re.compile(r"/(\d{14})\.gkg\.csv\.zip$")


def slice_time(url: str) -> Optional[float]:
    """GDELT's own timestamp for a slice, from its file name (UTC)."""
    found = _SLICE_STAMP.search(str(url))
    if not found:
        return None
    import calendar
    return float(calendar.timegm(time.strptime(found.group(1), "%Y%m%d%H%M%S")))


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
        self.kinds: Dict[str, str] = {}                      # entity -> person/organisation
        self.last_hot_count = 0
        self.last_new_hot_count = 0
        self.last_parse_seconds: Optional[float] = None
        self.last_detection_delay_s: Optional[float] = None
        self.slices = 0
        self.last_slice_at: Optional[float] = None
        self.launches_checked = 0
        self.launches_matched = 0
        self.recent_matches: deque = deque(maxlen=200)
        self.lags: deque = deque(maxlen=2000)

    # -- the news side ----------------------------------------------------

    def observe_slice(self, counts: Counter, at: Optional[float] = None,
                      kinds: Optional[Dict[str, str]] = None) -> List[str]:
        """Fold one slice in. Returns the entities that became or stayed hot.

        `at` is the slice's OWN time, so a spike is dated when the news broke,
        not when this desk got round to parsing it. The new state is built
        without the lock (this is the only writer) and swapped in under it,
        so a launch being matched never waits on a 200k-entity pass.
        """
        at = self.clock() if at is None else at
        decay = lambda since: 0.5 ** ((at - since) / self.half_life_s)
        slices = self.weight * decay(self.weight_at) if self.weight_at is not None else 0.0
        baseline = dict(self.baseline)
        hot = {key: dict(value) for key, value in self.hot.items()}
        spiking, new = [], 0
        scoring = self.slices >= WARMUP_SLICES and slices > 0
        for entity, count in counts.items():
            mentions, since = baseline.get(entity, (0.0, at))
            if scoring and count >= self.hot_min_count:
                # Scored against the mean BEFORE this slice is folded in, so
                # a spike cannot dilute its own surprise.
                expected = max(EXPECTED_FLOOR, mentions * decay(since) / slices)
                z = (count - expected) / math.sqrt(expected + 1.0)
                if z >= self.hot_z:
                    record = hot.get(entity)
                    if record is None or at - record["last_hot_at"] > self.hot_ttl_s:
                        record = {"entity": entity, "first_hot_at": at, "peak_z": z,
                                  "peak_count": count}
                        hot[entity] = record
                        new += 1
                    record["last_hot_at"] = at
                    record["peak_z"] = max(record["peak_z"], z)
                    record["peak_count"] = max(record["peak_count"], count)
                    spiking.append(entity)
            baseline[entity] = (mentions * decay(since) + count, at)
        hot = {key: value for key, value in hot.items() if at - value["last_hot_at"] <= self.hot_ttl_s}
        if len(baseline) > self.max_tracked:
            keep = sorted(baseline.items(), key=lambda item: item[1][0] * decay(item[1][1]),
                          reverse=True)[: self.max_tracked * 3 // 4]
            baseline = dict(keep)
        merged_kinds = {**self.kinds, **(kinds or {})}
        if len(merged_kinds) > len(baseline) + len(hot) + 10_000:
            merged_kinds = {key: merged_kinds[key] for key in set(baseline) | set(hot)
                            if key in merged_kinds}
        with self.lock:
            self.baseline, self.hot, self.kinds = baseline, hot, merged_kinds
            self.weight, self.weight_at = slices + 1.0, at
            self.slices += 1
            self.last_slice_at = at
            self.last_hot_count, self.last_new_hot_count = len(spiking), new
        return spiking

    def observe_gkg(self, payload: bytes, at: Optional[float] = None) -> List[str]:
        began = time.perf_counter()
        counts, kinds = gkg_entities(payload)
        result = self.observe_slice(counts, at, kinds)
        self.last_parse_seconds = time.perf_counter() - began
        self.last_detection_delay_s = (self.clock() - at) if at is not None else None
        return result

    # -- the launch side --------------------------------------------------

    def match(self, mint: str, name: str = "", symbol: str = "",
              launched_at: Optional[float] = None) -> List[Dict[str, Any]]:
        """Hot entities this launch is named after, with the lag since each began.

        A PERSON matches on the surname (the last distinctive word: "lamine
        yamal" -> "YAMAL"), also inside a joined name when it is 6+ letters
        ("GoRonaldoGo"). An ORGANISATION needs every distinctive word, so
        "white house" does not match WHITE CAT. Words that are too short or too
        common in token names are never distinctive. Unknown kinds are
        treated as organisations (the stricter rule).
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
                if self.kinds.get(entity) == "person":
                    surname = distinctive[-1]
                    if not (surname in words or (len(surname) >= 6 and surname in joined)):
                        continue
                elif not all(part in words or (len(part) >= 6 and part in joined)
                             for part in distinctive):
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
                # Measured false-positive pressure and cost, per slice.
                "hot_last_slice": self.last_hot_count,
                "new_hot_last_slice": self.last_new_hot_count,
                "last_parse_seconds": self.last_parse_seconds,
                # Slice time -> when this desk had it scored. GDELT's own
                # publication lag plus ours; the floor on any snipe it drives.
                "last_detection_delay_s": self.last_detection_delay_s,
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
