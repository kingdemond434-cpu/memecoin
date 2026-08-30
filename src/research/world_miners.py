"""Quota-free world-scale public sources.

What separates these from the miners in ``web_miners`` is the funding model
rather than the subject. Reddit, YouTube and the rest meter access with a key
and a monthly allowance that can be exhausted -- and on 2026-08-29 the desk's
Helius allowance WAS exhausted (HTTP 429, "max usage reached"), which is what
made T0 chain facts unavailable and blocked four feature groups at once.

The sources here are structurally different: they publish files or open
streams that anyone may read without registering, so there is no bucket to
empty. That is the property worth building on, not the subject matter.

Measured from this box on 2026-08-29:

  data.gdeltproject.org        HTTP 200   raw files reachable
  api.gdeltproject.org         HTTP 000   NOT reachable from this host

which is why the GDELT miner below reads the raw 15-minute files rather than
the DOC API. That is not merely a preference for open data: the API is not
reachable from here at all, so a miner built on it would have been dead on
arrival and reported as a healthy source producing zero.

Honest yield expectation, measured rather than hoped: a single 15-minute GKG
slice sampled 2026-08-29 carried 467 records and ZERO mentioning Solana,
memecoins or crypto. GDELT is worldwide MAINSTREAM news. It is a narrative
DISCOVERY layer -- what the world is talking about, in what language, before
a token is named after it -- and it is not a memecoin firehose. A miner that
returns nothing most passes is behaving correctly here, and the report must
not read that as a fault.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: The 15-minute publication manifest. Plain text, three fields per line:
#: size, md5, url.
GDELT_LASTUPDATE_URL = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"

#: How long an archive fetch may take. Generous by JSON standards and sized
#: for a multi-megabyte file on a shared box: the GKG slice measured ~2.08 MB
#: here and the first live pass died on the shared client's 10s limit. Still
#: bounded, because a stalled fetch must not pin the miner pool.
ARCHIVE_TIMEOUT_S = 90.0

#: GDELT publishes its manifest with http:// URLs. Fetched over https here
#: because plain http returned zero bytes from this host while the https form
#: of the identical URL returned 200 -- measured, not assumed.
_HTTP_PREFIX = "http://"
_HTTPS_PREFIX = "https://"

#: GKG v2.1 column positions actually read. Named rather than indexed inline
#: so a schema change fails somewhere legible instead of silently shifting
#: every field by one.
GKG_RECORD_ID = 0
GKG_DATE = 1
GKG_SOURCE_NAME = 3
GKG_DOCUMENT_ID = 4
GKG_THEMES = 7
GKG_PERSONS = 11
GKG_ORGANISATIONS = 13
GKG_MIN_COLUMNS = 14

#: Lowercased substrings that make a GKG row worth keeping. Deliberately
#: broad: this layer exists to notice a narrative BEFORE somebody mints a
#: coin named after it, so it watches the subject matter memecoins are made
#: from rather than only the word "memecoin".
DEFAULT_TERMS: Sequence[str] = (
    "solana", "memecoin", "meme coin", "pump.fun", "pumpfun", "crypto",
    "cryptocurrency", "bitcoin", "ethereum", "token launch", "airdrop",
    "dogecoin", "stablecoin", "blockchain",
)


def _https(url: str) -> str:
    return (_HTTPS_PREFIX + url[len(_HTTP_PREFIX):]
            if url.startswith(_HTTP_PREFIX) else url)


def parse_lastupdate(body: str) -> Dict[str, str]:
    """Map GDELT's manifest to {kind: https url}.

    Returns an empty mapping rather than raising for an unrecognised body:
    an upstream format change should stop this miner producing, not take the
    pool down with it.
    """
    feeds: Dict[str, str] = {}
    for line in (body or "").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        url = _https(parts[2])
        lowered = url.lower()
        for kind in ("gkg", "mentions", "export"):
            if f".{kind}." in lowered:
                feeds.setdefault(kind, url)
                break
    return feeds


def parse_gkg(payload: bytes, terms: Sequence[str] = DEFAULT_TERMS,
              max_records: int = 200) -> List[Dict[str, Any]]:
    """Rows from one GKG slice that mention any term, as flat records.

    Filtered here rather than downstream because a slice is a few thousand
    rows every fifteen minutes and almost none of it concerns this desk;
    carrying the whole thing through the pool would spend memory on a bound
    that has nothing to do with the signal.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, ValueError) as exc:
        raise RuntimeError(f"GKG payload was not a zip: {exc}") from exc
    names = archive.namelist()
    if not names:
        return []
    text = archive.read(names[0]).decode("utf-8", "replace")
    wanted = tuple(term.lower() for term in terms)
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if len(records) >= max_records:
            break
        lowered = line.lower()
        matched = [term for term in wanted if term in lowered]
        if not matched:
            continue
        columns = line.split("\t")
        if len(columns) < GKG_MIN_COLUMNS:
            continue
        records.append({
            "venue": "gdelt_gkg",
            "id": columns[GKG_RECORD_ID],
            "observed_at_raw": columns[GKG_DATE],
            "source": columns[GKG_SOURCE_NAME],
            "url": columns[GKG_DOCUMENT_ID],
            # Semicolon-delimited in GDELT; split so an entity resolver does
            # not have to know this file format.
            "themes": [item for item in columns[GKG_THEMES].split(";") if item],
            "persons": [item for item in columns[GKG_PERSONS].split(";") if item],
            "organisations": [item for item in columns[GKG_ORGANISATIONS].split(";")
                              if item],
            # Which term matched, so a source's lead time can later be
            # measured per subject rather than in aggregate.
            "matched_terms": matched,
            "data_status": "OK",
        })
    return records


def gdelt_world_news_miner(client: Any, terms: Sequence[str] = DEFAULT_TERMS
                           ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Worldwide mainstream news mentioning this desk's subjects.

    Two fetches per pass: the manifest, then the current GKG slice. No key,
    no account and no monthly allowance -- the property this source is here
    for. Returning zero records is the NORMAL case and is not a failure.
    """
    async def fetch() -> List[Dict[str, Any]]:
        # The manifest is tiny, but it is fetched on this miner's own timeout
        # rather than the shared client's 10s. Measured: the first live pass
        # failed with "timeout after 10.0s" on THIS call, not on the archive
        # -- the TLS handshake to a new host under box contention does not
        # reliably complete inside a limit tuned for warm JSON endpoints.
        body = (await _get_bytes(client, GDELT_LASTUPDATE_URL)).decode(
            "utf-8", "replace")
        feeds = parse_lastupdate(body)
        url = feeds.get("gkg")
        if not url:
            raise RuntimeError("GDELT manifest carried no gkg entry")
        payload = await _get_bytes(client, url)
        return parse_gkg(payload, terms)

    return fetch


async def _get_bytes(client: Any, url: str,
                     timeout_s: float = ARCHIVE_TIMEOUT_S) -> bytes:
    """Fetch raw bytes for an archive, on a timeout that fits the payload.

    The shared HTTP client decodes to text, which corrupts a zip. A miner
    owns how it talks to its own endpoint (see MinerSpec.endpoint), so this
    reaches for the client's session directly rather than widening a
    transport every other caller uses as text.

    The timeout is stated here rather than inherited. The shared client
    allows 10s, which is right for a JSON call and wrong for a 2 MB archive:
    measured on this box, the GKG slice is ~2.08 MB and the first live pass
    failed with "TransportError: timeout after 10.0s" while three unrelated
    JSON miners timed out in the same window. A timeout shorter than the
    payload can be fetched in is a miner that reports ERROR forever while
    the endpoint is healthy.
    """
    import aiohttp

    session = await client.session()
    async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status} from {url.split('?')[0]}")
        return await response.read()
