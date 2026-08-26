"""Probe candidate public endpoints and report which are real.

The source registry shipped as a seed with placeholder names in it, and the
obvious way to grow it -- writing a few hundred plausible URLs -- produces a
forest that looks impressive and is mostly fiction. A declaration that names an
endpoint nobody checked is worse than no declaration: the mesh reports it
NO_FETCHER or DEAD, an operator assumes it needs credentials, and the coverage
number stays wrong in the flattering direction.

So candidates are probed before they are declared. This writes nothing on its
own; it prints a verdict per endpoint, and only endpoints that actually
answered are worth putting in config/sources.yaml.

    python tools/verify_sources.py            # probe everything
    python tools/verify_sources.py --json     # machine-readable
    python tools/verify_sources.py --emit     # YAML for the ones that answered

Run it on the node the desk runs on. A sandbox with an egress allowlist will
report almost everything unreachable, and that verdict is about the sandbox
rather than about the endpoints -- which is precisely why the result is not
committed from wherever this happens to be executed.

Every endpoint here is a documented public interface requiring no account and
no access-control bypass: platform firehoses and public timelines, open news
and civic data APIs, and site-published feeds. Nothing here reads a private
channel, and nothing here should ever be extended to.
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple

TIMEOUT = 12
USER_AGENT = "memecoin-source-verifier/1.0 (+public endpoint reachability check)"

# (id, kind, language, region, url)
CANDIDATES: Tuple[Tuple[str, str, str, str, str], ...] = (
    # Federated and open social, public timelines only.
    ("mastodon-social", "mastodon", "en", "global",
     "https://mastodon.social/api/v1/timelines/public?limit=5"),
    ("mastodon-online", "mastodon", "en", "global",
     "https://mastodon.online/api/v1/timelines/public?limit=5"),
    ("mastodon-mstdn-jp", "mastodon", "ja", "jp",
     "https://mstdn.jp/api/v1/timelines/public?limit=5"),
    ("mastodon-pawoo", "mastodon", "ja", "jp",
     "https://pawoo.net/api/v1/timelines/public?limit=5"),
    ("mastodon-mas-to", "mastodon", "en", "global",
     "https://mas.to/api/v1/timelines/public?limit=5"),
    ("mastodon-fosstodon", "mastodon", "en", "global",
     "https://fosstodon.org/api/v1/timelines/public?limit=5"),
    ("mastodon-troet", "mastodon", "de", "de",
     "https://troet.cafe/api/v1/timelines/public?limit=5"),
    ("mastodon-piaille", "mastodon", "fr", "fr",
     "https://piaille.fr/api/v1/timelines/public?limit=5"),
    # Site-published feeds.
    ("reuters-tech-rss", "rss", "en", "global",
     "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"),
    ("bbc-world-rss", "rss", "en", "gb",
     "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("nhk-news-rss", "rss", "ja", "jp",
     "https://www.nhk.or.jp/rss/news/cat0.xml"),
    ("dw-rss", "rss", "de", "de", "https://rss.dw.com/rdf/rss-en-all"),
    ("lemonde-rss", "rss", "fr", "fr", "https://www.lemonde.fr/rss/une.xml"),
    ("elpais-rss", "rss", "es", "es",
     "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"),
    ("g1-rss", "rss", "pt", "br", "https://g1.globo.com/rss/g1/"),
    ("cnbc-rss", "rss", "en", "us",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("coindesk-rss", "rss", "en", "global",
     "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph-rss", "rss", "en", "global", "https://cointelegraph.com/rss"),
    ("theblock-rss", "rss", "en", "global",
     "https://www.theblock.co/rss.xml"),
    ("solana-blog-rss", "rss", "en", "global", "https://solana.com/rss.xml"),
    ("chainalysis-blog-rss", "rss", "en", "global", "https://blog.chainalysis.com/feed/"),
    ("chaincatcher-rss", "rss", "zh", "cn", "https://www.chaincatcher.com/rss/clist"),
    ("odaily-flash-rss", "rss", "zh", "cn", "https://rss.odaily.news/rss/newsflash"),
    ("odaily-post-rss", "rss", "zh", "cn", "https://rss.odaily.news/rss/post"),
    ("panews-zh-rss", "rss", "zh", "cn",
     "https://www.panewslab.com/rss.xml?lang=zh&type=NORMAL%2CNEWS"),
    ("panews-ja-rss", "rss", "ja", "jp",
     "https://www.panewslab.com/rss.xml?lang=ja&type=NORMAL%2CNEWS"),
    ("panews-ko-rss", "rss", "ko", "kr",
     "https://www.panewslab.com/rss.xml?lang=ko&type=NORMAL%2CNEWS"),
    # Chain and protocol data.
    ("solana-status", "rss", "en", "global", "https://status.solana.com/history.rss"),
    # Regulatory and official.
    ("sec-press-rss", "rss", "en", "us",
     "https://www.sec.gov/news/pressreleases.rss"),
    ("europa-rapid-rss", "rss", "en", "eu",
     "https://ec.europa.eu/commission/presscorner/api/rss?language=en"),
)


def probe(url: str) -> Tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            code = response.getcode()
            # Read a little to prove the body is real rather than a redirect
            # page that happened to return 200.
            body = response.read(2048)
            if code != 200:
                return False, f"HTTP {code}"
            if not body.strip():
                return False, "empty body"
            return True, f"HTTP {code}, {len(body)}+ bytes"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def declaration_options(kind: str, url: str) -> str:
    """The options THIS kind's transport actually requires.

    Emitting `url` for every kind produced declarations the transport builder
    rejects by name -- a mastodon source needs an instance, a code repo needs
    a repo. A verified endpoint that then fails to build is the same coverage
    hole as an unverified one, arrived at more slowly.
    """
    if kind == "mastodon":
        parsed = urllib.parse.urlparse(url)
        return f'{{instance: "{parsed.scheme}://{parsed.netloc}"}}'
    if kind == "nostr":
        return f'{{relay: "{url}"}}'
    if kind == "farcaster":
        return f'{{hub_url: "{url}"}}'
    if kind == "bluesky":
        return f'{{url: "{url}"}}'
    return f'{{url: "{url}"}}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--emit", action="store_true",
                        help="print sources.yaml declarations for verified endpoints")
    parser.add_argument("--out", default="",
                        help="write the emitted declarations to this file "
                             "(use config/sources.verified.yaml, which the loader "
                             "merges over the seed registry)")
    args = parser.parse_args()
    if args.out:
        args.emit = True
    results: List[Dict[str, Any]] = []
    for source_id, kind, language, region, url in CANDIDATES:
        ok, detail = probe(url)
        results.append({"id": source_id, "kind": kind, "language": language,
                        "region": region, "url": url, "verified": ok,
                        "detail": detail})
        if not args.json:
            print(f"{'OK  ' if ok else 'FAIL'} {source_id:28s} {detail}")
    verified = [item for item in results if item["verified"]]
    if args.json:
        print(json.dumps(results, indent=2))
    elif args.emit:
        # Emitted rather than merged into the seed: what belongs in the
        # registry is the operator's decision, made after reading what
        # actually answered on their node.
        import datetime

        lines = ["# Verified on this host at "
                 f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                 "# Only endpoints that answered here. Re-run to refresh.",
                 "schema_version: v1", "sources:"]
        for item in verified:
            lines.append(f"  - id: {item['id']}")
            lines.append(f"    kind: {item['kind']}")
            lines.append(f"    language: {item['language']}")
            lines.append(f"    region: {item['region']}")
            lines.append("    tier: 2")
            lines.append(f"    options: {declaration_options(item['kind'], item['url'])}")
        rendered = "\n".join(lines) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(rendered)
            print(f"wrote {len(verified)} verified declarations to {args.out}")
        else:
            print(rendered, end="")
    else:
        print(f"\n{len(verified)}/{len(results)} endpoints answered")
        if len(verified) < len(results):
            print("Unreachable endpoints may be blocked by this host's egress "
                  "policy rather than genuinely dead; re-run on the trading node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
