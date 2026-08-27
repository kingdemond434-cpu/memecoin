"""Build entity-registry declarations from the entity's OWN published page.

`config/entities.yaml` is empty on purpose, and the reason is in the file: an
entity declared there asserts that a specific account, domain or wallet
canonically IS a named person or organisation, and a wrong entry does not
degrade gracefully -- it makes an impersonator look verified, which is the
single most expensive error this system can make.

The obvious way to fill it is to type in handles from memory. That produces a
registry that looks complete and is unverifiable, and it is exactly the failure
the empty file exists to prevent. So this does the other thing: it FETCHES the
official domain the operator names, extracts the account links the domain's own
pages publish, and emits a declaration recording where each fact was read and
when.

    python tools/verify_entities.py --domain example.org --id example-org \
        --name "Example Organisation"
    python tools/verify_entities.py --candidates candidates.yaml \
        --out config/entities.verified.yaml

What it does NOT do, deliberately:

* It never invents an account. Only handles the fetched pages actually link to
  appear in the output.
* It never resolves a handle to a numeric account id. That needs each
  platform's API and the operator's own credentials; the emitted declaration
  carries the handle as a comment and leaves `accounts` for the operator to
  complete, because a display handle in an id field is an impersonation
  waiting to happen.
* It never adds a wallet. A wallet claim needs a public acknowledgement by the
  entity, read by a person.

Run it on the node with outbound access. A sandboxed host reports every domain
unreachable, and that verdict is about the sandbox.
"""

import argparse
import datetime
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

TIMEOUT = 15
USER_AGENT = "memecoin-entity-verifier/1.0 (+public page reader)"
MAX_BYTES = 512_000

# Paths an organisation's account links are usually on. Fetched in order and
# all of them contribute; a handle linked from two of an entity's own pages is
# better evidence than one linked from a single page.
DEFAULT_PATHS: Tuple[str, ...] = ("/", "/about", "/contact", "/press", "/social", "/links")

# platform -> pattern capturing the handle from a profile URL. Only networks
# whose profile URLs are unambiguous; a pattern that also matches a post URL
# would report the poster as the entity.
# A profile URL ENDS here: whitespace, a quote, a tag, or the end of the
# document. Without this bound the patterns also match a POST url -- and
# reporting the poster of one link as the entity itself is the whole mistake.
_END = r"/?(?=[\s\"'<>)\]]|$)"

HANDLE_PATTERNS: Dict[str, re.Pattern] = {
    "telegram": re.compile(r"https?://(?:t\.me|telegram\.me)/([A-Za-z0-9_]{5,32})" + _END),
    "youtube": re.compile(r"https?://(?:www\.)?youtube\.com/(?:@([A-Za-z0-9_.-]{3,30})"
                          r"|channel/(UC[A-Za-z0-9_-]{22}))" + _END),
    "mastodon": re.compile(r"https?://([a-z0-9.-]+)/@([A-Za-z0-9_]{1,30})" + _END),
    "bluesky": re.compile(r"https?://(?:www\.)?bsky\.app/profile/([A-Za-z0-9_.:-]{3,64})" + _END),
    "twitch": re.compile(r"https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]{4,25})" + _END),
    "github": re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9-]{1,39})" + _END),
}


def fetch(url: str) -> Tuple[bool, str, str]:
    """Body, or the reason it could not be read. Never raises."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            if response.getcode() != 200:
                return False, "", f"HTTP {response.getcode()}"
            body = response.read(MAX_BYTES).decode("utf-8", "replace")
            return True, body, f"HTTP 200, {len(body)} bytes"
    except urllib.error.HTTPError as exc:
        return False, "", f"HTTP {exc.code}"
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def handles_in(body: str) -> Dict[str, List[str]]:
    """Every platform handle this page links to, by platform."""
    found: Dict[str, List[str]] = {}
    for platform, pattern in HANDLE_PATTERNS.items():
        for match in pattern.finditer(body or ""):
            groups = [group for group in match.groups() if group]
            if not groups:
                continue
            handle = "@".join(reversed(groups)) if platform == "mastodon" else groups[0]
            bucket = found.setdefault(platform, [])
            if handle not in bucket:
                bucket.append(handle)
    return found


def verify_domain(domain: str, paths: Sequence[str] = DEFAULT_PATHS) -> Dict[str, Any]:
    """Fetch an entity's own pages and report what they publish.

    ``pages`` records every URL that answered and the hash of what it served,
    so a later re-run can say whether the page that vouched for a handle has
    since changed -- which is the difference between provenance and a URL.
    """
    scheme_domain = domain if "://" in domain else f"https://{domain}"
    parsed = urllib.parse.urlparse(scheme_domain)
    base = f"{parsed.scheme}://{parsed.netloc}"
    pages: List[Dict[str, str]] = []
    handles: Dict[str, List[str]] = {}
    for path in paths:
        url = urllib.parse.urljoin(base, path)
        ok, body, detail = fetch(url)
        if not ok:
            continue
        pages.append({"url": url, "detail": detail,
                      "sha256": hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()})
        for platform, values in handles_in(body).items():
            bucket = handles.setdefault(platform, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return {"domain": parsed.netloc, "reachable": bool(pages),
            "pages": pages, "handles": handles}


def declaration(entity_id: str, display_name: str, result: Dict[str, Any],
                aliases: Sequence[str] = ()) -> List[str]:
    """YAML for one entity, with the account ids left for a person to fill.

    The handles are emitted as comments rather than as `accounts` entries.
    `accounts` holds STABLE platform ids, and putting a display handle there
    would let a renamed or resold handle keep an entity's proof level -- the
    exact impersonation the id-not-name rule exists to stop.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    lines = [
        f"  - entity_id: {entity_id}",
        f"    display_name: {json.dumps(display_name)}",
        f"    official_domains: [{result['domain']}]",
        "    known_wallets: []          # only wallets the entity has publicly acknowledged",
    ]
    if aliases:
        lines.append(f"    aliases: {json.dumps(list(aliases))}")
    lines.append("    accounts:")
    if result["handles"]:
        lines.append("      # Handles this domain's own pages link to. Resolve each to the")
        lines.append("      # platform's STABLE numeric id before uncommenting it: a display")
        lines.append("      # handle here is an impersonation waiting to happen.")
        for platform, values in sorted(result["handles"].items()):
            for handle in values:
                lines.append(f"      # {platform}: {handle}")
    else:
        lines.append("      # This domain published no recognisable account links.")
    lines.append(f"    verified_from: {json.dumps(result['pages'][0]['url'] if result['pages'] else '')}")
    lines.append(f"    verified_at: \"{now.date().isoformat()}\"")
    lines.append("    metadata:")
    lines.append(f"      verified_pages: {json.dumps([page['url'] for page in result['pages']])}")
    lines.append(f"      page_hashes: {json.dumps({page['url']: page['sha256'] for page in result['pages']})}")
    return lines


def load_candidates(path: str) -> List[Dict[str, Any]]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return list(raw.get("candidates") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="", help="one official domain to verify")
    parser.add_argument("--id", default="", help="entity_id for --domain")
    parser.add_argument("--name", default="", help="display_name for --domain")
    parser.add_argument("--alias", action="append", default=[],
                        help="an alias that should trigger NAME_ONLY (repeatable)")
    parser.add_argument("--candidates", default="",
                        help="YAML with a `candidates:` list of "
                             "{entity_id, display_name, domain, aliases}")
    parser.add_argument("--out", default="", help="write the declarations here")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    candidates: List[Dict[str, Any]] = []
    if args.candidates:
        candidates = load_candidates(args.candidates)
    if args.domain:
        candidates.append({"entity_id": args.id or args.domain.replace(".", "-"),
                           "display_name": args.name or args.domain,
                           "domain": args.domain, "aliases": args.alias})
    if not candidates:
        parser.error("supply --domain or --candidates")

    results = []
    lines = [f"# Verified from each entity's own published pages at "
             f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
             "# Account ids are left commented: resolve every handle to the platform's",
             "# stable numeric id before the resolver is allowed to trust it.",
             "entities:"]
    for candidate in candidates:
        result = verify_domain(str(candidate["domain"]))
        results.append({**candidate, **result})
        status = "OK  " if result["reachable"] else "FAIL"
        handles = sum(len(values) for values in result["handles"].values())
        print(f"{status} {candidate['entity_id']:24s} "
              f"{len(result['pages'])} pages, {handles} handles", file=sys.stderr)
        if not result["reachable"]:
            continue
        lines += declaration(str(candidate["entity_id"]), str(candidate["display_name"]),
                             result, candidate.get("aliases") or ())

    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    rendered = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(rendered, end="")
    print("\nEvery emitted entity still needs a person to resolve its handles to "
          "stable ids and to confirm any wallet claim.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
