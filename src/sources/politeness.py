"""Crawl politeness — robots.txt consent + honest bot identity.

Why this exists
───────────────
A page rendering in a browser does NOT mean a server may fetch and store it.
Legitimate aggregators (Indeed, Google for Jobs) rely on the publisher *opting
in* through a machine-readable interface — a feed, an API, structured data, or
an explicit robots.txt allowance. We follow the same rule: fetch only what the
publisher permits, and identify ourselves honestly so they can contact or block
us.

Deliberately NOT done here: User-Agent spoofing, TLS-fingerprint mimicry,
residential-proxy rotation or CAPTCHA solving. Those defeat an access control
the publisher deliberately set; using them would turn a technical block into a
contractual/ToS problem for us *and* for the tenant whose KB holds the content.
When a site says no, the answer is "upload the file instead", never "evade".

What honest identity buys us: the browser-grade request below (HTTP/2 + the
header set a real client sends) is not a disguise — the UA still says
ShielvaBot with a contact URL. Many WAFs reject HTTP/1.1-with-no-Accept-Language
as malformed-looking traffic, so speaking the modern protocol correctly is what
gets Wikipedia (whose robots.txt explicitly welcomes "friendly, low-speed bots")
to answer us at all.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

# ── Identity ────────────────────────────────────────────────────────────────
# Wikimedia (and others) require a descriptive UA with a way to reach the
# operator; a bare token like "ShielvaBot/1.0" is explicitly against their
# policy. Format: Product/version (+info-url; contact) library/version
BOT_NAME = "ShielvaBot"
BOT_VERSION = "1.0"
BOT_INFO_URL = "https://shielva.ai/bot"
BOT_CONTACT = "support@shielva.ai"
USER_AGENT = (
    f"{BOT_NAME}/{BOT_VERSION} (+{BOT_INFO_URL}; {BOT_CONTACT}) httpx"
)

# A real client sends more than a UA. Sending only a UA over HTTP/1.1 is what
# most WAFs score as "not a browser and not a well-behaved bot".
BROWSER_HEADERS: Dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
}

# robots.txt is itself a fetch; keep it short so a blocked host fails fast.
_ROBOTS_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
_ROBOTS_TTL_SECONDS = 3600.0
# origin -> (parser or None, fetched_at). None = robots.txt unreachable/absent.
_robots_cache: Dict[str, Tuple[Optional[RobotFileParser], float]] = {}
_robots_lock = asyncio.Lock()


class RobotsDisallowed(Exception):
    """The publisher's robots.txt forbids us from fetching this URL."""


def _origin(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, "", "", "", ""))


async def _load_robots(origin: str) -> Optional[RobotFileParser]:
    """Fetch + parse an origin's robots.txt. Cached; None when unavailable."""
    now = time.monotonic()
    cached = _robots_cache.get(origin)
    if cached and (now - cached[1]) < _ROBOTS_TTL_SECONDS:
        return cached[0]

    async with _robots_lock:
        cached = _robots_cache.get(origin)
        if cached and (time.monotonic() - cached[1]) < _ROBOTS_TTL_SECONDS:
            return cached[0]

        parser: Optional[RobotFileParser] = None
        try:
            async with httpx.AsyncClient(
                timeout=_ROBOTS_TIMEOUT, follow_redirects=True, http2=True
            ) as client:
                resp = await client.get(f"{origin}/robots.txt", headers=BROWSER_HEADERS)
            if resp.status_code == 200 and resp.text.strip():
                parser = RobotFileParser()
                parser.parse(resp.text.splitlines())
            # 404 / empty  → no rules published → parser stays None (allowed).
            # 401 / 403    → the file itself is gated; treat as "no rules we can
            #                read" and let the page fetch decide. We do not
            #                invent consent, but we also don't block on a host
            #                that simply has no robots.txt.
        except Exception:
            parser = None  # unreachable robots.txt must not hard-fail ingestion

        _robots_cache[origin] = (parser, time.monotonic())
        return parser


async def assert_crawl_allowed(url: str) -> None:
    """Raise :class:`RobotsDisallowed` when robots.txt forbids this URL.

    Checked against our real UA first, then ``*`` — matching how a compliant
    crawler is expected to evaluate rules.
    """
    parser = await _load_robots(_origin(url))
    if parser is None:
        return
    if parser.can_fetch(USER_AGENT, url) or parser.can_fetch("*", url):
        return
    host = urlparse(url).hostname or url
    raise RobotsDisallowed(
        f"{host} does not allow automated access to this page (robots.txt). "
        f"Shielva honours robots.txt, so this URL cannot be ingested. Upload the "
        f"content as a file instead, or use a source the publisher permits "
        f"(their API, RSS feed or sitemap)."
    )


def crawl_delay(url: str) -> float:
    """Publisher-requested delay between hits, capped so a crawl can't stall."""
    parser = _robots_cache.get(_origin(url), (None, 0.0))[0]
    if parser is None:
        return 0.0
    try:
        delay = parser.crawl_delay(USER_AGENT) or parser.crawl_delay("*")
    except Exception:
        delay = None
    return min(float(delay), 5.0) if delay else 0.0
