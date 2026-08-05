"""RAG source adapters — turn a URL, database, or API into ingestable documents.

Each adapter returns a list of ``(title, content, doc_type)`` tuples that the
caller wraps in ``Document`` objects and runs through the normal parse → chunk →
embed → index pipeline. Three direct (non-connector) source types:

  - URL   : fetch a public page, or BFS-crawl a public site (depth/page capped).
  - DB    : connect to Postgres / MySQL / MongoDB, run a READ-ONLY query, one
            row/doc per record.
  - API   : GET/POST a REST endpoint, extract a list via a dotted JSON path.

Private / gated / OAuth sources are NOT handled here — those go through the
Integration Builder connectors. URL + API fetches are SSRF-guarded (public hosts
only); DB hosts are guarded too except in development (local DBs).
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import structlog

logger = structlog.get_logger(__name__)

# (title, content, doc_type)
SourceDoc = Tuple[str, str, str]

_IS_DEV = os.getenv("ENVIRONMENT", "development").lower() == "development"
_MAX_DOC_CHARS = 200_000

# Pooled SQLAlchemy engines keyed by connection URI — reused across ingest calls so
# we don't pay TCP + auth + TLS setup on every query (was create_engine + dispose
# per request). Pre-pinged + recycled hourly to survive idle DB timeouts.
_ENGINE_CACHE: Dict[str, Any] = {}


def _get_pooled_engine(connection_uri: str):
    engine = _ENGINE_CACHE.get(connection_uri)
    if engine is None:
        from sqlalchemy import create_engine

        engine = create_engine(
            connection_uri,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        _ENGINE_CACHE[connection_uri] = engine
    return engine


# ── SSRF guard ───────────────────────────────────────────────────────────────
def _host_is_public(host: str) -> bool:
    """Resolve a hostname and return True only if every address is public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False
    return True


def _assert_public_url(url: str) -> None:
    """Reject non-http(s) schemes and private/loopback hosts (SSRF defence)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http(s) URLs are allowed (got {parsed.scheme!r})")
    if not parsed.hostname:
        raise ValueError("URL has no host")
    if not _host_is_public(parsed.hostname):
        raise ValueError(f"Refusing to fetch a private/internal host: {parsed.hostname}")


def _assert_db_host(host: Optional[str]) -> None:
    """Block private DB hosts in production; allow them in development so local
    databases can be ingested during testing."""
    if _IS_DEV or not host:
        return
    if not _host_is_public(host):
        raise ValueError(f"Refusing to connect to a private/internal DB host: {host}")


# ── URL / crawl ──────────────────────────────────────────────────────────────
# Outbound page fetches are user-facing (someone is waiting on "Add source"), so
# fail fast rather than hanging: a healthy page answers well inside 10s. Sites that
# block server/datacenter traffic (bot mitigation) simply never reply — without a
# short read timeout that request hangs ~20s and the browser gives up first, which
# surfaced as a meaningless "Network Error" instead of a real reason.
_FETCH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


async def fetch_url(
    url: str,
    *,
    crawl: bool = False,
    max_pages: int = 20,
    max_depth: int = 2,
) -> List[SourceDoc]:
    """Fetch a single public page, or BFS-crawl same-host pages when ``crawl``.
    Returns each page's raw HTML as an ``html`` doc (the pipeline parses it)."""
    _assert_public_url(url)
    out: List[SourceDoc] = []
    if not crawl:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "ShielvaBot/1.0"})
            r.raise_for_status()
            out.append((url, r.text[:_MAX_DOC_CHARS], "html"))
        return out

    from bs4 import BeautifulSoup  # local import — only needed for crawl

    base_host = urlparse(url).hostname
    seen: set = set()
    queue: List[Tuple[str, int]] = [(url, 0)]
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True) as client:
        while queue and len(out) < max_pages:
            cur, depth = queue.pop(0)
            if cur in seen or depth > max_depth:
                continue
            seen.add(cur)
            try:
                if urlparse(cur).hostname != base_host or not _host_is_public(urlparse(cur).hostname or ""):
                    continue
                r = await client.get(cur, headers={"User-Agent": "ShielvaBot/1.0"})
                if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                    continue
                out.append((cur, r.text[:_MAX_DOC_CHARS], "html"))
                if depth < max_depth:
                    soup = BeautifulSoup(r.text, "html.parser")
                    for a in soup.find_all("a", href=True):
                        nxt = urljoin(cur, a["href"]).split("#")[0]
                        if nxt not in seen and urlparse(nxt).hostname == base_host:
                            queue.append((nxt, depth + 1))
            except Exception as e:  # noqa: BLE001 — skip a bad page, keep crawling
                logger.warning("crawl_page_failed", url=cur, error=str(e))
    return out


# ── Database ─────────────────────────────────────────────────────────────────
def _row_to_text(row: Dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)


async def read_database(
    db_type: str,
    connection_uri: str,
    *,
    query: Optional[str] = None,
    collection: Optional[str] = None,
    limit: int = 1000,
) -> List[SourceDoc]:
    """Read rows/documents from a database as text docs. SQL must be read-only
    (SELECT). Mongo reads a collection (optionally filtered by a JSON query)."""
    db_type = (db_type or "").lower()
    host = urlparse(connection_uri).hostname
    _assert_db_host(host)
    out: List[SourceDoc] = []

    if db_type in ("postgres", "postgresql", "mysql", "mariadb", "sql"):
        if not query or not query.strip().lower().startswith("select"):
            raise ValueError("A read-only SELECT query is required for SQL sources")
        import asyncio
        from sqlalchemy import text  # engine obtained from the pooled cache

        def _run() -> List[Dict[str, Any]]:
            engine = _get_pooled_engine(connection_uri)
            with engine.connect() as conn:
                res = conn.execute(text(query))
                cols = list(res.keys())
                rows = res.fetchmany(limit)
                return [dict(zip(cols, r)) for r in rows]

        records = await asyncio.to_thread(_run)
        for i, rec in enumerate(records):
            out.append((f"row {i + 1}", _row_to_text(rec)[:_MAX_DOC_CHARS], "text"))

    elif db_type in ("mongo", "mongodb"):
        if not collection:
            raise ValueError("A collection name is required for MongoDB sources")
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(connection_uri, serverSelectionTimeoutMS=8000)
        try:
            db = client.get_default_database()
            if db is None:
                raise ValueError("MongoDB URI must include a database name")
            filt: Dict[str, Any] = {}
            if query:
                try:
                    filt = json.loads(query)
                except Exception:
                    filt = {}
            cursor = db[collection].find(filt, {"_id": 0}).limit(limit)
            i = 0
            async for doc in cursor:
                i += 1
                out.append((f"{collection} #{i}", _row_to_text(doc)[:_MAX_DOC_CHARS], "text"))
        finally:
            client.close()
    else:
        raise ValueError(f"Unsupported database type: {db_type!r}")

    return out


# ── API ──────────────────────────────────────────────────────────────────────
def _json_path(data: Any, path: Optional[str]) -> Any:
    if not path:
        return data
    cur = data
    for part in path.split("."):
        if not part:
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


async def read_api(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    json_path: Optional[str] = None,
    limit: int = 500,
) -> List[SourceDoc]:
    """Fetch a REST endpoint and turn the response into docs. If ``json_path``
    resolves to a list, each item becomes a doc; otherwise the whole payload is
    one doc. SSRF-guarded."""
    _assert_public_url(url)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.request(method.upper(), url, headers=headers or {}, json=body)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return [(url, resp.text[:_MAX_DOC_CHARS], "text")]

    target = _json_path(data, json_path)
    out: List[SourceDoc] = []
    if isinstance(target, list):
        for i, item in enumerate(target[:limit]):
            out.append((f"item {i + 1}", json.dumps(item, ensure_ascii=False)[:_MAX_DOC_CHARS], "json"))
    else:
        out.append((url, json.dumps(target, ensure_ascii=False)[:_MAX_DOC_CHARS], "json"))
    return out
