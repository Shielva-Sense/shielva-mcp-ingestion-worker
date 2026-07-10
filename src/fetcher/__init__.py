"""
Fetcher Module
Document fetching from various sources.
"""

import asyncio
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


def _r2_coords() -> Tuple[str, str, str, str]:
    """R2/S3 connection coords from env (mirrors the canonical boto3-against-R2
    pattern in shielva-platform/app/account/r2_export.py and shielva-cdn adapter).

    Returns (endpoint_url, access_key, secret_key, region). Creds are AWS-standard
    R2 access/secret keys — sealed in production, plaintext .env in dev. Empty
    values are returned as-is so the caller can fail with a clear error instead of
    silently constructing an unauthenticated client.
    """
    endpoint = os.environ.get("R2_ENDPOINT_URL", "")
    access = os.environ.get("R2_ACCESS_KEY", "") or os.environ.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_KEY", "") or os.environ.get("R2_SECRET_ACCESS_KEY", "")
    region = os.environ.get("R2_REGION", "") or "auto"
    return endpoint, access, secret, region


def _r2_client():
    """Build a boto3 S3 client pointed at Cloudflare R2.

    SigV4 + path-style addressing (R2's virtual-host bucket DNS isn't always
    provisioned) — same config the platform/cdn services use. boto3 is synchronous;
    callers run it in a worker thread to keep the event loop unblocked.
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:  # pragma: no cover - dep-presence guard
        raise RuntimeError("boto3 is not installed — cannot fetch from R2. `pip install boto3`.") from exc

    endpoint, access, secret, region = _r2_coords()
    if not endpoint or not access or not secret:
        raise RuntimeError(
            "R2 is not configured (need R2_ENDPOINT_URL + R2_ACCESS_KEY + "
            "R2_SECRET_KEY). Cannot fetch the uploaded object."
        )
    boto_cfg = BotoConfig(
        signature_version="s3v4",
        s3={"addressing_style": "path"},  # R2-safe
        retries={"max_attempts": 3, "mode": "standard"},
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=boto_cfg,
    )


def _stream_r2_object_sync(bucket: str, key: str) -> Tuple[bytes, str]:
    """Synchronous boto3 GET that streams the object body in chunks (so the bytes
    are read incrementally off the wire rather than via a single buffered
    .content). Returns (body_bytes, content_type). Runs in a worker thread."""
    client = _r2_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    content_type = obj.get("ContentType") or "application/octet-stream"
    body_stream = obj["Body"]  # botocore StreamingBody — iterable in chunks
    buf = bytearray()
    try:
        for chunk in body_stream.iter_chunks(chunk_size=1024 * 1024):
            if chunk:
                buf.extend(chunk)
    finally:
        body_stream.close()
    return bytes(buf), content_type


async def fetch_r2_object(bucket: str, key: str) -> "FetchResult":
    """Stream an object from R2 by (bucket, key). The blocking boto3 GET runs in a
    worker thread; the StreamingBody is read in 1 MiB chunks so the object is
    pulled off the wire incrementally instead of one buffered read."""
    try:
        content, content_type = await asyncio.to_thread(_stream_r2_object_sync, bucket, key)
    except Exception as e:  # noqa: BLE001 — boto/botocore raise broadly
        logger.error("r2_fetch_failed", bucket=bucket, key=key, error=str(e))
        return FetchResult(content=b"", content_type="unknown", error=str(e))
    return FetchResult(
        content=content,
        content_type=content_type,
        metadata={"bucket": bucket, "key": key, "size": len(content)},
    )


@dataclass
class FetchResult:
    """Result of document fetch."""

    content: bytes
    content_type: str
    encoding: str = "utf-8"
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class DocumentFetcher:
    """
    Fetches documents from various sources.

    Supports:
    - HTTP/HTTPS URLs
    - S3 buckets
    - Local files
    - Base64 encoded content
    """

    def __init__(self, http_client=None):
        """
        Initialize fetcher.

        Args:
            http_client: HTTP client for URL fetching
        """
        self.http_client = http_client

        logger.info("DocumentFetcher initialized")

    async def fetch(self, source: str, source_type: str = "auto") -> FetchResult:
        """
        Fetch document from source.

        Args:
            source: Source location (URL, path, etc.)
            source_type: Type of source (url, s3, file, base64)

        Returns:
            Fetch result with content
        """
        if source_type == "auto":
            source_type = self._detect_source_type(source)

        try:
            if source_type == "url":
                return await self._fetch_url(source)
            elif source_type == "s3":
                return await self._fetch_s3(source)
            elif source_type == "file":
                return self._fetch_file(source)
            elif source_type == "base64":
                return self._decode_base64(source)
            else:
                return FetchResult(content=b"", content_type="unknown", error=f"Unknown source type: {source_type}")
        except Exception as e:
            logger.error("Fetch failed", source=source, error=str(e))
            return FetchResult(content=b"", content_type="unknown", error=str(e))

    def _detect_source_type(self, source: str) -> str:
        """Detect source type from URL/path."""
        if source.startswith("http://") or source.startswith("https://"):
            return "url"
        elif source.startswith("s3://"):
            return "s3"
        elif source.startswith("data:"):
            return "base64"
        else:
            return "file"

    async def _fetch_url(self, url: str) -> FetchResult:
        """Fetch document from URL."""
        if self.http_client:
            response = await self.http_client.get(url, timeout=60)
            return FetchResult(
                content=response.content,
                content_type=response.headers.get("content-type", "text/html"),
                metadata={"status_code": response.status_code},
            )

        # Mock
        return FetchResult(content=b"Mock content from URL", content_type="text/html")

    async def _fetch_s3(self, s3_uri: str) -> FetchResult:
        """Fetch (stream) a document from S3-compatible object storage (R2).

        Parses ``s3://bucket/key`` and streams the body via :func:`fetch_r2_object`
        — a real boto3 client pointed at R2 (SigV4, path-style, region 'auto'),
        replacing the previous mock.
        """
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        if not bucket or not key:
            return FetchResult(content=b"", content_type="unknown", error=f"Malformed s3 uri: {s3_uri}")
        return await fetch_r2_object(bucket, key)

    def _fetch_file(self, path: str) -> FetchResult:
        """Fetch document from local file."""
        try:
            with open(path, "rb") as f:
                content = f.read()

            # Detect content type
            content_type = self._detect_content_type(path)

            return FetchResult(content=content, content_type=content_type)
        except FileNotFoundError:
            return FetchResult(content=b"", content_type="unknown", error=f"File not found: {path}")

    def _decode_base64(self, data_uri: str) -> FetchResult:
        """Decode base64 encoded content."""
        import base64

        # Parse data:mime;base64,content
        if "," in data_uri:
            header, content = data_uri.split(",", 1)
            content_type = header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
        else:
            content = data_uri
            content_type = "application/octet-stream"

        try:
            decoded = base64.b64decode(content)
            return FetchResult(content=decoded, content_type=content_type)
        except Exception as e:
            return FetchResult(content=b"", content_type="unknown", error=str(e))

    def _detect_content_type(self, path: str) -> str:
        """Detect content type from file extension."""
        ext_map = {
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".csv": "text/csv",
        }

        for ext, content_type in ext_map.items():
            if path.lower().endswith(ext):
                return content_type

        return "application/octet-stream"


__all__ = ["DocumentFetcher", "FetchResult", "fetch_r2_object"]
