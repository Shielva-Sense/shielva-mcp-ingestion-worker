"""Document fetcher — source detection, base64, file, url, R2/S3 streaming."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.fetcher as fetcher
from src.fetcher import DocumentFetcher, FetchResult, fetch_r2_object


def test_detect_source_type():
    f = DocumentFetcher()
    assert f._detect_source_type("http://x") == "url"
    assert f._detect_source_type("https://x") == "url"
    assert f._detect_source_type("s3://b/k") == "s3"
    assert f._detect_source_type("data:text/plain;base64,aGk=") == "base64"
    assert f._detect_source_type("/tmp/file.txt") == "file"


def test_detect_content_type():
    f = DocumentFetcher()
    assert f._detect_content_type("a.pdf") == "application/pdf"
    assert f._detect_content_type("a.HTML") == "text/html"
    assert f._detect_content_type("a.unknown") == "application/octet-stream"


def test_decode_base64_with_header():
    f = DocumentFetcher()
    uri = "data:text/plain;base64," + base64.b64encode(b"hello").decode()
    res = f._decode_base64(uri)
    assert res.content == b"hello"
    assert res.content_type == "text/plain"


def test_decode_base64_invalid_returns_error():
    f = DocumentFetcher()
    res = f._decode_base64("data:text/plain;base64,!!!notb64!!!")
    assert res.error is not None


def test_fetch_file_reads_and_missing(tmp_path):
    f = DocumentFetcher()
    p = tmp_path / "x.txt"
    p.write_text("filedata")
    res = f._fetch_file(str(p))
    assert res.content == b"filedata"
    assert res.content_type == "text/plain"

    missing = f._fetch_file(str(tmp_path / "nope.txt"))
    assert missing.error is not None


async def test_fetch_dispatch_auto_url_with_http_client():
    resp = MagicMock()
    resp.content = b"body"
    resp.headers = {"content-type": "text/html"}
    resp.status_code = 200
    http = AsyncMock()
    http.get = AsyncMock(return_value=resp)
    f = DocumentFetcher(http_client=http)
    res = await f.fetch("http://example.com")
    assert res.content == b"body"
    assert res.metadata["status_code"] == 200


async def test_fetch_url_without_client_returns_mock():
    f = DocumentFetcher()
    res = await f._fetch_url("http://x")
    assert res.content == b"Mock content from URL"


async def test_fetch_unknown_source_type():
    f = DocumentFetcher()
    res = await f.fetch("whatever", source_type="martian")
    assert res.error is not None and "Unknown source type" in res.error


async def test_fetch_base64_dispatch():
    f = DocumentFetcher()
    uri = "data:application/octet-stream;base64," + base64.b64encode(b"z").decode()
    res = await f.fetch(uri, source_type="base64")
    assert res.content == b"z"


async def test_fetch_catches_exceptions():
    f = DocumentFetcher()
    with patch.object(f, "_fetch_file", side_effect=RuntimeError("disk")):
        res = await f.fetch("/some/path", source_type="file")
    assert res.error == "disk"


def test_r2_coords_from_env(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://r2")
    monkeypatch.setenv("R2_ACCESS_KEY", "ak")
    monkeypatch.setenv("R2_SECRET_KEY", "sk")
    monkeypatch.delenv("R2_REGION", raising=False)
    endpoint, access, secret, region = fetcher._r2_coords()
    assert (endpoint, access, secret, region) == ("https://r2", "ak", "sk", "auto")


def test_r2_client_requires_config(monkeypatch):
    monkeypatch.setattr(fetcher, "_r2_coords", lambda: ("", "", "", "auto"))
    with pytest.raises(RuntimeError):
        fetcher._r2_client()


def test_r2_client_builds_boto_client(monkeypatch):
    monkeypatch.setattr(fetcher, "_r2_coords", lambda: ("https://r2", "ak", "sk", "auto"))
    fake_boto = MagicMock()
    fake_boto.client = MagicMock(return_value="s3client")
    fake_botocore_config = MagicMock()
    with patch.dict(
        "sys.modules", {"boto3": fake_boto, "botocore": MagicMock(), "botocore.config": fake_botocore_config}
    ):
        client = fetcher._r2_client()
    assert client == "s3client"
    made_kwargs = fake_boto.client.call_args
    assert made_kwargs.kwargs["endpoint_url"] == "https://r2"


async def test_fetch_r2_object_streams(monkeypatch):
    body = MagicMock()
    body.iter_chunks.return_value = [b"ab", b"cd"]
    obj = {"ContentType": "text/plain", "Body": body}
    client = MagicMock()
    client.get_object.return_value = obj
    monkeypatch.setattr(fetcher, "_r2_client", lambda: client)
    res = await fetch_r2_object("bucket", "key")
    assert res.content == b"abcd"
    assert res.content_type == "text/plain"
    assert res.metadata["size"] == 4


async def test_fetch_r2_object_error(monkeypatch):
    def boom():
        raise RuntimeError("no r2")

    monkeypatch.setattr(fetcher, "_r2_client", boom)
    res = await fetch_r2_object("b", "k")
    assert res.error is not None
    assert res.content == b""


async def test_fetch_s3_malformed_uri():
    f = DocumentFetcher()
    res = await f._fetch_s3("s3://")
    assert res.error is not None


async def test_fetch_s3_delegates_to_r2(monkeypatch):
    async def fake_fetch(bucket, key):
        return FetchResult(content=b"data", content_type="text/plain")

    monkeypatch.setattr(fetcher, "fetch_r2_object", fake_fetch)
    f = DocumentFetcher()
    res = await f._fetch_s3("s3://bucket/path/key.txt")
    assert res.content == b"data"
