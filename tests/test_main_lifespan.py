"""Application lifespan wiring + the uvicorn runner entrypoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import main
import src.jobs.queue as queue_mod


class _FakeVectorStore:
    def __init__(self, *a, **k):
        self.connected = False

    async def connect(self):
        self.connected = True

    async def close(self):
        self.connected = False


async def test_lifespan_initializes_and_tears_down(monkeypatch):
    monkeypatch.setattr(main, "SupabaseVectorStore", _FakeVectorStore)
    # keep the real IngestQueue but make start/stop cheap (they run on the loop)
    async with main.lifespan(main.app):
        assert main.pipeline is not None
        assert main.processor is not None
        assert queue_mod.ingest_queue is not None
        # no REDIS_URL configured in tests -> token bucket disabled
        assert main.app.state.token_bucket is None
    # after exit the queue is stopped
    assert queue_mod.ingest_queue._started is False


async def test_lifespan_with_redis(monkeypatch):
    monkeypatch.setattr(main, "SupabaseVectorStore", _FakeVectorStore)

    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    import sys
    import types

    fake_redis_asyncio = types.ModuleType("redis.asyncio")
    fake_redis_asyncio.from_url = MagicMock(return_value=MagicMock())
    fake_redis = types.ModuleType("redis")
    fake_redis.asyncio = fake_redis_asyncio
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_redis_asyncio)

    async with main.lifespan(main.app):
        assert main.app.state.token_bucket is not None

    get_settings.cache_clear()


def test_main_runner_invokes_uvicorn(monkeypatch):
    monkeypatch.setattr(main, "find_free_port", lambda p: p)
    fake_run = MagicMock()
    monkeypatch.setattr(main.uvicorn, "run", fake_run)
    monkeypatch.delenv("CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERTFILE", raising=False)
    main.main()
    fake_run.assert_called_once()
    assert fake_run.call_args.args[0] == "main:app"


def test_main_runner_uses_alternate_port(monkeypatch, capsys):
    from src.config import get_settings

    target = get_settings().port
    monkeypatch.setattr(main, "find_free_port", lambda p: p + 5)  # port shifted
    monkeypatch.setattr(main.uvicorn, "run", MagicMock())
    monkeypatch.delenv("CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERTFILE", raising=False)
    main.main()
    out = capsys.readouterr().out
    assert str(target + 5) in out


def test_main_runner_with_tls(monkeypatch, tmp_path):
    cert = tmp_path / "c.pem"
    key = tmp_path / "k.pem"
    cert.write_text("cert")
    key.write_text("key")
    monkeypatch.setattr(main, "find_free_port", lambda p: p)
    fake_run = MagicMock()
    monkeypatch.setattr(main.uvicorn, "run", fake_run)
    monkeypatch.setenv("CERT_FILE", str(cert))
    monkeypatch.setenv("KEY_FILE", str(key))
    main.main()
    kwargs = fake_run.call_args.kwargs
    assert kwargs["ssl_certfile"] == str(cert)
    assert kwargs["ssl_keyfile"] == str(key)


def test_main_runner_no_free_port(monkeypatch):
    def boom(p):
        raise RuntimeError("no ports")

    monkeypatch.setattr(main, "find_free_port", boom)
    fake_run = MagicMock()
    monkeypatch.setattr(main.uvicorn, "run", fake_run)
    main.main()  # should log + return without calling uvicorn
    fake_run.assert_not_called()
