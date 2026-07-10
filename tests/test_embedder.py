"""Embedding client — provider routing, batching, mock + gemini fail-fast."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from src.embedder import EmbedderConfig, EmbeddingClient


async def test_embed_empty_returns_empty():
    client = EmbeddingClient(EmbedderConfig(provider="mock", dimension=3))
    assert await client.embed([]) == []


async def test_embed_mock_dimension_and_batching():
    cfg = EmbedderConfig(provider="unknown", dimension=5, batch_size=2)
    client = EmbeddingClient(cfg)
    out = await client.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 5 for v in out)


async def test_embed_single():
    client = EmbeddingClient(EmbedderConfig(provider="unknown", dimension=4))
    vec = await client.embed_single("hi")
    assert len(vec) == 4


async def test_embed_single_empty_input_list():
    client = EmbeddingClient(EmbedderConfig(provider="unknown", dimension=4))
    # embed([""]) yields one mock vector; embed_single returns it
    vec = await client.embed_single("")
    assert len(vec) == 4


async def test_gemini_success(monkeypatch):
    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = MagicMock()
    fake_genai.embed_content = MagicMock(return_value={"embedding": [[0.1, 0.2, 0.3]]})
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake_genai
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    cfg = EmbedderConfig(provider="gemini", model="models/embed", api_key="k", dimension=3)
    client = EmbeddingClient(cfg)
    out = await client.embed(["hello"])
    assert out == [[0.1, 0.2, 0.3]]
    # dimension pinned in kwargs
    _, kwargs = fake_genai.embed_content.call_args
    assert kwargs["output_dimensionality"] == 3


async def test_gemini_failure_raises_not_mock(monkeypatch):
    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = MagicMock()
    fake_genai.embed_content = MagicMock(side_effect=RuntimeError("api down"))
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake_genai
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    client = EmbeddingClient(EmbedderConfig(provider="gemini", api_key="k", dimension=3))
    with pytest.raises(RuntimeError):
        await client.embed(["x"])


async def test_openai_failure_falls_back_to_mock(monkeypatch):
    fake_openai = types.ModuleType("openai")

    class _Boom:
        def __init__(self, **kw):
            raise RuntimeError("no openai")

    fake_openai.AsyncOpenAI = _Boom
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    client = EmbeddingClient(EmbedderConfig(provider="openai", dimension=6))
    out = await client.embed(["a"])
    assert len(out[0]) == 6  # fell back to mock at correct dim


async def test_cohere_failure_falls_back_to_mock(monkeypatch):
    fake_cohere = types.ModuleType("cohere")

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no cohere")

    fake_cohere.AsyncClient = _Boom
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere)
    client = EmbeddingClient(EmbedderConfig(provider="cohere", dimension=7))
    out = await client.embed(["a"])
    assert len(out[0]) == 7


async def test_local_failure_falls_back_to_mock(monkeypatch):
    fake_st = types.ModuleType("sentence_transformers")

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no model")

    fake_st.SentenceTransformer = _Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    client = EmbeddingClient(EmbedderConfig(provider="local", dimension=8))
    out = await client.embed(["a"])
    assert len(out[0]) == 8
