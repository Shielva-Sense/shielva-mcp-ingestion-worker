"""Shared test fixtures + hermetic import shims.

The ingestion worker imports a private company package (``shielva_common``) and
the ``vecs`` pgvector client that are not available in CI's test image. Both are
injected as lightweight fakes into ``sys.modules`` BEFORE any application module
is imported, so every module under test imports cleanly and every external I/O
boundary (DB, R2, embedding providers, HTTP) is mocked in the individual tests.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# ── Put the repo root on sys.path so ``import src...`` / ``import main`` work ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Env the sealed settings + app import path need (set BEFORE app import) ─────
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://u:p@localhost:5432/test")
os.environ.setdefault("AUDIT_HMAC_SECRET", "test-audit-secret")
os.environ.setdefault("GATEWAY_PRINCIPAL_HMAC_KEY", "test-principal-key")
os.environ.setdefault("LOG_LEVEL", "INFO")


# ── Fake ``shielva_common`` package ───────────────────────────────────────────
def _install_shielva_common() -> None:
    if "shielva_common" in sys.modules:
        return

    from pydantic import Field, SecretStr
    from pydantic_settings import BaseSettings, SettingsConfigDict

    pkg = types.ModuleType("shielva_common")
    pkg.__path__ = []  # mark as package
    sys.modules["shielva_common"] = pkg

    # shielva_common.config + shielva_common.config.sealed
    config_mod = types.ModuleType("shielva_common.config")
    config_mod.__path__ = []
    sealed_mod = types.ModuleType("shielva_common.config.sealed")

    class SealedSettings(BaseSettings):
        model_config = SettingsConfigDict(
            extra="ignore",
            populate_by_name=True,
            env_file=None,
        )

    def sealed_field(default: Any = ..., *, env: Optional[str] = None, file_env: Optional[str] = None):
        if default is ...:
            return Field(..., validation_alias=env)
        return Field(default, validation_alias=env)

    sealed_mod.SealedSettings = SealedSettings
    sealed_mod.sealed_field = sealed_field
    sealed_mod.SecretStr = SecretStr
    config_mod.sealed = sealed_mod
    sys.modules["shielva_common.config"] = config_mod
    sys.modules["shielva_common.config.sealed"] = sealed_mod
    pkg.config = config_mod

    # shielva_common.envelope
    envelope_mod = types.ModuleType("shielva_common.envelope")
    envelope_mod.bootstrap = lambda *a, **k: None
    sys.modules["shielva_common.envelope"] = envelope_mod
    pkg.envelope = envelope_mod

    # shielva_common.auth
    auth_mod = types.ModuleType("shielva_common.auth")

    @dataclass
    class Principal:
        tenant_id: str = "tenant-test"
        user_id: str = "user-test"
        email: str = "tester@shielva.ai"
        roles: tuple = ()

    async def require_principal() -> "Principal":  # overridden in app tests
        return Principal()

    auth_mod.Principal = Principal
    auth_mod.require_principal = require_principal
    sys.modules["shielva_common.auth"] = auth_mod
    pkg.auth = auth_mod

    # shielva_common.ratelimit
    ratelimit_mod = types.ModuleType("shielva_common.ratelimit")

    def per_tenant_rate_limit(name: str, rps: float = 0.0, burst: float = 0.0):
        async def _dep() -> None:  # a no-op FastAPI dependency
            return None

        return _dep

    class PerTenantTokenBucket:
        def __init__(self, *a: Any, **k: Any) -> None:
            self._a = a

        async def take(self, *a: Any, **k: Any) -> bool:
            return True

    ratelimit_mod.per_tenant_rate_limit = per_tenant_rate_limit
    ratelimit_mod.PerTenantTokenBucket = PerTenantTokenBucket
    sys.modules["shielva_common.ratelimit"] = ratelimit_mod
    pkg.ratelimit = ratelimit_mod

    # shielva_common.tls
    tls_mod = types.ModuleType("shielva_common.tls")
    tls_mod.internal_ca_verify = lambda: False
    sys.modules["shielva_common.tls"] = tls_mod
    pkg.tls = tls_mod


def _install_vecs() -> None:
    """A stand-in for the ``vecs`` client so ``supabase_store`` imports.

    Every DB call is exercised in tests via a MagicMock assigned to
    ``store._client``; this fake only needs to satisfy the module-level import.
    """
    if "vecs" in sys.modules:
        return
    vecs_mod = types.ModuleType("vecs")
    vecs_mod.__path__ = []

    class Client:  # minimal marker type
        pass

    def create_client(url: str) -> "Client":
        return Client()

    vecs_mod.Client = Client
    vecs_mod.create_client = create_client

    collection_mod = types.ModuleType("vecs.collection")

    class Collection:  # marker type
        pass

    collection_mod.Collection = Collection
    vecs_mod.collection = collection_mod
    sys.modules["vecs"] = vecs_mod
    sys.modules["vecs.collection"] = collection_mod


_install_shielva_common()
_install_vecs()


# ── Reusable domain fixtures ──────────────────────────────────────────────────
@pytest.fixture()
def make_document():
    from src.models import Document, DocumentType

    def _make(content: str = "hello world", **kw: Any) -> Document:
        params: Dict[str, Any] = dict(
            id="doc-1",
            tenant_id="tenant-test",
            kb_id="kb-1",
            content=content,
            title="Doc",
            doc_type=DocumentType.TEXT,
        )
        params.update(kw)
        return Document(**params)

    return _make


@dataclass
class FakeEmbeddingClient:
    """Deterministic embedder — returns a fixed-dim vector per text."""

    dim: int = 4

    async def embed(self, texts):
        return [[float(len(t) % 7)] * self.dim for t in texts]

    async def embed_single(self, text):
        out = await self.embed([text])
        return out[0]


@dataclass
class FakeIndexer:
    calls: list = field(default_factory=list)

    async def index_chunks(self, chunks, tenant_id, kb_id):
        self.calls.append((len(chunks), tenant_id, kb_id))
        return chunks

    async def delete_by_document(self, document_id, tenant_id, kb_id):
        return 3


@pytest.fixture()
def fake_embedding_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture()
def fake_indexer() -> FakeIndexer:
    return FakeIndexer()
