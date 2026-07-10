"""Sealed settings — defaults, secret resolution, concurrency factories."""

from __future__ import annotations

from src.config import get_settings
from src.config.settings import (
    IngestionSettings,
    _default_ingest_concurrency,
    _default_max_concurrency,
)


def test_default_concurrency_factories_are_bounded():
    ic = _default_ingest_concurrency()
    assert 2 <= ic <= 8
    mc = _default_max_concurrency()
    assert 4 <= mc <= 16


def test_settings_load_with_required_secrets():
    # conftest exports SUPABASE_DB_URL + AUDIT_HMAC_SECRET
    s = IngestionSettings()
    assert s.service_name == "shielva-ingestion-worker"
    assert s.port == 8007
    assert s.embedding_dimensions == 768
    assert s.supabase_db_url.get_secret_value().startswith("postgresql://")
    assert s.audit_hmac_secret.get_secret_value() == "test-audit-secret"


def test_defaults_for_caps_and_chunking():
    s = IngestionSettings()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50
    assert s.max_entries_per_batch == 100
    assert s.max_bytes_per_document == 25 * 1024 * 1024
    assert s.chunking_strategy == "recursive"


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("INGESTION_PORT", "9099")
    s = IngestionSettings()
    assert s.chunk_size == 256
    assert s.port == 9099
