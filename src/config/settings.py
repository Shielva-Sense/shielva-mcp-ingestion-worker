"""Ingestion Worker — SealedSettings.

Embedding API keys + Supabase URI carry secrets — must come from envelope
decryption, file-mount, or kwargs. No plaintext defaults.
"""
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, SecretStr

from shielva_common.config.sealed import SealedSettings, sealed_field


def _default_ingest_concurrency() -> int:
    """INITIAL concurrency target. The pool is elastic (see queue.py) — it floats
    between ``ingest_min_concurrency`` and ``ingest_max_concurrency`` based on live
    system load, so this is only the starting point, not a hard cap. Start modest
    (~cpu/2) and let the controller scale up if there's backlog + CPU headroom."""
    cpus = os.cpu_count() or 4
    return max(2, min(8, cpus // 2))


def _default_max_concurrency() -> int:
    """Ceiling for the elastic pool. Ingest is mostly I/O (embedding/index network),
    so concurrency above core count is useful, but bound it so we never thrash —
    the controller only reaches this when load is low and backlog is high."""
    cpus = os.cpu_count() or 4
    return max(4, min(16, cpus * 2))


class IngestionSettings(SealedSettings):
    # ── Service identity ────────────────────────────────────────────────
    service_name: str = Field("shielva-ingestion-worker", validation_alias="SERVICE_NAME")
    environment: str = Field("development", validation_alias="ENVIRONMENT")

    # ── HTTP ────────────────────────────────────────────────────────────
    port: int = Field(8007, validation_alias="INGESTION_PORT")
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "https://localhost:3010",
            "https://localhost:3001",
            "https://localhost:3000",
        ],
        validation_alias="CORS_ORIGINS_LIST",
    )

    # ── Embedding (SECRET — API keys) ───────────────────────────────────
    embedding_provider: str = Field("gemini", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        "models/gemini-embedding-001", validation_alias="EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(768, validation_alias="EMBEDDING_DIMENSIONS")
    gemini_api_key: SecretStr = sealed_field(
        SecretStr(""), env="GEMINI_API_KEY", file_env="GEMINI_API_KEY_FILE",
    )
    openai_api_key: SecretStr = sealed_field(
        SecretStr(""), env="OPENAI_API_KEY", file_env="OPENAI_API_KEY_FILE",
    )

    # ── Chunking (non-secret) ───────────────────────────────────────────
    chunk_size: int = Field(512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(50, validation_alias="CHUNK_OVERLAP")
    chunking_strategy: str = Field("recursive", validation_alias="CHUNKING_STRATEGY")

    # ── Vector store (SECRET) ───────────────────────────────────────────
    supabase_db_url: SecretStr = sealed_field(
        ...,
        env="SUPABASE_DB_URL",
        file_env="SUPABASE_DB_URL_FILE",
    )
    supabase_collection_prefix: str = Field(
        "shielva_kb_", validation_alias="SUPABASE_COLLECTION_PREFIX",
    )

    # ── Ingest payload caps (CC6.6 — bound blast radius) ────────────────
    max_entries_per_batch: int = Field(
        100, validation_alias="INGEST_MAX_ENTRIES",
    )
    # Per-document cap for the DIRECT multipart path (/ingest/file, /ingest/sync).
    # The ARC UI routes files > 8 MB to the streaming R2 path (which is NOT byte-capped),
    # and falls back to this direct path when R2 is unavailable — so this cap must
    # comfortably exceed the UI's 8 MB threshold or legitimate documents 413. The old
    # 256 KB default rejected nearly every real PDF/DOCX/deck (→ worker 413 → core-api 502).
    # 25 MB covers typical KB documents; large files still stream via R2. Override per
    # deployment with INGEST_MAX_DOC_BYTES / INGEST_MAX_TOTAL_BYTES.
    max_total_bytes_per_batch: int = Field(
        50 * 1024 * 1024, validation_alias="INGEST_MAX_TOTAL_BYTES",
    )
    max_bytes_per_document: int = Field(
        25 * 1024 * 1024, validation_alias="INGEST_MAX_DOC_BYTES",
    )
    # Per-tenant rate limit
    ingest_rps: float = Field(10.0, validation_alias="INGEST_RPS")
    ingest_burst: float = Field(20.0, validation_alias="INGEST_BURST")

    # ── Async ingest queue (bounded executor) ───────────────────────────
    # Jobs run on a fixed pool of workers (the "work queue" = active slots)
    # draining a bounded waiting queue. A burst of ingests queues instead of
    # crashing the worker; when the waiting queue is full, /ingest/* returns
    # 429 (retry later). Embedding is the bottleneck, so keep concurrency low.
    ingest_concurrency: int = Field(default_factory=_default_ingest_concurrency, validation_alias="INGEST_CONCURRENCY")
    ingest_waiting_queue_max: int = Field(100, validation_alias="INGEST_WAITING_QUEUE_MAX")
    # Elastic pool bounds + load watermarks. The number of concurrent workers floats
    # between min and max driven by the 1-min system load average per core: scale up
    # below `load_low` when there's backlog, scale down above `load_high` (system-wide
    # load, so we yield to other VM processes too). All overridable; nothing hardcoded.
    ingest_min_concurrency: int = Field(1, validation_alias="INGEST_MIN_CONCURRENCY")
    ingest_max_concurrency: int = Field(default_factory=_default_max_concurrency, validation_alias="INGEST_MAX_CONCURRENCY")
    ingest_load_high: float = Field(1.0, validation_alias="INGEST_LOAD_HIGH")   # loadavg/core ≥ this → scale down
    ingest_load_low: float = Field(0.7, validation_alias="INGEST_LOAD_LOW")     # loadavg/core < this → may scale up
    ingest_control_interval: float = Field(3.0, validation_alias="INGEST_CONTROL_INTERVAL")

    # ── Audit + principal HMAC (SECRET) ─────────────────────────────────
    audit_hmac_secret: SecretStr = sealed_field(
        ...,
        env="AUDIT_HMAC_SECRET",
        file_env="AUDIT_HMAC_SECRET_FILE",
    )
    gateway_principal_hmac_key: SecretStr = sealed_field(
        SecretStr(""),
        env="GATEWAY_PRINCIPAL_HMAC_KEY",
        file_env="GATEWAY_PRINCIPAL_HMAC_KEY_FILE",
    )

    # ── Redis (SECRET) ──────────────────────────────────────────────────
    redis_url: SecretStr = sealed_field(
        SecretStr(""), env="REDIS_URL", file_env="REDIS_URL_FILE",
    )

    # ── Observability ────────────────────────────────────────────────────
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")


@lru_cache()
def get_settings() -> IngestionSettings:
    return IngestionSettings()
