"""Ingestion Worker — SealedSettings.

Embedding API keys + Supabase URI carry secrets — must come from envelope
decryption, file-mount, or kwargs. No plaintext defaults.
"""
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, SecretStr

from shielva_common.config.sealed import SealedSettings, sealed_field


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
    max_total_bytes_per_batch: int = Field(
        5 * 1024 * 1024, validation_alias="INGEST_MAX_TOTAL_BYTES",
    )
    max_bytes_per_document: int = Field(
        256 * 1024, validation_alias="INGEST_MAX_DOC_BYTES",
    )
    # Per-tenant rate limit
    ingest_rps: float = Field(10.0, validation_alias="INGEST_RPS")
    ingest_burst: float = Field(20.0, validation_alias="INGEST_BURST")

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
