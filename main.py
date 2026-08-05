"""
Ingestion Worker - FastAPI Service
Handles document ingestion jobs from connectors.

Hardening (2026-05-29):
  * Tenant is derived from the gateway-verified Principal — never from
    a raw ``X-Tenant-ID`` header — so a process that reaches the
    sidecar port cannot impersonate.
  * Per-batch caps bound blast radius (max entries, max total bytes,
    max bytes per document) — see config.settings.
  * Per-tenant token-bucket rate limit on ingest endpoints prevents
    one noisy tenant from monopolising the worker.
  * Settings come from SealedSettings — secrets MUST come from
    envelope decryption / file mount.
"""

# ── Envelope decryption (must run BEFORE any settings/env-reading imports) ──
import os as _envelope_os

_envelope_os.environ.setdefault("VAULT_SIDECAR_URL", "https://localhost:8054")
from dotenv import load_dotenv as _envelope_load_dotenv

_envelope_load_dotenv(".env", override=True)  # ciphertext + REDIS_URL passthrough
from shielva_common.envelope import bootstrap as _envelope_bootstrap

_envelope_bootstrap()
# ──────────────────────────────────────────────────────────────────────────

import uuid as _uuid_mod
import structlog.contextvars as _structlog_cv

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import httpx
from urllib.parse import urlparse
import json
import structlog
import uvicorn

from shielva_common.auth import Principal, require_principal
from shielva_common.ratelimit import per_tenant_rate_limit, PerTenantTokenBucket
from shielva_common.tls import internal_ca_verify

from src.config import get_settings
import os  # noqa: E402 — plain os (the envelope alias above shadows the name)
from src.pipeline import IngestionPipeline
from src.sources import fetch_url, read_database, read_api
from src.sources.politeness import RobotsDisallowed
from src.models import (
    Document,
    DocumentType,
    ChunkingStrategy,
)
from src.chunker import Chunker
from src.fetcher import DocumentFetcher
from src.indexer import VectorIndexer
from src.embedder import EmbeddingClient, EmbedderConfig
from src.jobs.manager import job_manager
from src.jobs.processor import JobProcessor
from src.jobs import queue as _ingest_queue_mod
from src.jobs.queue import IngestQueue, QueueFull

from src.vectorstore import SupabaseVectorStore

# Logging setup — canonical structlog config with contextvars support
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # NOTE: add_logger_name is intentionally omitted — it requires a stdlib
        # logger with a `.name`, which PrintLoggerFactory's PrintLogger lacks
        # (AttributeError at startup). Same fix applied in presence-core /
        # knowledge-manager.
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        __import__("logging").getLevelName(__import__("os").environ.get("LOG_LEVEL", "INFO").upper())
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger(__name__)


# ===== Request/Response Models =====


class IngestDocumentRequest(BaseModel):
    """Single document ingestion request"""

    id: str
    content: str
    title: str
    doc_type: str = "text"
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = {}


class IngestBatchRequest(BaseModel):
    """Batch document ingestion request"""

    kb_id: str
    documents: List[IngestDocumentRequest]
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    webhook_url: Optional[str] = None


class IngestResponse(BaseModel):
    """Ingestion response"""

    job_id: str
    status: str
    message: str
    documents_queued: int = 0


class JobStatusResponse(BaseModel):
    """Job status response"""

    job_id: str
    status: str
    documents_total: int
    documents_processed: int
    documents_failed: int
    chunks_created: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = []


class DeleteDocumentRequest(BaseModel):
    """Delete document request"""

    document_id: str
    kb_id: str


# ===== Global State =====

pipeline: Optional[IngestionPipeline] = None
processor: Optional[JobProcessor] = None


# ===== Helpers =====


def _validate_batch_caps(batch: IngestBatchRequest) -> None:
    """Reject oversized ingest payloads — CC6.6 bounds blast radius."""
    settings = get_settings()
    if len(batch.documents) > settings.max_entries_per_batch:
        raise HTTPException(
            status_code=413,
            detail=(f"too many entries (limit {settings.max_entries_per_batch}, got {len(batch.documents)})"),
        )
    total = sum(len(d.content or "") for d in batch.documents)
    if total > settings.max_total_bytes_per_batch:
        raise HTTPException(
            status_code=413,
            detail=(f"batch too large (limit {settings.max_total_bytes_per_batch} bytes, got {total})"),
        )
    for d in batch.documents:
        if len(d.content or "") > settings.max_bytes_per_document:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"document {d.id!r} too large "
                    f"(limit {settings.max_bytes_per_document} bytes, "
                    f"got {len(d.content or '')})"
                ),
            )


# ===== Application =====


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    global pipeline, processor

    logger.info("Starting Ingestion Worker")

    settings = get_settings()

    # Resolve secrets via SealedSettings — fail-closed if absent.
    supabase_db_url = settings.supabase_db_url.get_secret_value()
    embedding_api_key = settings.gemini_api_key.get_secret_value() or settings.openai_api_key.get_secret_value()

    vector_store = SupabaseVectorStore(
        db_url=supabase_db_url,
        collection_prefix=settings.supabase_collection_prefix,
        embedding_dim=settings.embedding_dimensions,
    )
    await vector_store.connect()

    embedding_config = EmbedderConfig(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=embedding_api_key,
        dimension=settings.embedding_dimensions,
    )

    embedding_client = EmbeddingClient(config=embedding_config)

    # External HTTP client uses the internal Shielva CA bundle —
    # TLS verification is mandatory in production (CC6.6 / C1.1).
    http_client = httpx.AsyncClient(timeout=60.0, verify=internal_ca_verify())
    fetcher = DocumentFetcher(http_client=http_client)

    indexer = VectorIndexer(vector_store=vector_store)

    strategy = ChunkingStrategy(settings.chunking_strategy)
    chunker = Chunker(
        strategy=strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_client=embedding_client,
        chunker=chunker,
        indexer=indexer,
        fetcher=fetcher,
    )
    processor = JobProcessor(pipeline)

    # Elastic, load-aware async ingest executor: concurrency floats between
    # min/max driven by system load; bounded waiting queue; decoupled completion
    # queue for response delivery. Started here so it binds to the running loop.
    _ingest_queue_mod.ingest_queue = IngestQueue(
        initial=settings.ingest_concurrency,
        min_workers=settings.ingest_min_concurrency,
        max_workers=settings.ingest_max_concurrency,
        waiting_max=settings.ingest_waiting_queue_max,
        load_high=settings.ingest_load_high,
        load_low=settings.ingest_load_low,
        control_interval=settings.ingest_control_interval,
    )
    _ingest_queue_mod.ingest_queue.start()

    # Redis-backed token bucket for per-tenant rate limiting. Falls
    # back to a local-only bucket when REDIS_URL is unset (dev only).
    redis_url = settings.redis_url.get_secret_value()
    if redis_url:
        try:
            import redis.asyncio as redis_asyncio

            redis_client = redis_asyncio.from_url(redis_url, decode_responses=False)
            app.state.token_bucket = PerTenantTokenBucket(redis_client)
            logger.info("rate_limit_backend", backend="redis")
        except Exception as e:  # noqa: BLE001
            logger.warning("rate_limit_redis_init_failed", error=str(e))
            app.state.token_bucket = None
    else:
        logger.warning("rate_limit_disabled_no_redis_url")
        app.state.token_bucket = None

    logger.info("Ingestion Worker initialized")

    yield

    # Cleanup
    if _ingest_queue_mod.ingest_queue is not None:
        await _ingest_queue_mod.ingest_queue.stop()
    await vector_store.close()
    await http_client.aclose()
    logger.info("Ingestion Worker shutdown")


app = FastAPI(
    title="Shielva Ingestion Worker",
    version="1.0.0",
    description="Document ingestion pipeline for Shielva ARC",
    lifespan=lifespan,
)

from src.core.error_handlers import install_exception_handlers as _install_exc

_install_exc(app)


@app.middleware("http")
async def _correlation_id_middleware(request: Request, call_next):
    cid = request.headers.get("X-Correlation-Id") or _uuid_mod.uuid4().hex
    _structlog_cv.clear_contextvars()
    _structlog_cv.bind_contextvars(correlation_id=cid)
    request.state.request_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = cid
    return response


def _cors_origins() -> List[str]:
    try:
        return get_settings().cors_origins
    except Exception:
        return [
            "https://localhost:3010",
            "https://localhost:3000",
        ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Endpoints =====


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "ingestion-worker",
        "pipeline_ready": pipeline is not None,
    }


# Build the rate-limit dependency once — it reads the settings (which
# could vary by env) at module import time.
_ingest_settings = get_settings()
_ingest_rate_limit = per_tenant_rate_limit(
    "mcp_ingest",
    rps=_ingest_settings.ingest_rps,
    burst=_ingest_settings.ingest_burst,
)


def _submit_to_queue(job, run) -> None:
    """Enqueue (job, run) on the bounded ingest queue. Raises 503 if the queue
    isn't ready, 429 (Retry-After) when the waiting queue is at capacity — so a
    burst of ingests queues or backs off cleanly instead of crashing the worker."""
    q = _ingest_queue_mod.ingest_queue
    if q is None:
        raise HTTPException(status_code=503, detail="Ingest queue not initialised")
    try:
        q.submit(job, run)
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"})


def _queued_job_response(job) -> "JobStatusResponse":
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        documents_total=job.documents_total,
        documents_processed=job.documents_processed,
        documents_failed=job.documents_failed,
        chunks_created=job.chunks_created,
        started_at=job.started_at,
        completed_at=job.completed_at,
        errors=job.errors,
    )


@app.get("/queue/stats")
async def queue_stats(principal: Principal = Depends(require_principal)):
    """Live queue depth: active (work queue) + waiting + capacity."""
    q = _ingest_queue_mod.ingest_queue
    return q.stats() if q is not None else {"error": "queue not initialised"}


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
    dependencies=[Depends(_ingest_rate_limit)],
)
async def ingest_documents(
    request: IngestBatchRequest,
    principal: Principal = Depends(require_principal),
):
    """Ingest a batch of documents (async background processing).

    Tenant is taken from the verified principal — body / header trust
    removed. Caps:
      * <= settings.max_entries_per_batch entries
      * <= settings.max_total_bytes_per_batch total bytes
      * <= settings.max_bytes_per_document per document
    """
    _validate_batch_caps(request)

    job = job_manager.create_job(
        tenant_id=principal.tenant_id,
        kb_id=request.kb_id,
        documents_count=len(request.documents),
        webhook_url=request.webhook_url,
    )

    documents = []
    for doc_req in request.documents:
        doc_type = DocumentType(doc_req.doc_type) if doc_req.doc_type else DocumentType.TEXT
        documents.append(
            Document(
                id=doc_req.id,
                tenant_id=principal.tenant_id,
                kb_id=request.kb_id,
                content=doc_req.content,
                title=doc_req.title,
                source_url=doc_req.source_url,
                doc_type=doc_type,
                metadata=doc_req.metadata,
            )
        )

    _submit_to_queue(job, lambda: processor.process_job(job, documents))

    return IngestResponse(
        job_id=job.job_id,
        status="queued",
        message="Ingestion queued",
        documents_queued=len(request.documents),
    )


@app.post(
    "/ingest/sync",
    response_model=JobStatusResponse,
    dependencies=[Depends(_ingest_rate_limit)],
)
async def ingest_documents_sync(
    request: IngestBatchRequest,
    principal: Principal = Depends(require_principal),
):
    """Synchronous ingest — same caps + rate limiting as async path."""
    _validate_batch_caps(request)

    job = job_manager.create_job(
        tenant_id=principal.tenant_id,
        kb_id=request.kb_id,
        documents_count=len(request.documents),
        webhook_url=request.webhook_url,
    )

    documents = []
    for doc_req in request.documents:
        doc_type = DocumentType(doc_req.doc_type) if doc_req.doc_type else DocumentType.TEXT
        documents.append(
            Document(
                id=doc_req.id,
                tenant_id=principal.tenant_id,
                kb_id=request.kb_id,
                content=doc_req.content,
                title=doc_req.title,
                source_url=doc_req.source_url,
                doc_type=doc_type,
                metadata=doc_req.metadata,
            )
        )

    await processor.process_job(job, documents)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        documents_total=job.documents_total,
        documents_processed=job.documents_processed,
        documents_failed=job.documents_failed,
        chunks_created=job.chunks_created,
        started_at=job.started_at,
        completed_at=job.completed_at,
        errors=job.errors,
    )


# Map an uploaded file's extension to its document type. Unknown → treated as text.
_EXT_TO_DOCTYPE = {
    "txt": DocumentType.TEXT,
    "text": DocumentType.TEXT,
    "log": DocumentType.TEXT,
    "md": DocumentType.MARKDOWN,
    "markdown": DocumentType.MARKDOWN,
    "html": DocumentType.HTML,
    "htm": DocumentType.HTML,
    "pdf": DocumentType.PDF,
    "docx": DocumentType.DOCX,
    "csv": DocumentType.CSV,
    "json": DocumentType.JSON,
    "xlsx": DocumentType.XLSX,
    "xlsm": DocumentType.XLSX,
    "pptx": DocumentType.PPTX,
}
# Formats whose parser needs raw bytes (the rest are decoded to text first).
_BINARY_DOCTYPES = {DocumentType.PDF, DocumentType.DOCX, DocumentType.XLSX, DocumentType.PPTX}


def _doctype_for_filename(name: str) -> DocumentType:
    ext = (name.rsplit(".", 1)[-1] if "." in name else "").lower()
    return _EXT_TO_DOCTYPE.get(ext, DocumentType.TEXT)


def _stable_doc_id(label: str, kb_id: str, key: str, content) -> str:
    """Deterministic document id from (kb, source key, content). Re-ingesting the
    SAME content yields the SAME id → the pipeline upserts (chunk ids are derived
    from doc id), so duplicates aren't created. Different content → different id
    (a new doc; the old one is superseded on the next full sync)."""
    import hashlib

    h = hashlib.sha256()
    h.update((kb_id or "").encode("utf-8"))
    h.update(b"|")
    h.update((key or "").encode("utf-8"))
    h.update(b"|")
    h.update(content if isinstance(content, (bytes, bytearray)) else (content or "").encode("utf-8", "ignore"))
    return f"{label}_{h.hexdigest()[:24]}"


@app.post(
    "/ingest/file",
    response_model=JobStatusResponse,
    status_code=202,
    dependencies=[Depends(_ingest_rate_limit)],
)
async def ingest_files(
    kb_id: str = Form(...),
    files: List[UploadFile] = File(...),
    guardrails: str = Form("{}"),
    webhook_url: str = Form(""),
    principal: Principal = Depends(require_principal),
):
    """Ingest uploaded files of ANY supported format into a KB.

    Each file is routed to its format's parser (PDF/DOCX/XLSX/PPTX parsed from raw
    bytes; text/markdown/html/csv/json decoded), then chunked, embedded and indexed
    into the per-(tenant, kb) pgvector collection. Tenant is the verified principal —
    never trusted from the body. Same size caps as the JSON ingest path.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    caps = get_settings()
    job = job_manager.create_job(
        tenant_id=principal.tenant_id,
        kb_id=kb_id,
        documents_count=len(files),
        webhook_url=webhook_url or None,
    )

    try:
        _guardrails = json.loads(guardrails) if guardrails else {}
    except Exception:
        _guardrails = {}

    documents = []
    total_bytes = 0
    for f in files:
        raw = await f.read()
        total_bytes += len(raw)
        if len(raw) > caps.max_bytes_per_document:
            raise HTTPException(status_code=413, detail=f"{f.filename} exceeds per-document size cap")
        doc_type = _doctype_for_filename(f.filename or "")
        content = raw if doc_type in _BINARY_DOCTYPES else raw.decode("utf-8", errors="ignore")
        documents.append(
            Document(
                id=_stable_doc_id("upload", kb_id, f.filename or "", raw),
                tenant_id=principal.tenant_id,
                kb_id=kb_id,
                content=content,
                title=f.filename or "uploaded-file",
                source_url=None,
                doc_type=doc_type,
                metadata={"upload": True, "filename": f.filename, "size": len(raw), "_guardrails": _guardrails},
            )
        )
    if total_bytes > caps.max_total_bytes_per_batch:
        raise HTTPException(status_code=413, detail="Upload exceeds total batch size cap")

    # Files are already read into memory above (the multipart body can't be read
    # after we respond), so enqueue the parse/chunk/embed/index work and return 202.
    _submit_to_queue(job, lambda: processor.process_job(job, documents))
    return _queued_job_response(job)


class IngestR2Request(BaseModel):
    """Ingest a single object already uploaded to R2 by the browser.

    The win over /ingest/file: core-api never touches the bytes (the browser PUTs
    straight to R2 via a presigned URL), and the worker STREAMS the object from R2
    instead of receiving a buffered multipart body — so a multi-GB file never gets
    fully buffered in core-api's memory.
    """

    kb_id: str
    key: str
    filename: str
    bucket: Optional[str] = None
    guardrails: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None


@app.post(
    "/ingest/r2",
    response_model=JobStatusResponse,
    status_code=202,
    dependencies=[Depends(_ingest_rate_limit)],
)
async def ingest_r2(
    body: IngestR2Request,
    principal: Principal = Depends(require_principal),
):
    """Stream a single uploaded object from R2 and ingest it into a KB.

    Mirrors ``ingest_files`` (same Document build / doc_type routing / _guardrails
    metadata / _stable_doc_id / process_job / return shape) — the only difference
    is the bytes come from a streamed R2 GET instead of a multipart upload, so
    core-api never buffers them. Tenant is the verified principal; the object key
    is expected to be tenant-scoped (core-api validates the prefix before
    forwarding) but the worker re-checks it as defence in depth.
    """
    from src.fetcher import fetch_r2_object

    if not body.key or not body.filename:
        raise HTTPException(status_code=400, detail="key and filename are required")

    # Defence in depth: the object key MUST start with this tenant's prefix. Reject
    # path traversal and cross-tenant keys even though core-api already validated —
    # the worker must never trust that an upstream check ran.
    if ".." in body.key or body.key.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid object key")
    expected_prefix = f"{principal.tenant_id}/{body.kb_id}/"
    if not body.key.startswith(expected_prefix):
        raise HTTPException(status_code=403, detail="Object key is outside this tenant/KB prefix")

    caps = get_settings()
    bucket = body.bucket or os.environ.get("R2_KNOWLEDGE_BUCKET", "") or "shielva-arc-knowledge"
    doc_type = _doctype_for_filename(body.filename or "")

    job = job_manager.create_job(
        tenant_id=principal.tenant_id,
        kb_id=body.kb_id,
        documents_count=1,
        webhook_url=body.webhook_url,
    )

    # All R2 ingest work runs on the bounded queue and returns 202 immediately; the
    # completion webhook is fired by the queue worker loop. TRUE streaming (constant
    # RAM, never fully buffered): TEXT-like → read R2 in 1 MiB windows, decode + chunk
    # + embed incrementally; PDF → temp file then page-by-page. docx/xlsx + fallback
    # use a bytes fetch. Failures here mark the job failed (the client already has 202).
    from src.streaming import (
        STREAMABLE_DOCTYPES,
        OFFICE_STREAMABLE_DOCTYPES,
        stream_ingest_r2,
        stream_ingest_pdf_r2,
        stream_ingest_office_r2,
    )

    _stream_fn = (
        stream_ingest_r2
        if doc_type in STREAMABLE_DOCTYPES
        else stream_ingest_pdf_r2
        if doc_type == DocumentType.PDF
        else stream_ingest_office_r2
        if doc_type in OFFICE_STREAMABLE_DOCTYPES
        else None
    )

    async def _run() -> None:
        if _stream_fn is not None:
            document = Document(
                id=_stable_doc_id("upload", body.kb_id, body.key, b""),  # stable from the uuid'd key
                tenant_id=principal.tenant_id,
                kb_id=body.kb_id,
                content="",  # streamed, not buffered
                title=body.filename or "uploaded-file",
                source_url=None,
                doc_type=doc_type,
                metadata={"upload": True, "filename": body.filename, "r2_key": body.key, "streamed": True},
            )
            try:
                chunks = await _stream_fn(
                    bucket=bucket,
                    key=body.key,
                    document=document,
                    guardrails=body.guardrails or {},
                    pipeline=pipeline,
                )
                job.chunks_created = chunks
                job.documents_processed = 1
                job.documents_failed = 0
                job.status = "completed"
            except Exception as exc:  # noqa: BLE001
                logger.error("stream_ingest_r2_failed", key=body.key, error=str(exc))
                job.documents_failed = 1
                job.status = "failed"
                job.errors.append(str(exc))
            job.completed_at = datetime.utcnow()
            return

        # Binary (pdf/docx/xlsx) → full-load path: the parser needs the whole bytes.
        result = await fetch_r2_object(bucket, body.key)
        if result.error:
            job.status = "failed"
            job.documents_failed = 1
            job.errors.append(f"R2 fetch failed: {result.error}")
            job.completed_at = datetime.utcnow()
            return

        raw = result.content
        if len(raw) > caps.max_bytes_per_document:
            job.status = "failed"
            job.documents_failed = 1
            job.errors.append(f"{body.filename} exceeds per-document size cap")
            job.completed_at = datetime.utcnow()
            return

        content = raw if doc_type in _BINARY_DOCTYPES else raw.decode("utf-8", errors="ignore")
        document = Document(
            id=_stable_doc_id("upload", body.kb_id, body.filename or body.key, raw),
            tenant_id=principal.tenant_id,
            kb_id=body.kb_id,
            content=content,
            title=body.filename or "uploaded-file",
            source_url=None,
            doc_type=doc_type,
            metadata={
                "upload": True,
                "filename": body.filename,
                "size": len(raw),
                "r2_key": body.key,
                "_guardrails": body.guardrails or {},
            },
        )
        await processor.process_job(job, [document])

    _submit_to_queue(job, _run)
    return _queued_job_response(job)


class IngestUrlRequest(BaseModel):
    kb_id: str
    url: str
    crawl: bool = False
    max_pages: int = 20
    max_depth: int = 2
    webhook_url: Optional[str] = None


class IngestDatabaseRequest(BaseModel):
    kb_id: str
    db_type: str
    connection_uri: str
    query: Optional[str] = None
    collection: Optional[str] = None
    limit: int = 1000
    webhook_url: Optional[str] = None


class IngestApiRequest(BaseModel):
    kb_id: str
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict[str, Any]] = None
    json_path: Optional[str] = None
    limit: int = 500
    webhook_url: Optional[str] = None


async def _ingest_source_docs(
    kb_id: str, tenant_id: str, docs, label: str, webhook_url: Optional[str] = None
) -> "JobStatusResponse":
    """Wrap (title, content, doc_type) tuples from a source adapter into Documents
    and enqueue the embed/index work on the bounded queue (returns 202/queued)."""
    if not docs:
        raise HTTPException(status_code=422, detail="Source returned no content")
    job = job_manager.create_job(tenant_id=tenant_id, kb_id=kb_id, documents_count=len(docs), webhook_url=webhook_url)
    documents = []
    for title, content, doc_type in docs:
        try:
            dt = DocumentType(doc_type)
        except ValueError:
            dt = DocumentType.TEXT
        documents.append(
            Document(
                id=_stable_doc_id(label, kb_id, title, content),
                tenant_id=tenant_id,
                kb_id=kb_id,
                content=content,
                title=title,
                source_url=None,
                doc_type=dt,
                metadata={"source": label},
            )
        )
    _submit_to_queue(job, lambda: processor.process_job(job, documents))
    return _queued_job_response(job)


def _source_fetch_error(url: str, exc: Exception) -> HTTPException:
    """Turn an outbound fetch failure into a clear, actionable client error.

    Only ``ValueError`` (the SSRF guard) used to be caught here, so every real
    network failure — timeout, 403, DNS — escaped as a bare 500 with a traceback.
    core-api then relayed "Ingestion enqueue failed: <traceback>" and the operator
    saw an unexplained "Network Error". These map to 400 so core-api surfaces the
    reason verbatim ("Source error: ..."), because they are all *input* problems:
    the URL the user supplied cannot be fetched.
    """
    host = urlparse(url).hostname or url
    if isinstance(exc, httpx.TimeoutException):
        return HTTPException(
            status_code=400,
            detail=(
                f"{host} did not respond in time. The site may be slow, or it may block "
                f"automated access from servers (many large sites do). Try a different URL, "
                f"or upload the content as a file instead."
            ),
        )
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403, 429):
            return HTTPException(
                status_code=400,
                detail=(
                    f"{host} refused the request (HTTP {code}) — the site blocks automated "
                    f"access. Try a different URL, or upload the content as a file instead."
                ),
            )
        return HTTPException(status_code=400, detail=f"{host} returned HTTP {code} for that URL.")
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)):
        # Connected, then the peer dropped us mid-conversation. Almost always a
        # bot-mitigation tarpit rather than a bad URL — say so, so nobody wastes
        # time re-checking a URL that is perfectly correct.
        return HTTPException(
            status_code=400,
            detail=(
                f"{host} closed the connection without responding. The site blocks "
                f"automated access (its bot protection drops non-browser clients). "
                f"Upload the content as a file instead, or use a source the publisher "
                f"offers for machines (their API, RSS feed or sitemap)."
            ),
        )
    return HTTPException(
        status_code=400,
        detail=f"Could not reach {host} ({type(exc).__name__}). Check the URL is public and reachable.",
    )


@app.post("/ingest/url", response_model=JobStatusResponse, dependencies=[Depends(_ingest_rate_limit)])
async def ingest_url(body: IngestUrlRequest, principal: Principal = Depends(require_principal)):
    """Ingest a public URL — a single page, or a same-host BFS crawl. SSRF-guarded."""
    try:
        docs = await fetch_url(body.url, crawl=body.crawl, max_pages=body.max_pages, max_depth=body.max_depth)
    except RobotsDisallowed as e:
        # The publisher declined automated access. Deliberately NOT retried or
        # worked around — we honour robots.txt.
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPError as e:
        raise _source_fetch_error(body.url, e)
    return await _ingest_source_docs(body.kb_id, principal.tenant_id, docs, "url", body.webhook_url)


@app.post("/ingest/database", response_model=JobStatusResponse, dependencies=[Depends(_ingest_rate_limit)])
async def ingest_database(body: IngestDatabaseRequest, principal: Principal = Depends(require_principal)):
    """Ingest rows/documents from a database via a read-only query."""
    try:
        docs = await read_database(
            body.db_type,
            body.connection_uri,
            query=body.query,
            collection=body.collection,
            limit=body.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await _ingest_source_docs(body.kb_id, principal.tenant_id, docs, "db", body.webhook_url)


@app.post("/ingest/api", response_model=JobStatusResponse, dependencies=[Depends(_ingest_rate_limit)])
async def ingest_api(body: IngestApiRequest, principal: Principal = Depends(require_principal)):
    """Ingest a REST API response (optionally a list via a dotted JSON path)."""
    try:
        docs = await read_api(
            body.url,
            method=body.method,
            headers=body.headers,
            body=body.body,
            json_path=body.json_path,
            limit=body.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPError as e:
        raise _source_fetch_error(body.url, e)
    return await _ingest_source_docs(body.kb_id, principal.tenant_id, docs, "api", body.webhook_url)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    principal: Principal = Depends(require_principal),
):
    """Get ingestion job status"""
    job = job_manager.get_job(job_id, principal.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        documents_total=job.documents_total,
        documents_processed=job.documents_processed,
        documents_failed=job.documents_failed,
        chunks_created=job.chunks_created,
        started_at=job.started_at,
        completed_at=job.completed_at,
        errors=job.errors,
    )


@app.get("/jobs")
async def list_jobs(
    principal: Principal = Depends(require_principal),
    status: Optional[str] = None,
):
    """List ingestion jobs for tenant"""
    jobs = job_manager.list_jobs(principal.tenant_id, status)

    tenant_jobs = [
        {
            "job_id": job.job_id,
            "kb_id": job.kb_id,
            "status": job.status,
            "documents_processed": job.documents_processed,
            "chunks_created": job.chunks_created,
            "started_at": job.started_at,
        }
        for job in jobs
    ]

    return {"jobs": tenant_jobs}


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Delete a document and its chunks"""
    try:
        deleted = await pipeline.delete_document(
            document_id=document_id,
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
        )

        return {
            "status": "deleted",
            "document_id": document_id,
            "chunks_deleted": deleted,
        }

    except Exception as e:
        logger.error("Delete failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/{kb_id}/initialize")
async def initialize_kb(
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Initialize a knowledge base (create collection)"""
    try:
        collection_name = await pipeline.vector_store.create_collection(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
        )

        return {
            "status": "initialized",
            "kb_id": kb_id,
            "collection": collection_name,
        }

    except Exception as e:
        logger.error("KB initialization failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/kb/{kb_id}")
async def delete_kb(
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Delete a knowledge base and all its data"""
    try:
        result = await pipeline.vector_store.delete_collection(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
        )

        return {
            "status": "deleted" if result else "not_found",
            "kb_id": kb_id,
        }

    except Exception as e:
        logger.error("KB deletion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/{kb_id}/info")
async def get_kb_info(
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Get knowledge base information"""
    try:
        info = await pipeline.vector_store.get_collection_info(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
        )

        return {
            "kb_id": kb_id,
            "tenant_id": principal.tenant_id,
            **info,
        }

    except Exception as e:
        logger.error("KB info failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/{kb_id}/documents")
async def list_kb_documents(
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """List the individual documents (files) ingested into a KB — id, title, chunk count."""
    try:
        docs = await pipeline.vector_store.list_documents(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
        )
        return {"kb_id": kb_id, "documents": docs}
    except Exception as e:
        logger.error("List documents failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/{kb_id}/storage")
async def kb_storage(
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Storage stats for a KB: document count, chunk (vector) count, cumulative
    file bytes. Vector byte size is derived by the caller from chunks × dim×4."""
    try:
        stats = await pipeline.vector_store.kb_storage(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
        )
        return {"kb_id": kb_id, **stats}
    except Exception as e:
        logger.error("KB storage failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/{kb_id}/cancel")
async def cancel_kb_ingestion(
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Kill in-flight / queued ingestion job(s) for a KB.

    A running job's worker task is cancelled at its current await point; a still-
    queued job is skipped when it would dispatch. Either way the job reports
    completion (status ``cancelled``) so core-api moves the KB out of "ingesting".
    """
    q = _ingest_queue_mod.ingest_queue
    n = q.cancel_by_kb(kb_id) if q is not None else 0
    logger.info("ingest_cancel_requested", kb_id=kb_id, tenant=principal.tenant_id, cancelled=n)
    return {"kb_id": kb_id, "cancelled": n}


class UpdateChunkRequest(BaseModel):
    kb_id: str
    content: str


@app.get("/kb/{kb_id}/chunks")
async def list_kb_chunks(
    kb_id: str,
    document_id: str = None,
    principal: Principal = Depends(require_principal),
):
    """List the individual chunks (fragments) of a KB — for content curation."""
    try:
        chunks = await pipeline.vector_store.list_chunks(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
            document_id=document_id,
        )
        return {"kb_id": kb_id, "chunks": chunks}
    except Exception as e:
        logger.error("List chunks failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chunks/{chunk_id}")
async def delete_chunk(
    chunk_id: str,
    kb_id: str,
    principal: Principal = Depends(require_principal),
):
    """Remove a single chunk (fragment) from a KB."""
    try:
        deleted = await pipeline.vector_store.delete_chunk(
            tenant_id=principal.tenant_id,
            kb_id=kb_id,
            chunk_id=chunk_id,
        )
        return {"status": "deleted", "chunk_id": chunk_id, "chunks_deleted": deleted}
    except Exception as e:
        logger.error("Delete chunk failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/chunks/{chunk_id}")
async def update_chunk(
    chunk_id: str,
    body: UpdateChunkRequest,
    principal: Principal = Depends(require_principal),
):
    """Edit a chunk's text and RE-EMBED it so vector retrieval stays consistent."""
    try:
        embedding = await pipeline.embedding_client.embed_single(body.content)
        ok = await pipeline.vector_store.update_chunk(
            tenant_id=principal.tenant_id,
            kb_id=body.kb_id,
            chunk_id=chunk_id,
            content=body.content,
            embedding=embedding,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return {"status": "updated", "chunk_id": chunk_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Update chunk failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ===== Run Server =====


def find_free_port(start_port: int, max_retries: int = 100) -> int:
    """Find a free port starting from start_port"""
    import socket

    for port in range(start_port, start_port + max_retries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_retries}")


def main():
    """Run the ingestion worker"""
    target_port = get_settings().port

    try:
        port = find_free_port(target_port)
    except Exception as e:
        logger.error("Failed to find free port", error=str(e))
        return

    if port != target_port:
        logger.warning(
            "Default port occupied. Using alternative port.",
            default_port=target_port,
            actual_port=port,
        )
        print(
            f"WARNING: Ingestion Worker listening on port {port} instead of {target_port}. "
            f"Update configuration if necessary."
        )

    # Serve HTTPS when the dev cert/key are present in the environment (the whole
    # local stack runs TLS with the localhost cert, and callers reach the worker
    # at https://localhost:8007). Falls back to plain HTTP when no certs are set.
    ssl_certfile = os.environ.get("CERT_FILE") or os.environ.get("SSL_CERTFILE")
    ssl_keyfile = os.environ.get("KEY_FILE") or os.environ.get("SSL_KEYFILE")
    run_kwargs = {"host": "0.0.0.0", "port": port, "reload": True}
    if ssl_certfile and ssl_keyfile and os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
        run_kwargs["ssl_certfile"] = ssl_certfile
        run_kwargs["ssl_keyfile"] = ssl_keyfile

    uvicorn.run("main:app", **run_kwargs)


if __name__ == "__main__":
    main()
