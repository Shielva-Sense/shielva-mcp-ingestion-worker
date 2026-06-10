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
_envelope_load_dotenv(".env", override=True)   # ciphertext + REDIS_URL passthrough
from shielva_common.envelope import bootstrap as _envelope_bootstrap
_envelope_bootstrap()
# ──────────────────────────────────────────────────────────────────────────

import uuid as _uuid_mod
import structlog.contextvars as _structlog_cv

from fastapi import (
    BackgroundTasks,
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
import json
import structlog
import uvicorn
import uuid

from shielva_common.auth import Principal, require_principal
from shielva_common.ratelimit import per_tenant_rate_limit, PerTenantTokenBucket
from shielva_common.tls import internal_ca_verify

from src.config import get_settings
import os  # noqa: E402 — plain os (the envelope alias above shadows the name)
from src.pipeline import IngestionPipeline
from src.sources import fetch_url, read_database, read_api
from src.models import (
    IngestionJob, Document, DocumentType, ChunkingStrategy,
)
from src.chunker import Chunker
from src.fetcher import DocumentFetcher
from src.indexer import VectorIndexer
from src.embedder import EmbeddingClient, EmbedderConfig
from src.jobs.manager import job_manager
from src.jobs.processor import JobProcessor

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
        __import__("logging").getLevelName(
            __import__("os").environ.get("LOG_LEVEL", "INFO").upper()
        )
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
            detail=(
                f"too many entries (limit {settings.max_entries_per_batch}, "
                f"got {len(batch.documents)})"
            ),
        )
    total = sum(len(d.content or "") for d in batch.documents)
    if total > settings.max_total_bytes_per_batch:
        raise HTTPException(
            status_code=413,
            detail=(
                f"batch too large (limit {settings.max_total_bytes_per_batch} bytes, "
                f"got {total})"
            ),
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
    embedding_api_key = (
        settings.gemini_api_key.get_secret_value()
        or settings.openai_api_key.get_secret_value()
    )

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


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(_ingest_rate_limit)],
)
async def ingest_documents(
    request: IngestBatchRequest,
    background_tasks: BackgroundTasks,
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
        doc_type = (
            DocumentType(doc_req.doc_type) if doc_req.doc_type else DocumentType.TEXT
        )
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

    background_tasks.add_task(processor.process_job, job, documents)

    return IngestResponse(
        job_id=job.job_id,
        status="queued",
        message="Ingestion started in background",
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
        doc_type = (
            DocumentType(doc_req.doc_type) if doc_req.doc_type else DocumentType.TEXT
        )
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
    "txt": DocumentType.TEXT, "text": DocumentType.TEXT, "log": DocumentType.TEXT,
    "md": DocumentType.MARKDOWN, "markdown": DocumentType.MARKDOWN,
    "html": DocumentType.HTML, "htm": DocumentType.HTML,
    "pdf": DocumentType.PDF,
    "docx": DocumentType.DOCX,
    "csv": DocumentType.CSV,
    "json": DocumentType.JSON,
    "xlsx": DocumentType.XLSX, "xlsm": DocumentType.XLSX,
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
    dependencies=[Depends(_ingest_rate_limit)],
)
async def ingest_files(
    kb_id: str = Form(...),
    files: List[UploadFile] = File(...),
    guardrails: str = Form("{}"),
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
        webhook_url=None,
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


class IngestUrlRequest(BaseModel):
    kb_id: str
    url: str
    crawl: bool = False
    max_pages: int = 20
    max_depth: int = 2


class IngestDatabaseRequest(BaseModel):
    kb_id: str
    db_type: str
    connection_uri: str
    query: Optional[str] = None
    collection: Optional[str] = None
    limit: int = 1000


class IngestApiRequest(BaseModel):
    kb_id: str
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict[str, Any]] = None
    json_path: Optional[str] = None
    limit: int = 500


async def _ingest_source_docs(kb_id: str, tenant_id: str, docs, label: str) -> "JobStatusResponse":
    """Wrap (title, content, doc_type) tuples from a source adapter into Documents
    and run them through the pipeline (sync)."""
    if not docs:
        raise HTTPException(status_code=422, detail="Source returned no content")
    job = job_manager.create_job(tenant_id=tenant_id, kb_id=kb_id, documents_count=len(docs), webhook_url=None)
    documents = []
    for title, content, doc_type in docs:
        try:
            dt = DocumentType(doc_type)
        except ValueError:
            dt = DocumentType.TEXT
        documents.append(Document(
            id=_stable_doc_id(label, kb_id, title, content), tenant_id=tenant_id, kb_id=kb_id,
            content=content, title=title, source_url=None, doc_type=dt,
            metadata={"source": label},
        ))
    await processor.process_job(job, documents)
    return JobStatusResponse(
        job_id=job.job_id, status=job.status, documents_total=job.documents_total,
        documents_processed=job.documents_processed, documents_failed=job.documents_failed,
        chunks_created=job.chunks_created, started_at=job.started_at,
        completed_at=job.completed_at, errors=job.errors,
    )


@app.post("/ingest/url", response_model=JobStatusResponse, dependencies=[Depends(_ingest_rate_limit)])
async def ingest_url(body: IngestUrlRequest, principal: Principal = Depends(require_principal)):
    """Ingest a public URL — a single page, or a same-host BFS crawl. SSRF-guarded."""
    try:
        docs = await fetch_url(body.url, crawl=body.crawl, max_pages=body.max_pages, max_depth=body.max_depth)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await _ingest_source_docs(body.kb_id, principal.tenant_id, docs, "url")


@app.post("/ingest/database", response_model=JobStatusResponse, dependencies=[Depends(_ingest_rate_limit)])
async def ingest_database(body: IngestDatabaseRequest, principal: Principal = Depends(require_principal)):
    """Ingest rows/documents from a database via a read-only query."""
    try:
        docs = await read_database(
            body.db_type, body.connection_uri, query=body.query,
            collection=body.collection, limit=body.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await _ingest_source_docs(body.kb_id, principal.tenant_id, docs, "db")


@app.post("/ingest/api", response_model=JobStatusResponse, dependencies=[Depends(_ingest_rate_limit)])
async def ingest_api(body: IngestApiRequest, principal: Principal = Depends(require_principal)):
    """Ingest a REST API response (optionally a list via a dotted JSON path)."""
    try:
        docs = await read_api(
            body.url, method=body.method, headers=body.headers, body=body.body,
            json_path=body.json_path, limit=body.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await _ingest_source_docs(body.kb_id, principal.tenant_id, docs, "api")


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
            tenant_id=principal.tenant_id, kb_id=kb_id, document_id=document_id,
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
            tenant_id=principal.tenant_id, kb_id=kb_id, chunk_id=chunk_id,
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
            tenant_id=principal.tenant_id, kb_id=body.kb_id, chunk_id=chunk_id,
            content=body.content, embedding=embedding,
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
