"""
Ingestion Worker - FastAPI Service
Handles document ingestion jobs from connectors.
"""
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import httpx
import json
import os
import structlog
import uvicorn
import uuid

from src.pipeline import IngestionPipeline
from src.models import (
    IngestionJob, Document, DocumentType, ChunkingStrategy
)
from src.chunker import Chunker
from src.fetcher import DocumentFetcher
from src.indexer import VectorIndexer
from src.embedder import EmbeddingClient, EmbedderConfig
from src.jobs.manager import job_manager
from src.jobs.processor import JobProcessor

from src.vectorstore import SupabaseVectorStore

# Logging setup
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger(__name__)


# ===== Configuration =====

class Settings:
    """Ingestion worker settings"""
    embedding_provider: str = "gemini"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_api_key: str = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "recursive"
    supabase_db_url: str = None
    supabase_collection_prefix: str = "shielva_kb_"
    embedding_dimensions: int = 768
    
    def __init__(self):
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        self.embedding_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.supabase_db_url = os.getenv("SUPABASE_DB_URL")
        self.supabase_collection_prefix = os.getenv("SUPABASE_COLLECTION_PREFIX", "shielva_kb_")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))


settings = Settings()


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


# ===== Application =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    global pipeline, processor
    
    logger.info("Starting Ingestion Worker")
    
    # Initialize vector store
    vector_store = SupabaseVectorStore(
        db_url=settings.supabase_db_url,
        collection_prefix=settings.supabase_collection_prefix,
        embedding_dim=settings.embedding_dimensions
    )
    await vector_store.connect()
    
    # Initialize embedding client
    embedding_config = EmbedderConfig(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        dimension=settings.embedding_dimensions
    )
    
    embedding_client = EmbeddingClient(config=embedding_config)
    
    # Initialize HTTP client and fetcher
    http_client = httpx.AsyncClient(timeout=60.0)
    fetcher = DocumentFetcher(http_client=http_client)
    
    # Initialize indexer
    indexer = VectorIndexer(vector_store=vector_store)
    
    # Initialize chunker
    strategy = ChunkingStrategy(settings.chunking_strategy)
    chunker = Chunker(
        strategy=strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_client=embedding_client,
        chunker=chunker,
        indexer=indexer,
        fetcher=fetcher
    )
    processor = JobProcessor(pipeline)
    
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
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("CORS_ORIGINS", '["https://localhost:3010","https://localhost:3001","http://localhost:3010","http://localhost:3000","https://localhost:3000","https://localhost:3005","https://127.0.0.1:3010","http://127.0.0.1:3000"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Dependencies =====

def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from headers"""
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID")
    return tenant_id


# ===== Endpoints =====

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "ingestion-worker",
        "pipeline_ready": pipeline is not None
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestBatchRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Ingest a batch of documents.
    
    Documents are processed in the background.
    """
    # Create job
    job = job_manager.create_job(
        tenant_id=tenant_id,
        kb_id=request.kb_id,
        documents_count=len(request.documents),
        webhook_url=request.webhook_url
    )
    
    # Convert to Document objects
    documents = []
    for doc_req in request.documents:
        doc_type = DocumentType(doc_req.doc_type) if doc_req.doc_type else DocumentType.TEXT
        
        documents.append(Document(
            id=doc_req.id,
            tenant_id=tenant_id,
            kb_id=request.kb_id,
            content=doc_req.content,
            title=doc_req.title,
            source_url=doc_req.source_url,
            doc_type=doc_type,
            metadata=doc_req.metadata
        ))
    
    # Run ingestion using processor
    background_tasks.add_task(processor.process_job, job, documents)
    
    return IngestResponse(
        job_id=job.job_id,
        status="queued",
        message="Ingestion started in background",
        documents_queued=len(request.documents)
    )


@app.post("/ingest/sync", response_model=JobStatusResponse)
async def ingest_documents_sync(
    request: IngestBatchRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Ingest documents synchronously.
    
    Waits for completion before returning.
    """
    # Create job
    job = job_manager.create_job(
        tenant_id=tenant_id,
        kb_id=request.kb_id,
        documents_count=len(request.documents),
        webhook_url=request.webhook_url
    )
    
    # Convert documents
    documents = []
    for doc_req in request.documents:
        doc_type = DocumentType(doc_req.doc_type) if doc_req.doc_type else DocumentType.TEXT
        
        documents.append(Document(
            id=doc_req.id,
            tenant_id=tenant_id,
            kb_id=request.kb_id,
            content=doc_req.content,
            title=doc_req.title,
            source_url=doc_req.source_url,
            doc_type=doc_type,
            metadata=doc_req.metadata
        ))
    
    # Run synchronously
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
        errors=job.errors
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Get ingestion job status"""
    job = job_manager.get_job(job_id, tenant_id)
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
        errors=job.errors
    )


@app.get("/jobs")
async def list_jobs(
    tenant_id: str = Depends(get_tenant_id),
    status: Optional[str] = None
):
    """List ingestion jobs for tenant"""
    jobs = job_manager.list_jobs(tenant_id, status)
    
    tenant_jobs = [
        {
            "job_id": job.job_id,
            "kb_id": job.kb_id,
            "status": job.status,
            "documents_processed": job.documents_processed,
            "chunks_created": job.chunks_created,
            "started_at": job.started_at
        }
        for job in jobs
    ]
    
    return {"jobs": tenant_jobs}


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    kb_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a document and its chunks"""
    try:
        deleted = await pipeline.delete_document(
            document_id=document_id,
            tenant_id=tenant_id,
            kb_id=kb_id
        )
        
        return {
            "status": "deleted",
            "document_id": document_id,
            "chunks_deleted": deleted
        }
        
    except Exception as e:
        logger.error("Delete failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/{kb_id}/initialize")
async def initialize_kb(
    kb_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Initialize a knowledge base (create collection)"""
    try:
        collection_name = await pipeline.vector_store.create_collection(
            tenant_id=tenant_id,
            kb_id=kb_id
        )
        
        return {
            "status": "initialized",
            "kb_id": kb_id,
            "collection": collection_name
        }
        
    except Exception as e:
        logger.error("KB initialization failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/kb/{kb_id}")
async def delete_kb(
    kb_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a knowledge base and all its data"""
    try:
        result = await pipeline.vector_store.delete_collection(
            tenant_id=tenant_id,
            kb_id=kb_id
        )
        
        return {
            "status": "deleted" if result else "not_found",
            "kb_id": kb_id
        }
        
    except Exception as e:
        logger.error("KB deletion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/{kb_id}/info")
async def get_kb_info(
    kb_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Get knowledge base information"""
    try:
        info = await pipeline.vector_store.get_collection_info(
            tenant_id=tenant_id,
            kb_id=kb_id
        )
        
        return {
            "kb_id": kb_id,
            "tenant_id": tenant_id,
            **info
        }
        
    except Exception as e:
        logger.error("KB info failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ===== Run Server =====

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
    import os
    target_port = int(os.getenv("INGESTION_PORT", 8007))
    
    try:
        port = find_free_port(target_port)
    except Exception as e:
        logger.error("Failed to find free port", error=str(e))
        return

    if port != target_port:
        logger.warning(
            "Default port occupied. Using alternative port.",
            default_port=target_port,
            actual_port=port
        )
        # Print for visibility in console/logs
        print(f"WARNING: Ingestion Worker listening on port {port} instead of {target_port}. Update configuration if necessary.")
        
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )


if __name__ == "__main__":
    main()
