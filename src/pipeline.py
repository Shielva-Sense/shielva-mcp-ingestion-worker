"""
Shielva Ingestion Worker
Document parsing, chunking, embedding, and indexing pipeline.
"""
from typing import Dict, Any, List, Optional, Callable, Coroutine
from datetime import datetime
import structlog

from .models import (
    Document, Chunk, IngestionJob, DocumentType, ChunkingStrategy
)
from .cleaner import TextCleaner
from .embedder import EmbeddingClient, EmbedderConfig
from .fetcher import DocumentFetcher
from .parser import BaseParser, PARSERS, TextParser
from .chunker import Chunker
from .indexer import VectorIndexer, IndexerConfig
import re

# ── Ingest guardrails ───────────────────────────────────────────────────────
_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CARD]"),
]


def apply_guardrails(text: str, guardrails: Dict[str, Any]) -> str:
    """Apply a KB's ingest guardrails to a document's text BEFORE chunking:
    redact PII and drop lines containing excluded keywords. Runs on every ingest
    (file / url / db / api) so the policy persists across re-syncs."""
    if not text or not guardrails:
        return text
    if guardrails.get("redact_pii"):
        for pattern, repl in _PII_PATTERNS:
            text = pattern.sub(repl, text)
    keywords = [str(k).strip().lower() for k in (guardrails.get("exclude_keywords") or []) if str(k).strip()]
    if keywords:
        kept = [line for line in text.split("\n") if not any(k in line.lower() for k in keywords)]
        text = "\n".join(kept)
    return text

logger = structlog.get_logger(__name__)


# ===== Components are now imported from modules =====


# ===== Embedding =====

# ===== Embedding Client imported from .embedder =====


# ===== Ingestion Pipeline =====

class IngestionPipeline:
    """
    Complete document ingestion pipeline.
    
    Steps:
    1. Parse document (extract text)
    2. Chunk document
    3. Generate embeddings
    4. Store in vector DB
    """
    
    def __init__(
        self,
        vector_store,
        embedding_client: EmbeddingClient,
        chunker: Chunker = None,
        indexer: VectorIndexer = None,
        fetcher: DocumentFetcher = None
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.chunker = chunker or Chunker()
        
        # Initialize Indexer
        if indexer:
            self.indexer = indexer
        else:
            self.indexer = VectorIndexer(vector_store=vector_store)
            
        # Initialize Fetcher
        self.fetcher = fetcher or DocumentFetcher()
        
        logger.info("IngestionPipeline initialized")
    
    async def ingest_document(self, document: Document) -> int:
        """
        Ingest a single document.
        
        Returns number of chunks created.
        """
        logger.info(
            "Ingesting document",
            document_id=document.id,
            title=document.title
        )
        
        try:
            print(f"DEBUG: Starting ingest_document for {document.id}")
            # Step 0: Fetch content if needed
            if not document.content and document.source_url:
                fetch_result = await self.fetcher.fetch(document.source_url)
                if fetch_result.content:
                    document.content = await self._parse_fetched_content(fetch_result, document)
            
            # Step 1: Parse. Run for EVERY doc type (incl. TEXT) so that binary
            # uploads (PDF/DOCX/XLSX/PPTX) arriving as raw bytes are decoded by
            # their parser, and text formats arriving as bytes are decoded too.
            # TextParser is a no-op for str content, so the JSON-text path is safe.
            parser = PARSERS.get(document.doc_type, TextParser())
            document.content = await parser.parse(document.content)
            
            # Step 1.5: Clean
            document.content = TextCleaner.clean(document.content)

            # Step 1.6: Guardrails — redact PII / drop excluded lines BEFORE
            # chunking. Carried on the document's metadata by the ingest endpoint.
            _gr = (document.metadata or {}).pop("_guardrails", None)
            if _gr:
                document.content = apply_guardrails(document.content, _gr)

            # Step 2: Chunk
            print(f"DEBUG: Chunking document {document.id}...")
            chunks = self.chunker.chunk(document)
            print(f"DEBUG: Chunked into {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("No chunks created", document_id=document.id)
                return 0
            
            # Step 3: Embed
            # The new EmbeddingClient has embed_single and embed methods, but maybe not embed_chunks
            # Let's check init file. It has embed(texts).
            # We need to map chunks to texts, embed, then assign back.
            
            chunk_texts = [c.content for c in chunks]
            print(f"DEBUG: Embedding {len(chunk_texts)} chunks...")
            embeddings = await self.embedding_client.embed(chunk_texts)
            print(f"DEBUG: Embeddings generated. Count: {len(embeddings)}")
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Step 4: Index (Store in vector DB)
            # The Indexer expects a list of dicts (chunks)
            chunk_dicts = [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in chunks
            ]
            
            print(f"DEBUG: Indexing {len(chunk_dicts)} chunks...")
            await self.indexer.index_chunks(
                chunks=chunk_dicts,
                tenant_id=document.tenant_id,
                kb_id=document.kb_id
            )
            print("DEBUG: Indexing completed")
            
            logger.info(
                "Document ingested",
                document_id=document.id,
                chunks=len(chunks)
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error(
                "Document ingestion failed",
                document_id=document.id,
                error=str(e)
            )
            raise
    
    async def ingest_batch(
        self,
        documents: List[Document],
        job: IngestionJob,
        on_progress: Optional[Callable[[IngestionJob], Coroutine[Any, Any, None]]] = None
    ) -> IngestionJob:
        """
        Ingest a batch of documents.
        
        Updates job status as it progresses.
        """
        job.status = "processing"
        job.started_at = datetime.utcnow()
        job.documents_total = len(documents)
        
        logger.info(
            "Starting batch ingestion",
            job_id=job.job_id,
            total=len(documents)
        )
        
        for document in documents:
            try:
                chunks = await self.ingest_document(document)
                job.documents_processed += 1
                job.chunks_created += chunks
                
                # Optional: trigger progress callback
                if on_progress:
                    await on_progress(job)
                
            except Exception as e:
                job.documents_failed += 1
                job.errors.append(f"{document.id}: {str(e)}")
                
                if on_progress:
                    await on_progress(job)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        
        logger.info(
            "Batch ingestion completed",
            job_id=job.job_id,
            processed=job.documents_processed,
            failed=job.documents_failed,
            chunks=job.chunks_created
        )
        
        return job
    
    async def _parse_fetched_content(self, fetch_result, document: Document) -> str:
        """Helper to parse content fetched as bytes"""
        # If fetcher returns bytes, we might need to update doc_type based on content_type
        # For now, simplistic handling
        if isinstance(fetch_result.content, bytes):
            return fetch_result.content.decode('utf-8', errors='ignore')
        return str(fetch_result.content)

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
        kb_id: str
    ) -> int:
        """Delete all chunks for a document"""
        logger.info(
            "Deleting document",
            document_id=document_id,
            tenant_id=tenant_id
        )
        
        return await self.indexer.delete_by_document(
            document_id=document_id,
            tenant_id=tenant_id,
            kb_id=kb_id
        )
