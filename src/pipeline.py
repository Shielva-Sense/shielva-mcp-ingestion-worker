"""
Shielva Ingestion Worker
Document parsing, chunking, embedding, and indexing pipeline.
"""

from typing import Dict, Any, List, Optional, Callable, Coroutine
from datetime import datetime
import structlog

from .models import Document, IngestionJob
from .cleaner import TextCleaner
from .embedder import EmbeddingClient
from .fetcher import DocumentFetcher
from .parser import PARSERS, TextParser
from .chunker import Chunker
from .indexer import VectorIndexer
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
        fetcher: DocumentFetcher = None,
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
        logger.info("Ingesting document", document_id=document.id, title=document.title)

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

            # 🚨 THE FORK. Extraction reads the document HERE — after parsing,
            # before guardrails — and that order is the whole reason it is a
            # fork rather than a later step.
            #
            # `apply_guardrails` below redacts email, phone, SSN and card
            # before chunking, permanently, for everything downstream of it.
            # Those four are the payload of KYC, invoices, loans and claims. A
            # knowledge base is right to strip them; an extractor running after
            # that would find [REDACTED_EMAIL] where the document says a name.
            #
            # Everything it produces is carried on the document's metadata and
            # delivered by the caller, because this function owns one document
            # and the delivery is a batch.
            await self._extract_if_asked(document)

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
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ]

            print(f"DEBUG: Indexing {len(chunk_dicts)} chunks...")
            await self.indexer.index_chunks(chunks=chunk_dicts, tenant_id=document.tenant_id, kb_id=document.kb_id)
            print("DEBUG: Indexing completed")

            logger.info("Document ingested", document_id=document.id, chunks=len(chunks))

            return len(chunks)

        except Exception as e:
            logger.error("Document ingestion failed", document_id=document.id, error=str(e))
            raise

    async def _extract_if_asked(self, document: Document) -> None:
        """Read this document against a category, when the ingest asked for one.

        Best-effort in the strictest sense: a failure here must never cost the
        RAG ingest that shares this parse. The vectors are the feature that
        already worked, and an extraction that raised would take the knowledge
        base down with it — so the row records the failure and the chunking
        below carries on.

        The result rides on `document.metadata`, which `ingest_batch` reads.
        Returning it would mean changing this method's contract (it returns a
        chunk count) for a caller that does not want it.
        """
        spec = (document.metadata or {}).get("_extract")
        if not spec:
            return

        from .extractor import extract_document
        from .extractor.client import completer

        try:
            result = await extract_document(
                text=document.content or "",
                category_name=str(spec.get("category_name") or ""),
                fields=list(spec.get("fields") or []),
                complete=completer(document.tenant_id),
            )
        except Exception as exc:  # pragma: no cover — extract_document swallows its own
            logger.warning("document_extract_unexpected", document_id=document.id, error=str(exc)[:200])
            return

        document.metadata["_extracted"] = {
            "document_id": document.id,
            "source_name": document.title or "",
            "source_key": document.source_url or "",
            "category_id": str(spec.get("category_id") or ""),
            **result.as_payload(),
        }

    async def _deliver_extracted(
        self, documents: List[Document], job: IngestionJob, rows: List[Dict[str, Any]]
    ) -> None:
        """Send the batch's extracted rows to core-api, which owns the store.

        The callback URL and collect come off the first document's extract
        spec — every document in one ingest belongs to one collect, because a
        collect is what the ingest was started against.
        """
        if not rows:
            return
        spec = next((d.metadata.get("_extract") for d in documents if (d.metadata or {}).get("_extract")), None)
        if not spec:
            return

        from .extractor.delivery import deliver_rows

        try:
            await deliver_rows(
                callback_url=str(spec.get("rows_callback_url") or ""),
                tenant_id=documents[0].tenant_id,
                collect_id=str(spec.get("collect_id") or ""),
                rows=rows,
            )
        except Exception as exc:  # pragma: no cover — deliver_rows swallows its own
            logger.error("extracted_rows_delivery_crashed", job_id=job.job_id, error=str(exc)[:200])

    async def ingest_batch(
        self,
        documents: List[Document],
        job: IngestionJob,
        on_progress: Optional[Callable[[IngestionJob], Coroutine[Any, Any, None]]] = None,
    ) -> IngestionJob:
        """
        Ingest a batch of documents.

        Updates job status as it progresses.
        """
        job.status = "processing"
        job.started_at = datetime.utcnow()
        job.documents_total = len(documents)

        logger.info("Starting batch ingestion", job_id=job.job_id, total=len(documents))

        # Rows the fork produced, delivered once at the end rather than per
        # document: one request per invoice would be fifty round-trips for a
        # batch that already knows it is a batch.
        extracted: List[Dict[str, Any]] = []

        for document in documents:
            try:
                chunks = await self.ingest_document(document)
                job.documents_processed += 1
                job.chunks_created += chunks

                row = (document.metadata or {}).get("_extracted")
                if row:
                    extracted.append(row)

                # Optional: trigger progress callback
                if on_progress:
                    await on_progress(job)

            except Exception as e:
                job.documents_failed += 1
                job.errors.append(f"{document.id}: {str(e)}")

                # 🚨 A document whose CHUNKING failed may still have been read.
                # Losing the extraction because the vectors failed would throw
                # away the more expensive half of the work — and the half the
                # reviewer is waiting for.
                row = (document.metadata or {}).get("_extracted")
                if row:
                    extracted.append(row)

                if on_progress:
                    await on_progress(job)

        await self._deliver_extracted(documents, job, extracted)

        job.status = "completed"
        job.completed_at = datetime.utcnow()

        logger.info(
            "Batch ingestion completed",
            job_id=job.job_id,
            processed=job.documents_processed,
            failed=job.documents_failed,
            chunks=job.chunks_created,
        )

        return job

    async def _parse_fetched_content(self, fetch_result, document: Document) -> str:
        """Helper to parse content fetched as bytes"""
        # If fetcher returns bytes, we might need to update doc_type based on content_type
        # For now, simplistic handling
        if isinstance(fetch_result.content, bytes):
            return fetch_result.content.decode("utf-8", errors="ignore")
        return str(fetch_result.content)

    async def delete_document(self, document_id: str, tenant_id: str, kb_id: str) -> int:
        """Delete all chunks for a document"""
        logger.info("Deleting document", document_id=document_id, tenant_id=tenant_id)

        return await self.indexer.delete_by_document(document_id=document_id, tenant_id=tenant_id, kb_id=kb_id)
