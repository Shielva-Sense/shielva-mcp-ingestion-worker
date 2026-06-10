"""
Indexer Module
Vector store indexing operations.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class IndexerConfig:
    """Indexer configuration."""
    vector_store_type: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    api_key: str = ""
    batch_size: int = 100


@dataclass
class IndexedChunk:
    """Indexed chunk result."""
    id: str
    document_id: str
    vector_id: str
    success: bool = True
    error: Optional[str] = None


class VectorIndexer:
    """
    Indexes document chunks to vector store.
    
    Features:
    - Batch indexing
    - Upsert support
    - Collection management
    - Metadata filtering support
    """
    
    def __init__(
        self,
        config: IndexerConfig = None,
        vector_store = None
    ):
        """
        Initialize indexer.
        
        Args:
            config: Indexer configuration
            vector_store: Vector store client
        """
        self.config = config or IndexerConfig()
        
        # Initialize Vector Store if not provided
        if not vector_store:
            # Default to Supabase
            from ..vectorstore import SupabaseVectorStore
            
            # Use SUPABASE_DB_URL from env or config
            db_url = self.config.host 
            if "postgresql://" not in db_url:
                # Fallback or assume it serves as DB URL holder
                pass
            
            self.vector_store = SupabaseVectorStore(
                db_url=db_url,
                collection_prefix="shielva_kb_"
            )
            # We need to connect
            # But __init__ is sync... we should probably rely on caller to connect or connect lazily.
            # SupabaseVectorStore.connect() is async.
            # We'll have to handle connection in index_chunks or make this async.
            # For now, let's assume it gets connected when used or we add a connect method.
        else:
            self.vector_store = vector_store
            
        logger.info(
            "VectorIndexer initialized",
            store_type="supabase"
        )
    
    async def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: str,
        kb_id: str
    ) -> List[IndexedChunk]:
        """
        Index document chunks to vector store.
        """
        if not chunks:
            return []
            
        # Ensure connection if not connected (idempotent-ish)
        if hasattr(self.vector_store, "connect"):
             # This is hacky if we don't await it. 
             # Ideally indexer should be async context manager.
             # We'll assume caller handles it or we do it here.
             await self.vector_store.connect()

        results = []
        
        # Process in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            
            try:
                batch_results = await self._index_batch(
                    batch, tenant_id, kb_id
                )
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(
                    "Batch indexing failed",
                    batch_start=i,
                    error=str(e)
                )
                # Mark all chunks in batch as failed
                for chunk in batch:
                    results.append(IndexedChunk(
                        id=chunk.get("id", ""),
                        document_id=chunk.get("document_id", ""),
                        vector_id="",
                        success=False,
                        error=str(e)
                    ))
        
        success_count = sum(1 for r in results if r.success)
        logger.info(
            "Indexing complete",
            total=len(chunks),
            success=success_count,
            failed=len(chunks) - success_count
        )
        
        return results
    
    async def _index_batch(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: str,
        kb_id: str
    ) -> List[IndexedChunk]:
        """Index a batch of chunks."""
        from ..vectorstore import VectorDocument
        results = []
        
        if self.vector_store:
            # Convert chunks to VectorDocument objects
            documents = []
            for chunk in chunks:
                # Prepare metadata
                metadata = chunk.get("metadata", {}).copy()
                metadata.update({
                    "document_id": chunk["document_id"],
                    "title": chunk.get("title", ""),
                    "chunk_index": chunk.get("chunk_index", 0)
                    # content is passed explicitly to VectorDocument
                })
                
                doc = VectorDocument(
                    id=chunk["id"],
                    content=chunk["content"],
                    embedding=chunk["embedding"],
                    metadata=metadata
                )
                documents.append(doc)
            
            # Upsert
            await self.vector_store.upsert(
                tenant_id=tenant_id,
                kb_id=kb_id,
                documents=documents
            )

            # Ensure the FTS + HNSW indexes exist on this collection. upsert only
            # get_or_create's the table (no ANN index), so without this the vector
            # column has no HNSW index and queries do O(n) exact scans. IF NOT
            # EXISTS makes this idempotent + cheap to call each batch. Best-effort —
            # never fail ingestion if index creation hiccups.
            try:
                create_idx = getattr(self.vector_store, "create_text_index", None)
                if create_idx:
                    await create_idx(tenant_id, kb_id)
            except Exception as _ie:
                logger.warning("index_ensure_failed", error=str(_ie))

            for chunk in chunks:
                results.append(IndexedChunk(
                    id=chunk["id"],
                    document_id=chunk["document_id"],
                    vector_id=chunk["id"],
                    success=True
                ))
        else:
            # Mock indexing
            for chunk in chunks:
                results.append(IndexedChunk(
                    id=chunk.get("id", "mock-id"),
                    document_id=chunk.get("document_id", "mock-doc"),
                    vector_id=f"vec-{chunk.get('id', 'mock')}",
                    success=True
                ))
        
        return results
    
    async def delete_by_document(
        self,
        document_id: str,
        tenant_id: str,
        kb_id: str
    ) -> int:
        """
        Delete all indexed chunks for a document.
        
        Returns:
            Number of deleted chunks
        """
        if self.vector_store:
            return await self.vector_store.delete_by_filter(
                tenant_id=tenant_id,
                kb_id=kb_id,
                filter={"document_id": document_id}
            )
        
        return 0
    
    async def delete_by_kb(
        self,
        tenant_id: str,
        kb_id: str
    ) -> bool:
        """Delete all indexed content for a knowledge base."""
        if self.vector_store:
            return await self.vector_store.delete_collection(
                tenant_id=tenant_id,
                kb_id=kb_id
            )
        
        return True


__all__ = ["VectorIndexer", "IndexerConfig", "IndexedChunk"]
