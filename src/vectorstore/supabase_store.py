"""
Vector Store Integration with Supabase (vecs)
Multi-tenant vector storage using pgvector via vecs
"""
from typing import Dict, Any, List, Optional
import structlog
import vecs
from vecs.collection import Collection

logger = structlog.get_logger(__name__)

from .models import VectorDocument, SearchResult


class SupabaseVectorStore:
    """
    Supabase vector store using vecs (pgvector).
    
    Features:
    - Collection management via vecs
    - Metadata filtering
    - Dot product / Cosine distance
    """
    
    def __init__(
        self,
        db_url: str,
        collection_prefix: str = "shielva_kb_",
        embedding_dim: int = 768  # Default for Gemini
    ):
        """
        Initialize Supabase vector store.
        
        Args:
            db_url: PostgreSQL connection string
            collection_prefix: Prefix for collection names
            embedding_dim: Embedding dimension
        """
        self.db_url = db_url
        self.collection_prefix = collection_prefix
        self.embedding_dim = embedding_dim
        
        self._client: Optional[vecs.Client] = None
        
        logger.info("SupabaseVectorStore initialized")
    
    async def connect(self):
        """Establish connection to Supabase (via vecs)."""
        # vecs.create_client is synchronous but lightweight (connection pool)
        try:
            self._client = vecs.create_client(self.db_url)
            logger.info("Connected to Supabase Vector")
        except Exception as e:
            logger.error("Failed to connect to Supabase Vector", error=str(e))
            raise e
    
    async def close(self):
        """Close connection."""
        if self._client:
            self._client.disconnect()
    
    def _get_collection_name(self, tenant_id: str, kb_id: str) -> str:
        """Get collection name for tenant and KB."""
        # Combine checks
        raw_name = f"{self.collection_prefix}{tenant_id}_{kb_id}"
        
        # PostgreSQL identifier limit is 63 bytes
        if len(raw_name) <= 63:
            return raw_name.replace("-", "_")
            
        # If too long, hash the unique components
        import hashlib
        # Keep prefix for readability (11 chars)
        # Hash tenant+kb for uniqueness (32 chars hex)
        # Total = 11 + 32 = 43 chars < 63
        unique_str = f"{tenant_id}_{kb_id}"
        hash_suffix = hashlib.sha256(unique_str.encode()).hexdigest()[:32]
        return f"{self.collection_prefix}{hash_suffix}"
    
    async def create_collection(
        self,
        tenant_id: str,
        kb_id: str
    ) -> str:
        """Create a new collection."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        
        # vecs creates the table/index if it doesn't exist
        # We perform this in a sync way as vecs is sync currently
        try:
            self._client.get_or_create_collection(
                name=collection_name,
                dimension=self.embedding_dim
            )
            logger.info("Created/Retrieved collection", collection=collection_name)
            
            # Ensure text index exists
            await self.create_text_index(tenant_id, kb_id)
            
            return collection_name
        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise e
    
    async def delete_collection(
        self,
        tenant_id: str,
        kb_id: str
    ) -> bool:
        """Delete a collection."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        
        try:
            self._client.delete_collection(collection_name)
            logger.info("Deleted collection", collection=collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
            return False
    
    async def upsert(
        self,
        tenant_id: str,
        kb_id: str,
        documents: List[VectorDocument]
    ) -> int:
        """Upsert documents."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        
        # Get collection
        collection = self._client.get_or_create_collection(
            name=collection_name,
            dimension=self.embedding_dim
        )
        
        # Prepare records: list of (id, vector, metadata)
        records = []
        for doc in documents:
            metadata = {
                "content": doc.content,
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                **doc.metadata
            }
            records.append((doc.id, doc.embedding, metadata))
        
        # Batch upsert is handled by vecs
        try:
            collection.upsert(records=records)
            logger.info(
                "Upserted documents",
                collection=collection_name,
                count=len(records)
            )
            return len(records)
        except Exception as e:
            logger.error("Upsert failed", error=str(e))
            raise e
    
    async def search(
        self,
        tenant_id: str,
        kb_ids: List[str],
        query_embedding: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search across KBs."""
        all_results = []
        
        for kb_id in kb_ids:
            collection_name = self._get_collection_name(tenant_id, kb_id)
            
            try:
                # Check if collection exists first to avoid error
                try:
                    collection = self._client.get_collection(name=collection_name)
                except KeyError:
                    continue  # Collection doesn't exist
                
                results = collection.query(
                    data=query_embedding,
                    limit=top_k,
                    filters=filters,
                    include_metadata=True
                )
                
                # vecs 0.4+ returns List[Tuple[str, dict]] if include_metadata=True
                # Older versions or if include_metadata=False might return List[str]
                if not results:
                    continue
                    
                if not isinstance(results[0], str):
                    found_ids = [r[0] for r in results]
                else:
                    found_ids = results

                if not found_ids:
                    continue
                    
                # Fetch records
                records = collection.fetch(ids=found_ids)
                # Map to dict
                record_map = {r.id: r for r in records}
                
                for rank, local_id in enumerate(found_ids):
                    if local_id not in record_map:
                        continue
                    
                    rec = record_map[local_id]
                    # Synthetic score for RRF stability
                    score = 1.0 / (1.0 + rank)
                    
                    content = rec.metadata.get("content", "")
                    
                    all_results.append(SearchResult(
                        id=rec.id,
                        content=content,
                        score=score, 
                        metadata={
                            k: v for k, v in rec.metadata.items()
                            if k not in ["content", "tenant_id", "kb_id"]
                        }
                    ))
                    
            except Exception as e:
                logger.error("Search failed", collection=collection_name, error=str(e))
                
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    async def create_text_index(self, tenant_id: str, kb_id: str):
        """
        Create a GIN index on metadata->'content' for Full Text Search.
        """
        collection_name = self._get_collection_name(tenant_id, kb_id)
        
        try:
            from sqlalchemy import text
            with self._client.Session() as sess:
                sql = text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{collection_name}_content_fts 
                    ON vecs."{collection_name}" 
                    USING GIN (to_tsvector('english', metadata->>'content'));
                """)
                sess.execute(sql)
                sess.commit()
                logger.info("Created FTS index", collection=collection_name)
        except Exception as e:
            logger.error("Failed to create FTS index", error=str(e))

    async def keyword_search(
        self,
        tenant_id: str,
        kb_ids: List[str],
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform keyword search using Postgres Full Text Search.
        """
        from sqlalchemy import text
        all_results = []
        
        for kb_id in kb_ids:
            collection_name = self._get_collection_name(tenant_id, kb_id)
            
            try:
                with self._client.Session() as sess:
                    sql = text(f"""
                        SELECT 
                            id, 
                            metadata, 
                            ts_rank(to_tsvector('english', metadata->>'content'), websearch_to_tsquery('english', :query)) as score
                        FROM vecs."{collection_name}"
                        WHERE to_tsvector('english', metadata->>'content') @@ websearch_to_tsquery('english', :query)
                        ORDER BY score DESC
                        LIMIT :limit;
                    """)
                    
                    result = sess.execute(sql, {"query": query, "limit": top_k})
                    
                    for row in result:
                        doc_id = row[0]
                        metadata = row[1] or {}
                        score = float(row[2])
                        
                        content = metadata.get("content", "")
                        
                        all_results.append(SearchResult(
                            id=doc_id,
                            content=content,
                            score=score,
                            metadata={
                                k: v for k, v in metadata.items()
                                if k not in ["content", "tenant_id", "kb_id"]
                            }
                        ))
                        
            except Exception as e:
                logger.warning("Keyword search failed", collection=collection_name, error=str(e))
                continue
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    async def delete_by_ids(
        self,
        tenant_id: str,
        kb_id: str,
        document_ids: List[str]
    ) -> int:
        """Delete by IDs."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        collection = self._client.get_collection(name=collection_name)
        collection.delete(ids=document_ids)
        return len(document_ids)
    
    async def get_collection_info(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> Dict[str, Any]:
        """Return vector count and metadata for a KB collection."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        try:
            collection = self._client.get_collection(name=collection_name)
            # vecs Collection exposes __len__ via the underlying table
            from sqlalchemy import text
            with self._client.Session() as sess:
                result = sess.execute(
                    text(f'SELECT COUNT(*) FROM vecs."{collection_name}"')
                )
                count = result.scalar() or 0
            return {
                "collection_name": collection_name,
                "vector_count": count,
                "document_count": count,
            }
        except KeyError:
            # Collection doesn't exist yet
            return {"collection_name": collection_name, "vector_count": 0, "document_count": 0}
        except Exception as e:
            logger.warning("get_collection_info failed", collection=collection_name, error=str(e))
            return {"collection_name": collection_name, "vector_count": 0, "document_count": 0}

    async def delete_by_filter(
        self,
        tenant_id: str,
        kb_id: str,
        filter: Dict[str, Any]
    ) -> int:
        """Delete documents by metadata filter."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        collection = self._client.get_collection(name=collection_name)
        
        # vecs delete supports filters
        # collection.delete(filters=...)
        collection.delete(filters=filter)
        return 0 # vecs doesn't return count?
