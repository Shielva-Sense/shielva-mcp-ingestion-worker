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
                sess.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{collection_name}_content_fts
                    ON vecs."{collection_name}"
                    USING GIN (to_tsvector('english', metadata->>'content'));
                """))
                # ANN index on the embedding column so vector search is sub-linear
                # (HNSW, cosine). Without it pgvector does an O(n) exact scan — the
                # query-latency cliff as the KB grows. vecs stores the vector in the
                # `vec` column. IF NOT EXISTS keeps this idempotent.
                sess.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{collection_name}_vec_hnsw
                    ON vecs."{collection_name}"
                    USING hnsw (vec vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """))
                sess.commit()
                logger.info("Created FTS + HNSW indexes", collection=collection_name)
        except Exception as e:
            logger.error("Failed to create indexes", error=str(e))

    async def list_documents(self, tenant_id: str, kb_id: str) -> List[Dict[str, Any]]:
        """List distinct documents (id, title, chunk count, file bytes) in a KB.

        Drives the "Files in this KB" UI so individual files can be removed and
        their size is shown. ``bytes`` is the document's original file size: it is
        stored identically on every chunk of the document (Document.metadata.size
        at ingest), so MAX over the group recovers it once. Returns an empty list
        if the collection doesn't exist yet (no files ingested).
        """
        from sqlalchemy import text
        collection_name = self._get_collection_name(tenant_id, kb_id)
        documents: List[Dict[str, Any]] = []
        try:
            with self._client.Session() as sess:
                sql = text(f"""
                    SELECT metadata->>'document_id' AS document_id,
                           COALESCE(NULLIF(MAX(metadata->>'title'), ''), NULLIF(MAX(metadata->>'filename'), '')) AS title,
                           COUNT(*)                  AS chunks,
                           MAX(COALESCE((metadata->>'size')::bigint, 0)) AS bytes
                    FROM vecs."{collection_name}"
                    WHERE COALESCE(metadata->>'document_id', '') <> ''
                    GROUP BY metadata->>'document_id'
                    ORDER BY 2;
                """)
                for row in sess.execute(sql):
                    doc_id = row[0]
                    title = (row[1] or "").strip() or doc_id
                    documents.append({
                        "document_id": doc_id,
                        "title": title,
                        "chunks": int(row[2]),
                        "bytes": int(row[3] or 0),
                    })
        except Exception as e:
            logger.warning("list_documents failed (collection may not exist)", collection=collection_name, error=str(e))
        return documents

    async def kb_storage(self, tenant_id: str, kb_id: str) -> Dict[str, int]:
        """Return ``{documents, chunks, file_bytes}`` for a KB's collection.

        ``file_bytes`` sums each DISTINCT document's original size (size lives once
        per document, replicated across its chunks — so per-document MAX then SUM).
        Returns zeros if the collection doesn't exist yet. Cheap single round-trip.
        """
        from sqlalchemy import text
        collection_name = self._get_collection_name(tenant_id, kb_id)
        out = {"documents": 0, "chunks": 0, "file_bytes": 0}
        try:
            with self._client.Session() as sess:
                row = sess.execute(text(f"""
                    SELECT
                      (SELECT COUNT(DISTINCT metadata->>'document_id')
                         FROM vecs."{collection_name}"
                        WHERE COALESCE(metadata->>'document_id','') <> '')            AS documents,
                      (SELECT COUNT(*) FROM vecs."{collection_name}")                  AS chunks,
                      (SELECT COALESCE(SUM(b), 0) FROM (
                          SELECT MAX(COALESCE((metadata->>'size')::bigint, 0)) AS b
                            FROM vecs."{collection_name}"
                           WHERE COALESCE(metadata->>'document_id','') <> ''
                           GROUP BY metadata->>'document_id'
                       ) t)                                                            AS file_bytes
                """)).first()
                if row:
                    out = {"documents": int(row[0] or 0), "chunks": int(row[1] or 0), "file_bytes": int(row[2] or 0)}
        except Exception as e:
            logger.warning("kb_storage failed (collection may not exist)", collection=collection_name, error=str(e))
        return out

    async def list_chunks(self, tenant_id: str, kb_id: str, document_id: str = None, limit: int = 500) -> List[Dict[str, Any]]:
        """List individual chunks (id, text, source doc, order) in a KB — for content curation."""
        from sqlalchemy import text
        collection_name = self._get_collection_name(tenant_id, kb_id)
        out: List[Dict[str, Any]] = []
        try:
            with self._client.Session() as sess:
                where = "WHERE metadata->>'document_id' = :doc" if document_id else ""
                params: Dict[str, Any] = {"lim": limit}
                if document_id:
                    params["doc"] = document_id
                sql = text(f"""
                    SELECT id,
                           metadata->>'content'     AS content,
                           metadata->>'document_id' AS document_id,
                           metadata->>'title'       AS title,
                           COALESCE((metadata->>'chunk_index')::int, 0) AS chunk_index
                    FROM vecs."{collection_name}"
                    {where}
                    ORDER BY metadata->>'document_id', COALESCE((metadata->>'chunk_index')::int, 0)
                    LIMIT :lim
                """)
                for row in sess.execute(sql, params):
                    out.append({"id": row[0], "content": row[1] or "", "document_id": row[2], "title": row[3], "chunk_index": row[4]})
        except Exception as e:
            logger.warning("list_chunks failed", collection=collection_name, error=str(e))
        return out

    async def delete_chunk(self, tenant_id: str, kb_id: str, chunk_id: str) -> int:
        """Delete a single chunk (fragment) by its row id."""
        from sqlalchemy import text
        collection_name = self._get_collection_name(tenant_id, kb_id)
        try:
            with self._client.Session() as sess:
                res = sess.execute(text(f'DELETE FROM vecs."{collection_name}" WHERE id = :id'), {"id": chunk_id})
                sess.commit()
                return res.rowcount or 0
        except Exception as e:
            logger.error("delete_chunk failed", collection=collection_name, error=str(e))
            return 0

    async def update_chunk(self, tenant_id: str, kb_id: str, chunk_id: str, content: str, embedding: List[float]) -> bool:
        """Replace a chunk's text AND its embedding (re-embed) so retrieval stays consistent."""
        from sqlalchemy import text
        collection_name = self._get_collection_name(tenant_id, kb_id)
        vec_literal = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
        try:
            with self._client.Session() as sess:
                res = sess.execute(text(f"""
                    UPDATE vecs."{collection_name}"
                    SET vec = CAST(:vec AS vector),
                        metadata = jsonb_set(metadata, '{{content}}', to_jsonb(CAST(:content AS text)))
                    WHERE id = :id
                """), {"vec": vec_literal, "content": content, "id": chunk_id})
                sess.commit()
                return (res.rowcount or 0) > 0
        except Exception as e:
            logger.error("update_chunk failed", collection=collection_name, error=str(e))
            return False

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
        """Delete chunks by metadata equality filter. Returns the rows deleted.

        Uses raw SQL on the vecs collection table — vecs' own ``collection.delete``
        needs operator-form filters (``{"k": {"$eq": v}}``) and returns no count, so
        a plain ``{"k": v}`` filter silently deleted nothing.
        """
        from sqlalchemy import text
        collection_name = self._get_collection_name(tenant_id, kb_id)
        # Keys are internal (e.g. "document_id"), never user input; values are bound.
        conds = " AND ".join(f"metadata->>'{k}' = :v{i}" for i, k in enumerate(filter.keys()))
        params = {f"v{i}": str(v) for i, v in enumerate(filter.values())}
        try:
            with self._client.Session() as sess:
                res = sess.execute(text(f'DELETE FROM vecs."{collection_name}" WHERE {conds}'), params)
                sess.commit()
                return res.rowcount or 0
        except Exception as e:
            logger.error("delete_by_filter failed", collection=collection_name, error=str(e))
            return 0
