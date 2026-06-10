from __future__ import annotations

"""Adapter — wraps the existing SupabaseVectorStore behind the domain port."""

from typing import List

from src.domain.ingestion.entities import Document
from src.domain.ingestion.ports import IVectorStore
from src.domain.ingestion.value_objects import TenantId
from src.vectorstore.supabase_store import SupabaseVectorStore


class VectorStoreImpl(IVectorStore):
    """Secondary adapter delegating to the existing Supabase store."""

    def __init__(self, supabase_store: SupabaseVectorStore) -> None:
        self._store = supabase_store

    async def connect(self) -> None:
        await self._store.connect()

    async def close(self) -> None:
        await self._store.close()

    async def upsert(self, tenant_id: TenantId, documents: List[Document]) -> int:
        # Delegate to the existing store interface.
        # The existing store is called per-document in the pipeline;
        # this adapter provides the port contract for future batch paths.
        count = 0
        for doc in documents:
            # The existing store's upsert signature may differ — this is a
            # thin shim; adapt as the underlying API evolves.
            await self._store.upsert(  # type: ignore[attr-defined]
                tenant_id=str(tenant_id),
                document_id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
            )
            count += 1
        return count

    async def delete(self, tenant_id: TenantId, document_id: str) -> bool:
        result = await self._store.delete(  # type: ignore[attr-defined]
            tenant_id=str(tenant_id),
            document_id=document_id,
        )
        return bool(result)
