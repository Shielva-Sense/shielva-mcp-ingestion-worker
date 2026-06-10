from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .entities import Document, IngestionJob
from .value_objects import JobId, TenantId


class IIngestionJobRepository(ABC):
    """Port — persistence adapter for ingestion jobs."""

    @abstractmethod
    async def save(self, job: IngestionJob) -> IngestionJob: ...

    @abstractmethod
    async def get(self, job_id: JobId, tenant_id: TenantId) -> Optional[IngestionJob]: ...

    @abstractmethod
    async def list_by_tenant(
        self, tenant_id: TenantId, status: Optional[str] = None
    ) -> List[IngestionJob]: ...


class IVectorStore(ABC):
    """Port — vector store adapter."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def upsert(self, tenant_id: TenantId, documents: List[Document]) -> int: ...

    @abstractmethod
    async def delete(self, tenant_id: TenantId, document_id: str) -> bool: ...
