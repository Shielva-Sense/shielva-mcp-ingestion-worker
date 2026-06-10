from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .value_objects import JobId, JobStatus, TenantId


@dataclass
class IngestionJob:
    """Aggregate root representing a document ingestion job."""

    id: JobId
    tenant_id: TenantId
    kb_id: str
    status: JobStatus = JobStatus.PENDING
    documents_total: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Document:
    """A document submitted for ingestion."""

    id: str
    tenant_id: TenantId
    kb_id: str
    content: str
    title: str
    source_url: Optional[str] = None
    doc_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
