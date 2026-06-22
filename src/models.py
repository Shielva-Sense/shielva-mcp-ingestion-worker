"""
Ingestion Worker Data Models
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    CSV = "csv"
    JSON = "json"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


@dataclass
class Document:
    """Document to be ingested"""
    id: str
    tenant_id: str
    kb_id: str
    content: str
    title: str
    source_url: Optional[str] = None
    doc_type: DocumentType = DocumentType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Chunk:
    """Document chunk with embedding"""
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class IngestionJob:
    """Ingestion job status"""
    job_id: str
    tenant_id: str
    kb_id: str
    status: str = "pending"
    documents_total: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    # Absolute cumulative file bytes for the WHOLE KB after this job (queried from
    # the vector store at completion). core-api stores it so the Knowledge page reads
    # per-KB file size without a worker round-trip per KB.
    kb_file_bytes: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
