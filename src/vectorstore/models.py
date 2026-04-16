"""
Vector Store Data Models
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class VectorDocument:
    """Document to be stored in vector DB"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector search"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
