from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SubmitIngestionJobCommand:
    tenant_id: str
    kb_id: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    webhook_url: Optional[str] = None
