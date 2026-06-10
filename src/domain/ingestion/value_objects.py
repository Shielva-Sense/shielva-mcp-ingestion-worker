from __future__ import annotations

from enum import Enum
from typing import NewType

JobId = NewType("JobId", str)
TenantId = NewType("TenantId", str)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
