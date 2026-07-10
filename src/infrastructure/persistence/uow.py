"""Simple coordinator for ingestion-worker.

The ingestion worker does not use a relational DB for job state (jobs are
held in-process by job_manager).  This UoW is a lightweight context manager
placeholder for future persistence needs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator


class UnitOfWork:
    """No-op Unit of Work placeholder for the ingestion worker.

    Replace with a real SQLAlchemy or asyncpg implementation when job state
    is moved to a persistent store.
    """

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        yield
