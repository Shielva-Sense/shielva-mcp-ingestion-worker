"""
Job Manager
Manages ingestion job lifecycle and state.
"""
from typing import Dict, List, Optional
import uuid
import structlog
from datetime import datetime

from ..models import IngestionJob

logger = structlog.get_logger(__name__)


class JobManager:
    """
    Manages ingestion jobs.
    
    Features:
    - Job creation and tracking
    - Status updates
    - History listing
    - Cleanup
    """
    
    def __init__(self):
        self._jobs: Dict[str, IngestionJob] = {}
        logger.info("JobManager initialized")
    
    def create_job(self, tenant_id: str, kb_id: str, documents_count: int, webhook_url: Optional[str] = None) -> IngestionJob:
        """Create a new ingestion job."""
        job_id = str(uuid.uuid4())
        
        job = IngestionJob(
            job_id=job_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            documents_total=documents_count,
            status="queued",
            started_at=datetime.utcnow(),
            webhook_url=webhook_url
        )
        
        self._jobs[job_id] = job
        
        logger.info(
            "Job created",
            job_id=job_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            docs=documents_count
        )
        
        return job
    
    def get_job(self, job_id: str, tenant_id: str) -> Optional[IngestionJob]:
        """Get job by ID."""
        job = self._jobs.get(job_id)
        if job and job.tenant_id == tenant_id:
            return job
        return None
    
    def list_jobs(self, tenant_id: str, status: Optional[str] = None) -> List[IngestionJob]:
        """List jobs for a tenant."""
        return [
            job for job in self._jobs.values()
            if job.tenant_id == tenant_id
            and (status is None or job.status == status)
        ]
    
    def update_status(self, job_id: str, status: str, error: str = None):
        """Update job status."""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        job.status = status
        if status in ["completed", "failed"]:
            job.completed_at = datetime.utcnow()
        
        if error:
            job.errors.append(error)
            
        logger.info("Job status updated", job_id=job_id, status=status)


job_manager = JobManager()
