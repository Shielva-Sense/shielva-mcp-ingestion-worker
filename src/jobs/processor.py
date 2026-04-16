"""
Job Processor
Handles the execution of ingestion jobs.
"""
import asyncio
import structlog
from typing import List

from ..pipeline import IngestionPipeline
from ..models import Document, IngestionJob

logger = structlog.get_logger(__name__)


class JobProcessor:
    """
    Processes ingestion jobs.
    
    Executes the ingestion pipeline for a given job and set of documents.
    """
    
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
    
    async def process_job(self, job: IngestionJob, documents: List[Document]):
        """
        Process a job with the given documents.
        """
        logger.info(
            "Processing job",
            job_id=job.job_id,
            docs=len(documents)
        )
        
        job.status = "processing"
        
        try:
            # Run pipeline
            await self.pipeline.ingest_batch(documents, job, on_progress=self._trigger_webhook)
            
            job.status = "completed"
            logger.info("Job execution completed", job_id=job.job_id)

            # Trigger Webhook if URL is present
            if job.webhook_url:
                await self._trigger_webhook(job)
            
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(
                "Job execution failed",
                job_id=job.job_id,
                error=str(e)
            )
            
            # Trigger Webhook for failure too
            if job.webhook_url:
                await self._trigger_webhook(job)

    async def _trigger_webhook(self, job: IngestionJob):
        """Send job stats to webhook."""
        import httpx
        try:
            payload = {
                "job_id": job.job_id,
                "kb_id": job.kb_id,
                "status": job.status,
                "documents_processed": job.documents_processed,
                "documents_failed": job.documents_failed,
                "chunks_created": job.chunks_created,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "errors": job.errors
            }
            
            async with httpx.AsyncClient() as client:
                # We need to pass tenant_id header if possible, but IngestionJob has it
                response = await client.post(
                    job.webhook_url,
                    json=payload,
                    headers={"X-Tenant-ID": job.tenant_id},
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    logger.warning(
                        "Webhook failed", 
                        url=job.webhook_url, 
                        status=response.status_code,
                        response=response.text
                    )
                else:
                    logger.info("Webhook triggered successfully", url=job.webhook_url)
                    
        except Exception as e:
            logger.error("Failed to trigger webhook", error=str(e))
