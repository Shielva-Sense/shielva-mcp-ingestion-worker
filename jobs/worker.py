"""
Standalone Worker
Entry point for running a standalone ingestion worker process.
Can be used for processing queues (e.g., Redis/Celery) in the future.
"""

import asyncio
import structlog
import signal

logger = structlog.get_logger(__name__)


async def run_worker():
    """Run the worker loop."""
    logger.info("Starting standalone ingestion worker...")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # In a real implementation, this would pull from a queue
    # For now, it's a placeholder for the worker process
    while running:
        await asyncio.sleep(1)
        # Check queue
        # Process job

    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(run_worker())
