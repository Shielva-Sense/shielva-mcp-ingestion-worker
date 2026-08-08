# Shielva MCP - Ingestion Worker

Document ingestion pipeline for the Shielva ARC platform. Handles parsing, chunking, embedding, and vector storage of documents from connectors.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION WORKER                             │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ Document │───│ Chunking │───│Embedding │───│  Vector  │     │
│  │ Parsers  │   │  Engine  │   │  Client  │   │  Store   │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       │              │              │              │            │
│     PDF           Recursive       LiteLLM        Qdrant        │
│     DOCX          Sentence        OpenAI         Multi-tenant  │
│     HTML          Paragraph       Azure                        │
│     Markdown      Fixed           Cohere                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Document Parsers
- **TextParser**: Plain text passthrough
- **HTMLParser**: HTML to clean text with tag removal
- **MarkdownParser**: Markdown preservation
- **PDFParser**: PDF extraction using PyMuPDF
- **DocxParser**: DOCX extraction using python-docx

### Chunking Strategies
- **Fixed Size**: Simple character-based chunking
- **Sentence**: Sentence-boundary aware chunking
- **Paragraph**: Paragraph-boundary chunking
- **Recursive**: Header → Paragraph → Sentence (best for structured docs)

### Embedding
- Uses LiteLLM for provider abstraction
- Supports OpenAI, Azure, Cohere
- Batch processing for efficiency

### Vector Storage
- Qdrant for vector storage
- Multi-tenant with namespace isolation
- Batch upsert support

## API Endpoints

### Ingestion

```bash
# Batch ingestion (async)
POST /ingest
{
  "kb_id": "kb-123",
  "documents": [
    {
      "id": "doc-1",
      "content": "...",
      "title": "Document Title",
      "doc_type": "text",
      "source_url": "https://..."
    }
  ]
}

# Synchronous ingestion
POST /ingest/sync
{...same as above...}
```

### Job Management

```bash
# Get job status
GET /jobs/{job_id}

# List jobs
GET /jobs?status=completed
```

### Knowledge Base Management

```bash
# Initialize KB
POST /kb/{kb_id}/initialize

# Delete KB
DELETE /kb/{kb_id}

# Get KB info
GET /kb/{kb_id}/info
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | API key for embeddings |
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `INGEST_WEBHOOK_MAX_ATTEMPTS` | 4 | Inline attempts for the terminal callback |
| `INGEST_WEBHOOK_BASE_DELAY` | 0.5 | First backoff delay (s), doubling |
| `INGEST_WEBHOOK_MAX_DELAY` | 8.0 | Backoff ceiling (s) |
| `INGEST_WEBHOOK_TIMEOUT` | 15.0 | Per-attempt HTTP timeout (s) |
| `INGEST_WEBHOOK_REDELIVERY_INTERVAL` | 60.0 | Sweep interval for undelivered results (s) |
| `INGEST_WEBHOOK_REDELIVERY_MAX_AGE` | 3600.0 | Give up on an undelivered result after (s) |

## Result delivery

An ingest job has **two** outcomes, and they can disagree:

* `status` — the pipeline result (`completed` / `failed` / `cancelled`).
* `delivery_status` — whether that result reached the caller's `webhook_url`
  (`pending` / `delivered` / `undelivered` / `skipped`).

The terminal callback is the only way a result reaches core-api — it never polls
`/jobs/{job_id}` — so a dropped callback leaves the KB inconsistent with the
vector store: `status=failed, chunks=None` in Mongo while the chunks are indexed
and queryable. Delivery is therefore retried, not fired and forgotten:

1. **Inline** — bounded exponential backoff on transport errors, 5xx and 429.
   A 4xx (e.g. a bad callback token) is permanent and is not retried.
2. **Redelivery sweep** — results that exhausted their inline attempts are held
   and re-sent every `INGEST_WEBHOOK_REDELIVERY_INTERVAL` until
   `INGEST_WEBHOOK_REDELIVERY_MAX_AGE`, so a core-api rollout heals itself
   without anyone re-uploading.
3. **Signal** — exhaustion logs `ingest_result_undelivered` at ERROR (and
   `ingest_result_delivery_abandoned` when the sweep gives up). `GET /queue/stats`
   exposes `undelivered`, `undelivered_total` and `abandoned_total`. Alert on the
   ERROR events: a non-zero `undelivered` means KBs are inconsistent *right now*.

Known gap: undelivered results live in process memory. A worker restart drops
them (logged as `ingest_results_undelivered_at_shutdown`); surviving a restart
needs a Redis-backed outbox.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m main

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

## Integration with Connectors

When a connector syncs documents:

1. Connector calls Gateway sync endpoint
2. Gateway fetches documents from source
3. Documents are sent to Ingestion Worker API
4. Ingestion Worker parses, chunks, embeds
5. Chunks are stored in Qdrant
6. MCP Server queries via RAG Engine

## Example Ingestion Flow

```python
import httpx

# Ingest documents
response = httpx.post(
    "http://localhost:8004/ingest",
    headers={"X-Tenant-ID": "tenant-123"},
    json={
        "kb_id": "kb-001",
        "documents": [
            {
                "id": "confluence-page-1",
                "content": "# Employee Handbook\\n\\n...",
                "title": "Employee Handbook",
                "doc_type": "markdown",
                "metadata": {
                    "source": "confluence",
                    "space": "HR"
                }
            }
        ]
    }
)

job_id = response.json()["job_id"]

# Check status
status = httpx.get(
    f"http://localhost:8004/jobs/{job_id}",
    headers={"X-Tenant-ID": "tenant-123"}
).json()

print(f"Processed: {status['documents_processed']}")
print(f"Chunks created: {status['chunks_created']}")
```

## Chunking Configuration

Customize chunking per request:

```json
{
  "kb_id": "kb-123",
  "documents": [...],
  "chunking_strategy": "recursive",
  "chunk_size": 1024
}
```

## Dependencies

- FastAPI for API
- LiteLLM for embeddings
- Qdrant for vectors
- PyMuPDF for PDFs
- python-docx for DOCX
