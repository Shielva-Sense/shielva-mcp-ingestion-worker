"""Reaching the tenant's own model, from the worker.

🚨 No provider is named here, and no key is read. MCP owns "which model does
this workspace use" — it resolves the provisioned configuration from the
principal on the request, the same way core-api's flow orchestrator does. A
client here that read `OPENAI_API_KEY` would bill the platform for every
tenant's extraction and ignore the Models screen entirely, which is the exact
failure `tenant_keys` in presence was written to fix.

The worker has no session, so it forwards a SERVICE principal — the same shape
the pre-render and recording paths already use when they act outside a request.
"""

from __future__ import annotations

import os

import httpx
import structlog

logger = structlog.get_logger(__name__)

#: Extraction reads a whole document and answers with an object. Longer than a
#: chat turn's budget and shorter than a summarisation job: an invoice's worth
#: of fields is a few hundred tokens, and a cap that cuts the JSON mid-object
#: produces a row that parses to nothing.
MAX_TOKENS = 1500

#: A document is read once, not streamed, and a model working through several
#: pages is slower than a chat turn. This is the wall-clock a single document
#: gets before the row says it could not be read.
TIMEOUT_S = 90.0


def _tls_verify() -> object:
    """The internal CA bundle, matching the rest of the worker's calls.

    A bare client fails the handshake against the self-signed in-cluster cert,
    which is how the ingest callback silently dropped before it was fixed.
    """
    try:
        from shielva_common.tls import internal_ca_verify

        return internal_ca_verify()
    except Exception:  # pragma: no cover — defensive; dev without the bundle
        return False


def service_headers(tenant_id: str) -> dict[str, str]:
    """The principal a document extraction runs as.

    🚨 The TENANT is the point of these headers, not the identity. MCP picks
    the model and the key from the workspace named here, so a wrong or missing
    tenant does not fail — it quietly bills somebody else, which is the kind of
    error nobody finds until an invoice arrives.
    """
    return {
        "X-Shielva-Tenant-Id": tenant_id,
        "X-Shielva-Email": "documents@shielva.ai",
        "X-Shielva-User-Id": "document-extractor",
        "X-Shielva-Roles": "service",
        "X-Shielva-Auth-Method": "service",
        "X-Tenant-Id": tenant_id,
    }


def completer(tenant_id: str):
    """An async `(prompt) -> str` bound to one workspace's model.

    Returned as a closure so `extract_document` stays testable without a
    network, and so the tenant is fixed at the call site that knows it rather
    than threaded through the extractor.
    """
    mcp_url = os.getenv("MCP_SERVICE_URL", "https://localhost:8004")
    headers = service_headers(tenant_id)

    async def complete(prompt: str) -> str:
        async with httpx.AsyncClient(verify=_tls_verify(), timeout=TIMEOUT_S) as client:
            r = await client.post(
                f"{mcp_url}/mcp/v1/llm/complete",
                json={"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS},
                headers=headers,
            )
        if r.status_code != 200:
            # Raised, so `extract_document` turns it into a row that says the
            # document could not be read — one row, not a lost batch.
            raise RuntimeError(f"model returned {r.status_code}")
        return str(r.json().get("text") or "")

    return complete
