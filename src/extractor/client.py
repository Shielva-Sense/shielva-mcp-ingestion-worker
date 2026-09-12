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

#: Reading PICTURES of pages is slower than reading text, by enough that the
#: text budget times out on a multi-page scan — which looks exactly like the
#: model being unable to see, and would have been diagnosed as that.
VISION_TIMEOUT_S = 180.0


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

    async def complete(prompt: str, images: list[str] | None = None) -> str:
        """Ask the workspace's model. With `images`, ask it to READ them.

        🚨 The images are sent as OpenAI content parts, which MCP accepts and
        LiteLLM forwards to whichever provider the workspace has provisioned.
        That is the entire scanned-page story: no OCR vendor, no second key, and
        the usage lands on the metering that already exists because this is an
        ordinary completion.
        """
        if images:
            content: object = [
                {"type": "text", "text": prompt},
                *({"type": "image_url", "image_url": {"url": uri}} for uri in images),
            ]
        else:
            content = prompt
        async with httpx.AsyncClient(verify=_tls_verify(), timeout=VISION_TIMEOUT_S if images else TIMEOUT_S) as client:
            r = await client.post(
                f"{mcp_url}/mcp/v1/llm/complete",
                json={"messages": [{"role": "user", "content": content}], "max_tokens": MAX_TOKENS},
                headers=headers,
            )
        if r.status_code != 200:
            # Raised, so `extract_document` turns it into a row that says the
            # document could not be read — one row, not a lost batch.
            #
            # 🚨 The BODY travels with it for a vision call. A model that cannot
            # see answers 400, and "model returned 400" cannot be told apart
            # from a malformed request — the caller needs to know it was the
            # modality, so the row can say the page is a scan the configured
            # model cannot read.
            detail = (r.text or "")[:300] if images else ""
            raise RuntimeError(f"model returned {r.status_code}{': ' + detail if detail else ''}")
        return str(r.json().get("text") or "")

    return complete
