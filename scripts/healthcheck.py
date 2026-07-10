#!/usr/bin/env python3
"""Docker HEALTHCHECK probe.

Hits the in-process /health endpoint and exits 0 on any HTTP response
in [200, 600). A connection error or timeout exits 1.

`/health` is the cheap "is the process alive" probe — distinct from
`/ready`, which is the "all dependencies pingable" gate. Docker uses
/health; Kubernetes uses both (livenessProbe → /health,
readinessProbe → /ready).

Kept dependency-free — only stdlib — so the runtime image carries no
extra wheels and so a broken httpx won't break the healthcheck.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    port = os.environ.get("PORT") or os.environ.get("INGESTION_PORT") or "8007"
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
            return 0 if 200 <= r.status < 600 else 1
    except urllib.error.HTTPError as e:
        return 0 if 200 <= e.code < 600 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
