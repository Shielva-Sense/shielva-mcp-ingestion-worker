# ─────────────────────────────────────────────────────────────────────
# shielva-mcp-ingestion-worker — production container image.
#
# Multi-stage build:
#   stage 1  builder  Python 3.11-slim + build toolchain. Installs
#                     deps from requirements.txt + the `shielva-common`
#                     GitHub release into an isolated /opt/venv.
#   stage 2  runtime  Python 3.11-slim, non-root UID 1000, copies
#                     /opt/venv + src/ + main.py + jobs/ + healthcheck.py.
#
# Build context:
#
#   cd shielva-mcp/shielva-mcp/ingestion-worker
#   docker build -t shielva-mcp-ingestion-worker:dev .
#
# Hardening (SOC2 CC6.8):
#   * non-root UID/GID 1000 (`appuser`)
#   * orchestrator-side: read_only=true, cap_drop=ALL,
#     seccomp=runtime/default, allowPrivilegeEscalation=false
#   * PYTHONDONTWRITEBYTECODE=1
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build-only toolchain. `git` for shielva-common @ git+...,
# `libffi-dev` + `libxml2-dev` + `libxslt1-dev` for lxml wheel build.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
        pkg-config \
        git \
        ca-certificates \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

WORKDIR /build

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt \
    && pip install \
        "shielva-common @ git+https://github.com/Shielva-AI/shielva-platform-core@shielva-common-v1.2.1#subdirectory=shielva-common"

# Sanity gate — fail the build if core imports cannot resolve.
RUN python -c "import fastapi, uvicorn, structlog, httpx; import shielva_common; print('builder ok')"

# ─────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libs for pdf/xml/html parsing.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid 1000 \
        --home-dir /home/appuser --create-home --shell /usr/sbin/nologin \
        appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY --chown=root:root src       /srv/src
COPY --chown=root:root jobs      /srv/jobs
COPY --chown=root:root main.py   /srv/main.py

COPY --chown=root:root --chmod=0555 scripts/healthcheck.py /usr/local/bin/healthcheck.py

RUN mkdir -p /var/lib/shielva-mcp-ingestion-worker \
    && chown -R appuser:appuser /var/lib/shielva-mcp-ingestion-worker \
    && chmod 0750 /var/lib/shielva-mcp-ingestion-worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/srv \
    SOP_ENABLED=false \
    HOST=0.0.0.0 \
    INGESTION_PORT=8007 \
    PORT=8007

EXPOSE 8007

USER appuser:appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python3", "/usr/local/bin/healthcheck.py"]

# `main:app` matches the in-tree uvicorn.run("main:app", ...) entry.
# Workers calibrated against platform reference image; capacity tuning
# uses N+1 horizontal pods.
CMD ["python3", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8007", \
     "--workers", "4", \
     "--backlog", "2048", \
     "--no-access-log", \
     "--log-level", "warning"]
