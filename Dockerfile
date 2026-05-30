FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install -e ".[dev]"

# ── Development target ───────────────────────────────────────────────
FROM base AS dev
COPY tests/ tests/
COPY notebooks/ notebooks/
CMD ["pytest", "--tb=short", "-q"]

# ── Production target ────────────────────────────────────────────────
FROM base AS prod
RUN pip install -e .
CMD ["python", "-m", "yenibot"]
