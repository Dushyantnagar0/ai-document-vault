# Verified multi-platform index digests; runtime evidence is recorded in the journal.
FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS tooling
COPY --from=uv /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never TIKTOKEN_CACHE_DIR=/app/tokenizer-cache \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

FROM tooling AS base
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --group ui
# Download hash-checked vocabulary assets at build time, before non-root runtime.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"
RUN groupadd --gid 10001 vault && useradd --uid 10001 --gid vault --no-create-home vault
ENV PYTHONPATH=/app
COPY app ./app
COPY migrations ./migrations
COPY scripts ./scripts
COPY ui ./ui
COPY alembic.ini ./
USER 10001:10001

FROM base AS test
ENV RUFF_CACHE_DIR=/tmp/ruff-cache MYPY_CACHE_DIR=/tmp/mypy-cache \
    PYTEST_ADDOPTS="-p no:cacheprovider"
USER root
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --group ui --group dev
COPY tests ./tests
COPY .env.example ./
USER 10001:10001
CMD ["pytest", "-m", "unit or api"]

FROM base AS runtime
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "95"]
