# Build context is the repository root, since uv.lock is shared by the workspace
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies go in a separate layer, so it is rebuilt only when the lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/api/pyproject.toml,target=apps/api/pyproject.toml \
    uv sync --frozen --no-dev --package api --no-install-workspace

COPY apps/api/src ./src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
# Only tokenizers are used, so the "PyTorch was not found" advice is noise
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ENTRYPOINT []

CMD ["fastapi", "run", "--entrypoint", "main:app", "--host", "0.0.0.0", "--port", "8000"]
