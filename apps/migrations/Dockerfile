# Build context is the repository root, since uv.lock is shared by the workspace
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies go in a separate layer, so it is rebuilt only when the lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/migrations/pyproject.toml,target=apps/migrations/pyproject.toml \
    uv sync --frozen --no-dev --package migrations --no-install-workspace

# Same layout as the repository, so the root alembic.ini works unchanged
COPY alembic.ini ./
COPY apps/migrations/env.py apps/migrations/settings.py ./apps/migrations/
COPY apps/migrations/versions ./apps/migrations/versions

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

CMD ["alembic", "upgrade", "head"]
