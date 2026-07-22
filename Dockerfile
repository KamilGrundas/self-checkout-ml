FROM python:3.12.13-slim-bookworm

ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app/
ENV PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY ./pyproject.toml ./uv.lock /app/
COPY ./app /app/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

CMD ["fastapi", "run", "--workers", "2", "app/main.py"]
