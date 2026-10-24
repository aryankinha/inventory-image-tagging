FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-group notebook
COPY src ./src
COPY scripts ./scripts
CMD ["uv", "run", "--no-sync", "python3", "scripts/reproduce.py", "--work-dir", "/app/runs/reproduction"]
