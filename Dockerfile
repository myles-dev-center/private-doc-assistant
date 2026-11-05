FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
# Build arg so users in regions with slow PyPI access can point at a mirror.
ARG UV_INDEX_URL=https://pypi.org/simple
ENV UV_INDEX_URL=${UV_INDEX_URL}

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ocr --no-install-project
COPY src ./src
COPY schemas ./schemas
COPY ui ./ui
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev --extra ocr

ENV PATH="/app/.venv/bin:$PATH" DOCASSIST_DATA_DIR=/data
VOLUME /data
EXPOSE 8000 8501
CMD ["uvicorn", "docassist.api:app", "--host", "0.0.0.0", "--port", "8000"]
