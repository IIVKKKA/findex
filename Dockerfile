# syntax=docker/dockerfile:1
# ---- stage 1: build the virtualenv from uv.lock, download a small corpus, build the index ----
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# data/ is not in git: fetch ~CORPUS_MB of Project Gutenberg books while building the image
ARG CORPUS_MB=15
COPY scripts/download_corpus.py scripts/
RUN .venv/bin/python scripts/download_corpus.py --out data/gutenberg --target-mb ${CORPUS_MB} \
 && .venv/bin/findex index data/gutenberg --out index.bin --positions --executor serial

# ---- stage 2: slim runtime image, non-root user ----
FROM python:3.13-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/data /app/data
COPY --from=builder --chown=app:app /app/index.bin /app/index.bin
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/index.bin PORT=8000 WEB_WORKERS=1 PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)"
CMD ["sh", "-c", "exec findex serve --host 0.0.0.0 --port ${PORT} --workers ${WEB_WORKERS}"]
