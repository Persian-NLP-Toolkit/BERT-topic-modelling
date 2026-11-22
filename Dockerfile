# syntax=docker/dockerfile:1

# ---------- 1. build the React frontend ----------
FROM node:22-alpine AS frontend
ARG NPM_REGISTRY=https://registry.npmjs.org/
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry="$NPM_REGISTRY"
COPY frontend/ ./
RUN npm run build

# ---------- 2. Python API + CLI (serves the built frontend) ----------
FROM python:3.12-slim
# EXTRAS=bertopic installs torch (CPU) + transformers + BERTopic (much larger image)
ARG EXTRAS=""
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static \
    HF_HOME=/data/huggingface

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bert_wordcloud ./bert_wordcloud
RUN if [ "$EXTRAS" = "bertopic" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url "$PIP_INDEX_URL"; \
    fi \
    && pip install ".${EXTRAS:+[$EXTRAS]}"

COPY --from=frontend /app/frontend/dist ./static
COPY examples ./examples

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["bert-wordcloud", "serve", "--host", "0.0.0.0", "--port", "8000"]
