# Dockerfile — ClaimAssist API v3 (Session 8)
# Session 2 discipline: pinned slim base, non-root user, cached deps layer,
# HEALTHCHECK, secrets only via environment (.env — never baked in).
# v3: requirements now include chromadb; the chroma store (rag/chroma/) and
# the data directory are runtime volume mounts, never baked into the image.
FROM python:3.11-slim

ARG HOST_UID=1000
ARG HOST_GID=1000

RUN groupadd --gid ${HOST_GID} appuser \
    && useradd --uid ${HOST_UID} --gid ${HOST_GID} --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /home/appuser/.cache/chroma \
    && chown -R appuser /home/appuser/.cache/chroma

RUN mkdir -p /app/logs \
    && chown -R appuser /app/logs

COPY app.py guardrails.py redact.py ./

RUN chown -R appuser /app

USER appuser

ENV PORT=8001

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
