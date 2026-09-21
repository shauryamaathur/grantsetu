# GrantSetu backend (FastAPI). Builds one image used for BOTH the AWS and
# the low-cost (Railway/Render) deployment -- the only difference between
# environments is which environment variables are injected at `docker run`
# time (see .env.example / GrantSetu_Deployment_Guide.pdf), never a
# different image or Dockerfile.
FROM python:3.10-slim AS base

# Needed to build psycopg2-binary's transitive deps on some base images and
# for pandas/scikit-learn wheels that fall back to source builds on
# platforms without a prebuilt wheel; kept minimal and removed from the
# final layer via a single RUN so the image doesn't carry build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY data/connectors/ data/connectors/
# Served directly by api/main.py's GET / and /assets mount -- see there.
COPY frontend/index.html frontend/index.html
COPY frontend/assets/ frontend/assets/

# data/processed/ is intentionally NOT copied in -- it's a regeneratable
# cache (the cleaned dataset + TF-IDF index), rebuilt from data/raw/ on
# first request/startup (src/ranking/corpus.py) and safe to lose on every
# container restart/redeploy. Mount a volume at /app/data/processed only if
# you want to skip that few-seconds rebuild across restarts; not required.
RUN mkdir -p data/processed

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# $PORT is honored for platforms that assign it dynamically (Railway,
# Render); defaults to 8000 for `docker run`/AWS. $WEB_CONCURRENCY controls
# gunicorn worker count -- keep this at 1 if ENABLE_SCHEDULER=true (see
# api/scheduler.py's docstring: more than one worker/process running the
# in-process scheduler would trigger duplicate source syncs).
CMD ["sh", "-c", "gunicorn api.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --timeout 120 --access-logfile - --error-logfile -"]
