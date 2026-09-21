# GrantSetu frontend: a single static HTML file, no build step. This image
# just serves it via nginx and renders env.js (window.GRANTSETU_API_BASE)
# from the API_BASE_URL environment variable at CONTAINER START (not build
# time), so the same image works for both the AWS and low-cost deployments
# -- only the env var passed to `docker run`/the platform's env settings
# changes.
FROM nginx:1.27-alpine

COPY index.html /usr/share/nginx/html/index.html
COPY env.example.js /usr/share/nginx/html/env.example.js

# Runs as one of nginx's standard docker-entrypoint.d/ scripts -- executed
# by the base image's own entrypoint BEFORE nginx starts, then control
# returns to that entrypoint to exec the real CMD (nginx). Must NOT start
# nginx itself here.
RUN printf '#!/bin/sh\nset -eu\n: "${API_BASE_URL:=http://localhost:8000}"\nsed "s#__API_BASE_URL__#$API_BASE_URL#g" /usr/share/nginx/html/env.example.js > /usr/share/nginx/html/env.js\n' > /docker-entrypoint.d/40-render-env.sh \
    && chmod +x /docker-entrypoint.d/40-render-env.sh

EXPOSE 80
