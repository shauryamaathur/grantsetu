#!/usr/bin/env sh
# Generates frontend/env.js from frontend/env.example.js by substituting
# the API_BASE_URL environment variable -- the one place the static
# frontend learns which backend to call in a given deployment (AWS,
# Railway/Render, or a plain `docker run`). Safe to re-run; always
# overwrites env.js.
#
# Usage:
#   API_BASE_URL=https://api.grantsetu.example ./scripts/render_frontend_env.sh
set -eu

: "${API_BASE_URL:?Set API_BASE_URL to this deployment's backend URL, e.g. https://api.grantsetu.example}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../frontend/env.example.js"
OUTPUT="$SCRIPT_DIR/../frontend/env.js"

sed "s#__API_BASE_URL__#${API_BASE_URL}#g" "$TEMPLATE" > "$OUTPUT"
echo "Wrote $OUTPUT with GRANTSETU_API_BASE=${API_BASE_URL}"
