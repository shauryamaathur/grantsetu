"""Centralized, environment-driven runtime configuration. Nothing here reads
a hardcoded localhost/dev value as if it were a production default -- every
setting either has a safe, explicit fallback appropriate for local
development, or is required and the app fails fast at startup if it's
missing (see api/main.py's `_check_production_config`)."""
import os

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"
DEBUG = os.environ.get("DEBUG", "false" if IS_PRODUCTION else "true").lower() == "true"

# Comma-separated list of allowed browser origins, e.g.
# "https://app.grantsetu.org,https://grantsetu.org". No default of "*" in
# production -- see api/main.py. Local dev defaults to the two ports the
# README's local frontend/backend commands actually use.
_default_cors = "http://localhost:5500,http://127.0.0.1:5500"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _default_cors).split(",")
    if origin.strip()
]

# Bearer token required (via the `X-Admin-Token` header) for operator-only
# endpoints (creating/syncing an individual grant source, forcing a full
# index rebuild). Left unset, these endpoints stay open -- convenient for
# local development, but api/main.py logs a loud warning if this is unset
# while ENVIRONMENT=production. The public "Refresh All Sources" flow
# (POST /admin/sync-all-sources, GET /admin/refresh-jobs/{id}) is
# deliberately NOT gated by this token -- it's an ordinary user-facing
# feature that happens to live under the /admin prefix for historical
# reasons, not an operator-only endpoint.
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN", "").strip() or None

# Optional in-process scheduler for source refresh (api/scheduler.py) --
# off by default so it never double-runs alongside an external cron/
# EventBridge trigger hitting the same endpoint unless explicitly enabled.
ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "false").lower() == "true"
SYNC_INTERVAL_HOURS = float(os.environ.get("SYNC_INTERVAL_HOURS", "6"))

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5500").rstrip("/")
