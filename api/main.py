import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Portable path (not a hard-coded absolute one) so this works regardless of
# where the repo is cloned or which directory the server is launched from.
#
# override=False (the default) is deliberate, not an oversight: it means
# .env only fills in variables that AREN'T already set in the process
# environment, rather than unconditionally clobbering them. This matters
# concretely -- tests/conftest.py sets DATABASE_URL to a temp SQLite file
# and EMAIL_PROVIDER=mock BEFORE importing this module specifically so the
# test suite never touches the real dev database or makes real network
# calls; `override=True` would silently defeat that isolation (a real bug,
# caught live: an earlier version of this line used override=True, and the
# test suite started failing/attempting real Brevo calls the moment a
# real .env with EMAIL_PROVIDER=brevo existed on disk).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api.admin_auth import require_admin
from api.config import (
    ADMIN_API_TOKEN,
    CORS_ORIGINS,
    ENABLE_SCHEDULER,
    ENVIRONMENT,
    IS_PRODUCTION,
)
from api.db.migrations import create_all_tables
from api.db.models import GrantSource
from api.db.session import SessionLocal
from api.routers import (
    ai_config,
    analytics,
    assistant,
    auth,
    feedback,
    grant_sources,
    ngo_profile,
    opportunities,
    saved,
    subscribers,
    website_analysis,
)
from src.ranking.corpus import get_corpus, reset_corpus

# Structured-enough for production log aggregation (CloudWatch, Railway/
# Render's log viewers) without adding a JSON-logging dependency: a
# timestamp + level + logger name + message is enough to filter "API
# failures" from "source refresh failures" from "AI provider failures" by
# logger name (grantsetu.api / grantsetu.sync / grantsetu.ai / ...) in any
# of those platforms' log search. LOG_LEVEL is the one env var that tunes
# verbosity -- never logs API keys/passwords (see src/ai/crypto.py,
# src/email/tokens.py: secrets are hashed/encrypted at rest and only ever
# logged as "***" or not at all).
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("grantsetu.api")

# Idempotently seeded at startup so /grant-sources always has something to show,
# without ever claiming live data is flowing (status starts "not_configured").
#
# The sample/fixture connector is DELIBERATELY NOT seeded here. It exists
# (src/connectors/sample_connector.py) purely to prove the sync/dedup/
# notification pipeline in automated tests, which construct a GrantSource
# row for it directly -- see tests/test_grant_sync.py. Seeding it into every
# real deployment's database would put a permanently-visible, samples-only
# "source" in front of real users, which is exactly the kind of fake-data-
# filling-the-interface this project has committed not to do. An admin who
# wants to exercise the pipeline manually can still create one via
# POST /admin/grant-sources -- it just isn't there by default.
DEFAULT_GRANT_SOURCES = [
    {
        "name": "Grants.gov (U.S. federal opportunities)",
        "source_url": "https://www.grants.gov/",
        "connector_type": "grants_gov_api",
        # A conservative default: fetchOpportunity is called once per hit
        # (to get description/eligibility text), sequentially, so a larger
        # "rows" value makes the FIRST sync noticeably slow (each detail
        # call is a real network round-trip to grants.gov). 10 keeps a
        # first sync to roughly the time of ~10 sequential API calls;
        # increase via PATCH-equivalent (update the GrantSource row's
        # config) once you've seen it work.
        "config": {"rows": 10, "opp_statuses": "forecasted|posted", "fetch_details": True},
        "notes": (
            "REAL, LIVE integration against grants.gov's public search2/"
            "fetchOpportunity APIs -- no API key required (these are the same "
            "public endpoints grants.gov's own website calls). U.S. federal "
            "data only; does not make GrantSetu's opportunity data India-"
            "specific. Trigger a sync via POST /admin/sync-source/{source_id}."
        ),
    },
    {
        "name": "NGOBox (Indian NGO grant announcements)",
        "source_url": "https://ngobox.org/grant_announcement_listing.php",
        "connector_type": "listing_page_discovery",
        # Conservative default (same reasoning as grants.gov's "rows"):
        # each of the 15 detail pages is a real, separate network fetch.
        "config": {
            "listing_url": "https://ngobox.org/grant_announcement_listing.php",
            "link_contains": "full_grant_announcement",
            "max_detail_pages": 15,
        },
        "notes": (
            "REAL, LIVE web-discovery source -- ngobox.org's public grant-"
            "announcement listing (India-focused NGO/CSR grants), no login, "
            "no API key. Verified live during development: robots.txt "
            "returns 404 (no restriction declared) and the listing page has "
            "no linked Terms of Use restricting automated access -- neither "
            "is the same as an affirmative permission, so this stays "
            "not_configured (never auto-synced) until an operator confirms "
            "acceptable use directly with the site, same as every other "
            "connector in this project. Extraction is rule-based (same "
            "extractor as WebDiscoveryConnector); fields not stated on a "
            "given detail page (e.g. a specific funding amount) are left "
            "None rather than guessed. Trigger a sync via "
            "POST /admin/sync-source/{source_id}."
        ),
    },
]


def _seed_grant_sources():
    db = SessionLocal()
    try:
        for entry in DEFAULT_GRANT_SOURCES:
            if not db.query(GrantSource).filter_by(name=entry["name"]).first():
                db.add(GrantSource(**entry))
        db.commit()
    finally:
        db.close()


def _check_production_config():
    """Fails fast (rather than degrading silently at request time) on
    misconfiguration that's only acceptable in local development. Runs once
    at startup so a bad deploy is caught in the platform's boot/health-check
    logs, not discovered later as a confusing runtime error."""
    if not IS_PRODUCTION:
        return
    problems = []
    if CORS_ORIGINS == []:
        problems.append("CORS_ORIGINS is empty -- no browser origin will be able to call this API")
    if ADMIN_API_TOKEN is None:
        logger.warning(
            "ADMIN_API_TOKEN is not set in a production environment -- "
            "POST /admin/grant-sources and POST /admin/sync-source/{id} are "
            "reachable by anyone. Set ADMIN_API_TOKEN to require the "
            "X-Admin-Token header on those endpoints."
        )
    encryption_key = os.environ.get("GRANTSETU_ENCRYPTION_KEY", "")
    if not encryption_key or "YOUR_FERNET" in encryption_key.upper():
        problems.append(
            "GRANTSETU_ENCRYPTION_KEY is unset or a placeholder -- a real key is required in "
            "production so encrypted AI provider keys survive a restart (see .env.example)"
        )
    if problems:
        raise RuntimeError(
            "Refusing to start with ENVIRONMENT=production and unsafe configuration:\n- "
            + "\n- ".join(problems)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_production_config()
    create_all_tables()
    _seed_grant_sources()
    get_corpus()  # warm the TF-IDF index once at startup rather than per-request
    if ENABLE_SCHEDULER:
        from api.scheduler import start_scheduler

        start_scheduler()
    logger.info("GrantSetu API ready (environment=%s)", ENVIRONMENT)
    yield
    if ENABLE_SCHEDULER:
        from api.scheduler import stop_scheduler

        stop_scheduler()


app = FastAPI(
    title="GrantSetu API",
    description=(
        "NGO funding discovery and intelligence platform: profile matching, live "
        "opportunity discovery, website-driven NGO analysis, and explainable "
        "ranking. Every opportunity carries accurate, item-level status and "
        "source labels (see status_label/source_label on each result) rather "
        "than a single blanket disclaimer -- see README.md and "
        "docs/data_audit_report.md for the full data-source breakdown."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS_ORIGINS (api/config.py) is a comma-separated allow-list from the
# environment -- never "*" in production. Local dev defaults to the ports
# README's local frontend/backend commands actually use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,  # auth is a Bearer token, never a cookie -- see api/routers/auth.py
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the existing static frontend (frontend/index.html + frontend/assets/)
# directly from this same FastAPI process -- for a single-service deployment
# (e.g. one Railway service) where there's no separate static-site host.
# Deliberately just two routes + one static mount, not a second server or a
# rewrite of the frontend: every API route below is registered on this same
# `app` and is completely unaffected, since none of them match "/",
# "/env.js", or "/assets/*".
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/env.js", include_in_schema=False)
def serve_frontend_env():
    """The frontend (frontend/index.html's `<script src="env.js">`) expects
    this to set window.GRANTSETU_API_BASE -- normally generated per-
    deployment by scripts/render_frontend_env.sh or frontend/Dockerfile's
    entrypoint from API_BASE_URL (see .env.example) for a deployment where
    the frontend and API are on different origins. Here they're the SAME
    origin (this FastAPI process serves both), so an empty string is the
    correct value: it makes the frontend's relative fetches
    (`${API_BASE}/health`, etc.) resolve against whatever domain actually
    served this page -- Railway's generated domain, a later custom domain,
    or localhost during local testing -- with nothing hardcoded and no
    CORS involved, since same-origin requests aren't subject to it."""
    return Response('window.GRANTSETU_API_BASE = "";', media_type="application/javascript")


app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="frontend-assets")


@app.get("/health", tags=["meta"])
def health():
    from src.email.service import MockEmailProvider, get_email_provider

    corpus = get_corpus()
    email_provider = get_email_provider()
    return {
        "status": "ok",
        "opportunities_indexed": len(corpus.df),
        # Lets the frontend show accurate, state-aware messaging ("emails
        # are not configured yet" vs "check your inbox") instead of a
        # hardcoded claim either way. Cheap: reuses the same provider-
        # selection logic already used to send, no extra network call.
        "email_delivery_configured": not isinstance(email_provider, MockEmailProvider),
        "email_provider": email_provider.name,
    }


@app.post("/admin/refresh-index", tags=["admin"], dependencies=[Depends(require_admin)])
def refresh_index():
    corpus = reset_corpus(force_rebuild=True)
    return {"status": "refreshed", "opportunities_indexed": len(corpus.df)}


app.include_router(ngo_profile.router)
# grant_sources.router registers GET /opportunities/live -- it MUST be
# included before opportunities.router, whose GET /opportunities/{opportunity_id}
# has no int converter in the path string and would otherwise swallow
# "/opportunities/live" as opportunity_id="live" (then fail int validation)
# because Starlette matches routes in registration order.
app.include_router(grant_sources.router)
app.include_router(assistant.router)
app.include_router(auth.router)
app.include_router(opportunities.router)
app.include_router(saved.router)
app.include_router(subscribers.router)
app.include_router(ai_config.router)
app.include_router(website_analysis.router)
app.include_router(feedback.router)
app.include_router(analytics.router)
