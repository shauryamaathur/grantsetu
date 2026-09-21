# GrantSetu — Architecture Notes

## Layers

1. **Ingestion** (`src/ingestion/`) — loads `data/raw/grants_raw.csv` with explicit
   dtypes for monetary/date columns and validates it against the documented schema
   (`validate_schema.py`). The raw file is never mutated.
2. **Cleaning** (`src/cleaning/`) — text normalization, date parsing, currency
   parsing, and de-duplication, composed in `pipeline.py::clean_dataset`.
3. **Feature engineering** (`src/features/`) — award-size buckets (relative
   quantiles, not INR-comparable), recency flags, a combined text field for
   TF-IDF, and a normalized eligibility-text field. `tfidf_index.py` fits a
   `TfidfVectorizer` over the combined text field.
4. **Ranking** (`src/ranking/`) — `score.py` computes a transparent weighted sum
   of text relevance, category overlap, and recency, multiplied by an
   eligibility compatibility factor from `eligibility_rules.py`. `explain.py`
   turns the named score components into a plain-language explanation.
   `corpus.py` is an in-process singleton that loads/caches the cleaned+featured
   corpus and its TF-IDF index once per process (not per request).
5. **API** (`api/`) — FastAPI app (`main.py`) with routers for NGO profiles,
   opportunity search/recommendation/detail, and saved-opportunity tracking.
   Persistence for NGO profiles / saved opportunities uses SQLite via
   SQLAlchemy (`api/db/`); the opportunity corpus itself is not stored in the
   relational DB -- it lives in the in-process `Corpus` object, rebuilt from
   `data/raw/` (or its cached parquet snapshot) via `scripts/refresh_index.py`.
6. **Frontend** (`frontend/index.html`) — a single static HTML/vanilla-JS page
   (no build step) that calls the API directly. Chosen over a framework because
   the MVP only needs four screens (blueprint Section M).

## Why no ML model

No outcome-label data exists in the source dataset (no application/award/rejection
history), so a supervised "will this NGO win this grant" model cannot be trained
honestly. Ranking is a transparent weighted-sum of interpretable signals instead of a
trained model. See `docs/data_audit_report.md` for the specifics.

## Deliberately excluded from this MVP build

Sentence embeddings/vector search, Elasticsearch, Kubernetes, multi-user roles, and
automated deadline alerts are out of scope per the blueprint (Sections F.2, L.2, Q.2)
and were not built here.

## V2 additions (live discovery, AI, auth) -- layered on top, not a rewrite

The MVP layers above are unchanged and still serve the historical corpus. Everything
below was added as new, separate layers:

7. **Live discovery** (`src/connectors/`, `src/discovery/`, `src/ingestion/
   sync_grants.py`) -- a `GrantSourceConnector` interface (grants.gov API, a
   generic web-discovery crawler, a generic listing-page crawler, a sample/test
   fixture) feeding a sync pipeline that deduplicates (two layers: same-source
   external ID, and a cross-source content hash), detects and logs field-level
   changes on re-sync (`OpportunityChange`), and persists results as
   `ImportedGrant` rows -- a SEPARATE table from the historical corpus, never
   mixed with it. `src/discovery/fetcher.py` is the one place that fetches
   arbitrary external URLs, and is SSRF-safe (private-IP/cloud-metadata
   blocking, validated on every redirect hop), robots.txt-aware, and
   rate-limited.
8. **Live ranking** (`src/ranking/live_match.py`) -- a second, lighter-weight
   scorer (not `score.py`, which is corpus-specific) producing an explainable
   0-100 integer score (`LiveMatchScore`) from four weighted, named components
   (topical TF-IDF relevance, focus-area/beneficiary keyword overlap,
   geography, deadline urgency) times an eligibility multiplier -- same
   transparent-weighted-sum philosophy as `score.py`, adapted to live
   opportunities' different available fields.
9. **AI provider abstraction** (`src/ai/`) -- one adapter per vendor family
   (`providers.py`), a per-NGO encrypted config (`AIProviderConfig` /
   `crypto.py`), and an AI-first/rule-based-fallback pattern used consistently
   everywhere AI could help: opportunity/website field extraction
   (`research.py`), search-query intent parsing (`query_understanding.py`),
   AI-vs-historical/live retrieval and grounded per-result explanations
   (`ai_search.py`, `grounded_explain.py`), and the floating assistant
   (`assistant.py`). AI is opt-in for LIVE DISCOVERY specifically (a
   `GrantSource.config["ai_ngo_id"]`, since a source isn't owned by one NGO)
   and never required for ANY feature to work -- every AI-capable function
   has a fully-functional deterministic fallback.
10. **The floating AI assistant** (`api/routers/assistant.py`,
    `src/ai/assistant.py`) -- a single grounded-Q&A endpoint. The router
    resolves real facts from the database FIRST based on what the user is
    looking at (an opportunity, their NGO profile, a search they ran, or a
    Grant Intelligence chart, passed by the frontend), then hands ONLY those
    facts to the model with a no-invention instruction; without a configured
    provider, the same real facts are returned directly instead of AI prose,
    rather than disabling the widget.
11. **Optional Google sign-in** (`api/routers/auth.py`, `src/auth/`) -- guest
    usage (an NGO profile keyed by a browser-local ID, the ORIGINAL and still
    fully-supported path) never depends on this working. Session tokens follow
    the same hash-at-rest pattern as email verification tokens
    (`src/email/tokens.py`): a high-entropy raw token goes to the browser once,
    only its SHA-256 hash is persisted (`UserSession`). Google's ID token is
    verified against Google's real JWKS (signature + issuer + audience +
    expiry) before any claim in it is trusted. Signing in only ever attaches
    an EXISTING guest `NGOProfile` to a `User` via an explicit,
    user-confirmed `POST /auth/migrate-guest` call -- never automatically.
12. **Grant Intelligence** (`api/routers/analytics.py`) -- unchanged from the
    prior phase (real on-demand pandas/scikit-learn aggregations over the
    historical corpus: overview, trends, funders, categories, a KMeans+PCA
    cluster analysis) but now feeds the floating assistant (frontend passes
    the already-computed analytics payload as grounding context) and gained
    one interactive drill-down (clicking a category jumps to the live feed
    filtered by it) -- "data science calculates, AI explains," never the
    reverse.
