# GrantSetu

An NGO funding discovery and intelligence platform.

GrantSetu helps NGOs find funding that actually fits them. An organization
describes its focus areas, location, and beneficiaries once, and GrantSetu
ranks opportunities against that profile with a transparent, explainable
score — combining a historical reference corpus, continuously discovered
live opportunities, and natural-language search.

## Features

- **Explainable matching** — every opportunity gets a transparent score
  (text relevance, category overlap, recency, eligibility), never a black box
- **Live opportunity discovery** — connectors continuously pull in current
  funding opportunities from real sources (e.g. Grants.gov, NGOBox), kept
  fresh via background refresh jobs
- **Ask GrantSetu** — natural-language search ("healthcare funding for NGOs
  in India closing this month"), AI-assisted when configured, with a fully
  working deterministic fallback when it isn't
- **NGO profiles** — guest-friendly by default; optional Google sign-in adds
  persistence, never a requirement
- **Website-driven profile building** — point GrantSetu at your NGO's
  website and it proposes focus areas/locations/beneficiaries to review
- **Saved opportunities & application tracking** — bookmark and move a
  grant through your own pipeline
- **Grant Intelligence** — interactive analytics over the historical corpus
  (trends, funders, categories, clustering)
- **Multi-provider AI, always optional** — OpenAI, Anthropic, Gemini,
  DeepSeek, Groq, Mistral, Together AI, OpenRouter, or Cohere, configured
  per-NGO; every AI-assisted feature degrades gracefully without one

## Tech stack

- **Backend**: FastAPI (Python), SQLAlchemy (SQLite for local dev,
  PostgreSQL for production)
- **Frontend**: a single static HTML/JS page — no build step, no framework
- **Deployment**: Docker images for both backend and frontend, deployable
  to AWS or any container-friendly platform (Render, Railway, etc.)

## Getting started

```bash
git clone https://github.com/shauryamaathur/grantsetu.git
cd grantsetu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in what you need — see below
```

This repository doesn't include the historical opportunity dataset (a large
CSV) to keep it lightweight. Place your own copy at
`data/raw/grants_raw.csv` before running the backend for the first time.

**Run the backend:**

```bash
uvicorn api.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs` · Health check: `http://localhost:8000/health`

**Run the frontend** (in a second terminal):

```bash
cd frontend
python3 -m http.server 5500
```

Then open `http://localhost:5500`.

## Configuration

Every environment variable GrantSetu reads is documented in
[`.env.example`](.env.example) — copy it to `.env` and fill in only what you
need. Nothing is required for the app to run locally; AI providers, real
email delivery, and Google sign-in are all optional and degrade gracefully
when unconfigured.

## Running tests

```bash
pytest
```

## Deployment

A `Dockerfile` (backend) and `frontend/Dockerfile` are included, along with
`docker-compose.yml` for local production-like testing against PostgreSQL.
The same images deploy to AWS (EC2 + RDS) or a managed platform like Render
— configuration is entirely environment-variable driven, so no code changes
are needed between environments.
