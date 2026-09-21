"""Tests for src/ranking/ai_search.py: retrieval + ranking across LIVE
ImportedGrant data ONLY -- this module never queries the historical
GrantSetu corpus (see its module docstring's hard architectural rule), so
these tests seed only ImportedGrant rows, same pattern as
tests/test_saved.py / tests/test_live_opportunities.py."""
from datetime import datetime, timedelta

import pytest

from api.db.models import GrantSource, ImportedGrant
from api.db.session import SessionLocal
from src.ai.query_understanding import extract_search_intent
from src.ranking.ai_search import search


@pytest.fixture
def live_grant(client):  # `client` ensures tables exist (create_all_tables), regardless of test run order
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()

    source = GrantSource(name="AI Search Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    grant = ImportedGrant(
        source_id=source.source_id,
        external_id="ai-search-1",
        title="India Education Access Grant",
        funder="Test Foundation",
        category="Education",
        description="Supports education programs in India for youth.",
        eligibility_text="Registered nonprofits in India.",
        status="open",
        dedup_hash="ai-search-hash-1",
        is_sample_data=False,
        deadline=datetime.utcnow() + timedelta(days=10),
    )
    db.add(grant)
    db.commit()
    grant_id = grant.imported_grant_id
    db.close()

    yield grant_id

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


@pytest.fixture
def db_session(client):  # `client` ensures tables exist regardless of test run order
    db = SessionLocal()
    yield db
    db.close()


def test_topical_query_finds_live_grant(db_session, live_grant):
    intent = extract_search_intent("Find education grants for NGOs in India.", None)
    results = search(db_session, intent, limit=10)
    assert any(r.ref == live_grant for r in results)
    assert all(r.score > 0 for r in results)


def test_pure_deadline_query_returns_hard_filtered_results_without_relevance_score(db_session, live_grant):
    """Regression test: a query with no topic keywords at all (e.g. 'Find
    grants with deadlines in the next 30 days') must not be pruned to an
    empty list by the score>0 relevance filter -- found while building this
    feature. relevance_known must be False for these (no real text-
    relevance signal exists)."""
    intent = extract_search_intent("Find grants with deadlines in the next 30 days.", None)
    results = search(db_session, intent, limit=10)
    assert any(r.ref == live_grant for r in results)
    matched = next(r for r in results if r.ref == live_grant)
    assert matched.relevance_known is False
    assert "deadline_within_days" in matched.hard_filters_applied


def test_irrelevant_query_returns_empty_not_padded(db_session, live_grant):
    intent = extract_search_intent("wildlife conservation marine biology tigers", None)
    results = search(db_session, intent, limit=10)
    assert results == []


def test_no_sample_data_ever_returned(db_session):
    db = db_session
    source = GrantSource(name="Sample Source For AI Search Test", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    db.add(ImportedGrant(
        source_id=source.source_id, external_id="sample-1", title="Education Grant Sample",
        category="Education", status="open", dedup_hash="sample-ai-search-hash",
        is_sample_data=True,
    ))
    db.commit()

    intent = extract_search_intent("education grants", None)
    results = search(db, intent, limit=10)
    assert all(not r.ref == "sample-1" for r in results)

    db.query(ImportedGrant).filter_by(external_id="sample-1").delete()
    db.commit()


def test_closed_live_grant_excluded(db_session):
    db = db_session
    source = GrantSource(name="Closed Grant Source For AI Search", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    db.add(ImportedGrant(
        source_id=source.source_id, external_id="closed-1", title="Closed Education Grant",
        category="Education", status="closed", dedup_hash="closed-ai-search-hash",
        is_sample_data=False,
    ))
    db.commit()

    intent = extract_search_intent("education grants", None)
    results = search(db, intent, limit=10)
    assert all(r.title != "Closed Education Grant" for r in results)

    db.query(ImportedGrant).filter_by(external_id="closed-1").delete()
    db.commit()


def test_eligibility_known_flag_reflects_real_data(db_session, live_grant):
    intent = extract_search_intent("education grants in India", None)
    results = search(db_session, intent, limit=10)
    # tests/conftest's live_grant fixture sets a real eligibility_text.
    assert any(r.ref == live_grant and r.eligibility_known for r in results)


def test_search_never_touches_historical_corpus_regardless_of_scope(db_session, live_grant):
    """The hard architectural rule from src/ranking/ai_search.py's module
    docstring: no result returned by search() may ever be source=
    "historical", no matter what search_scope the query classifies as --
    there is no historical retrieval path in this module at all."""
    for query in [
        "education grants in India",  # current
        "previous education grants in India",  # historical
        "who funds education NGOs",  # research
    ]:
        intent = extract_search_intent(query, None)
        results = search(db_session, intent, limit=10)
        assert all(r.source == "imported" for r in results)
