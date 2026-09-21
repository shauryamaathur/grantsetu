"""Tests for the floating AI assistant (POST /assistant/ask,
src/ai/assistant.py). No real network calls -- a fake AIProvider stub
stands in for a real one, same pattern as tests/test_ai.py."""
from datetime import datetime, timedelta

import pytest

from api.db.models import GrantSource, ImportedGrant
from api.db.session import SessionLocal
from src.ai.assistant import answer_question


class _StubProvider:
    def __init__(self, text: str, ok: bool = True):
        self._text = text
        self._ok = ok

    def complete(self, prompt, max_tokens=700):
        from src.ai.providers import AIResult

        return AIResult(ok=self._ok, text=self._text if self._ok else None, error=None if self._ok else "failed")


def test_answer_question_without_provider_returns_facts_directly():
    facts = {"opportunity_title": "Rural Health Grant", "deadline": "2027-01-01"}
    result = answer_question("What is the deadline?", facts, ["Live opportunity"], provider=None)
    assert result.method == "rule_based"
    assert "Rural Health Grant" in result.answer
    assert "2027-01-01" in result.answer
    assert result.data_available is True


def test_answer_question_with_no_facts_says_information_unavailable():
    result = answer_question("What is the funding amount?", {}, [], provider=None)
    assert result.data_available is False
    assert "does not currently have that information" in result.answer


def test_answer_question_uses_ai_when_provider_available():
    provider = _StubProvider("This grant supports rural health programs with a 2027-01-01 deadline.")
    facts = {"opportunity_title": "Rural Health Grant", "deadline": "2027-01-01"}
    result = answer_question("Explain this opportunity", facts, ["Live opportunity"], provider=provider)
    assert result.method == "ai"
    assert result.ai_disclaimer is not None


def test_answer_question_falls_back_when_ai_fails():
    provider = _StubProvider("", ok=False)
    facts = {"opportunity_title": "Rural Health Grant"}
    result = answer_question("Explain this", facts, [], provider=provider)
    assert result.method == "rule_based"


@pytest.fixture
def live_grant_for_assistant():
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    source = GrantSource(name="Assistant Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    grant = ImportedGrant(
        source_id=source.source_id, external_id="assistant-1", title="Rural Health Access Grant",
        funder="Ocean Trust", category="Health", country="India",
        description="Supports rural health access programs.",
        eligibility_text="nonprofit organizations",
        status="open", dedup_hash="assistant-hash-1", is_sample_data=False,
        deadline=datetime.utcnow() + timedelta(days=20),
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    grant_id = grant.imported_grant_id
    db.close()

    yield grant_id

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_assistant_endpoint_opportunity_context(client, live_grant_for_assistant):
    res = client.post("/assistant/ask", json={
        "message": "What is this opportunity about?",
        "context": {"type": "opportunity", "opportunity_source": "imported", "opportunity_ref": live_grant_for_assistant},
    })
    assert res.status_code == 200
    body = res.json()
    assert body["method"] == "rule_based"  # no AI configured
    assert "Rural Health Access Grant" in body["answer"]
    assert body["data_available"] is True


def test_assistant_endpoint_opportunity_not_found(client):
    res = client.post("/assistant/ask", json={
        "message": "What is this?",
        "context": {"type": "opportunity", "opportunity_source": "imported", "opportunity_ref": "does-not-exist"},
    })
    assert res.status_code == 404


def test_assistant_endpoint_general_context_no_body(client):
    res = client.post("/assistant/ask", json={"message": "What can GrantSetu do?"})
    assert res.status_code == 200
    body = res.json()
    assert "GrantSetu" in body["answer"] or body["data_available"]


def test_assistant_endpoint_ngo_context(client):
    ngo_res = client.post("/ngo-profile", json={
        "name": "Test NGO", "registration_type": "trust", "focus_areas": ["health"], "beneficiaries": ["rural communities"],
    })
    ngo_id = ngo_res.json()["ngo_id"]

    res = client.post("/assistant/ask", json={
        "message": "What opportunities match my NGO?",
        "ngo_id": ngo_id,
        "context": {"type": "ngo"},
    })
    assert res.status_code == 200
    body = res.json()
    assert "Test NGO" in body["answer"]


def test_assistant_endpoint_unknown_ngo_404(client):
    res = client.post("/assistant/ask", json={"message": "hi", "ngo_id": "does-not-exist"})
    assert res.status_code == 404


def test_assistant_endpoint_search_context(client):
    res = client.post("/assistant/ask", json={
        "message": "Which of these close soonest?",
        "context": {
            "type": "search",
            "search_query": "health grants",
            "search_results": [{"title": "Grant A", "deadline": "2027-01-01", "match_tier": "Strong match"}],
        },
    })
    assert res.status_code == 200
    assert "Grant A" in res.json()["answer"]


def test_assistant_endpoint_analytics_context(client):
    res = client.post("/assistant/ask", json={
        "message": "What does this trend mean?",
        "context": {"type": "analytics", "analytics": {"total_rows": 100, "open_count": 40}},
    })
    assert res.status_code == 200
    assert res.json()["data_available"] is True


def test_assistant_endpoint_invalid_context_type_rejected(client):
    res = client.post("/assistant/ask", json={"message": "hi", "context": {"type": "not_a_real_type"}})
    assert res.status_code == 422


def test_assistant_endpoint_rejects_empty_message(client):
    res = client.post("/assistant/ask", json={"message": ""})
    assert res.status_code == 422
