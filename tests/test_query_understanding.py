"""Tests for src/ai/query_understanding.py -- both the rule-based fallback
(deterministic, no network) and the AI path (mocked provider, matching the
pattern in tests/test_ai.py)."""
from src.ai.providers import AIResult
from src.ai.query_understanding import extract_search_intent


class _StubProvider:
    def __init__(self, result: AIResult):
        self._result = result

    def complete(self, prompt, max_tokens=700):
        return self._result


def test_rule_based_extracts_cause_and_location():
    intent = extract_search_intent("Find education grants for NGOs in India.", None)
    assert intent.method == "rule_based"
    assert "education" in intent.causes
    assert "india" in intent.locations
    assert intent.applicant_type == "nonprofit"


def test_rule_based_handles_hyphenated_cause_phrases():
    """Regression test: FOCUS_AREA_KEYWORDS/BENEFICIARY_KEYWORDS use spaces
    ("climate change"), but a natural query commonly hyphenates
    ("climate-change grants") -- found live while implementing this feature."""
    intent = extract_search_intent("Show climate-change grants open to Indian nonprofits.", None)
    assert "climate change" in intent.causes
    assert "india" in intent.locations

    intent2 = extract_search_intent("Find women-empowerment grants available globally.", None)
    assert "women empowerment" in intent2.causes
    assert intent2.international_ok is True


def test_rule_based_extracts_deadline_window():
    intent = extract_search_intent("Find grants with deadlines in the next 30 days.", None)
    assert intent.deadline_within_days == 30


def test_rule_based_extracts_closing_within_phrasing():
    """Regression test: "closing within N days" (the exact phrasing the
    product brief's own example query uses -- "CSR healthcare funding for
    Indian NGOs closing within 60 days") was NOT recognized by the original
    "next N days"-only pattern; a real query using this common phrasing
    silently got no deadline filter at all."""
    intent = extract_search_intent("Find health grants closing within 60 days.", None)
    assert intent.deadline_within_days == 60

    intent2 = extract_search_intent("Show grants due in 14 days.", None)
    assert intent2.deadline_within_days == 14


def test_rule_based_extracts_country_and_international_flag():
    intent = extract_search_intent("Find USA grants that allow international applicants.", None)
    assert "usa" in intent.locations
    assert intent.international_ok is True


def test_rule_based_never_assumes_international_ok_when_not_mentioned():
    intent = extract_search_intent("Find education grants for NGOs in India.", None)
    assert intent.international_ok is None


def test_ai_path_parses_json_and_reports_method():
    provider = _StubProvider(AIResult(ok=True, text=(
        '{"keywords":["education","india"],"locations":["india"],"causes":["education"],'
        '"applicant_type":"nonprofit","deadline_within_days":null,"international_ok":null,'
        '"funding_min":null,"funding_max":null}'
    )))
    intent = extract_search_intent("Find education grants for NGOs in India.", provider)
    assert intent.method == "ai"
    assert intent.causes == ["education"]
    assert intent.applicant_type == "nonprofit"


def test_ai_path_falls_back_to_rule_based_on_provider_failure():
    provider = _StubProvider(AIResult(ok=False, error="rate limited"))
    intent = extract_search_intent("Find education grants for NGOs in India.", provider)
    assert intent.method == "rule_based"
    assert "education" in intent.causes


def test_ai_path_falls_back_to_rule_based_on_unparseable_response():
    provider = _StubProvider(AIResult(ok=True, text="not json at all"))
    intent = extract_search_intent("Find education grants for NGOs in India.", provider)
    assert intent.method == "rule_based"


def test_search_scope_defaults_to_current_for_normal_discovery_queries():
    for query in [
        "Find grants for education NGOs",
        "Find funding for women empowerment",
        "Open healthcare grants in India",
        "Grants available this month",
        "Find climate funding under 50 lakh",
    ]:
        assert extract_search_intent(query, None).search_scope == "current", query


def test_search_scope_detects_historical_queries():
    for query in [
        "Show me previous grants",
        "What grants did this organization receive?",
        "Historical funding opportunities for education",
        "Past U.S. government grants for India",
    ]:
        assert extract_search_intent(query, None).search_scope == "historical", query


def test_search_scope_detects_research_queries():
    for query in [
        "Who funds climate NGOs?",
        "What foundations fund women's health?",
    ]:
        assert extract_search_intent(query, None).search_scope == "research", query


def test_ai_path_parses_search_scope_and_validates_it():
    provider = _StubProvider(AIResult(ok=True, text=(
        '{"keywords":["education"],"locations":[],"causes":["education"],'
        '"applicant_type":null,"deadline_within_days":null,"international_ok":null,'
        '"funding_min":null,"funding_max":null,"search_scope":"historical"}'
    )))
    intent = extract_search_intent("What grants did this organization receive?", provider)
    assert intent.search_scope == "historical"


def test_ai_path_falls_back_to_rule_based_scope_on_invalid_value():
    """An AI response with a search_scope outside the known set must not be
    trusted verbatim -- falls back to the same deterministic classifier the
    rule-based path uses, so an out-of-range value never silently becomes
    "current" and mis-scopes an explicitly historical query."""
    provider = _StubProvider(AIResult(ok=True, text=(
        '{"keywords":["education"],"locations":[],"causes":["education"],'
        '"applicant_type":null,"deadline_within_days":null,"international_ok":null,'
        '"funding_min":null,"funding_max":null,"search_scope":"archived"}'
    )))
    intent = extract_search_intent("Show me previous grants for education.", provider)
    assert intent.search_scope == "historical"
