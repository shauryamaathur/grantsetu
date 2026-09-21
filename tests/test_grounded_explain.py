"""Tests for src/ai/grounded_explain.py: the match-tier logic (must always
be deterministic, never AI-driven) and the AI-first/rule-based-fallback
explanation text."""
from datetime import date

from src.ai.grounded_explain import (
    TIER_ELIGIBILITY_UNCLEAR,
    TIER_POSSIBLE,
    TIER_STRONG,
    TIER_WEAK,
    explain_result,
)
from src.ai.providers import AIResult
from src.ranking.ai_search import AISearchResult


class _StubProvider:
    def __init__(self, result: AIResult):
        self._result = result

    def complete(self, prompt, max_tokens=700):
        return self._result


def _make_result(**overrides) -> AISearchResult:
    defaults = dict(
        source="imported", ref="r1", title="Education Grant", funder="Test Fund", category="Education",
        deadline=date(2027, 1, 1), url="https://example.org", status_label="Open",
        source_label="Discovered opportunity", is_expired=False, score=0.7,
        hard_filters_applied=[], relevance_known=True, eligibility_known=True,
    )
    defaults.update(overrides)
    return AISearchResult(**defaults)


def test_strong_match_tier():
    result = _make_result(score=0.7)
    explanation = explain_result("education grants", result, None)
    assert explanation.tier == TIER_STRONG
    assert explanation.method == "rule_based"
    assert explanation.ai_disclaimer is None


def test_weak_match_tier():
    result = _make_result(score=0.05)
    explanation = explain_result("education grants", result, None)
    assert explanation.tier == TIER_WEAK


def test_possible_match_tier():
    result = _make_result(score=0.2)
    explanation = explain_result("education grants", result, None)
    assert explanation.tier == TIER_POSSIBLE


def test_eligibility_unclear_overrides_score_based_tier():
    """A high-scoring result with no eligibility text must never be called
    Strong -- eligibility being unclear is a real limitation of the record,
    not something a high relevance score should paper over."""
    result = _make_result(score=0.95, eligibility_known=False)
    explanation = explain_result("education grants", result, None)
    assert explanation.tier == TIER_ELIGIBILITY_UNCLEAR


def test_filter_only_match_never_claims_strong():
    """Regression test: a filter-only match (e.g. a pure deadline-window
    query with no topic keywords) uses a 1.0 sentinel score internally --
    this must never be presented as 'Strong match' since there's no real
    relevance signal behind it."""
    result = _make_result(score=1.0, relevance_known=False, hard_filters_applied=["deadline_within_days"])
    explanation = explain_result("", result, None)
    assert explanation.tier == TIER_POSSIBLE


def test_ai_explanation_used_when_provider_available():
    provider = _StubProvider(AIResult(ok=True, text="This grant supports education work and closes soon."))
    result = _make_result(score=0.7)
    explanation = explain_result("education grants", result, provider)
    assert explanation.method == "ai"
    assert explanation.summary == "This grant supports education work and closes soon."
    assert explanation.ai_disclaimer is not None
    assert explanation.tier == TIER_STRONG  # AI never overrides the deterministic tier


def test_ai_failure_falls_back_to_template():
    provider = _StubProvider(AIResult(ok=False, error="timeout"))
    result = _make_result(score=0.7)
    explanation = explain_result("education grants", result, provider)
    assert explanation.method == "rule_based"
    assert explanation.tier == TIER_STRONG


def test_ai_empty_response_falls_back_to_template():
    provider = _StubProvider(AIResult(ok=True, text=""))
    result = _make_result(score=0.7)
    explanation = explain_result("education grants", result, provider)
    assert explanation.method == "rule_based"
