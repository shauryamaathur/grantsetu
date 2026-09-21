"""Tests for the explainable 0-100 live-opportunity match score
(src/ranking/live_match.py) -- the redesign that replaced a raw decimal
relevance score with an integer score, a 6-band tier, and a named-component
breakdown (see interview.txt's match-score redesign and its later
"more lenient scoring curve" follow-up)."""
from datetime import datetime, timedelta

from api.db.models import ImportedGrant
from src.ranking.live_match import (
    TIER_EXCEPTIONAL,
    TIER_GOOD,
    TIER_LOW,
    TIER_MINIMAL,
    TIER_MODERATE,
    TIER_STRONG,
    TIER_VERY_STRONG,
    build_match_score,
    rank_live_grants,
)


def _grant(**overrides) -> ImportedGrant:
    defaults = dict(
        source_id="src-1",
        external_id="ext-1",
        title="Rural Health Access Grant",
        funder="Ocean Trust",
        description="Supports rural health access programs for underserved communities.",
        category="Health",
        eligibility_text="nonprofit organizations",
        status="open",
        country="India",
        dedup_hash="h1",
        is_sample_data=False,
        deadline=datetime.utcnow() + timedelta(days=30),
    )
    defaults.update(overrides)
    return ImportedGrant(**defaults)


def test_score_is_integer_0_to_100():
    grant = _grant()
    score = build_match_score(grant, text_relevance=0.8, focus_areas=["health"], beneficiaries=["rural communities"], locations=[], registration_type="trust")
    assert isinstance(score.score_100, int)
    assert 0 <= score.score_100 <= 100


def test_high_relevance_strong_match_tier():
    """Very high raw relevance (near-total text similarity + exact focus/
    beneficiary/geography match) should land in the top bands the
    leniency curve reserves for that -- "very strong" or "exceptional",
    not merely "strong" (see live_match.py's _LENIENCY_CURVE_ANCHORS)."""
    grant = _grant(description="Supports health programs for rural communities in Maharashtra.")
    score = build_match_score(
        grant, text_relevance=0.95, focus_areas=["health"], beneficiaries=["rural communities"],
        locations=["Maharashtra"], registration_type="trust",
    )
    assert score.tier in {TIER_VERY_STRONG, TIER_EXCEPTIONAL}
    assert score.score_100 >= 85


def test_low_relevance_low_tier():
    grant = _grant(country="unknown", deadline=None)
    score = build_match_score(
        grant, text_relevance=0.02, focus_areas=["health"], beneficiaries=[], locations=[], registration_type=None,
    )
    assert score.tier == TIER_LOW


def test_eligibility_incompatible_reduces_score_and_is_explained():
    """An opportunity restricted to a structure an Indian NGO can't be
    (see eligibility_rules.HARD_EXCLUSION_KEYWORDS) must score lower than
    the identical opportunity with compatible eligibility text, and the
    breakdown must show WHY."""
    compatible = _grant(eligibility_text="nonprofit organizations")
    incompatible = _grant(eligibility_text="Eligible applicants: State governments only")

    compatible_score = build_match_score(compatible, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=[], registration_type="trust")
    incompatible_score = build_match_score(incompatible, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=[], registration_type="trust")

    assert incompatible_score.score_100 < compatible_score.score_100
    elig_component = next(c for c in incompatible_score.components if c.label == "Eligibility")
    assert elig_component.points < 0
    assert "state governments" in elig_component.detail.lower()


def test_geography_match_scores_higher_than_no_location_signal():
    grant_with_location = _grant(description="Supports programs in Maharashtra and Gujarat.")
    grant_without = _grant(description="Supports programs for underserved communities.", country="unknown")

    with_geo = build_match_score(grant_with_location, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=["Maharashtra"], registration_type="trust")
    without_geo = build_match_score(grant_without, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=["Maharashtra"], registration_type="trust")

    assert with_geo.score_100 > without_geo.score_100


def test_breakdown_never_shows_zero_value_components():
    """A component that contributed zero points is omitted entirely rather
    than shown as a fabricated "0 points, no reason" line -- e.g. no
    text-relevance component when text_relevance is exactly 0."""
    grant = _grant()
    score = build_match_score(grant, text_relevance=0.0, focus_areas=["health"], beneficiaries=[], locations=[], registration_type="trust")
    labels = [c.label for c in score.components]
    assert "Topical relevance" not in labels
    # Geography and Deadline relevance always appear (never zero-value by
    # design -- "unknown" location/deadline gets a neutral, not zero, score).
    assert "Geography" in labels
    assert "Deadline relevance" in labels


def test_deadline_urgency_scoring():
    far = _grant(deadline=datetime.utcnow() + timedelta(days=30))
    too_soon = _grant(deadline=datetime.utcnow() + timedelta(days=2))
    very_far = _grant(deadline=datetime.utcnow() + timedelta(days=300))

    far_score = build_match_score(far, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=[], registration_type="trust")
    soon_score = build_match_score(too_soon, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=[], registration_type="trust")
    very_far_score = build_match_score(very_far, text_relevance=0.5, focus_areas=["health"], beneficiaries=[], locations=[], registration_type="trust")

    # A practical 30-day window scores better than either extreme.
    assert far_score.score_100 >= soon_score.score_100
    assert far_score.score_100 >= very_far_score.score_100


def test_rank_live_grants_excludes_grants_with_no_textual_relevance():
    relevant = _grant(external_id="e1", title="Rural Health Access Grant", dedup_hash="h1")
    irrelevant = _grant(external_id="e2", title="Urban Infrastructure Road Grant", description="Road construction funding.", category="Infrastructure", dedup_hash="h2")

    matches = rank_live_grants([relevant, irrelevant], focus_areas=["health"], beneficiaries=["rural communities"], limit=10)
    titles = {m.grant.title for m in matches}
    assert "Rural Health Access Grant" in titles
    assert "Urban Infrastructure Road Grant" not in titles


def test_rank_live_grants_returns_match_score_on_each_result():
    grant = _grant()
    matches = rank_live_grants([grant], focus_areas=["health"], beneficiaries=[], locations=["Maharashtra"], registration_type="trust", limit=10)
    assert len(matches) == 1
    assert matches[0].match_score.score_100 > 0
    assert matches[0].match_score.tier in {
        TIER_MINIMAL, TIER_LOW, TIER_MODERATE, TIER_GOOD, TIER_STRONG, TIER_VERY_STRONG, TIER_EXCEPTIONAL,
    }
