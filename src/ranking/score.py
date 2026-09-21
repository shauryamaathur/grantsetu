"""Transparent weighted hybrid ranking: text relevance + category overlap +
recency + eligibility compatibility.

Every component is named and interpretable (blueprint Section E.9) -- this is
NOT a trained model and scores are not probabilities. Weights are fixed
constants, documented here, and can be tuned without retraining anything.
"""
from dataclasses import dataclass, field

import pandas as pd

from src.ranking.concepts import concept_tags_for_text, normalize, term_match, term_weight
from src.ranking.eligibility_rules import check_eligibility, eligibility_score

# Documented weights for the final weighted-sum score. Must sum to 1.0 across
# the three continuous components; eligibility is applied as a multiplier
# (see eligibility_rules.eligibility_score) rather than an additive term,
# because it represents a hard-ish constraint, not a soft preference.
WEIGHT_TEXT_RELEVANCE = 0.55
WEIGHT_CATEGORY_OVERLAP = 0.25
WEIGHT_RECENCY = 0.20

MIN_RELEVANCE_THRESHOLD = 0.03  # results below this are not shown (blueprint E.8)

# Same per-matched-term cap as src/ranking/live_match.py::OVERLAP_MATCH_CAP --
# kept in sync so a strong single match reads the same way in both scoring
# paths (historical corpus here, live opportunities there).
OVERLAP_MATCH_CAP = 0.55


@dataclass
class ScoredOpportunity:
    opportunity_id: int
    total_score: float
    text_relevance: float
    category_overlap: float
    recency_score: float
    eligibility_multiplier: float
    eligibility_reason: str
    matched_categories: list = field(default_factory=list)


def _category_overlap_score(ngo_focus_areas: list[str], opportunity_category_text: str) -> tuple[float, list[str]]:
    """Weighted, partial-relevance-aware overlap (see
    src/ranking/concepts.py + live_match.py::_concept_overlap for the full
    rationale) -- an exact/synonym/partial match on even one or two of the
    NGO's focus areas produces a meaningful score via diminishing-returns
    combination, rather than requiring most of the NGO's focus areas to
    literally appear in the category text."""
    if not ngo_focus_areas or not opportunity_category_text:
        return 0.0, []
    text = opportunity_category_text.lower()
    text_norm = normalize(text)
    text_concepts = concept_tags_for_text(text)

    remaining = 1.0
    matched: list[str] = []
    for area in ngo_focus_areas:
        is_match, strength, kind = term_match(area, text_norm, text_concepts)
        if not is_match:
            continue
        w = OVERLAP_MATCH_CAP * term_weight(area) * strength
        remaining *= (1 - w)
        matched.append(area)

    if not matched:
        return 0.0, []
    return round(1 - remaining, 4), matched


def _recency_score(is_recent: bool, is_expired: bool) -> float:
    if is_expired:
        return 0.0
    return 1.0 if is_recent else 0.4


def score_opportunities(
    ngo_focus_areas: list[str],
    ngo_registration_type: str | None,
    tfidf_scores: pd.Series,
    opportunities_df: pd.DataFrame,
) -> list[ScoredOpportunity]:
    """Combine per-opportunity TF-IDF similarity with structured signals into
    a single explainable score, for every opportunity id present in `tfidf_scores`.

    `opportunities_df` must be indexed/lookupable by opportunity_id and contain
    columns: category_of_funding_activity, opportunity_category, is_recent,
    is_expired, eligibility_text.
    """
    df_by_id = opportunities_df.set_index("opportunity_id", drop=False)

    results = []
    for opp_id, text_relevance in tfidf_scores.items():
        if opp_id not in df_by_id.index:
            continue
        row = df_by_id.loc[opp_id]

        category_text = " ".join(
            str(row.get(c, "")) for c in ["category_of_funding_activity", "opportunity_category"] if pd.notna(row.get(c))
        )
        cat_score, matched_categories = _category_overlap_score(ngo_focus_areas, category_text)

        recency = _recency_score(bool(row.get("is_recent", False)), bool(row.get("is_expired", False)))

        elig_check = check_eligibility(ngo_registration_type, row.get("eligibility_text"))
        elig_multiplier = eligibility_score(elig_check)

        base_score = (
            WEIGHT_TEXT_RELEVANCE * float(text_relevance)
            + WEIGHT_CATEGORY_OVERLAP * cat_score
            + WEIGHT_RECENCY * recency
        )
        total = base_score * elig_multiplier

        results.append(
            ScoredOpportunity(
                opportunity_id=int(opp_id),
                total_score=round(total, 4),
                text_relevance=round(float(text_relevance), 4),
                category_overlap=round(cat_score, 4),
                recency_score=round(recency, 4),
                eligibility_multiplier=round(elig_multiplier, 4),
                eligibility_reason=elig_check.reason,
                matched_categories=matched_categories,
            )
        )

    results.sort(key=lambda r: (-r.total_score, df_by_id.loc[r.opportunity_id].get("close_date") or pd.Timestamp.max))
    return [r for r in results if r.total_score >= MIN_RELEVANCE_THRESHOLD]
