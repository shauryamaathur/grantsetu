"""Turn a ScoredOpportunity's named components into a human-readable explanation.

The explanation is generated directly from the transparent scoring components
(blueprint Section E.9) -- it is not phrased by an LLM and does not invent
reasons beyond what the score actually used.
"""
from src.ranking.score import ScoredOpportunity


def build_explanation(scored: ScoredOpportunity, opportunity_row) -> dict:
    reasons = []

    if scored.text_relevance >= 0.15:
        reasons.append("Strong topical match between your profile and this opportunity's title/category")
    elif scored.text_relevance >= 0.05:
        reasons.append("Some topical overlap with your profile")

    if scored.matched_categories:
        reasons.append(f"Matched focus area(s): {', '.join(scored.matched_categories)}")

    if scored.recency_score >= 1.0:
        reasons.append("Posted within the last 5 years (relatively recent for this historical corpus)")
    elif scored.recency_score == 0.0:
        reasons.append("This opportunity's deadline has passed (historical record)")

    award_bucket = opportunity_row.get("award_size_bucket")
    if award_bucket and award_bucket != "unknown":
        reasons.append(f"Award size bucket: {award_bucket} (relative to this dataset, not INR-comparable)")

    reasons.append(f"Eligibility: {scored.eligibility_reason}")

    if not reasons:
        reasons.append("Included as a low-confidence match; review eligibility details manually")

    return {
        "summary": "Matched on: " + "; ".join(reasons[:3]),
        "reasons": reasons,
        "score_breakdown": {
            "total_score": scored.total_score,
            "text_relevance": scored.text_relevance,
            "category_overlap": scored.category_overlap,
            "recency_score": scored.recency_score,
            "eligibility_multiplier": scored.eligibility_multiplier,
        },
    }
