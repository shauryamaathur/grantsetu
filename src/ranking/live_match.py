"""Relevance ranking for LIVE (ImportedGrant) opportunities against one
NGO's profile -- what powers GET /opportunities/recommended.

This is deliberately a separate, lighter-weight module from src/ranking/
corpus.py + score.py: those are built around the ~75k-row historical corpus
(a precomputed, disk-cached TF-IDF index over structured columns like
opportunity_category/eligible_applicants_type). Live opportunities are far
fewer (typically tens to low hundreds after a sync), have no such
structured taxonomy, and change between syncs -- so a fresh, small TF-IDF
fit per request is both correct and cheap (sub-millisecond at this scale),
with no precomputed index to keep in sync.

A grant is only returned if it has NONZERO textual relevance to the NGO's
profile -- there is no "everyone gets a fallback recommendation" path. If
nothing in the live set shares any vocabulary with the profile, the honest
result is an empty list, not a forced/arbitrary top-N.

SCORE DESIGN (mirrors src/ranking/score.py's historical-corpus design:
named, weighted, documented components + an eligibility multiplier -- never
a trained/opaque model):

  raw_100 = round(100 * (
        WEIGHT_TEXT_RELEVANCE   * TF-IDF cosine similarity(profile, grant)
      + WEIGHT_FOCUS_BENEFICIARY* keyword-overlap(focus_areas+beneficiaries, grant text)
      + WEIGHT_GEOGRAPHY        * geography_score(ngo.locations, grant.country/text)
      + WEIGHT_DEADLINE         * deadline_urgency(grant.deadline)
  ) * eligibility_multiplier)

  score_100 = LENIENCY_CURVE(raw_100)   -- see _apply_leniency_curve

Every component is named and shown in the breakdown (LiveMatchScore.
components) -- nothing is fabricated, and a component contributing zero
points is simply omitted from the breakdown rather than shown as "0 for
some invented reason." Component points are rescaled by the same factor
the leniency curve applies to the total, so the breakdown still sums to
`score_100` -- the curve reshapes confidence, it does not change which
component earned the credit. `score_100` is a RELEVANCE score, never framed
as a probability of winning funding.

LENIENCY CURVE: raw_100 (the literal weighted-sum score above) clusters low
even for genuinely relevant opportunities, because TF-IDF cosine similarity
against a short profile is mathematically small and partial/synonym
overlap (src/ranking/concepts.py) still only closes part of that gap. Per
product requirement, ~30 should read as "the practical floor of a
meaningful match" and scores above that floor should compress upward much
faster than raw_100 does on its own, while genuinely weak matches (roughly
raw_100 < 20) must stay low. _apply_leniency_curve is a fixed, monotonic,
piecewise-linear remapping (a calibration table, not a trained model) that
implements exactly that shape -- see its docstring for the anchor points.
"""
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from api.db.models import ImportedGrant
from src.ranking.concepts import concept_tag_tokens, concept_tags_for_text, normalize, term_match, term_weight
from src.ranking.eligibility_rules import check_eligibility, eligibility_score

# Documented weights for the four additive components -- must sum to 1.0.
# Eligibility is applied as a MULTIPLIER on top (same design as
# src/ranking/score.py), because it represents a hard-ish constraint, not a
# soft preference: an opportunity a INCOMPATIBLE-registration-type NGO
# structurally cannot apply to should never outscore a compatible one purely
# on topical relevance.
#
# Weighted UP relative to geography/deadline (was 0.35/0.30/0.20/0.15) so
# thematic/beneficiary relevance -- the two components partial/synonym
# matching (src/ranking/concepts.py) actually improves -- dominates the
# score, per the product requirement that a few genuinely relevant concepts
# should move the needle more than they used to.
WEIGHT_TEXT_RELEVANCE = 0.40
WEIGHT_FOCUS_BENEFICIARY_OVERLAP = 0.35
WEIGHT_GEOGRAPHY = 0.15
WEIGHT_DEADLINE = 0.10

# Per-matched-term cap applied before combining overlap matches (see
# _concept_overlap) -- keeps a single exact match strong-but-not-total
# (0.55, not 1.0) so the curve still rewards a SECOND and THIRD relevant
# concept rather than maxing out on the first one, while still giving that
# first meaningful match a real, noticeable jump (not a 1-of-10 fraction).
OVERLAP_MATCH_CAP = 0.55

# Integer score bands shown to the user as a 6-state visual -- see
# interview.txt's match-score redesign. Never shown as a decimal or a
# claimed win-probability; always "relevance" in copy. Bucketed off the
# CURVED score_100 (post leniency curve), matching the product's stated
# interpretation of the 0-100 scale:
#   0-29  very weak / minimal relevance
#   30-49 low match       (the practical floor of a "meaningful" match)
#   50-64 moderate match
#   65-74 good potential
#   75-84 strong match
#   85-94 very strong match
#   95-100 exceptional match
TIER_MINIMAL = "minimal"
TIER_LOW = "low"
TIER_MODERATE = "moderate"
TIER_GOOD = "good"
TIER_STRONG = "strong"
TIER_VERY_STRONG = "very_strong"
TIER_EXCEPTIONAL = "exceptional"
TIER_LABELS = {
    TIER_MINIMAL: "Weak match",
    TIER_LOW: "Low match",
    TIER_MODERATE: "Moderate match",
    TIER_GOOD: "Good potential",
    TIER_STRONG: "Strong match",
    TIER_VERY_STRONG: "Very strong match",
    TIER_EXCEPTIONAL: "Exceptional match",
}
# Kept for backward compatibility with older callers/tests that only knew
# the original 3-tier scheme -- TIER_POTENTIAL now maps onto the "moderate"
# band, the rough midpoint of the old "potential" range.
TIER_POTENTIAL = TIER_MODERATE

# (lower-bound score_100, tier) pairs, descending -- first match wins.
_TIER_THRESHOLDS = [
    (95, TIER_EXCEPTIONAL),
    (85, TIER_VERY_STRONG),
    (75, TIER_STRONG),
    (65, TIER_GOOD),
    (50, TIER_MODERATE),
    (30, TIER_LOW),
    (0, TIER_MINIMAL),
]

# Leniency curve anchors: (raw_100, curved_100). Fixed, monotonic,
# deterministic calibration table (piecewise-linear interpolation between
# points) -- not a trained model. Raw scores below ~20 stay under the 30
# "meaningful match" floor; raw scores from ~20 up are compressed upward
# aggressively so that a raw ~50 (meaningful partial relevance under the
# component weights above) lands in the 80s, while raw ~30 only reaches the
# high-40s/50 -- i.e. "a current 30 should not suddenly become 80" but "a
# current 51 should be capable of becoming 80+".
_LENIENCY_CURVE_ANCHORS = [
    (0, 0), (5, 8), (10, 16), (15, 24), (20, 33), (25, 38), (30, 48),
    (35, 58), (40, 68), (45, 76), (50, 83), (55, 87), (60, 90), (65, 93),
    (70, 97), (80, 99), (90, 100), (100, 100),
]


def _apply_leniency_curve(raw_100: float) -> float:
    """Piecewise-linear remap of a raw 0-100 score onto the more generous
    curve described above. Deterministic and monotonic (never reorders two
    opportunities relative to each other), so this only changes HOW
    generous the number looks, never the ranking."""
    raw_100 = max(0.0, min(100.0, raw_100))
    anchors = _LENIENCY_CURVE_ANCHORS
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= raw_100 <= x1:
            if x1 == x0:
                return float(y0)
            t = (raw_100 - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 100.0

# Only these statuses are "currently applicable or genuinely unknown" per
# the product requirement that Recommended never shows closed/expired
# grants. A status of "unknown" is included deliberately -- it means the
# source didn't confirm a status either way, which is different from a
# CONFIRMED closed/expired grant and should not be hidden from the user.
RECOMMENDABLE_STATUSES = {"open", "forecasted", "unknown"}


@dataclass
class ScoreComponent:
    label: str
    points: int  # points out of 100 actually contributed by this component (can be negative, for the eligibility penalty)
    detail: str


@dataclass
class LiveMatchScore:
    score_100: int
    tier: str  # strong | potential | low
    tier_label: str
    components: list[ScoreComponent] = field(default_factory=list)


@dataclass
class LiveMatch:
    grant: ImportedGrant
    score: float  # raw 0..1 relevance, kept for sorting/backward compatibility
    match_reasons: list[str]
    match_score: LiveMatchScore


def _combined_text(grant: ImportedGrant) -> str:
    parts = [grant.title, grant.category, grant.funder, grant.description, grant.eligibility_text]
    return " ".join(p for p in parts if p)


def _concept_overlap(terms: list[str], text_lower: str) -> tuple[float, list[str]]:
    """Weighted, partial-relevance-aware overlap between the NGO's own
    terms (focus areas + beneficiaries) and the grant's text -- replaces a
    plain "fraction of terms found verbatim" count, which required nearly
    ALL of an NGO's terms to literally appear before the component meant
    anything (the root cause of scores clustering low even for NGOs with
    several genuinely relevant focus areas).

    Each term is matched via src/ranking/concepts.py::term_match, which
    tries exact phrase, then shared-concept (synonym), then partial-word
    overlap -- never requiring identical wording. Matched terms are
    combined with diminishing returns (1 - product(1 - w_i)) rather than
    a linear fraction, so the score curve is FRONT-LOADED: the first
    strong match moves the score a lot, the second/third add real but
    shrinking increments, and a profile with many terms doesn't need most
    of them to match to reach a meaningful score. Generic single words
    ("community", "development", ...) are capped to low weight via
    term_weight so they can't drive a match alone (interview.txt: "avoid
    keyword stuffing")."""
    text_norm = normalize(text_lower)
    text_concepts = _text_concepts(text_lower)
    cleaned = [t for t in terms if t and t.strip()]
    if not cleaned:
        return 0.0, []

    remaining = 1.0
    matched_labels: list[str] = []
    for term in cleaned:
        matched, strength, kind = term_match(term, text_norm, text_concepts)
        if not matched:
            continue
        w = OVERLAP_MATCH_CAP * term_weight(term) * strength
        remaining *= (1 - w)
        if kind == "exact":
            matched_labels.append(term)
        else:
            matched_labels.append(f"{term} ({kind} match)")

    if not matched_labels:
        return 0.0, []
    return round(1 - remaining, 4), matched_labels


def _text_concepts(text_lower: str) -> set[str]:
    return concept_tags_for_text(text_lower)


def _geography_score(locations: list[str], grant: ImportedGrant, text_lower: str) -> tuple[float, str]:
    """Never penalizes an opportunity just for not confirming a country --
    "unknown" gets a neutral score, not a low one. A grant that literally
    mentions one of the NGO's own operating locations is the strongest,
    most concrete signal available; short of that, grant.country (see
    src/ai/research.py::_detect_country -- never fabricated) is the next
    best signal."""
    cleaned = [l for l in locations if l and l.strip()]
    if cleaned:
        matched = [l for l in cleaned if l.lower().strip() in text_lower]
        if matched:
            return 1.0, f"Mentions your operating location(s): {', '.join(matched[:3])}"
    if grant.country == "India":
        return 0.6, "This opportunity is based in India"
    if grant.country and grant.country != "unknown":
        return 0.4, f"This opportunity is based in {grant.country} (may still accept international applicants)"
    return 0.5, "This opportunity's location is not confirmed by the source"


def _deadline_score(deadline) -> tuple[float, str]:
    """Deadline RELEVANCE for actionability, not a claim about the
    opportunity's quality -- a deadline that's neither too soon to prepare
    for nor so distant it's not worth tracking yet scores highest. An
    opportunity with no known deadline gets a neutral score (never
    penalized for a source that simply didn't state one)."""
    if deadline is None:
        return 0.5, "Deadline not specified by the source"
    days = (deadline - datetime.utcnow()).days
    if days < 0:
        return 0.0, "Deadline has passed"
    if days <= 7:
        return 0.5, f"Deadline in {days} day(s) -- limited time to prepare an application"
    if days <= 45:
        return 1.0, f"Deadline in {days} days -- a practical window to prepare"
    if days <= 120:
        return 0.8, f"Deadline in {days} days"
    return 0.6, f"Deadline in {days}+ days -- plenty of lead time"


def _match_reasons(grant: ImportedGrant, focus_areas: list[str], beneficiaries: list[str]) -> list[str]:
    text_lower = _combined_text(grant).lower()
    reasons = []
    _, matched_terms = _concept_overlap([*focus_areas, *beneficiaries], text_lower)
    if matched_terms:
        reasons.append(f"Overlaps on your focus area(s)/beneficiary group(s): {', '.join(matched_terms[:4])}")
    if grant.category:
        reasons.append(f"Category: {grant.category}")
    if not reasons:
        reasons.append("Related to your profile description by overall text relevance")
    return reasons


def _score_to_tier(score_100: int) -> tuple[str, str]:
    for lower_bound, tier in _TIER_THRESHOLDS:
        if score_100 >= lower_bound:
            return tier, TIER_LABELS[tier]
    return TIER_MINIMAL, TIER_LABELS[TIER_MINIMAL]


def build_match_score(
    grant: ImportedGrant,
    text_relevance: float,
    focus_areas: list[str],
    beneficiaries: list[str],
    locations: list[str],
    registration_type: str | None,
) -> LiveMatchScore:
    """Builds the explainable 0-100 score + breakdown for one grant, given
    its already-computed TF-IDF text_relevance (0..1, computed in batch
    across all candidates by rank_live_grants -- TF-IDF requires the whole
    candidate set for IDF weighting, so it can't be computed per-grant in
    isolation here)."""
    text_lower = _combined_text(grant).lower()

    overlap_score, matched_terms = _concept_overlap([*focus_areas, *beneficiaries], text_lower)
    geo_score, geo_detail = _geography_score(locations, grant, text_lower)
    deadline_score, deadline_detail = _deadline_score(grant.deadline)

    elig_check = check_eligibility(registration_type, grant.eligibility_text)
    elig_multiplier = eligibility_score(elig_check)

    base = (
        WEIGHT_TEXT_RELEVANCE * text_relevance
        + WEIGHT_FOCUS_BENEFICIARY_OVERLAP * overlap_score
        + WEIGHT_GEOGRAPHY * geo_score
        + WEIGHT_DEADLINE * deadline_score
    )
    raw_before_eligibility = base * 100
    raw_after_eligibility = base * elig_multiplier * 100
    curved_100 = _apply_leniency_curve(raw_after_eligibility)
    score_100 = round(curved_100)

    # Rescale every component by the SAME factor the leniency curve applied
    # to the total, so the displayed breakdown still adds up to score_100 --
    # the curve recalibrates how generous the number looks overall, it does
    # not change which component earned how much of the credit.
    scale = curved_100 / raw_after_eligibility if raw_after_eligibility > 1e-9 else 0.0

    components = []
    text_points = round(WEIGHT_TEXT_RELEVANCE * text_relevance * 100 * scale)
    if text_points > 0:
        components.append(ScoreComponent("Topical relevance", text_points, "Overall text similarity to your profile"))
    overlap_points = round(WEIGHT_FOCUS_BENEFICIARY_OVERLAP * overlap_score * 100 * scale)
    if overlap_points > 0:
        components.append(ScoreComponent("Focus area / beneficiary match", overlap_points, f"Matches: {', '.join(matched_terms[:4])}"))
    geo_points = round(WEIGHT_GEOGRAPHY * geo_score * 100 * scale)
    components.append(ScoreComponent("Geography", geo_points, geo_detail))
    deadline_points = round(WEIGHT_DEADLINE * deadline_score * 100 * scale)
    components.append(ScoreComponent("Deadline relevance", deadline_points, deadline_detail))

    if elig_multiplier < 1.0:
        penalty = round((raw_before_eligibility - raw_after_eligibility) * scale)
        components.append(ScoreComponent("Eligibility", -penalty, elig_check.reason))
    else:
        components.append(ScoreComponent("Eligibility", 0, elig_check.reason))

    tier, tier_label = _score_to_tier(score_100)
    return LiveMatchScore(score_100=score_100, tier=tier, tier_label=tier_label, components=components)


def rank_live_grants(
    grants: list[ImportedGrant],
    focus_areas: list[str],
    beneficiaries: list[str],
    locations: list[str] | None = None,
    registration_type: str | None = None,
    limit: int = 20,
) -> list[LiveMatch]:
    """Ranks `grants` (already filtered to non-sample, recommendable-status
    rows by the caller) by relevance to the NGO's profile. Returns only
    grants with a nonzero relevance score, highest first -- never pads the
    result with irrelevant grants just to fill a quota."""
    locations = locations or []
    query_text = " ".join([*focus_areas, *beneficiaries]).strip()
    if not query_text or not grants:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [_combined_text(g) for g in grants]
    if not any(doc.strip() for doc in documents):
        return []

    # Append concept tags (src/ranking/concepts.py) to both the query and
    # every document before fitting TF-IDF. This is the deterministic
    # stand-in for semantic similarity the module docstring above
    # describes: e.g. a profile saying "women empowerment" and a grant
    # saying "women's economic inclusion" share no raw vocabulary, but both
    # get tagged "__concept_women_empowerment__", so TF-IDF cosine
    # similarity picks up the relationship without exact wording.
    tagged_query = f"{query_text} {concept_tag_tokens(query_text)}".strip()
    tagged_documents = [f"{doc} {concept_tag_tokens(doc)}".strip() for doc in documents]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([tagged_query, *tagged_documents])
    except ValueError:
        # All documents + query were empty/stop-words-only after tokenizing
        # -- no meaningful vocabulary to compare, so no honest match exists.
        return []

    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    matches = []
    for g, text_relevance in zip(grants, similarities):
        text_relevance = float(text_relevance)
        text_lower = _combined_text(g).lower()
        overlap_score, _ = _concept_overlap([*focus_areas, *beneficiaries], text_lower)
        if text_relevance <= 0 and overlap_score <= 0:
            continue  # no genuine textual relevance at all -- never a forced match
        match_score = build_match_score(g, text_relevance, focus_areas, beneficiaries, locations, registration_type)
        matches.append(LiveMatch(
            grant=g,
            score=text_relevance,
            match_reasons=_match_reasons(g, focus_areas, beneficiaries),
            match_score=match_score,
        ))

    matches.sort(key=lambda m: m.match_score.score_100, reverse=True)
    return matches[:limit]


def build_canonical_match_scores(
    db: Session,
    focus_areas: list[str],
    beneficiaries: list[str],
    locations: list[str] | None,
    registration_type: str | None,
) -> dict[str, LiveMatch]:
    """THE single source of truth for an NGO's match score against a live
    opportunity, keyed by imported_grant_id. Every surface that shows a
    live grant's score to a given NGO -- GET /opportunities/recommended,
    GET /opportunities/live/{id}, and POST /opportunities/ai-search -- MUST
    read from this same function's output rather than computing its own
    score, so the identical opportunity never shows two different numbers
    depending on which page it's viewed from (a real bug found in
    production: the detail page used to rank a grant against only itself,
    a different TF-IDF fit than the batch fit the list view used).

    Always ranks over the FULL recommendable candidate pool (is_sample_data
    =False, status in RECOMMENDABLE_STATUSES) -- deliberately NOT filtered
    by this NGO's hidden-opportunity feedback, because a grant's score is a
    property of (grant, NGO profile), not of what else the NGO happens to
    have hidden; callers that need to exclude hidden opportunities from a
    LISTING (e.g. /recommended) do that filtering on the output list, never
    by changing the scoring corpus itself, which would make the same
    grant's score silently depend on unrelated hide actions.

    Returns only grants with a nonzero relevance score (same honesty rule
    as rank_live_grants) -- a grant absent from this dict has no genuine
    match for this profile, and callers must treat that as "no score" (None),
    never fabricate one."""
    candidates = (
        db.query(ImportedGrant)
        .filter_by(is_sample_data=False)
        .filter(ImportedGrant.status.in_(RECOMMENDABLE_STATUSES))
        .all()
    )
    matches = rank_live_grants(
        candidates, focus_areas, beneficiaries,
        locations=locations, registration_type=registration_type,
        limit=len(candidates),
    )
    return {m.grant.imported_grant_id: m for m in matches}
