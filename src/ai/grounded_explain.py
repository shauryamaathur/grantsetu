"""Generates a per-result match tier and explanation for AI-powered search
results (src/ranking/ai_search.py::AISearchResult) -- the LAST step of the
ai-search pipeline, run only on an already-shortlisted, already-ranked
top-N (never the full corpus; see module docstrings in ai_search.py and
query_understanding.py for why).

Match tier is ALWAYS computed deterministically from real signals already on
the result (relevance score, whether relevance is even known, whether the
record has any eligibility text at all) -- never an AI judgment call, so a
"Strong match" claim is always backed by a real number, and "Eligibility
unclear" always means the underlying record genuinely has no eligibility
text, not an AI guess that it might not apply.

The explanation TEXT may be AI-phrased (if the NGO has a provider
configured) using ONLY the result's own real fields as context, with a
strict prompt forbidding invented facts -- but the tier itself is computed
BEFORE and independently of any AI call, and the AI is never allowed to
override it.
"""
from dataclasses import dataclass

from src.ai.providers import AIProvider
from src.ranking.ai_search import AISearchResult

STRONG_THRESHOLD = 0.5
POSSIBLE_THRESHOLD = 0.15

TIER_STRONG = "Strong match"
TIER_POSSIBLE = "Possible match"
TIER_WEAK = "Weak match"
TIER_ELIGIBILITY_UNCLEAR = "Eligibility unclear"


@dataclass
class GroundedExplanation:
    tier: str
    summary: str
    method: str  # "ai" | "rule_based"
    ai_disclaimer: str | None = None


def compute_match_tier(result: AISearchResult) -> str:
    """Deterministic, never AI-driven -- see module docstring."""
    if not result.eligibility_known:
        return TIER_ELIGIBILITY_UNCLEAR
    if not result.relevance_known:
        # A filter-only match (e.g. a pure "deadlines in the next 30 days"
        # query with no topic keywords) has no real relevance score to
        # judge strength from -- never call this "Strong" on the strength
        # of the 1.0 sentinel used internally for that case.
        return TIER_POSSIBLE
    if result.score >= STRONG_THRESHOLD:
        return TIER_STRONG
    if result.score >= POSSIBLE_THRESHOLD:
        return TIER_POSSIBLE
    return TIER_WEAK


def _template_summary(result: AISearchResult, tier: str) -> str:
    parts = []
    if result.hard_filters_applied:
        if "deadline_within_days" in result.hard_filters_applied:
            parts.append("its deadline falls within the window you asked about")
        if "funding_min" in result.hard_filters_applied or "funding_max" in result.hard_filters_applied:
            parts.append("its funding amount matches the range you asked about")
    if result.relevance_known and result.score > 0:
        parts.append(f"a text-relevance score of {result.score:.2f} against your search terms")
    if not parts:
        parts.append("it matched your search")
    reason = "; ".join(parts)
    # Checked directly off the real field, NOT off the tier string -- when a
    # tier_override is supplied (see explain_result), `tier` is a canonical
    # src/ranking/live_match.py tier LABEL (e.g. "Strong match"), which will
    # never literally equal TIER_ELIGIBILITY_UNCLEAR even when eligibility
    # genuinely is unknown for that grant.
    tier_note = " Eligibility details are not stated by the source -- verify before applying." if not result.eligibility_known else ""
    historical_note = _historical_note(result)
    return f"Matched because {reason}.{tier_note}{historical_note}"


def _historical_note(result: AISearchResult) -> str:
    """A record surfaced from GrantSetu's historical archive rather than a
    live source must say so in the explanation itself, not just a badge --
    interview.txt: current results must never be confused with historical
    ones. Empty string for a live result (nothing to caveat)."""
    if result.status_bucket == "historical":
        return " This is a historical GrantSetu record, not a confirmed currently-open opportunity -- verify its current status with the funder before applying."
    return ""


_AI_EXPLAIN_PROMPT = """You are explaining, in one or two short sentences, why a grant search result matched an NGO's query. Use ONLY the facts given below -- do not invent a funder, deadline, eligibility rule, or amount that isn't listed. If eligibility is not listed, say it is unclear rather than guessing. Do not claim certainty about whether the NGO qualifies. If STATUS BUCKET is "historical", you MUST explicitly say this is a historical/archived record and its current availability is not confirmed -- never imply it is currently open.

QUERY: {query}
GRANT TITLE: {title}
FUNDER: {funder}
CATEGORY: {category}
DEADLINE: {deadline}
STATUS: {status_label}
STATUS BUCKET: {status_bucket}
MATCH TIER (already decided, do not contradict it): {tier}

Respond with ONLY the one-or-two-sentence explanation, no preamble, no JSON."""


def explain_result(
    query_text: str,
    result: AISearchResult,
    provider: AIProvider | None,
    tier_override: str | None = None,
) -> GroundedExplanation:
    """`tier_override`, when given, is used INSTEAD of compute_match_tier --
    the caller (api/routers/opportunities.py::ai_search) passes the
    canonical src/ranking/live_match.py tier_label for a live opportunity
    the NGO already has a real profile-based score for, so the explanation
    text is generated against the SAME tier the score badge will show,
    never a locally-recomputed one that could quietly disagree with it."""
    tier = tier_override or compute_match_tier(result)

    if provider is not None:
        prompt = _AI_EXPLAIN_PROMPT.format(
            query=query_text,
            title=result.title,
            funder=result.funder or "not stated",
            category=result.category or "not stated",
            deadline=result.deadline.isoformat() if result.deadline else "not stated",
            status_label=result.status_label,
            status_bucket=result.status_bucket,
            tier=tier,
        )
        ai_result = provider.complete(prompt, max_tokens=150)
        if ai_result.ok and ai_result.text and ai_result.text.strip():
            return GroundedExplanation(
                tier=tier,
                summary=ai_result.text.strip(),
                method="ai",
                ai_disclaimer="This explanation was generated by AI from the grant's real listed fields -- verify against the source before relying on it.",
            )
        # AI configured but failed/empty -- fall through to the template,
        # same fallback discipline as every other AI-capable function here.

    return GroundedExplanation(tier=tier, summary=_template_summary(result, tier), method="rule_based")
