"""Turns a free-text natural-language grant search query into a structured
`SearchIntent` -- powers POST /opportunities/ai-search
(src/ranking/ai_search.py). Follows the exact AI-first/rule-based-fallback
pattern already used throughout src/ai/research.py: try the NGO's configured
provider first, ALWAYS fall back to a deterministic keyword/regex extractor
if no provider is configured or the call fails/returns unparseable output,
and always report which path ran (`method`) so the caller can be honest
about it. This module does not talk to the database -- callers resolve the
AIProvider (via src/ai/factory.py::get_provider_for_ngo) and pass it in.
"""
import re
from dataclasses import dataclass, field

from src.ai.providers import AIProvider
from src.ai.research import (
    BENEFICIARY_KEYWORDS,
    FOCUS_AREA_KEYWORDS,
    LOCATION_KEYWORDS,
    _parse_json_block,
)

# LOCATION_KEYWORDS (src/ai/research.py) is India-focused (states/cities) --
# the example queries this feature must support ("USA grants", "available
# globally") need a few extra, deliberately small markers not tied to India.
_INTERNATIONAL_MARKERS = ["global", "globally", "worldwide", "international"]
_COUNTRY_MARKERS = {
    "usa": ["usa", "u.s.a", "united states", "u.s."],
    "uk": ["uk", "united kingdom", "u.k."],
}

_DEADLINE_WINDOW_PATTERNS = [
    (re.compile(r"next\s+(\d+)\s+days?"), lambda m: int(m.group(1))),
    # "closing/deadline/due within/in N days" -- the phrasing a real query
    # (e.g. "grants closing within 60 days") actually uses, distinct from
    # "next N days" above.
    (re.compile(r"(?:closing|deadline|due)?\s*(?:within|in)\s+(\d+)\s+days?"), lambda m: int(m.group(1))),
    (re.compile(r"this\s+week"), lambda m: 7),
    (re.compile(r"this\s+month"), lambda m: 30),
    (re.compile(r"next\s+week"), lambda m: 14),
    (re.compile(r"next\s+month"), lambda m: 60),
]

_APPLICANT_TYPE_MARKERS = {
    "nonprofit": ["ngo", "ngos", "nonprofit", "non-profit", "non profit", "charity", "charities"],
    "institution": ["school", "schools", "university", "universities", "institution", "institutions"],
    "individual": ["individual", "individuals", "researcher", "researchers"],
}

# search_scope classification -- decides whether ai_search.py should treat
# the historical corpus as the primary source (HISTORICAL), deprioritize it
# heavily in favor of live/current opportunities (CURRENT, the default), or
# treat both symmetrically for funder/landscape research (RESEARCH). See
# src/ranking/ai_search.py's module docstring for how each scope changes
# ranking. Checked in this order -- HISTORICAL phrasing is the most
# specific and must win over a RESEARCH marker if a query somehow contains
# both (e.g. "who has historically funded climate NGOs").
SCOPE_CURRENT = "current"
SCOPE_HISTORICAL = "historical"
SCOPE_RESEARCH = "research"
_VALID_SCOPES = {SCOPE_CURRENT, SCOPE_HISTORICAL, SCOPE_RESEARCH}

_HISTORICAL_MARKERS = [
    "historical", "historically", "previous", "past grants", "in the past",
    "prior grants", "earlier grants", "old grants", "did fund", "did receive",
    "has funded", "has received", "used to fund", "what grants did",
    "grants received", "funding history", "previously funded", "previously received",
]
_HISTORICAL_WORD_PATTERN = re.compile(r"\b(past|prior)\b")

_RESEARCH_MARKERS = [
    "who funds", "who supports", "who is funding", "who are the funders",
    "which foundations fund", "what foundations fund", "which funders",
    "what funders", "funder research", "funding landscape", "which organizations fund",
    "what organizations fund", "who funded",
]


def _classify_search_scope(text_lower: str) -> str:
    if any(m in text_lower for m in _HISTORICAL_MARKERS) or _HISTORICAL_WORD_PATTERN.search(text_lower):
        return SCOPE_HISTORICAL
    if any(m in text_lower for m in _RESEARCH_MARKERS):
        return SCOPE_RESEARCH
    return SCOPE_CURRENT


@dataclass
class SearchIntent:
    method: str  # "ai" | "rule_based"
    keywords: list[str] = field(default_factory=list)  # free-text terms for the retrieval query
    locations: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    applicant_type: str | None = None
    deadline_within_days: int | None = None
    international_ok: bool | None = None  # None = unknown/not asked about, never assumed
    funding_min: float | None = None
    funding_max: float | None = None
    # "current" (default) | "historical" | "research" -- see
    # _classify_search_scope above and src/ranking/ai_search.py.
    search_scope: str = SCOPE_CURRENT


def _extract_intent_rule_based(query_text: str) -> SearchIntent:
    text_lower = query_text.lower()
    # FOCUS_AREA_KEYWORDS/BENEFICIARY_KEYWORDS/LOCATION_KEYWORDS use spaces
    # ("climate change", "women empowerment"), but a natural query very
    # commonly hyphenates the same phrase ("climate-change grants",
    # "women-empowerment grants") -- match against a hyphen-normalized copy
    # so both spellings hit the same vocabulary term.
    text_normalized = text_lower.replace("-", " ")

    causes = [term for term in FOCUS_AREA_KEYWORDS if term in text_normalized]
    causes += [term for term in BENEFICIARY_KEYWORDS if term in text_normalized and term not in causes]

    locations = [term for term in LOCATION_KEYWORDS if term in text_normalized]
    for country, markers in _COUNTRY_MARKERS.items():
        if any(m in text_lower for m in markers):
            locations.append(country)

    international_ok = True if any(m in text_lower for m in _INTERNATIONAL_MARKERS) else None
    if "international applicants" in text_lower or "allow international" in text_lower:
        international_ok = True

    deadline_within_days = None
    for pattern, resolver in _DEADLINE_WINDOW_PATTERNS:
        m = pattern.search(text_lower)
        if m:
            deadline_within_days = resolver(m)
            break

    applicant_type = None
    for label, markers in _APPLICANT_TYPE_MARKERS.items():
        if any(m in text_lower for m in markers):
            applicant_type = label
            break

    # Keywords: the causes/locations already found, plus any remaining
    # meaningful words -- used as the free-text TF-IDF retrieval query so a
    # query with no recognized vocabulary still searches on something
    # rather than returning nothing.
    stopwords = {
        "find", "show", "grants", "grant", "for", "ngos", "ngo", "open", "to",
        "in", "with", "deadlines", "the", "next", "days", "that", "allow",
        "applicants", "available", "and", "a", "an", "of", "on",
    }
    words = re.findall(r"[a-z][a-z\-]{2,}", text_lower)
    leftover = [w for w in words if w not in stopwords]
    keywords = list(dict.fromkeys(causes + locations + leftover))  # dedupe, keep order

    return SearchIntent(
        method="rule_based",
        keywords=keywords,
        locations=locations,
        causes=causes,
        applicant_type=applicant_type,
        deadline_within_days=deadline_within_days,
        international_ok=international_ok,
        search_scope=_classify_search_scope(text_lower),
    )


_AI_PROMPT_TEMPLATE = """You are extracting structured search filters from an NGO's natural-language grant search query. Read the query and respond with ONLY a JSON object with these keys:
"keywords" (list of up to 8 short search terms capturing the core topic),
"locations" (list of countries/states/regions mentioned or implied, empty list if none),
"causes" (list of thematic focus areas mentioned, e.g. "education", "climate change"),
"applicant_type" ("nonprofit", "institution", "individual", or null),
"deadline_within_days" (integer number of days if a deadline window is mentioned, e.g. "next 30 days" -> 30, otherwise null),
"international_ok" (true if the query asks about international/global eligibility, false if it asks for local-only, null if not mentioned),
"funding_min" (number or null), "funding_max" (number or null),
"search_scope": one of "current", "historical", "research" --
  "current" is the default for any query looking for funding to apply to now (e.g. "find education grants", "open healthcare grants in India", "grants closing this month"),
  "historical" is ONLY for queries explicitly about past/prior funding records (e.g. "what grants did X receive", "previous grants for education", "historical funding for India"),
  "research" is ONLY for queries asking who funds a cause rather than asking for opportunities to apply to (e.g. "who funds climate NGOs", "what foundations fund women's health").
Base every field ONLY on the query text -- do not invent locations, causes, or amounts not implied by it.

QUERY: {query}"""


def extract_search_intent(query_text: str, provider: AIProvider | None) -> SearchIntent:
    """AI-first, rule-based-fallback -- see module docstring. Never raises;
    a provider error/unparseable response falls through to the rule-based
    path rather than failing the search."""
    if provider is not None:
        result = provider.complete(_AI_PROMPT_TEMPLATE.format(query=query_text), max_tokens=300)
        if result.ok and result.text:
            parsed = _parse_json_block(result.text)
            if parsed:
                # The AI's search_scope is trusted only when it's one of the
                # three known values -- an invalid/missing value falls back
                # to the same deterministic classifier the rule-based path
                # uses, rather than letting a malformed AI response silently
                # default to "current" and mis-scope a historical/research
                # query.
                ai_scope = parsed.get("search_scope")
                scope = ai_scope if ai_scope in _VALID_SCOPES else _classify_search_scope(query_text.lower())
                return SearchIntent(
                    method="ai",
                    keywords=[str(x) for x in parsed.get("keywords", [])][:8],
                    locations=[str(x) for x in parsed.get("locations", [])][:6],
                    causes=[str(x) for x in parsed.get("causes", [])][:6],
                    applicant_type=parsed.get("applicant_type"),
                    deadline_within_days=parsed.get("deadline_within_days"),
                    international_ok=parsed.get("international_ok"),
                    funding_min=parsed.get("funding_min"),
                    funding_max=parsed.get("funding_max"),
                    search_scope=scope,
                )
        # AI configured but failed or returned unparseable output -- fall
        # through to rule-based rather than erroring the whole search.

    return _extract_intent_rule_based(query_text)
