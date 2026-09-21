"""AI-first, rule-based-fallback research functions used by website analysis
and opportunity extraction. Every public function here follows the same
pattern: try the NGO's configured AI provider (if any) first, and ALWAYS
fall back to a deterministic, keyword/regex-based method if the provider is
unavailable, disabled, errors, rate-limits, or returns output that cannot be
parsed. The caller never has to know which path ran -- but the RESULT always
says which one did (`method: "ai" | "rule_based"`), so provenance is honest.

This module intentionally does NOT talk to the database or HTTP directly --
it receives already-fetched text and an optional AIProvider, so it is easy
to unit-test deterministically without a live network call.
"""
import json
import re
from dataclasses import dataclass, field

from src.ai.providers import AIProvider

# A deliberately small, common-sense vocabulary -- not exhaustive, not a
# trained classifier. Good enough for keyword-based signal without claiming
# to be an NLP model.
FOCUS_AREA_KEYWORDS = [
    "education", "health", "environment", "water", "sanitation", "livelihood",
    "agriculture", "disaster relief", "human rights", "child welfare",
    "women empowerment", "gender equality", "disability", "elderly care",
    "mental health", "nutrition", "housing", "skill development",
    "youth development", "animal welfare", "arts and culture",
    "poverty alleviation", "rural development", "climate change",
    "conservation", "wildlife", "renewable energy", "technology access",
]

BENEFICIARY_KEYWORDS = [
    "women", "children", "youth", "elderly", "senior citizens",
    "persons with disabilities", "farmers", "tribal communities",
    "refugees", "migrants", "students", "patients", "families",
    "adolescents", "girls", "widows", "unemployed", "slum communities",
    "indigenous communities", "orphans",
]

LOCATION_KEYWORDS = [
    "india", "andhra pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "mumbai", "bengaluru", "bangalore", "chennai",
    "kolkata", "hyderabad", "pune", "ahmedabad",
]

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "are",
    "was", "were", "has", "our", "their", "your", "you", "not", "all",
    "can", "will", "also", "more", "about", "into", "than", "such", "its",
    "who", "how", "when", "what", "which", "these", "those", "been",
}


def _keyword_hits(text_lower: str, vocabulary: list[str], limit: int) -> list[str]:
    hits = [term for term in vocabulary if term in text_lower]
    return hits[:limit]


def _top_keywords(text_lower: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[a-z]{4,}", text_lower)
    counts: dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [w for w, _ in ranked[:limit]]


def _extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    meaningful = [s for s in sentences if len(s.split()) >= 6][:max_sentences]
    return " ".join(meaningful) if meaningful else (text[:280].strip() + ("..." if len(text) > 280 else ""))


def _parse_json_block(text: str) -> dict | None:
    """AI responses often wrap JSON in prose or a ```json fence. Extract the
    first {...} block and parse it; return None (never raise) on failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class WebsiteAnalysisResult:
    method: str  # "ai" | "rule_based"
    summary: str
    focus_areas: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    beneficiaries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    error: str | None = None


def analyze_website_text(combined_text: str, provider: AIProvider | None) -> WebsiteAnalysisResult:
    """`combined_text` is the already-fetched, already-cleaned text from an
    NGO's website (About/Programs/etc pages, concatenated -- see
    src/discovery/website_analyzer.py). Never fetches anything itself."""
    if provider is not None:
        prompt = (
            "You are analyzing an NGO's website content to help build a funding-matching profile. "
            "Read the text below and respond with ONLY a JSON object with these keys: "
            '"summary" (2-3 sentence plain-language summary of the organization\'s mission), '
            '"focus_areas" (list of up to 6 short thematic focus areas, e.g. "education", "water sanitation"), '
            '"locations" (list of up to 6 places/regions the NGO appears to operate in), '
            '"beneficiaries" (list of up to 6 groups the NGO serves, e.g. "children", "farmers"). '
            "Base every field ONLY on the text given -- do not invent information not present in it.\n\n"
            f"WEBSITE TEXT:\n{combined_text[:6000]}"
        )
        result = provider.complete(prompt, max_tokens=500)
        if result.ok and result.text:
            parsed = _parse_json_block(result.text)
            if parsed:
                return WebsiteAnalysisResult(
                    method="ai",
                    summary=str(parsed.get("summary", "")).strip(),
                    focus_areas=[str(x) for x in parsed.get("focus_areas", [])][:6],
                    locations=[str(x) for x in parsed.get("locations", [])][:6],
                    beneficiaries=[str(x) for x in parsed.get("beneficiaries", [])][:6],
                    keywords=_top_keywords(combined_text.lower()),
                )
        # AI was configured but failed or returned unparseable output --
        # fall through to the rule-based path rather than erroring out.

    text_lower = combined_text.lower()
    return WebsiteAnalysisResult(
        method="rule_based",
        summary=_extractive_summary(combined_text),
        focus_areas=_keyword_hits(text_lower, FOCUS_AREA_KEYWORDS, 6),
        locations=_keyword_hits(text_lower, LOCATION_KEYWORDS, 6),
        beneficiaries=_keyword_hits(text_lower, BENEFICIARY_KEYWORDS, 6),
        keywords=_top_keywords(text_lower),
    )


@dataclass
class ExtractedOpportunityFields:
    method: str  # "ai" | "rule_based"
    title: str | None = None
    funder: str | None = None
    description: str | None = None
    eligibility_text: str | None = None
    category: str | None = None
    deadline: str | None = None  # ISO date string, best-effort
    url: str | None = None
    country: str = "unknown"  # "India" | "unknown" -- see _detect_country
    evidence_snippet: str | None = None
    extraction_status: str = "complete"  # complete | partial | failed | rejected


_MONTH_NAMES = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_ABBR = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec"

_DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
    rf"\b((?:{_MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{4}})\b",
    rf"\b(\d{{1,2}}\s+(?:{_MONTH_NAMES})\s*,?\s*\d{{4}})\b",
    # Abbreviated months (e.g. "13 Oct 2026", "Oct 13, 2026") -- a real gap
    # found live against ngobox.org's "Apply By: 13 Oct 2026" phrasing;
    # without this, a written-out-but-abbreviated deadline was silently
    # missed even though _has_grant_signal/_find_near_marker found the
    # surrounding "Apply By:" context fine.
    rf"\b(\d{{1,2}}\s+(?:{_MONTH_ABBR})\.?\s*,?\s*\d{{4}})\b",
    rf"\b((?:{_MONTH_ABBR})\.?\s+\d{{1,2}},?\s+\d{{4}})\b",
]

_MONTH_LOOKUP = {
    name.lower()[:3]: i + 1
    for i, name in enumerate(_MONTH_NAMES.split("|"))
}


def _normalize_date_to_iso(raw: str) -> str | None:
    """Converts any of the _DATE_PATTERNS match shapes to a real ISO
    (YYYY-MM-DD) string -- previously, a matched "Month Day, Year" (or any
    non-ISO, non-MM/DD/YYYY) string was returned as-is and silently failed
    to parse downstream (src/ingestion/sync_grants.py::_parse_deadline uses
    datetime.fromisoformat, which rejects "October 13, 2026" outright) --
    the deadline was found by the regex but then lost. Never raises --
    returns None (caller treats as "no deadline found") if it can't
    confidently normalize."""
    raw = raw.strip().rstrip(".")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        month, day, year = (int(x) for x in m.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"

    # "Month[.,] Day, Year" or "Day Month[.,] Year" (full or abbreviated)
    m = re.fullmatch(rf"({_MONTH_NAMES}|{_MONTH_ABBR})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", raw, re.IGNORECASE)
    if m:
        month_word, day, year = m.groups()
    else:
        m = re.fullmatch(rf"(\d{{1,2}})\s+({_MONTH_NAMES}|{_MONTH_ABBR})\.?,?\s*(\d{{4}})", raw, re.IGNORECASE)
        if not m:
            return None
        day, month_word, year = m.groups()

    month = _MONTH_LOOKUP.get(month_word.lower()[:3])
    if month is None:
        return None
    try:
        return f"{int(year):04d}-{month:02d}-{int(day):02d}"
    except ValueError:
        return None

_ELIGIBILITY_MARKERS = ["eligib", "who can apply", "applicants must", "criteria"]
_DEADLINE_MARKERS = ["deadline", "closes on", "closing date", "last date", "due date", "apply by"]
_FUNDER_MARKERS = [
    "organization:", "organisation:", "funded by", "funder:", "donor:",
    "sponsored by", "grant maker:", "grantmaker:", "offered by",
]
# Words that signal the funder name has ended and the page has moved on to
# a different field -- needed because extract_page() collapses ALL
# whitespace to single spaces (src/discovery/extract.py), so there is no
# double-space/newline left to detect a field boundary; a plain character
# window (the first version of this function) ran straight through into
# the next field's text (e.g. "ChangeX Apply By: 13 Oct 2026 Apply for the
# ServiceNow Comm..." instead of just "ChangeX" -- caught via a real live
# sync against ngobox.org, not a synthetic test).
_FUNDER_BOUNDARY_WORDS = {
    "apply", "deadline", "grant", "amount", "eligib", "close", "closing",
    "last", "due", "about", "the", "organization", "organisation",
}
_FUNDER_MAX_WORDS = 6

# A page must show at least one of these signals to be treated as
# describing a funding/grant opportunity at all. This is deliberately a
# LOW bar (any one hit is enough) rather than a strict classifier -- the
# goal is only to reject pages with NO grant-like vocabulary whatsoever
# (a Wikipedia article, a news story, a blog post, a generic "About us"
# page), not to second-guess borderline cases. See
# extract_opportunity_fields_rule_based's "rejected" extraction_status.
_GRANT_SIGNAL_MARKERS = [
    "grant", "funding", "fund for", "award of", "scholarship", "fellowship",
    "call for proposals", "request for proposals", "rfp", "apply now",
    "application deadline", "eligible applicants", "who can apply",
    "funding opportunity", "financial assistance", "sponsorship",
]


_INDIA_DOMAIN_SUFFIXES = (".in", ".gov.in", ".org.in", ".nic.in", ".co.in")


def _detect_country(text_lower: str, page_url: str) -> str:
    """Never fabricates a country -- "India" only when the page text names
    India or an Indian state/city (LOCATION_KEYWORDS, already used for NGO
    website analysis above), or the domain itself is under an Indian TLD.
    Otherwise "unknown": there is no reliable, evidence-based way to guess a
    country for a generic funding page that never mentions one, and
    guessing (e.g. defaulting to the connector operator's own country) would
    be exactly the kind of invented field this project has committed not to
    produce."""
    from urllib.parse import urlparse

    hostname = (urlparse(page_url).hostname or "").lower()
    if any(hostname.endswith(suffix) for suffix in _INDIA_DOMAIN_SUFFIXES):
        return "India"
    if any(term in text_lower for term in LOCATION_KEYWORDS):
        return "India"
    return "unknown"


def _has_grant_signal(text_lower: str) -> bool:
    # A page that clearly states eligibility criteria or a deadline is
    # opportunity-like even if it never uses the literal word "grant" (e.g.
    # some fellowship/scholarship pages) -- so those marker sets count as
    # grant signal too, not just _GRANT_SIGNAL_MARKERS.
    all_markers = _GRANT_SIGNAL_MARKERS + _ELIGIBILITY_MARKERS + _DEADLINE_MARKERS
    return any(marker in text_lower for marker in all_markers)


def _find_near_marker(text: str, markers: list[str], window: int = 220) -> str | None:
    lower = text.lower()
    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(text), idx + window)
            return text[start:end].strip()
    return None


def _extract_funder(text: str) -> str | None:
    """A short, targeted extraction (unlike _find_near_marker's wide context
    window, meant for a name, not a paragraph) -- looks for a marker like
    "Organization:"/"Funded by" and takes the first few words after it, up
    to _FUNDER_MAX_WORDS or the first word that signals a new field has
    started (see _FUNDER_BOUNDARY_WORDS). Real gap found live: the
    rule-based path never populated `funder` at all before this -- only the
    AI-assisted path (extract_opportunity_fields, below) did, via the LLM's
    own judgment -- so a real funder name (e.g. ngobox.org's "Organization:
    ChangeX") was silently dropped for every web-discovered/listing-page
    grant when no AI provider is configured, despite being right there in
    the text."""
    lower = text.lower()
    for marker in _FUNDER_MARKERS:
        idx = lower.find(marker)
        if idx == -1:
            continue
        remainder = text[idx + len(marker):].strip(" :-–—")
        words = remainder.split()
        name_words = []
        for word in words[:_FUNDER_MAX_WORDS]:
            bare = word.strip(".,:;()").lower()
            if bare in _FUNDER_BOUNDARY_WORDS or bare.isdigit():
                break
            name_words.append(word)
        candidate = " ".join(name_words).strip(" .,:;-–—")
        if candidate:
            return candidate
    return None


def extract_opportunity_fields_rule_based(page_text: str, page_title: str | None, page_url: str) -> ExtractedOpportunityFields:
    text_lower = page_text.lower()

    if not _has_grant_signal(text_lower):
        # No grant/funding-related vocabulary anywhere on the page -- this
        # is the "reject non-grant pages" requirement: a Wikipedia article,
        # news story, or blog post must never be presented to a user as a
        # discovered funding opportunity just because it was fetched. The
        # caller (WebDiscoveryConnector) drops "rejected" records entirely
        # rather than persisting them.
        return ExtractedOpportunityFields(
            method="rule_based",
            title=(page_title or "").strip() or None,
            url=page_url,
            extraction_status="rejected",
        )

    deadline = None
    deadline_context = _find_near_marker(page_text, _DEADLINE_MARKERS)
    search_scope = deadline_context or page_text
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, search_scope)
        if m:
            deadline = _normalize_date_to_iso(m.group(1))
            if deadline:
                break

    eligibility_snippet = _find_near_marker(page_text, _ELIGIBILITY_MARKERS)
    funder = _extract_funder(page_text)

    category_hits = _keyword_hits(text_lower, FOCUS_AREA_KEYWORDS, 1)
    category = category_hits[0] if category_hits else None

    status = "complete" if (deadline or eligibility_snippet) else "partial"

    # Country detection deliberately scans only the parts of the page that
    # are actually ABOUT this specific opportunity (title + the
    # eligibility/deadline context windows already extracted above), not
    # the full page_text -- extract_page() concatenates a listing/aggregator
    # site's nav/footer/sidebar/other-article-teasers into page_text too, and
    # a real bug found live against ngobox.org showed a page for a
    # U.S.-based funder getting tagged "India" purely because NGOBox's OWN
    # site chrome (its India CSR Summit ad, its Gujarat office address in
    # the footer) mentions India/Gujarat/Mumbai -- content that has nothing
    # to do with the specific grant on that page.
    country_scope = " ".join(filter(None, [page_title, eligibility_snippet, deadline_context])).lower()
    country = _detect_country(country_scope, page_url)

    return ExtractedOpportunityFields(
        method="rule_based",
        title=(page_title or "").strip() or None,
        funder=funder,
        description=_extractive_summary(page_text, max_sentences=2),
        eligibility_text=eligibility_snippet,
        category=category,
        deadline=deadline,
        url=page_url,
        country=country,
        evidence_snippet=deadline_context or eligibility_snippet,
        extraction_status=status,
    )


def extract_opportunity_fields(
    page_text: str, page_title: str | None, page_url: str, provider: AIProvider | None
) -> ExtractedOpportunityFields:
    if provider is not None:
        prompt = (
            "You are extracting structured funding-opportunity data from a web page. "
            "Read the text below and respond with ONLY a JSON object with these keys: "
            '"title", "funder" (the organization offering the funding, if identifiable), '
            '"description" (1-2 sentence summary), "eligibility_text" (who can apply, if stated), '
            '"category" (one short theme), "deadline" (an ISO date YYYY-MM-DD if a specific date is '
            'stated, otherwise null), "evidence_snippet" (a short quote from the text supporting the '
            "deadline or eligibility you extracted, or null). "
            "If the page does NOT appear to describe a funding/grant opportunity at all, respond with "
            '{"not_an_opportunity": true} and nothing else. Never invent a deadline, funder, or '
            "eligibility criterion that is not actually present in the text.\n\n"
            f"PAGE TITLE: {page_title or 'unknown'}\nPAGE URL: {page_url}\n\nPAGE TEXT:\n{page_text[:6000]}"
        )
        result = provider.complete(prompt, max_tokens=500)
        if result.ok and result.text:
            parsed = _parse_json_block(result.text)
            if parsed and not parsed.get("not_an_opportunity"):
                return ExtractedOpportunityFields(
                    method="ai",
                    title=parsed.get("title") or page_title,
                    funder=parsed.get("funder"),
                    description=parsed.get("description"),
                    eligibility_text=parsed.get("eligibility_text"),
                    category=parsed.get("category"),
                    deadline=parsed.get("deadline"),
                    url=page_url,
                    # Deliberately NOT taken from the AI response -- country
                    # is always resolved deterministically (_detect_country),
                    # even on the AI path, so this field can never be an AI
                    # guess. Scoped to the AI's own extracted title/
                    # eligibility_text/description (already specific to
                    # THIS opportunity) rather than the raw page_text, for
                    # the same reason as the rule-based path: a listing/
                    # aggregator page's full text includes site-wide nav/
                    # footer/sidebar content that can mention a country with
                    # nothing to do with the specific opportunity on the page.
                    country=_detect_country(
                        " ".join(filter(None, [
                            parsed.get("title"), parsed.get("eligibility_text"), parsed.get("description"),
                        ])).lower(),
                        page_url,
                    ),
                    evidence_snippet=parsed.get("evidence_snippet"),
                    extraction_status="complete",
                )
            if parsed and parsed.get("not_an_opportunity"):
                # Same vocabulary as the rule-based rejection path below --
                # "rejected" means "confirmed not a grant page," a decision
                # the caller acts on by dropping the record, distinct from
                # "failed" (couldn't tell either way).
                return ExtractedOpportunityFields(method="ai", url=page_url, extraction_status="rejected")
        # fall through to rule-based on any AI failure/unparseable output

    return extract_opportunity_fields_rule_based(page_text, page_title, page_url)


def explain_recommendation_text(
    ngo_summary: str, opportunity_summary: str, rule_based_reasons: list[str], provider: AIProvider | None
) -> str:
    """Turns the EXISTING rule-based score reasons (src/ranking/explain.py)
    into a more fluent sentence, WITHOUT inventing new reasons -- the
    underlying score is still the transparent weighted formula; AI is only
    ever used to rephrase, never to justify the ranking itself (same
    boundary the project blueprint set for LLM use from the very first MVP)."""
    if provider is not None:
        prompt = (
            "Rewrite the following bullet-point match reasons as one fluent, natural sentence "
            "for an NGO fundraising lead. Do not add any reason that is not already listed. "
            "Do not mention probability of winning funding.\n\n"
            f"NGO: {ngo_summary}\nOpportunity: {opportunity_summary}\n"
            f"Reasons: {'; '.join(rule_based_reasons)}"
        )
        result = provider.complete(prompt, max_tokens=150)
        if result.ok and result.text:
            return result.text.strip()

    return "Matched on: " + "; ".join(rule_based_reasons[:3])
