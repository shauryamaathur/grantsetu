"""Deterministic concept/synonym normalization used to make keyword and
TF-IDF matching sensitive to PARTIAL, SEMANTICALLY-RELATED overlap instead
of requiring exact string equality (see interview.txt's "matching too
conservative" fix request).

No embedding model is loaded here -- the project has no embeddings/vector
index anywhere in the pipeline (checked before writing this), and pulling
one in only for this would mean a non-deterministic, hard-to-audit
dependency for what a small curated synonym map already solves cleanly at
this data scale (a few hundred live grants, short profile phrases). Every
match this module produces is traceable to an explicit rule, which is what
lets `_match_reasons`/explain.py show a real "why" rather than an opaque
similarity number.

Two things live here:

1. CONCEPT_SYNONYMS -- canonical concept id -> phrases that mean roughly
   the same thing (e.g. "women empowerment" ~ "women's economic
   inclusion"). Both `_concept_tags_for_text` (used to enrich TF-IDF
   documents with shared concept tokens, the closest deterministic
   equivalent to a semantic-similarity boost) and `term_match` (used for
   the explicit focus-area/beneficiary overlap component) key off this.

2. GENERIC_TERMS -- single, low-signal words ("community", "development",
   "people", ...) that must NOT drive a match by themselves, so an NGO
   profile stuffed with generic words doesn't get an inflated score.
"""
import re

# canonical concept id -> phrases that should all be treated as (roughly)
# the same underlying concept. Keep phrases lowercase; matching normalizes
# input the same way. Order/spelling here is illustrative, not exhaustive --
# extend freely, this is a plain data table, not an algorithm.
CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "women_empowerment": [
        "women empowerment", "women's empowerment", "women empowered",
        "women's economic inclusion", "women's economic empowerment",
        "female empowerment", "gender equality", "gender equity",
        "women led", "women-led", "girls empowerment",
    ],
    "rural_development": [
        "rural development", "rural communities", "rural community",
        "rural livelihoods", "rural areas", "rural population",
    ],
    "livelihoods": [
        "livelihoods", "livelihood support", "income generation",
        "income generating", "sustainable livelihoods", "economic empowerment",
        "employment generation",
    ],
    "climate_resilience": [
        "climate resilience", "climate adaptation", "climate change adaptation",
        "climate change mitigation", "climate risk", "resilience building",
    ],
    "biodiversity_conservation": [
        "biodiversity", "biodiversity conservation", "ecosystem conservation",
        "ecosystem-based adaptation", "ecosystem based adaptation",
        "conservation", "natural resource management", "environmental conservation",
    ],
    "healthcare": [
        "healthcare", "health care", "public health", "health services",
        "health access", "health outcomes", "primary health",
    ],
    "skill_development": [
        "skill development", "skills development", "vocational training",
        "skills training", "capacity building", "technical training",
    ],
    "education": [
        "education", "literacy", "schooling", "quality education", "learning outcomes",
    ],
    "vulnerable_communities": [
        "vulnerable communities", "vulnerable populations", "marginalized communities",
        "marginalised communities", "underserved communities", "disadvantaged communities",
        "at-risk populations", "at risk populations", "low-income communities",
        "low income communities",
    ],
    "water_sanitation": [
        "water and sanitation", "wash", "clean water", "sanitation", "water access",
    ],
    "child_welfare": [
        "child welfare", "child protection", "children", "child rights", "child development",
    ],
    "disaster_management": [
        "disaster management", "disaster risk reduction", "humanitarian relief",
        "emergency response", "disaster preparedness",
    ],
    "agriculture": [
        "agriculture", "farming", "agri-business", "agribusiness", "farmers", "smallholder farmers",
    ],
    "gender": [
        "gender", "women", "girls", "women and girls",
    ],
    "youth": [
        "youth", "youth development", "adolescents", "young people",
    ],
    "financial_inclusion": [
        "financial inclusion", "microfinance", "access to finance", "financial literacy",
    ],
    "disability_inclusion": [
        "disability", "persons with disabilities", "disability inclusion", "differently abled",
    ],
    "indigenous_communities": [
        "indigenous communities", "tribal communities", "adivasi", "indigenous peoples",
    ],
}

# Low-signal single words that must not, by themselves, drive a match --
# keeps a keyword-stuffed profile (lots of generic words) from outscoring a
# profile with a few precise, specific terms (interview.txt: "avoid keyword
# stuffing"). Still counted, but at heavily reduced weight (see term_weight).
GENERIC_TERMS = {
    "community", "communities", "development", "people", "support", "project",
    "projects", "program", "programs", "programme", "programmes", "initiative",
    "initiatives", "organization", "organisation", "ngo", "work", "service",
    "services", "activities", "activity", "area", "areas", "sector", "group",
    "groups", "welfare", "empowerment", "based", "general", "society",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, strip possessives/punctuation, collapse whitespace --
    the shared normalization step every matcher below relies on, so
    "women's" and "womens" and "women" all compare equal."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("'s", "").replace("’s", "")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text))


def term_weight(term: str) -> float:
    """How much discriminative weight a single NGO profile term deserves.
    A specific multi-word phrase (e.g. "rural livelihoods") gets full
    weight; a single generic word (e.g. "development") gets a small
    fraction so it can nudge but never carry a match on its own."""
    norm = normalize(term)
    words = norm.split()
    if len(words) == 1 and words[0] in GENERIC_TERMS:
        return 0.25
    return 1.0


def concepts_for_term(term: str) -> set[str]:
    """Canonical concept id(s) a profile term belongs to, if any."""
    norm = normalize(term)
    if not norm:
        return set()
    hits = set()
    for concept_id, phrases in CONCEPT_SYNONYMS.items():
        for phrase in phrases:
            if phrase in norm or norm in phrase:
                hits.add(concept_id)
                break
    return hits


def concept_tags_for_text(text: str) -> set[str]:
    """All concept ids whose synonym phrases appear anywhere in `text`.
    Used to (a) tag TF-IDF documents with shared concept tokens so
    semantically-related phrasing raises cosine similarity, and (b) match
    an NGO term against a grant via concept rather than literal substring."""
    norm = normalize(text)
    if not norm:
        return set()
    tags = set()
    for concept_id, phrases in CONCEPT_SYNONYMS.items():
        for phrase in phrases:
            if phrase in norm:
                tags.add(concept_id)
                break
    return tags


def concept_tag_tokens(text: str) -> str:
    """Space-joined pseudo-tokens (one per matched concept) to append to a
    document before TF-IDF fitting -- e.g. a grant mentioning "ecosystem
    conservation" and a profile mentioning "biodiversity" both gain the
    token "__concept_biodiversity_conservation__", so they share vocabulary
    a plain TF-IDF over the raw text would miss entirely."""
    return " ".join(f"__concept_{c}__" for c in sorted(concept_tags_for_text(text)))


def term_match(term: str, text_norm: str, text_concepts: set[str]) -> tuple[bool, float, str]:
    """Match one NGO profile term against already-normalized grant text
    (+ its precomputed concept tags). Returns (matched, match_strength,
    match_kind) where match_strength in (0, 1] feeds the weighted overlap
    curve and match_kind is one of "exact"/"concept"/"partial" for the
    explanation text.

    Strength is intentionally tiered rather than binary: an exact phrase is
    the strongest, honest signal; a shared concept (different wording, same
    idea) is still strong; a partial word overlap on a non-generic word is
    the weakest signal that still deserves SOME credit rather than none.
    """
    norm = normalize(term)
    if not norm:
        return False, 0.0, ""

    if norm in text_norm:
        return True, 1.0, "exact"

    term_concepts = concepts_for_term(norm)
    if term_concepts and (term_concepts & text_concepts):
        return True, 0.85, "concept"

    term_words = _words(norm) - GENERIC_TERMS
    if term_words:
        text_words = _words(text_norm)
        if term_words & text_words:
            return True, 0.5, "partial"

    return False, 0.0, ""
