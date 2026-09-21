"""Tests for src/ai/: encryption, the provider factory's fallback behavior,
and the AI-first/rule-based-fallback research functions. No real network
calls -- a fake AIProvider stub stands in for a real one so these tests are
deterministic and don't require API credentials."""
from src.ai.crypto import decrypt_api_key, encrypt_api_key
from src.ai.providers import AIResult
from src.ai.research import (
    _detect_country,
    _extract_funder,
    _normalize_date_to_iso,
    analyze_website_text,
    extract_opportunity_fields,
    extract_opportunity_fields_rule_based,
)


class _StubProvider:
    """A fake AIProvider for tests -- returns whatever the test configures,
    without ever making a network call."""

    def __init__(self, result: AIResult):
        self._result = result

    def complete(self, prompt, max_tokens=700):
        return self._result

    def test_connection(self):
        return self._result


def test_encrypt_decrypt_round_trip():
    ciphertext = encrypt_api_key("sk-test-12345")
    assert ciphertext != "sk-test-12345"
    assert decrypt_api_key(ciphertext) == "sk-test-12345"


def test_decrypt_garbage_returns_none_not_raise():
    assert decrypt_api_key("not-a-real-token") is None


def test_analyze_website_text_rule_based_when_no_provider():
    text = (
        "We are an environmental organization working on water sanitation for rural "
        "communities in Maharashtra. Our beneficiaries include women and children."
    )
    result = analyze_website_text(text, provider=None)
    assert result.method == "rule_based"
    assert "environment" in result.focus_areas
    assert "water" in result.focus_areas or "sanitation" in result.focus_areas
    assert "women" in result.beneficiaries
    assert "children" in result.beneficiaries
    assert "maharashtra" in result.locations


def test_analyze_website_text_uses_ai_when_provider_returns_valid_json():
    ai_text = (
        '{"summary": "An education NGO.", "focus_areas": ["education"], '
        '"locations": ["Delhi"], "beneficiaries": ["students"]}'
    )
    provider = _StubProvider(AIResult(ok=True, text=ai_text))
    result = analyze_website_text("some raw website text", provider=provider)
    assert result.method == "ai"
    assert result.summary == "An education NGO."
    assert result.focus_areas == ["education"]


def test_analyze_website_text_falls_back_when_ai_fails():
    provider = _StubProvider(AIResult(ok=False, error="rate limited"))
    text = "health clinic serving elderly patients in Kerala"
    result = analyze_website_text(text, provider=provider)
    assert result.method == "rule_based"  # fell back despite a provider being configured
    assert "health" in result.focus_areas


def test_analyze_website_text_falls_back_when_ai_returns_unparseable_text():
    provider = _StubProvider(AIResult(ok=True, text="I'm not going to give you JSON, sorry."))
    text = "youth skill development program in Punjab"
    result = analyze_website_text(text, provider=provider)
    assert result.method == "rule_based"


def test_extract_opportunity_fields_rule_based_finds_deadline_and_eligibility():
    page_text = (
        "Community Health Grant. Eligibility: nonprofit organizations registered in India may apply. "
        "The application deadline is 2027-06-30. Apply through our online portal."
    )
    result = extract_opportunity_fields_rule_based(page_text, "Community Health Grant", "https://example.org/grant")
    assert result.method == "rule_based"
    assert result.deadline == "2027-06-30"
    assert result.eligibility_text is not None
    assert "nonprofit" in result.eligibility_text.lower()
    assert result.extraction_status == "complete"


def test_extract_opportunity_fields_rule_based_finds_funder():
    """Regression test: the rule-based path never populated `funder` at all
    before this -- found via a real live sync against ngobox.org, not a
    synthetic test (see interview.txt)."""
    page_text = "Community Solar Fund. Organization: ChangeX Apply By: 13 Oct 2026 Eligibility criteria apply."
    result = extract_opportunity_fields_rule_based(page_text, "Community Solar Fund", "https://example.org/grant")
    assert result.funder == "ChangeX"


def test_extract_opportunity_fields_rule_based_handles_abbreviated_month_deadline():
    """Regression test: 'Apply By: 13 Oct 2026' (abbreviated month) used to
    be found by the old deadline regex but then silently fail to normalize
    into an ISO string usable downstream (sync_grants.py::_parse_deadline
    uses datetime.fromisoformat, which rejects non-ISO strings outright) --
    also found via the real ngobox.org sync."""
    page_text = "Grant. Eligibility: nonprofits. Apply By: 13 Oct 2026."
    result = extract_opportunity_fields_rule_based(page_text, "Grant", "https://example.org/grant")
    assert result.deadline == "2026-10-13"


def test_normalize_date_to_iso_handles_multiple_formats():
    assert _normalize_date_to_iso("2026-10-13") == "2026-10-13"
    assert _normalize_date_to_iso("10/13/2026") == "2026-10-13"
    assert _normalize_date_to_iso("October 13, 2026") == "2026-10-13"
    assert _normalize_date_to_iso("13 October, 2026") == "2026-10-13"
    assert _normalize_date_to_iso("Oct 13, 2026") == "2026-10-13"
    assert _normalize_date_to_iso("13 Oct 2026") == "2026-10-13"
    assert _normalize_date_to_iso("not a date") is None


def test_extract_funder_finds_marker_and_stops_at_next_field():
    assert _extract_funder("Organization: ChangeX Apply By: 13 Oct 2026") == "ChangeX"
    assert _extract_funder("Donor: Ford Foundation Grant Amount: 50000") == "Ford Foundation"
    assert _extract_funder("no marker present here") is None


def test_detect_country_finds_india_from_text_mention():
    assert _detect_country("this grant supports ngos in maharashtra", "https://example.org/grant") == "India"


def test_detect_country_finds_india_from_in_domain():
    assert _detect_country("no location mentioned at all", "https://somefoundation.org.in/grants") == "India"


def test_detect_country_defaults_to_unknown_without_evidence():
    """Never guesses -- a page with no India signal (text or domain) must
    report "unknown", not fabricate a country like "United States" just
    because that's a common default."""
    assert _detect_country("a generic funding page with no location", "https://example.org/grant") == "unknown"


def test_extract_opportunity_fields_rule_based_country_ignores_site_chrome():
    """Real bug found live against ngobox.org: a page for a U.S.-based
    funder was tagged country="India" purely because the SITE'S OWN nav/
    footer text ("India CSR Summit", a Gujarat office address) mentions
    India -- content unrelated to the specific opportunity on the page.
    Country must only be detected from the opportunity's own title/
    eligibility/deadline context, not the whole page dump."""
    filler = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore. "
    page_text = (
        "India CSR Summit | Advertise | Contact us: A-404, Shela, Gujarat 380058\n"
        + filler * 3
        + "Water Research Foundation Grant\n"
        "Eligibility: US-based nonprofit organizations only.\n"
        "Grant application deadline: 2027-10-01.\n"
        + filler * 3
        + "Related: Mumbai eye-care camp raises funds."
    )
    result = extract_opportunity_fields_rule_based(page_text, "Water Research Foundation Grant", "https://ngobox.org/wrf-grant")
    assert result.country == "unknown"


def test_extract_opportunity_fields_rule_based_rejects_pages_with_no_grant_signal():
    """A page with no grant/funding vocabulary at all (a Wikipedia article,
    news story, blog post, etc) must be classified "rejected", not
    persisted as a partially-extracted opportunity -- see
    WebDiscoveryConnector.fetch_records, which drops "rejected" records
    entirely rather than importing them."""
    result = extract_opportunity_fields_rule_based("This page has no relevant information at all.", "Untitled", "https://example.org/x")
    assert result.extraction_status == "rejected"
    assert result.deadline is None


def test_extract_opportunity_fields_rule_based_partial_when_signal_present_but_incomplete():
    """A page WITH grant vocabulary but no discoverable deadline/eligibility
    text is "partial" (still a real, importable opportunity, just with
    incomplete fields) -- distinct from "rejected" (not a grant page at all)."""
    result = extract_opportunity_fields_rule_based(
        "This foundation offers grant funding to nonprofit organizations each year.",
        "Foundation Grants", "https://example.org/grants",
    )
    assert result.extraction_status == "partial"
    assert result.deadline is None


def test_extract_opportunity_fields_uses_ai_when_valid():
    ai_text = (
        '{"title": "Youth Grant", "funder": "Example Fund", "description": "desc", '
        '"eligibility_text": "NGOs", "category": "youth", "deadline": "2027-01-01", '
        '"evidence_snippet": "deadline is 2027-01-01"}'
    )
    provider = _StubProvider(AIResult(ok=True, text=ai_text))
    result = extract_opportunity_fields("raw page text", "Youth Grant", "https://example.org", provider)
    assert result.method == "ai"
    assert result.deadline == "2027-01-01"
    assert result.funder == "Example Fund"


def test_extract_opportunity_fields_ai_says_not_an_opportunity():
    provider = _StubProvider(AIResult(ok=True, text='{"not_an_opportunity": true}'))
    result = extract_opportunity_fields("this is just a blog post", "Blog", "https://example.org/blog", provider)
    assert result.method == "ai"
    assert result.extraction_status == "rejected"


def test_extract_opportunity_fields_falls_back_on_ai_error():
    provider = _StubProvider(AIResult(ok=False, error="invalid api key"))
    page_text = "Eligibility: registered trusts. Deadline: 2027-09-01."
    result = extract_opportunity_fields(page_text, "Some Grant", "https://example.org/g", provider)
    assert result.method == "rule_based"
    assert result.deadline == "2027-09-01"
