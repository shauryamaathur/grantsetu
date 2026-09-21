"""Tests for src/discovery/ (HTML extraction, robots.txt-respecting fetch)
and the web-discovery connector. No real network calls: fetch_url is
monkeypatched everywhere a live HTTP request would otherwise happen -- the
exceptions are the SSRF-validation tests below, which call the REAL
validate_url_is_safe()/fetch_url() against hostnames that resolve locally
(localhost, 127.0.0.1) via the OS resolver rather than the network, so they
stay fast and deterministic without needing internet access."""
from src.connectors.web_discovery_connector import WebDiscoveryConnector
from src.discovery import fetcher
from src.discovery.extract import extract_page
from src.discovery.fetcher import FetchResult, validate_url_is_safe

SAMPLE_HTML = """
<html>
<head><title>Rural Health Access Grant</title>
<meta name="description" content="A grant for rural health access programs.">
</head>
<body>
<script>console.log("should be stripped")</script>
<h1>Rural Health Access Grant</h1>
<p>Eligibility: registered nonprofit organizations and trusts may apply.</p>
<p>The application deadline is 2027-04-15.</p>
<a href="/about">About us</a>
<a href="/programs">Our Programs</a>
<a href="https://external-site.example/other">External link</a>
<a href="mailto:contact@example.org">Email us</a>
</body>
</html>
"""


def test_extract_page_strips_scripts_and_returns_clean_text():
    page = extract_page(SAMPLE_HTML, "https://example.org/grant")
    assert page.title == "Rural Health Access Grant"
    assert "console.log" not in page.text
    assert "Eligibility" in page.text
    assert page.meta_description == "A grant for rural health access programs."


def test_extract_page_resolves_relative_links_and_filters_external_and_mailto():
    page = extract_page(SAMPLE_HTML, "https://example.org/grant")
    assert "https://example.org/about" in page.links
    assert "https://example.org/programs" in page.links
    assert not any("external-site.example" in link for link in page.links)
    assert not any(link.startswith("mailto:") for link in page.links)


def test_extract_page_handles_empty_html_gracefully():
    page = extract_page("<html><body></body></html>", "https://example.org")
    assert page.text == ""
    assert page.links == []


def test_validate_url_is_safe_rejects_localhost():
    reason = validate_url_is_safe("http://localhost:8000/admin")
    assert reason is not None
    assert "private" in reason.lower() or "internal" in reason.lower()


def test_validate_url_is_safe_rejects_loopback_ip_literal():
    reason = validate_url_is_safe("http://127.0.0.1/secret")
    assert reason is not None


def test_validate_url_is_safe_rejects_private_network_ip():
    reason = validate_url_is_safe("http://10.0.0.5/internal-dashboard")
    assert reason is not None


def test_validate_url_is_safe_rejects_cloud_metadata_address():
    reason = validate_url_is_safe("http://169.254.169.254/latest/meta-data/")
    assert reason is not None


def test_validate_url_is_safe_rejects_non_http_scheme():
    reason = validate_url_is_safe("file:///etc/passwd")
    assert reason is not None
    assert "scheme" in reason.lower()


def test_validate_url_is_safe_allows_a_real_public_hostname():
    # grants.gov is a real, known-public domain (already used live elsewhere
    # in this project) -- resolving it exercises the real DNS path and
    # confirms the check doesn't over-block legitimate public sites.
    reason = validate_url_is_safe("https://www.grants.gov/")
    assert reason is None


def test_is_unsafe_ip_allows_nat64_synthesized_public_address():
    """Regression test for a real reported bug: on an IPv6-only network path
    (DNS64/NAT64 -- common in some sandboxed/CI/cloud environments), a
    hostname with only an IPv4 address gets a SYNTHESIZED IPv6 address in
    the RFC 6052 Well-Known Prefix 64:ff9b::/96, with the real IPv4 address
    embedded in the low 32 bits. Python's ipaddress module marks the WHOLE
    64:ff9b::/96 block is_reserved=True regardless of what's embedded,
    which was wrongly refusing genuinely public sites reached this way --
    e.g. POST /ngo-profile/{id}/analyze-website returning 502 for
    https://helpeachother.org.in/, which resolves to 64:ff9b::5219:6b0e
    (embedding the real public address 82.25.107.14)."""
    from src.discovery.fetcher import _is_unsafe_ip

    assert _is_unsafe_ip("64:ff9b::5219:6b0e") is False  # embeds 82.25.107.14 (real public IP)


def test_is_unsafe_ip_still_rejects_nat64_synthesized_private_addresses():
    """The NAT64 unwrap must not become a blanket bypass: a private/
    internal destination reached THROUGH NAT64 is still refused, evaluated
    against its embedded IPv4 address."""
    from src.discovery.fetcher import _is_unsafe_ip

    assert _is_unsafe_ip("64:ff9b::7f00:1") is True    # embeds 127.0.0.1 (loopback)
    assert _is_unsafe_ip("64:ff9b::a00:1") is True      # embeds 10.0.0.1 (private)
    assert _is_unsafe_ip("64:ff9b::a9fe:a9fe") is True  # embeds 169.254.169.254 (cloud metadata)


def test_is_unsafe_ip_still_rejects_genuine_reserved_ipv6_ranges():
    """The NAT64 fix is scoped to exactly one prefix -- other reserved/
    private IPv6 ranges must still be blocked, unaffected."""
    from src.discovery.fetcher import _is_unsafe_ip

    assert _is_unsafe_ip("::1") is True              # loopback
    assert _is_unsafe_ip("fe80::1") is True           # link-local
    assert _is_unsafe_ip("fc00::1") is True           # unique local (private)
    assert _is_unsafe_ip("2001:db8::1") is True       # documentation (reserved)


def test_fetch_url_refuses_to_fetch_a_private_address(monkeypatch):
    """End-to-end: fetch_url() itself must refuse (never attempt an HTTP
    request against) a private-address URL, not just the validator function
    in isolation."""
    called = False

    class _FailIfCalledClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            nonlocal called
            called = True
            raise AssertionError("should never make a real request for a blocked URL")

    monkeypatch.setattr(fetcher.httpx, "Client", _FailIfCalledClient)
    result = fetcher.fetch_url("http://127.0.0.1/internal")
    assert result.ok is False
    assert called is False


def test_web_discovery_connector_not_configured_without_seed_urls():
    connector = WebDiscoveryConnector(config={})
    assert connector.is_configured() is False


def test_web_discovery_connector_configured_with_seed_urls():
    connector = WebDiscoveryConnector(config={"seed_urls": ["https://example.org/grant"]})
    assert connector.is_configured() is True


def test_web_discovery_connector_extracts_records_from_fetched_pages(monkeypatch):
    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html=SAMPLE_HTML)

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch)
    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    connector = WebDiscoveryConnector(config={"seed_urls": ["https://example.org/grant"]})
    records = connector.fetch_records()

    assert len(records) == 1
    record = records[0]
    assert record.title == "Rural Health Access Grant"
    assert record.deadline == "2027-04-15"
    assert record.extraction_method == "rule_based"
    assert record.url == "https://example.org/grant"


def test_web_discovery_connector_rejects_pages_with_no_grant_signal(monkeypatch):
    """A successfully-fetched page that isn't actually about a grant (a
    Wikipedia-style article, in this case) must never be imported as an
    opportunity -- see extract_opportunity_fields_rule_based's "rejected"
    classification and WebDiscoveryConnector.fetch_records."""
    non_grant_html = """
    <html><head><title>History of Rural Development</title></head>
    <body><h1>History of Rural Development</h1>
    <p>Rural development has been studied by economists for decades, with
    many theories about urbanization and migration patterns over time.</p>
    </body></html>
    """

    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html=non_grant_html)

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    connector = WebDiscoveryConnector(config={"seed_urls": ["https://example.org/wiki/rural-development"]})
    records = connector.fetch_records()
    assert records == []  # rejected as not a grant page -- never imported


def test_web_discovery_connector_skips_unreachable_pages(monkeypatch):
    def fake_fetch(url):
        return FetchResult(ok=False, url=url, error="Connection timed out")

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    connector = WebDiscoveryConnector(config={"seed_urls": ["https://unreachable.example/grant"]})
    records = connector.fetch_records()
    assert records == []  # never fabricates a record for a page that failed to load


def test_web_discovery_connector_raises_when_unconfigured():
    from src.connectors.base import ConnectorNotConfiguredError

    connector = WebDiscoveryConnector(config={})
    try:
        connector.fetch_records()
        assert False, "expected ConnectorNotConfiguredError"
    except ConnectorNotConfiguredError:
        pass


def test_web_discovery_connector_uses_ai_provider_when_given(monkeypatch):
    """When sync_grants.py resolves an AIProvider for this source (opt-in via
    GrantSource.config["ai_ngo_id"] -- see src/ingestion/sync_grants.py) and
    passes it into fetch_records(), the connector must use it via
    extract_opportunity_fields and record extraction_method="ai_inferred" --
    not silently stay on the rule-based path."""
    from src.ai.providers import AIResult

    class _StubProvider:
        def complete(self, prompt, max_tokens=700):
            return AIResult(
                ok=True,
                text='{"title": "AI-Extracted Grant Title", "funder": "AI Foundation", '
                '"description": "AI summary.", "eligibility_text": "Nonprofits.", '
                '"category": "health", "deadline": "2027-04-15", "evidence_snippet": "deadline text"}',
            )

    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html=SAMPLE_HTML)

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    connector = WebDiscoveryConnector(config={"seed_urls": ["https://example.org/grant"]})
    records = connector.fetch_records(provider=_StubProvider())

    assert len(records) == 1
    record = records[0]
    assert record.title == "AI-Extracted Grant Title"
    assert record.funder == "AI Foundation"
    assert record.extraction_method == "ai_inferred"


def test_web_discovery_connector_falls_back_to_rule_based_when_ai_fails(monkeypatch):
    """A provider that errors (bad key, rate limit, unparseable output) must
    never break discovery -- extract_opportunity_fields already guarantees
    this fallback; this test locks in that WebDiscoveryConnector doesn't
    bypass it."""
    from src.ai.providers import AIResult

    class _FailingProvider:
        def complete(self, prompt, max_tokens=700):
            return AIResult(ok=False, error="rate limited")

    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html=SAMPLE_HTML)

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    connector = WebDiscoveryConnector(config={"seed_urls": ["https://example.org/grant"]})
    records = connector.fetch_records(provider=_FailingProvider())

    assert len(records) == 1
    assert records[0].extraction_method == "rule_based"
    assert records[0].title == "Rural Health Access Grant"


# --- ListingPageDiscoveryConnector -----------------------------------------
# Fixture HTML shapes match what was verified live against ngobox.org during
# development (see interview.txt) -- captured/reconstructed here, never a
# live network call in the automated suite, same discipline as every other
# connector test in this project.

_LISTING_HTML = """
<html><body>
<div class="grant-card"><a href="/full_grant_announcement_First-Grant_1001">First Grant</a></div>
<div class="grant-card"><a href="/full_grant_announcement_Second-Grant_1002">Second Grant</a></div>
<div class="footer"><a href="/about">About us</a></div>
</body></html>
"""

_DETAIL_HTML_TEMPLATE = """
<html><head><title>{title}</title></head><body>
<h1>{title}</h1>
<p>Organization: {funder} Apply By: {deadline}</p>
<p>Eligibility: registered nonprofit organizations may apply.</p>
<p>This grant offers funding for community projects.</p>
</body></html>
"""

_NON_GRANT_DETAIL_HTML = """
<html><head><title>About Us</title></head><body>
<h1>About Us</h1>
<p>We are a community organization founded in 2005 with a mission to connect people.</p>
</body></html>
"""


def test_listing_page_connector_not_configured_without_listing_url():
    from src.connectors.listing_page_connector import ListingPageDiscoveryConnector

    connector = ListingPageDiscoveryConnector(config={})
    assert connector.is_configured() is False


def test_listing_page_connector_raises_when_unconfigured():
    from src.connectors.base import ConnectorNotConfiguredError
    from src.connectors.listing_page_connector import ListingPageDiscoveryConnector

    connector = ListingPageDiscoveryConnector(config={})
    try:
        connector.fetch_records()
        assert False, "expected ConnectorNotConfiguredError"
    except ConnectorNotConfiguredError:
        pass


def test_listing_page_connector_extracts_records_from_detail_pages(monkeypatch):
    import src.connectors.listing_page_connector as lpc_module

    def fake_fetch(url):
        if url == "https://example.org/listing":
            return FetchResult(ok=True, url=url, status_code=200, html=_LISTING_HTML)
        if "First-Grant" in url:
            html = _DETAIL_HTML_TEMPLATE.format(title="First Grant", funder="Example Fund", deadline="15 Nov 2026")
            return FetchResult(ok=True, url=url, status_code=200, html=html)
        if "Second-Grant" in url:
            html = _DETAIL_HTML_TEMPLATE.format(title="Second Grant", funder="Other Fund", deadline="20 Dec 2026")
            return FetchResult(ok=True, url=url, status_code=200, html=html)
        return FetchResult(ok=False, url=url, error="not found")

    monkeypatch.setattr(lpc_module, "fetch_url", fake_fetch)

    from src.connectors.listing_page_connector import ListingPageDiscoveryConnector

    connector = ListingPageDiscoveryConnector(config={
        "listing_url": "https://example.org/listing",
        "link_contains": "full_grant_announcement",
    })
    records = connector.fetch_records()

    assert len(records) == 2
    titles = {r.title for r in records}
    assert titles == {"First Grant", "Second Grant"}
    first = next(r for r in records if r.title == "First Grant")
    assert first.funder == "Example Fund"
    assert first.deadline == "2026-11-15"
    assert first.eligibility_text is not None
    assert first.extraction_status == "complete"


def test_listing_page_connector_skips_non_grant_detail_pages(monkeypatch):
    import src.connectors.listing_page_connector as lpc_module

    def fake_fetch(url):
        if url == "https://example.org/listing":
            html = '<html><body><a href="/full_grant_announcement_About_1">About</a></body></html>'
            return FetchResult(ok=True, url=url, status_code=200, html=html)
        return FetchResult(ok=True, url=url, status_code=200, html=_NON_GRANT_DETAIL_HTML)

    monkeypatch.setattr(lpc_module, "fetch_url", fake_fetch)

    from src.connectors.listing_page_connector import ListingPageDiscoveryConnector

    connector = ListingPageDiscoveryConnector(config={"listing_url": "https://example.org/listing"})
    records = connector.fetch_records()
    assert records == []  # rejected as not a grant page -- never imported


def test_listing_page_connector_raises_when_listing_unreachable(monkeypatch):
    """The listing page itself (this connector's one entry point) failing
    to fetch must be reported as a real failure, not silently swallowed
    into an empty list that reads identically to "checked, nothing new" --
    see ConnectorUnavailableError's docstring and sync_grants.py, which
    turns this into an honest GrantSource.status="error" with the real
    reason (e.g. an HTTP 403 from the site's own bot protection)."""
    import src.connectors.listing_page_connector as lpc_module
    from src.connectors.base import ConnectorUnavailableError

    def fake_fetch(url):
        return FetchResult(ok=False, url=url, error="Connection timed out")

    monkeypatch.setattr(lpc_module, "fetch_url", fake_fetch)

    from src.connectors.listing_page_connector import ListingPageDiscoveryConnector

    connector = ListingPageDiscoveryConnector(config={"listing_url": "https://example.org/listing"})
    try:
        connector.fetch_records()
        assert False, "expected ConnectorUnavailableError"
    except ConnectorUnavailableError as exc:
        assert "Connection timed out" in str(exc)


def test_listing_page_connector_respects_max_detail_pages(monkeypatch):
    import src.connectors.listing_page_connector as lpc_module

    many_links_html = "<html><body>" + "".join(
        f'<a href="/full_grant_announcement_Grant-{i}_100{i}">Grant {i}</a>' for i in range(10)
    ) + "</body></html>"

    def fake_fetch(url):
        if url == "https://example.org/listing":
            return FetchResult(ok=True, url=url, status_code=200, html=many_links_html)
        html = _DETAIL_HTML_TEMPLATE.format(title="Some Grant", funder="A Fund", deadline="1 Jan 2027")
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    monkeypatch.setattr(lpc_module, "fetch_url", fake_fetch)

    from src.connectors.listing_page_connector import ListingPageDiscoveryConnector

    connector = ListingPageDiscoveryConnector(config={
        "listing_url": "https://example.org/listing",
        "link_contains": "full_grant_announcement",
        "max_detail_pages": 3,
    })
    records = connector.fetch_records()
    assert len(records) == 3


# --- PDFListingDiscoveryConnector -------------------------------------------
# Real-world shape verified live against grants-msje.gov.in (MSJE's
# e-ANUDAAN portal) during development: a real HTML listing/announcements
# page whose individual notices are direct PDF downloads, not HTML detail
# pages -- see interview.txt Section 23 and this connector's module
# docstring. No live network call or real PDF bytes in the automated suite;
# fetch_url and extract_pdf_text are both monkeypatched.

_PDF_LISTING_HTML = """
<html><body>
<a href="/notice_First-Notice_1001">First Notice</a>
<a href="/notice_Second-Notice_1002">Second Notice</a>
<a href="/about">About us</a>
</body></html>
"""


def test_pdf_listing_connector_not_configured_without_listing_url():
    from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector

    connector = PDFListingDiscoveryConnector(config={})
    assert connector.is_configured() is False


def test_pdf_listing_connector_raises_when_unconfigured():
    from src.connectors.base import ConnectorNotConfiguredError
    from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector

    connector = PDFListingDiscoveryConnector(config={})
    try:
        connector.fetch_records()
        assert False, "expected ConnectorNotConfiguredError"
    except ConnectorNotConfiguredError:
        pass


def test_pdf_listing_connector_extracts_records_from_pdf_detail_pages(monkeypatch):
    import src.connectors.pdf_listing_connector as plc_module

    def fake_fetch(url, accepted_content_types=None, max_bytes=None):
        if url == "https://example.org/listing":
            return FetchResult(ok=True, url=url, status_code=200, html=_PDF_LISTING_HTML)
        if "First-Notice" in url:
            return FetchResult(ok=True, url=url, status_code=200, content=b"%PDF-fake-first", content_type="application/pdf")
        if "Second-Notice" in url:
            return FetchResult(ok=True, url=url, status_code=200, content=b"%PDF-fake-second", content_type="application/pdf")
        return FetchResult(ok=False, url=url, error="not found")

    _PDF_TEXT = {
        b"%PDF-fake-first": (
            "Expression of Interest Ministry of Social Justice and Empowerment invites NGOs "
            "Organization: MSJE Apply By: 15 Nov 2026 Eligibility: registered nonprofit organizations may apply. "
            "This grant offers funding for shelter home projects."
        ),
        b"%PDF-fake-second": (
            "Expression of Interest Ministry of Social Justice and Empowerment invites NGOs "
            "Organization: MSJE Apply By: 20 Dec 2026 Eligibility: registered nonprofit organizations may apply. "
            "This grant offers funding for de-addiction centre projects."
        ),
    }

    def fake_extract_pdf_text(data: bytes) -> str:
        return _PDF_TEXT.get(data, "")

    monkeypatch.setattr(plc_module, "fetch_url", fake_fetch)
    monkeypatch.setattr(plc_module, "extract_pdf_text", fake_extract_pdf_text)

    from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector

    connector = PDFListingDiscoveryConnector(config={
        "listing_url": "https://example.org/listing",
        "link_contains": "notice",
    })
    records = connector.fetch_records()

    assert len(records) == 2
    deadlines = {r.deadline for r in records}
    assert deadlines == {"2026-11-15", "2026-12-20"}
    for r in records:
        assert r.eligibility_text is not None
        assert r.extraction_status == "complete"


def test_pdf_listing_connector_skips_pdfs_with_no_extractable_text(monkeypatch):
    import src.connectors.pdf_listing_connector as plc_module

    def fake_fetch(url, accepted_content_types=None, max_bytes=None):
        if url == "https://example.org/listing":
            html = '<html><body><a href="/notice_Scanned_1">Scanned Notice</a></body></html>'
            return FetchResult(ok=True, url=url, status_code=200, html=html)
        return FetchResult(ok=True, url=url, status_code=200, content=b"%PDF-scanned-image-only", content_type="application/pdf")

    monkeypatch.setattr(plc_module, "fetch_url", fake_fetch)
    monkeypatch.setattr(plc_module, "extract_pdf_text", lambda data: "")  # e.g. a scanned/image-only PDF -- no OCR

    from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector

    connector = PDFListingDiscoveryConnector(config={"listing_url": "https://example.org/listing"})
    records = connector.fetch_records()
    assert records == []


def test_pdf_listing_connector_raises_when_listing_unreachable(monkeypatch):
    """See listing_page_connector.py's identical test/fix."""
    import src.connectors.pdf_listing_connector as plc_module
    from src.connectors.base import ConnectorUnavailableError

    def fake_fetch(url, accepted_content_types=None, max_bytes=None):
        return FetchResult(ok=False, url=url, error="Connection timed out")

    monkeypatch.setattr(plc_module, "fetch_url", fake_fetch)

    from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector

    connector = PDFListingDiscoveryConnector(config={"listing_url": "https://example.org/listing"})
    try:
        connector.fetch_records()
        assert False, "expected ConnectorUnavailableError"
    except ConnectorUnavailableError as exc:
        assert "Connection timed out" in str(exc)


def test_pdf_listing_connector_handles_mixed_html_and_pdf_links(monkeypatch):
    """Real government listing pages mix HTML and PDF detail links without
    warning (verified live at grants-msje.gov.in) -- the connector must
    handle either per-link, not assume the whole source is one format."""
    import src.connectors.pdf_listing_connector as plc_module

    def fake_fetch(url, accepted_content_types=None, max_bytes=None):
        if url == "https://example.org/listing":
            html = (
                '<html><body>'
                '<a href="/notice_HTML-one_1">HTML Notice</a>'
                '<a href="/notice_PDF-one_2">PDF Notice</a>'
                '</body></html>'
            )
            return FetchResult(ok=True, url=url, status_code=200, html=html)
        if "HTML-one" in url:
            html = _DETAIL_HTML_TEMPLATE.format(title="HTML Notice Grant", funder="Example Fund", deadline="1 Jan 2027")
            return FetchResult(ok=True, url=url, status_code=200, html=html)
        return FetchResult(ok=True, url=url, status_code=200, content=b"%PDF-one", content_type="application/pdf")

    monkeypatch.setattr(plc_module, "fetch_url", fake_fetch)
    monkeypatch.setattr(
        plc_module,
        "extract_pdf_text",
        lambda data: (
            "Expression of Interest Organization: PDF Fund Apply By: 5 Feb 2027 "
            "Eligibility: registered nonprofit organizations may apply. Funding for community projects."
        ) if data == b"%PDF-one" else "",
    )

    from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector

    connector = PDFListingDiscoveryConnector(config={"listing_url": "https://example.org/listing"})
    records = connector.fetch_records()

    assert len(records) == 2
    titles = {r.title for r in records}
    assert "HTML Notice Grant" in titles


# --- src/discovery/pdf_extract.py -------------------------------------------

def test_extract_pdf_text_returns_empty_string_for_garbage_bytes():
    from src.discovery.pdf_extract import extract_pdf_text

    assert extract_pdf_text(b"not a real pdf at all") == ""


def test_extract_pdf_text_returns_empty_string_for_blank_pdf():
    import io

    from pypdf import PdfWriter

    from src.discovery.pdf_extract import extract_pdf_text

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    assert extract_pdf_text(buf.getvalue()) == ""


def test_extract_pdf_title_guess_skips_letterhead_boilerplate():
    from src.discovery.pdf_extract import _looks_like_a_title

    assert _looks_like_a_title("Government of India") is False
    assert _looks_like_a_title("Government of lndia") is False  # real OCR I/l artifact
    assert _looks_like_a_title("Ministry of Social Justice & Empowerment") is False
    assert _looks_like_a_title("No.12/28/2021-DP-II (EO 95308)") is False
    assert _looks_like_a_title("Expression of Interest") is True
