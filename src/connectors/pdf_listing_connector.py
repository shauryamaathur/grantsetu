"""A connector for sites whose listing page links to PDF notices instead of
HTML detail pages -- the pattern found on MSJE's e-ANUDAAN portal
(grants-msje.gov.in) during live verification for this phase: real,
server-rendered HTML on the listing/home page, but every individual
announcement is a direct PDF download (an Expression of Interest, a
corrigendum, a scheme guideline), which src/connectors/listing_page_connector.py
cannot use -- src/discovery/fetcher.py deliberately refuses any
content-type outside text/html and text/plain for that connector's fetches.

Mechanically identical to ListingPageDiscoveryConnector (same
GrantSource.config shape: {"listing_url", "link_contains",
"max_detail_pages"}) except the per-link fetch accepts application/pdf in
addition to text/html, and a PDF response's text comes from
src/discovery/pdf_extract.py::extract_pdf_text() (no OCR -- a scanned/
image-only PDF yields no text and is skipped, same as an empty HTML page)
rather than src/discovery/extract.py. A link that turns out to be HTML
instead of a PDF is still handled correctly (same extract_page() path as
the HTML connector), since real government listing pages mix the two
without warning.

NOT YET VERIFIED before any source using this connector is enabled in
production: like ListingPageDiscoveryConnector's note on ngobox.org, no
Terms of Use was found linked from the MSJE homepage during development,
but that is not the same as an affirmative "automated access permitted" --
an operator should confirm this with the source before syncing at volume.
`max_detail_pages` defaults conservatively low for exactly this reason, and
each PDF fetch is capped well above HTML page sizes (real MSJE notices
verified during development ran 1-2.5MB) but still bounded, so one huge
attachment can't make a sync run unbounded.

KNOWN LIMITATION, verified live and left honest rather than papered over:
several real MSJE notices are scanned-letterhead-style PDFs whose OCR text
opens with generic boilerplate ("Department of Social Justice &
Empowerment") rather than the notice's actual subject, even after
src/discovery/pdf_extract.py::extract_pdf_title_guess() skips the most
common letterhead lines -- a title guess for these can end up generic
(never fabricated, just unhelpfully generic) until a real per-notice
subject line is found further down the document. The eligibility/deadline/
description fields, extracted from the FULL page text rather than just the
first line, were verified accurate against the same real documents.
"""
import hashlib
from urllib.parse import urlparse

from src.ai.providers import AIProvider
from src.ai.research import extract_opportunity_fields
from src.connectors.base import ConnectorNotConfiguredError, GrantSourceConnector, RawGrantRecord
from src.discovery.extract import extract_page
from src.discovery.fetcher import fetch_url
from src.discovery.pdf_extract import extract_pdf_text, extract_pdf_title_guess

DEFAULT_MAX_DETAIL_PAGES = 15
MAX_PDF_BYTES = 8_000_000
_DETAIL_ACCEPTED_CONTENT_TYPES = {"text/html", "text/plain", "application/pdf"}

# See listing_page_connector.py -- same vocabulary mismatch between
# ExtractedOpportunityFields.method and RawGrantRecord.extraction_method.
_EXTRACTION_METHOD_MAP = {"ai": "ai_inferred", "rule_based": "rule_based"}


class PDFListingDiscoveryConnector(GrantSourceConnector):
    connector_type = "pdf_listing_discovery"
    is_sample_data = False

    def is_configured(self) -> bool:
        return bool(self.config.get("listing_url"))

    def fetch_records(self, provider: AIProvider | None = None) -> list[RawGrantRecord]:
        listing_url = self.config.get("listing_url")
        if not listing_url:
            raise ConnectorNotConfiguredError(
                "No listing_url configured for this pdf_listing_discovery source. "
                'Set GrantSource.config = {"listing_url": "https://..."} via '
                "POST /admin/grant-sources."
            )
        link_contains = self.config.get("link_contains")
        max_detail_pages = int(self.config.get("max_detail_pages", DEFAULT_MAX_DETAIL_PAGES))

        listing_result = fetch_url(listing_url)
        if not listing_result.ok:
            # Source unreachable/blocked this sync -- an honest "nothing
            # fetched," not a fabricated record.
            return []

        listing_page = extract_page(listing_result.html, listing_result.url)
        detail_urls = listing_page.links
        if link_contains:
            detail_urls = [u for u in detail_urls if link_contains in u]
        detail_urls = detail_urls[:max_detail_pages]

        records: list[RawGrantRecord] = []
        for url in detail_urls:
            result = fetch_url(url, accepted_content_types=_DETAIL_ACCEPTED_CONTENT_TYPES, max_bytes=MAX_PDF_BYTES)
            if not result.ok:
                continue

            if result.html is not None:
                page = extract_page(result.html, result.url)
                page_text, page_title = page.text, page.title
            else:
                page_text = extract_pdf_text(result.content or b"")
                page_title = extract_pdf_title_guess(result.content or b"")
            if not page_text.strip():
                continue

            extracted = extract_opportunity_fields(page_text, page_title, result.url, provider)
            if extracted.extraction_status == "rejected":
                continue

            external_id = hashlib.sha256(result.url.encode("utf-8")).hexdigest()[:24]
            records.append(
                RawGrantRecord(
                    external_id=external_id,
                    title=extracted.title or page_title or urlparse(result.url).path,
                    funder=extracted.funder,
                    description=extracted.description,
                    eligibility_text=extracted.eligibility_text,
                    category=extracted.category,
                    deadline=extracted.deadline,
                    url=result.url,
                    country=extracted.country,
                    extraction_method=_EXTRACTION_METHOD_MAP.get(extracted.method, "rule_based"),
                    evidence_snippet=extracted.evidence_snippet,
                    extraction_status=extracted.extraction_status,
                )
            )
        return records
