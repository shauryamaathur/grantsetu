"""A connector for sites that publish grant opportunities as a LISTING page
(many opportunities per page, each linking to its own detail page) rather
than one-opportunity-per-configured-URL, which is what WebDiscoveryConnector
already handles. Deliberately generic -- configured per GrantSource via
`config: {"listing_url": "https://...", "link_contains": "optional-substring",
"max_detail_pages": 15}` -- so it is NOT source-specific code; the first real
source configured through it is ngobox.org's public grant-announcement
listing (verified live during development: real page, no login, no
robots.txt restriction found -- see interview.txt), but adding a second
listing-style source later is a new GrantSource row, not new connector code.

Pipeline: fetch the listing page (SSRF-safe, robots.txt-aware, rate-limited
-- src/discovery/fetcher.py, same as every other fetch in this project) ->
extract_page() gives same-domain links already filtered/deduped -> narrow to
links matching `link_contains` if given -> fetch each detail page (same
safety pipeline, one grant-signal-classified extraction per page via
src/ai/research.py::extract_opportunity_fields -- rule-based by default,
AI-assisted if sync_grants.py resolved a provider for this source; see that
module's `ai_ngo_id` opt-in, and this module's `fetch_records`) -> skip pages
the extractor classifies as "rejected" (not actually a grant page) -> emit
one RawGrantRecord per surviving detail page.

NOT YET VERIFIED before this connector is used in production: ngobox.org's
Terms of Use (none were found linked from the listing page during
development, but that is not the same as an affirmative "automated access
permitted" -- an operator enabling this source should confirm this
directly with the site before syncing at any real volume). `max_detail_pages`
defaults conservatively low for exactly this reason.
"""
import hashlib
from urllib.parse import urlparse

from src.ai.providers import AIProvider
from src.ai.research import extract_opportunity_fields
from src.connectors.base import ConnectorNotConfiguredError, GrantSourceConnector, RawGrantRecord
from src.discovery.extract import extract_page
from src.discovery.fetcher import fetch_url

DEFAULT_MAX_DETAIL_PAGES = 15

# See src/connectors/web_discovery_connector.py -- same vocabulary mismatch
# between ExtractedOpportunityFields.method and RawGrantRecord.extraction_method.
_EXTRACTION_METHOD_MAP = {"ai": "ai_inferred", "rule_based": "rule_based"}


class ListingPageDiscoveryConnector(GrantSourceConnector):
    connector_type = "listing_page_discovery"
    is_sample_data = False

    def is_configured(self) -> bool:
        return bool(self.config.get("listing_url"))

    def fetch_records(self, provider: AIProvider | None = None) -> list[RawGrantRecord]:
        listing_url = self.config.get("listing_url")
        if not listing_url:
            raise ConnectorNotConfiguredError(
                "No listing_url configured for this listing_page_discovery source. "
                'Set GrantSource.config = {"listing_url": "https://..."} via '
                "POST /admin/grant-sources."
            )
        link_contains = self.config.get("link_contains")
        max_detail_pages = int(self.config.get("max_detail_pages", DEFAULT_MAX_DETAIL_PAGES))

        listing_result = fetch_url(listing_url)
        if not listing_result.ok:
            # Source unreachable/blocked this sync -- an honest "nothing
            # fetched," not a fabricated record. sync_grants.py still marks
            # the source "active" with fetched=0, same as any other
            # temporarily-empty real sync.
            return []

        listing_page = extract_page(listing_result.html, listing_result.url)
        detail_urls = listing_page.links
        if link_contains:
            detail_urls = [u for u in detail_urls if link_contains in u]
        detail_urls = detail_urls[:max_detail_pages]

        records: list[RawGrantRecord] = []
        for url in detail_urls:
            result = fetch_url(url)
            if not result.ok:
                continue

            page = extract_page(result.html, result.url)
            if not page.text.strip():
                continue

            extracted = extract_opportunity_fields(page.text, page.title, result.url, provider)
            if extracted.extraction_status == "rejected":
                continue

            external_id = hashlib.sha256(result.url.encode("utf-8")).hexdigest()[:24]
            records.append(
                RawGrantRecord(
                    external_id=external_id,
                    title=extracted.title or page.title or urlparse(result.url).path,
                    funder=extracted.funder,
                    description=extracted.description or page.meta_description,
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
