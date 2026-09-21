"""A GENERIC web-discovery connector: given a list of seed URLs (configured
per GrantSource via the `config: {"seed_urls": [...]}` JSON column), fetches
each page (respecting robots.txt / rate limits via src/discovery/fetcher.py),
extracts its readable text (src/discovery/extract.py), and parses it into a
RawGrantRecord via src/ai/research.py::extract_opportunity_fields, which
ALWAYS works with zero AI configuration (rule-based extraction is the
default/fallback path, so discovery never depends on an API key -- Section 4
of the "real product" upgrade brief) and additionally uses an AI provider,
when one is passed in, to improve extraction quality. The caller
(src/ingestion/sync_grants.py) is responsible for resolving which provider
(if any) applies to this GrantSource -- this connector itself has no opinion
and works identically either way, just passing `provider` straight through.

WHY NOT A REAL SEARCH ENGINE: this project has no credentials for a search
API (Google/Bing/etc all require paid API keys), and scraping a general
search engine's results page is both fragile and against most search
engines' terms of use. So "discovery" here means "crawl the specific pages
an admin configures," not "search the whole web for grant listings." This is
the honest, "at minimum" generic workflow the brief calls for -- adding a
real search-API-backed source later just means adding another connector
that also emits RawGrantRecord, without touching this one or the sync
pipeline.
"""
import hashlib

from src.ai.providers import AIProvider
from src.ai.research import extract_opportunity_fields
from src.connectors.base import ConnectorNotConfiguredError, GrantSourceConnector, RawGrantRecord
from src.discovery.extract import extract_page
from src.discovery.fetcher import fetch_url

# ExtractedOpportunityFields.method ("ai"/"rule_based") uses different
# vocabulary than RawGrantRecord.extraction_method ("ai_inferred"/
# "rule_based"/"direct") -- both are documented, real distinctions (the
# latter also has to cover connectors that never touch AI at all), so this
# maps between them rather than merging the two into one vocabulary.
_EXTRACTION_METHOD_MAP = {"ai": "ai_inferred", "rule_based": "rule_based"}


class WebDiscoveryConnector(GrantSourceConnector):
    connector_type = "web_discovery"
    is_sample_data = False

    def is_configured(self) -> bool:
        return bool(self.config.get("seed_urls"))

    def fetch_records(self, provider: AIProvider | None = None) -> list[RawGrantRecord]:
        seed_urls = self.config.get("seed_urls", [])
        if not seed_urls:
            raise ConnectorNotConfiguredError(
                "No seed_urls configured for this web_discovery source. "
                'Set GrantSource.config = {"seed_urls": ["https://..."]} via '
                "POST /admin/grant-sources."
            )

        records: list[RawGrantRecord] = []
        for url in seed_urls:
            result = fetch_url(url)
            if not result.ok:
                # A single unreachable/blocked source must not fail the whole
                # sync -- skip it and let the sync summary report it. No
                # fabricated record is ever produced for a page that could
                # not be fetched.
                continue

            page = extract_page(result.html, result.url)
            if not page.text.strip():
                continue

            extracted = extract_opportunity_fields(page.text, page.title, result.url, provider)

            if extracted.extraction_status == "rejected":
                # No grant/funding vocabulary anywhere on the page -- this
                # is NOT a grant opportunity (a Wikipedia article, news
                # story, blog post, generic "About us" page, etc). Never
                # persist it as one just because a URL was configured and
                # fetched successfully.
                continue

            external_id = hashlib.sha256(result.url.encode("utf-8")).hexdigest()[:24]
            records.append(
                RawGrantRecord(
                    external_id=external_id,
                    title=extracted.title or page.title or url,
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
