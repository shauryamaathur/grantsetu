"""Fetches an NGO's website and a handful of its most relevant internal pages
(About/Programs/Impact/Contact-style pages, found by simple link-text
matching -- no AI needed for this step), then hands the combined text to
src/ai/research.py::analyze_website_text for the actual summarization/
keyword-detection (AI if the NGO has a configured provider, rule-based
otherwise)."""
import logging

from src.ai.providers import AIProvider
from src.ai.research import WebsiteAnalysisResult, analyze_website_text
from src.discovery.extract import extract_page
from src.discovery.fetcher import fetch_url

logger = logging.getLogger("grantsetu.discovery.website_analyzer")

RELEVANT_LINK_KEYWORDS = ["about", "program", "project", "impact", "mission", "work", "cause", "focus"]
MAX_ADDITIONAL_PAGES = 4


def _looks_relevant(url: str) -> bool:
    lower = url.lower()
    return any(kw in lower for kw in RELEVANT_LINK_KEYWORDS)


def analyze_ngo_website(website_url: str, provider: AIProvider | None) -> tuple[WebsiteAnalysisResult, list[str], str | None]:
    """Returns (analysis_result, pages_actually_fetched, error).
    `error` is set (and analysis_result is a minimal empty result) only if
    even the homepage could not be fetched at all -- e.g. blocked by
    robots.txt, DNS failure, timeout. Partial success (homepage fetched,
    some sub-pages failed) still returns a real analysis."""
    home = fetch_url(website_url)
    if not home.ok:
        reason = "blocked by the site's robots.txt" if home.blocked_by_robots else (home.error or "fetch failed")
        return (
            WebsiteAnalysisResult(method="rule_based", summary="", error=reason),
            [],
            f"Could not fetch {website_url}: {reason}",
        )

    home_page = extract_page(home.html, home.url)
    pages_fetched = [home.url]
    combined_text = home_page.text

    candidate_links = [link for link in home_page.links if _looks_relevant(link)][:MAX_ADDITIONAL_PAGES]
    for link in candidate_links:
        result = fetch_url(link)
        if not result.ok:
            logger.info("Skipping sub-page %s: %s", link, result.error)
            continue
        sub_page = extract_page(result.html, result.url)
        combined_text += " " + sub_page.text
        pages_fetched.append(result.url)

    analysis = analyze_website_text(combined_text, provider)
    return analysis, pages_fetched, None
