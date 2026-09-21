"""Pluggable web-search provider seam for Ask AI (src/ranking/ai_search.py).

No search-API credentials exist in this deployment. As
src/connectors/web_discovery_connector.py's docstring already explains for
the sync-side discovery connectors: search APIs (Google/Bing/Brave/etc) all
require a paid key, none is configured here, and scraping a general search
engine's results page is both fragile and against most engines' terms of
use. This module is therefore a clean, real extension POINT, not a working
integration -- `get_web_search_provider()` returns None unless a provider
is actually configured, and every caller already treats "no provider" as a
normal, fully-supported state (the same AI-optional discipline used
everywhere else in this codebase, e.g. src/ai/factory.py::
get_provider_for_ngo). Ask AI works completely without this wired up; it
just can't discover opportunities GrantSetu hasn't indexed from some other
source yet.

TO ACTUALLY ENABLE OPEN-WEB DISCOVERY:
1. Get an API key for a real search provider (e.g. Brave Search -- has a
   free tier at https://brave.com/search/api/).
2. Set SEARCH_PROVIDER=brave and BRAVE_SEARCH_API_KEY=... in .env.
3. Implement BraveSearchProvider.search() below (a plain HTTP GET to
   https://api.search.brave.com/res/v1/web/search) -- it currently raises
   NotImplementedError so a misconfigured deployment fails loudly rather
   than silently returning nothing.
4. In src/ranking/ai_search.py::_search_web(), feed each hit's URL through
   the SAME extraction pipeline web_discovery_connector.py already uses
   (src/discovery/fetcher.py -- robots.txt-aware, SSRF-safe -- then
   src/ai/research.py::extract_opportunity_fields) so a web-search result
   becomes a real, structured, source-linked AISearchResult -- never a raw
   search snippet displayed as if it were a verified grant.
"""
import os
from dataclasses import dataclass


@dataclass
class WebSearchHit:
    title: str
    url: str
    snippet: str


class WebSearchProvider:
    def search(self, query: str, limit: int = 10) -> list[WebSearchHit]:
        raise NotImplementedError


class BraveSearchProvider(WebSearchProvider):
    """Stub adapter for Brave Search's web search API -- intentionally not
    implemented (there are no credentials in this environment to build or
    test a real integration against). Present so the seam in
    src/ranking/ai_search.py has something concrete to call once a real key
    exists, and so enabling it later is a small, well-scoped change instead
    of a new architecture."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, limit: int = 10) -> list[WebSearchHit]:
        raise NotImplementedError(
            "BraveSearchProvider is a stub -- implement the actual Brave Search API call "
            "(GET https://api.search.brave.com/res/v1/web/search) before enabling SEARCH_PROVIDER=brave."
        )


_PROVIDERS = {
    "brave": lambda: BraveSearchProvider(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()),
}


def get_web_search_provider() -> WebSearchProvider | None:
    """Returns a configured provider, or None if web search isn't set up.
    None is the default, fully-supported state (see module docstring) --
    callers must never treat it as an error."""
    provider_name = os.environ.get("SEARCH_PROVIDER", "").strip().lower()
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if provider_name in _PROVIDERS and api_key:
        return _PROVIDERS[provider_name]()
    return None
