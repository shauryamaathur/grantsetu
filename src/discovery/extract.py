"""Turns raw HTML into readable text plus discovered internal links.
Uses BeautifulSoup with Python's built-in html.parser (no lxml dependency)."""
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ExtractedPage:
    title: str | None
    text: str
    meta_description: str | None
    links: list[str] = field(default_factory=list)


def extract_page(html: str, base_url: str) -> ExtractedPage:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_desc_tag.get("content", "").strip() if meta_desc_tag else None

    text = soup.get_text(separator=" ")
    text = _WHITESPACE_RE.sub(" ", text).strip()

    links = []
    base_domain = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        absolute = urljoin(base_url, href)
        # Only keep same-domain links -- this connector is a same-site crawler,
        # not an open-ended web spider (deliberately bounded scope/politeness).
        if urlparse(absolute).netloc == base_domain:
            links.append(absolute)

    return ExtractedPage(title=title, text=text, meta_description=meta_description, links=list(dict.fromkeys(links)))
