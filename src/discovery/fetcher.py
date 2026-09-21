"""A small, polite HTTP fetcher for the web-discovery pipeline: respects
robots.txt, rate-limits per domain, retries transient failures, always
identifies itself with a real User-Agent, and validates every URL (INCLUDING
every redirect hop) against SSRF before making a request. This is the ONLY
place in the project that fetches arbitrary external web pages -- website
analysis and the web-discovery connector both go through this module so the
politeness/safety rules are enforced in one place, not reimplemented per
caller.

SSRF matters here specifically because GrantSource.config's seed_urls (for
the web-discovery connector) and an NGO's website_url are both
admin/user-supplied strings that this module will actually fetch -- without
validation, either could point at an internal service (localhost, a cloud
metadata endpoint, a private-network admin panel) and use this server as a
proxy to reach it. `follow_redirects` is deliberately OFF at the httpx-client
level; redirects are followed manually, one hop at a time, with the SAME
validation applied to each hop's Location header -- a URL that looks safe
but 302s to an internal address is blocked at the hop, not just the start.
"""
import ipaddress
import logging
import socket
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger("grantsetu.discovery.fetcher")

USER_AGENT = "GrantSetuBot/1.0 (NGO funding-discovery research; contact: project maintainer)"
REQUEST_TIMEOUT = 12.0
MIN_SECONDS_BETWEEN_REQUESTS_PER_DOMAIN = 1.5
MAX_RETRIES = 2
MAX_PAGE_BYTES = 2_000_000  # don't try to parse huge downloads (e.g. accidental PDFs/binaries)
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_ACCEPTED_CONTENT_TYPES = {"text/html", "text/plain"}
_TEXT_CONTENT_TYPES = {"text/html", "text/plain"}

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
_last_request_at: dict[str, float] = {}


def _domain_of(url: str) -> str:
    return urlparse(url).netloc.lower()


_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _is_unsafe_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # couldn't parse -- refuse rather than guess

    # RFC 6052 NAT64 Well-Known Prefix: on an IPv6-only network path (DNS64/
    # NAT64 -- common in some sandboxed/CI/cloud environments), a hostname
    # that only has an IPv4 address gets a SYNTHESIZED IPv6 address in this
    # block, with the real IPv4 address embedded in the low 32 bits. Python's
    # `ipaddress` module marks the entire 64:ff9b::/96 block `is_reserved =
    # True` regardless of what's embedded in it (it's true of the block as a
    # special-purpose range, but not of any specific real destination
    # reached through it) -- so without unwrapping, a genuinely public IPv4
    # site (e.g. reached over NAT64) would be wrongly refused as if it were
    # an internal address. Evaluate safety on the EMBEDDED IPv4 address
    # instead when this prefix is detected; a real private/internal
    # destination synthesized into this same block (e.g. NAT64-mapped
    # 127.0.0.1 or 169.254.169.254) is still correctly caught below, since
    # the checks then run against that embedded address.
    if isinstance(ip, ipaddress.IPv6Address) and ip in _NAT64_WELL_KNOWN_PREFIX:
        ip = ipaddress.IPv4Address(ip.packed[-4:])

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        # The cloud-metadata address (AWS/GCP/Azure all use it) is the
        # single most common real-world SSRF target -- called out
        # explicitly even though is_link_local already covers it, as a
        # belt-and-suspenders check against this exact address.
        or str(ip) == "169.254.169.254"
    )


class SSRFBlockedError(RuntimeError):
    """Raised (and caught internally, turned into a FetchResult) when a URL
    resolves to a disallowed destination."""


def validate_url_is_safe(url: str) -> str | None:
    """Returns None if `url` is safe to fetch, or a human-readable reason
    string if it must be refused. Checks: scheme is http(s), hostname is
    present and resolves, and NONE of its resolved IP addresses are
    private/loopback/link-local/reserved/multicast/the cloud metadata
    address -- a hostname that round-robins between a public and a private
    IP is refused entirely, not fetched against whichever IP happens to be
    "safe" on a given lookup."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return f"Unsupported URL scheme '{parsed.scheme}' (only http/https allowed)"
    hostname = parsed.hostname
    if not hostname:
        return "Not a valid absolute URL"

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return f"Could not resolve hostname: {exc}"

    resolved_ips = {info[4][0] for info in infos}
    if not resolved_ips:
        return "Hostname did not resolve to any address"
    unsafe = [ip for ip in resolved_ips if _is_unsafe_ip(ip)]
    if unsafe:
        return f"Refusing to fetch {hostname}: resolves to a private/internal address ({unsafe[0]})"
    return None


def _get_robot_parser(url: str) -> urllib.robotparser.RobotFileParser:
    domain = _domain_of(url)
    if domain in _robots_cache:
        return _robots_cache[domain]

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(robots_url)
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
            else:
                # No robots.txt, or it errored -- treat as "allow" (the
                # standard convention: a missing robots.txt permits crawling).
                parser.parse([])
    except Exception as exc:  # noqa: BLE001 -- a robots.txt fetch failure must not block crawling
        logger.warning("Could not fetch robots.txt for %s: %s -- defaulting to allow", domain, exc)
        parser.parse([])

    _robots_cache[domain] = parser
    return parser


def _respect_rate_limit(domain: str):
    last = _last_request_at.get(domain)
    if last is not None:
        elapsed = time.monotonic() - last
        wait = MIN_SECONDS_BETWEEN_REQUESTS_PER_DOMAIN - elapsed
        if wait > 0:
            time.sleep(wait)
    _last_request_at[domain] = time.monotonic()


@dataclass
class FetchResult:
    ok: bool
    url: str
    status_code: int | None = None
    html: str | None = None  # populated only for a text/html or text/plain response
    content: bytes | None = None  # raw response body, populated for any accepted content-type (e.g. a PDF)
    content_type: str | None = None
    error: str | None = None
    blocked_by_robots: bool = False


def _fetch_one_hop(url: str, client: httpx.Client) -> httpx.Response:
    return client.get(url)


def fetch_url(
    url: str,
    accepted_content_types: set[str] | None = None,
    max_bytes: int | None = None,
) -> FetchResult:
    """Fetch one URL, honoring robots.txt and per-domain rate limiting, with
    a couple of retries on transient network errors. Never raises -- always
    returns a FetchResult describing what happened, including WHY a fetch
    was refused or failed, so callers can surface an honest status rather
    than silently producing nothing.

    `accepted_content_types` defaults to text/html + text/plain (the
    original, still-default behavior for every existing caller). Passing a
    different set -- e.g. {"application/pdf"} for a source whose real
    detail pages are PDF notices (see src/connectors/pdf_listing_connector.py)
    -- widens what's accepted for THIS call only; a response whose
    content-type isn't in the set is still refused exactly as before.
    `html` is only ever populated for a text/html or text/plain response;
    `content` (raw bytes) is populated for any accepted response, so a
    binary format like PDF is available to the caller without this module
    knowing how to parse it.

    SSRF protection: the URL (and, manually, every redirect hop) is checked
    with validate_url_is_safe() before any request is made -- see that
    function and the module docstring for what's blocked and why."""
    accepted_content_types = accepted_content_types or DEFAULT_ACCEPTED_CONTENT_TYPES
    max_bytes = max_bytes or MAX_PAGE_BYTES
    unsafe_reason = validate_url_is_safe(url)
    if unsafe_reason:
        logger.warning("Refusing to fetch %s: %s", url, unsafe_reason)
        return FetchResult(ok=False, url=url, error=unsafe_reason)

    domain = _domain_of(url)
    if not domain:
        return FetchResult(ok=False, url=url, error="Not a valid absolute URL")

    robots = _get_robot_parser(url)
    if not robots.can_fetch(USER_AGENT, url):
        logger.info("Skipping %s -- disallowed by robots.txt", url)
        return FetchResult(ok=False, url=url, blocked_by_robots=True, error="Disallowed by robots.txt")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        _respect_rate_limit(domain)
        try:
            with httpx.Client(
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=False,
            ) as client:
                current_url = url
                resp = _fetch_one_hop(current_url, client)
                for _ in range(MAX_REDIRECTS):
                    if resp.status_code not in (301, 302, 303, 307, 308):
                        break
                    location = resp.headers.get("location")
                    if not location:
                        break
                    current_url = urljoin(current_url, location)
                    hop_unsafe_reason = validate_url_is_safe(current_url)
                    if hop_unsafe_reason:
                        logger.warning("Refusing to follow redirect to %s: %s", current_url, hop_unsafe_reason)
                        return FetchResult(ok=False, url=current_url, error=f"Redirect blocked: {hop_unsafe_reason}")
                    hop_domain = _domain_of(current_url)
                    hop_robots = _get_robot_parser(current_url)
                    if not hop_robots.can_fetch(USER_AGENT, current_url):
                        return FetchResult(ok=False, url=current_url, blocked_by_robots=True, error="Redirect target disallowed by robots.txt")
                    _respect_rate_limit(hop_domain)
                    resp = _fetch_one_hop(current_url, client)
                else:
                    return FetchResult(ok=False, url=current_url, error="Too many redirects")

            if resp.status_code >= 400:
                return FetchResult(ok=False, url=current_url, status_code=resp.status_code, error=f"HTTP {resp.status_code}")
            content_type = resp.headers.get("content-type", "")
            if not any(accepted in content_type for accepted in accepted_content_types):
                return FetchResult(ok=False, url=current_url, status_code=resp.status_code, error=f"Unsupported content-type: {content_type}")
            if len(resp.content) > max_bytes:
                return FetchResult(ok=False, url=current_url, status_code=resp.status_code, error="Page too large, skipped")
            is_text = any(t in content_type for t in _TEXT_CONTENT_TYPES)
            return FetchResult(
                ok=True,
                url=current_url,
                status_code=resp.status_code,
                html=resp.text if is_text else None,
                content=resp.content,
                content_type=content_type,
            )
        except Exception as exc:  # noqa: BLE001 -- retry loop must catch broadly to retry/report cleanly
            last_error = str(exc)
            logger.warning("Fetch attempt %d/%d for %s failed: %s", attempt, MAX_RETRIES, url, exc)

    return FetchResult(ok=False, url=url, error=last_error or "Unknown fetch error")
