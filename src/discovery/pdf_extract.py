"""Extracts plain text from a PDF's raw bytes, for sources (e.g. MSJE
e-ANUDAAN -- see interview.txt Section 23) whose real "detail pages" are PDF
notices rather than HTML. Deliberately minimal: text extraction only, no
OCR -- a scanned/image-only PDF yields "" (an honest empty result), never a
guess at what it might say.
"""
import io
import logging
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger("grantsetu.discovery.pdf")


def _open_reader(data: bytes) -> PdfReader | None:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001 -- can't open it; treat as unreadable
                logger.warning("PDF is encrypted and could not be opened with an empty password")
                return None
        return reader
    except (PdfReadError, Exception) as exc:  # noqa: BLE001 -- malformed PDFs raise all sorts of things
        logger.warning("Could not open PDF: %s", exc)
        return None


def extract_pdf_text(data: bytes) -> str:
    """Never raises -- a corrupt/encrypted/unreadable PDF returns "",
    the same "nothing extracted" signal an empty HTML page would produce,
    so callers don't need PDF-specific error handling."""
    reader = _open_reader(data)
    if reader is None:
        return ""
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = " ".join(pages_text)
    return " ".join(text.split())


_MIN_TITLE_LEN = 6
_MAX_TITLE_LEN = 200
_MIN_ALPHA_WORDS = 2  # words of 3+ letters -- filters out reference-number headers

# Letterhead boilerplate that real MSJE notices were verified live to open
# with -- genuinely present text, but never the actual subject of the
# notice, so a title guess must skip past it rather than stop on it.
_BOILERPLATE_LINE_RE = re.compile(
    # [il]ndia tolerates the OCR'd source PDFs' own I/l confusion ("lndia")
    # -- a real, observed artifact in these documents, not a typo here.
    r"^(government of [il]ndia|govt\.? of [il]ndia|ministry of [a-z &]+(,? government of [il]ndia)?)\.?$",
    re.IGNORECASE,
)


def _looks_like_a_title(line: str) -> bool:
    """Real government notices verified live during development often open
    with a file-reference-number header line ("No.12/28/2021-DP-II (EO
    95308)") or a letterhead line ("Government of India") BEFORE the
    actual title -- genuinely-extracted but useless as a "title." Reject
    lines that are mostly digits/punctuation rather than words, and known
    letterhead boilerplate; a scanned/OCR'd PDF with no usable title line
    at all should fall through to None (the caller's URL fallback), not
    surface noise."""
    if _BOILERPLATE_LINE_RE.match(line.strip()):
        return False
    alpha_words = [w for w in re.findall(r"[A-Za-z]+", line) if len(w) >= 3]
    return len(alpha_words) >= _MIN_ALPHA_WORDS


def extract_pdf_title_guess(data: bytes) -> str | None:
    """A PDF has no <title> tag equivalent that's reliably populated, so a
    connector needs SOME title source -- this takes the first plausible
    non-blank line of the first page's text (real government notices
    verified live during development typically open with a line like
    "Expression of Interest" or the scheme name), never invented. Returns
    None (letting the caller fall back to the page URL) rather than an
    empty, implausibly short/long, or reference-number-only line."""
    reader = _open_reader(data)
    if reader is None or not reader.pages:
        return None
    first_page_text = reader.pages[0].extract_text() or ""
    for raw_line in first_page_text.splitlines():
        line = raw_line.strip()
        if _MIN_TITLE_LEN <= len(line) <= _MAX_TITLE_LEN and _looks_like_a_title(line):
            return line
    return None
