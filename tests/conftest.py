"""Shared pytest fixtures: a tiny synthetic opportunity corpus (used in place
of the full 75k-row dataset so tests are fast/deterministic) and a FastAPI
TestClient wired to a temporary SQLite database, reused across all API-level
test files."""
import os
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
os.environ["EMAIL_PROVIDER"] = "mock"
# A real, valid Fernet key generated once for the test session -- without
# this, api/main.py's load_dotenv(override=False) (deliberately) leaves
# whatever is in a developer's real .env untouched if THIS var isn't
# already set here first, and a real .env commonly has a placeholder like
# "YOUR_FERNET_ENCRYPTION_KEY_HERE" (not a valid Fernet key), which would
# break every AI-config test with a ValueError unrelated to the code being
# tested. Setting a real key here keeps the test suite deterministic and
# isolated from whatever a developer's local .env happens to contain.
from cryptography.fernet import Fernet  # noqa: E402

os.environ["GRANTSETU_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from api.main import app  # noqa: E402
from src.features.tfidf_index import build_tfidf_index  # noqa: E402
from src.ranking import corpus as corpus_module  # noqa: E402


def _fake_corpus():
    df = pd.DataFrame({
        "opportunity_id": [101, 102, 103],
        "opportunity_title": ["Environmental Youth Grant", "Rural Health Initiative", "Urban Roads Fund"],
        "opportunity_number": ["A-1", "A-2", "A-3"],
        "opportunity_category": ["Environment", "Health", "Infrastructure"],
        "funding_instrument_type": ["Grant", "Grant", "Grant"],
        "category_of_funding_activity": ["Environment", "Health", "Infrastructure"],
        "eligible_applicants": ["Nonprofits", "Nonprofits", "State governments"],
        "eligible_applicants_type": ["Non-Government Organization", "Non-Government Organization", "State"],
        "agency_name": ["Agency A", "Agency B", "Agency C"],
        "post_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2010-01-01"]),
        "close_date": pd.to_datetime(["2030-01-01", "2030-01-01", "2011-01-01"]),
        "award_ceiling": [50000.0, 80000.0, 2000000.0],
        "award_floor": [10000.0, 20000.0, 500000.0],
        "estimated_total_program_funding": [50000.0, 80000.0, 2000000.0],
        "additional_information_url": [None, None, None],
        "is_expired": [False, False, True],
        "is_recent": [True, True, False],
        "award_size_bucket": ["small", "medium", "large"],
        "combined_text": [
            "Environmental Youth Grant Environment Environment Agency A",
            "Rural Health Initiative Health Health Agency B",
            "Urban Roads Fund Infrastructure Infrastructure Agency C",
        ],
        "eligibility_text": ["nonprofits non-government organization", "nonprofits non-government organization", "state governments state"],
    })
    c = corpus_module.Corpus()
    c.df = df
    c.index = build_tfidf_index(df)
    return c


@pytest.fixture(autouse=True)
def patch_corpus(monkeypatch):
    fake = _fake_corpus()
    monkeypatch.setattr(corpus_module, "_corpus_singleton", fake)
    monkeypatch.setattr(corpus_module, "get_corpus", lambda: fake)
    import api.main as main_module
    import api.routers.opportunities as opp_router
    import api.routers.saved as saved_router
    monkeypatch.setattr(opp_router, "get_corpus", lambda: fake)
    monkeypatch.setattr(saved_router, "get_corpus", lambda: fake)
    monkeypatch.setattr(main_module, "get_corpus", lambda: fake)
    yield fake


@pytest.fixture
def client():
    from api.db.migrations import create_all_tables
    create_all_tables()
    with TestClient(app) as c:
        yield c


class _CapturingEmailProvider:
    """A test double that records every email 'sent' through it, so tests
    can extract the REAL raw verification/unsubscribe token from the email
    body -- exactly how a real user would get it (the token is deliberately
    NOT readable from the database once src/email/tokens.py hashes it at
    rest; see api/db/models.py's Subscriber docstring)."""

    name = "test-capture"

    def __init__(self):
        self.sent = []

    def send(self, to_email, subject, body_text, body_html=None):
        from src.email.service import EmailSendResult

        self.sent.append({"to": to_email, "subject": subject, "text": body_text, "html": body_html})
        return EmailSendResult(status="sent_mock", provider=self.name)


@pytest.fixture
def capture_emails(monkeypatch):
    """Patches src.email.notify.get_email_provider (the only place email-
    sending functions look up a provider) to return one shared capturing
    instance for the duration of the test."""
    provider = _CapturingEmailProvider()
    import src.email.notify as notify_module
    monkeypatch.setattr(notify_module, "get_email_provider", lambda: provider)
    yield provider


def extract_token(email_text: str, param_name: str) -> str:
    """Pulls e.g. `verify_token=<...>` or `unsubscribe_token=<...>` out of a
    captured email body -- the same way a user extracts it by clicking the
    real link."""
    import re

    match = re.search(rf"{param_name}=([^\s&]+)", email_text)
    assert match, f"{param_name} not found in email text: {email_text!r}"
    return match.group(1)
