"""Tests for src/email/service.py: provider selection, the Brevo adapter's
real HTTP-call logic (mocked at the httpx boundary), and the honest-fallback
behavior when a provider is requested but not fully configured."""
import httpx
import pytest

from src.email.service import (
    BrevoEmailProvider,
    MockEmailProvider,
    ResendEmailProvider,
    get_email_provider,
)


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200, text=""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.content = b"x" if json_data is not None else b""
        self.text = text or str(json_data)

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.brevo.com/v3/smtp/email")
            raise httpx.HTTPStatusError("error", request=request, response=httpx.Response(self.status_code, request=request, text=self.text))


# --- Provider selection / factory ----------------------------------------

def test_get_email_provider_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("EMAIL_PROVIDER", raising=False)
    provider = get_email_provider()
    assert isinstance(provider, MockEmailProvider)


def test_get_email_provider_falls_back_to_mock_when_brevo_not_fully_configured(monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "brevo")
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("EMAIL_FROM_ADDRESS", raising=False)
    provider = get_email_provider()
    assert isinstance(provider, MockEmailProvider)  # never silently activates a half-configured provider


def test_get_email_provider_returns_brevo_when_fully_configured(monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "brevo")
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-fake-key")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "noreply@example.org")
    monkeypatch.setenv("EMAIL_FROM_NAME", "GrantSetu")
    provider = get_email_provider()
    assert isinstance(provider, BrevoEmailProvider)
    assert provider.from_email == "noreply@example.org"


def test_get_email_provider_falls_back_to_mock_when_resend_not_fully_configured(monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "resend")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    provider = get_email_provider()
    assert isinstance(provider, MockEmailProvider)


def test_mock_provider_never_reports_a_real_send():
    provider = MockEmailProvider()
    result = provider.send("someone@example.org", "Subject", "Body")
    assert result.status == "sent_mock"
    assert result.provider == "mock"


def test_mock_provider_health_check_reports_not_delivering():
    health = MockEmailProvider().health_check()
    assert health.ok is False
    assert "not delivered" in health.message.lower()


# --- Brevo adapter: real request shape, mocked at the network boundary ----

def test_brevo_provider_requires_api_key_and_from_email():
    with pytest.raises(ValueError):
        BrevoEmailProvider(api_key="", from_email="a@b.com", from_name="X")
    with pytest.raises(ValueError):
        BrevoEmailProvider(api_key="key", from_email="", from_name="X")


def test_brevo_send_success(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse({"messageId": "abc-123"})

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = BrevoEmailProvider(api_key="xkeysib-real", from_email="noreply@example.org", from_name="GrantSetu")
    result = provider.send("subscriber@example.org", "Confirm your email", "text body", "<p>html body</p>")

    assert result.status == "sent"
    assert result.provider == "brevo"
    assert result.provider_message_id == "abc-123"
    assert captured["url"] == BrevoEmailProvider.SEND_URL
    assert captured["headers"]["api-key"] == "xkeysib-real"
    assert captured["json"]["to"] == [{"email": "subscriber@example.org"}]
    assert captured["json"]["sender"] == {"name": "GrantSetu", "email": "noreply@example.org"}
    assert captured["json"]["htmlContent"] == "<p>html body</p>"


def test_brevo_send_never_exposes_api_key_in_error_path(monkeypatch, caplog):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(status_code=401, text="Unauthorized")

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = BrevoEmailProvider(api_key="xkeysib-secret-value", from_email="noreply@example.org", from_name="GrantSetu")
    result = provider.send("subscriber@example.org", "Subject", "Body")

    assert result.status == "failed"
    assert "xkeysib-secret-value" not in (result.error or "")
    assert "xkeysib-secret-value" not in caplog.text


def test_brevo_send_handles_timeout_gracefully(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise httpx.TimeoutException("simulated timeout")

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = BrevoEmailProvider(api_key="key", from_email="noreply@example.org", from_name="GrantSetu")
    result = provider.send("subscriber@example.org", "Subject", "Body")
    assert result.status == "failed"
    assert "timed out" in result.error.lower()


def test_brevo_health_check_ok(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=None: _FakeResponse({"email": "account@example.org"}))
    provider = BrevoEmailProvider(api_key="key", from_email="noreply@example.org", from_name="GrantSetu")
    health = provider.health_check()
    assert health.ok is True


def test_brevo_health_check_reports_failure_honestly(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=None: _FakeResponse(status_code=401))
    provider = BrevoEmailProvider(api_key="bad-key", from_email="noreply@example.org", from_name="GrantSetu")
    health = provider.health_check()
    assert health.ok is False


def test_resend_provider_still_works_as_an_alternative(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({"id": "resend-msg-1"})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = ResendEmailProvider(api_key="re_key", from_email="noreply@example.org")
    result = provider.send("someone@example.org", "Subject", "Body")
    assert result.status == "sent"
    assert result.provider == "resend"
