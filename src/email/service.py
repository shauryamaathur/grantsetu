"""Email provider abstraction.

`get_email_provider()` is the single factory every caller uses (the
subscription API and the notification/sync logic) -- swapping providers is
a one-line env-var change, never a code change.

- `BrevoEmailProvider` is the PRIMARY, free-first real provider (Brevo's
  free tier sends real transactional email with no credit card required),
  used when `EMAIL_PROVIDER=brevo` and `BREVO_API_KEY` is set.
- `ResendEmailProvider` is a supported alternative (`EMAIL_PROVIDER=resend`).
- `MockEmailProvider` is the EXPLICIT development fallback: it never
  contacts a real network service, logs the full email content instead,
  and every result it returns is honestly labelled `sent_mock` -- never
  presented as a real delivery. It is used whenever no real provider is
  configured, or a configured one is missing required settings; in both
  cases a warning is logged so the gap is visible in server logs, not
  silently hidden.

Nothing in this module ever silently reports success for an email that
was not actually sent.
"""
import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("grantsetu.email")

REQUEST_TIMEOUT = 10.0


@dataclass
class EmailSendResult:
    status: str  # "sent_mock" | "sent" | "failed"
    provider: str
    error: str | None = None
    provider_message_id: str | None = None


@dataclass
class ProviderHealth:
    ok: bool
    provider: str
    message: str


class EmailProvider:
    name = "base"

    def send(self, to_email: str, subject: str, body_text: str, body_html: str | None = None) -> EmailSendResult:
        raise NotImplementedError

    def health_check(self) -> ProviderHealth:
        """Cheap call that verifies the provider is actually reachable and
        the credentials are valid, WITHOUT sending a real email. Default
        implementation: no real check possible, report unknown."""
        return ProviderHealth(ok=True, provider=self.name, message="No live health check implemented for this provider.")


class MockEmailProvider(EmailProvider):
    """Explicit local-development fallback. Never contacts a real network
    service. Logs the full email so a developer can see exactly what would
    have been sent, and reports it as `sent_mock`, not `sent`."""

    name = "mock"

    def send(self, to_email: str, subject: str, body_text: str, body_html: str | None = None) -> EmailSendResult:
        logger.info(
            "[MOCK EMAIL -- NOT DELIVERED] to=%s subject=%r\n--- body ---\n%s\n------------",
            to_email, subject, body_text,
        )
        return EmailSendResult(status="sent_mock", provider=self.name)

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            ok=False,
            provider=self.name,
            message="Development mode: emails are logged, not delivered. Configure EMAIL_PROVIDER=brevo (or resend) to send real email.",
        )


class BrevoEmailProvider(EmailProvider):
    """Real integration with Brevo's transactional email API
    (https://developers.brevo.com/reference/sendtransacemail). Brevo's free
    tier (as of this writing) sends real email with no credit card required,
    making it the free-first default for this project. Requires
    BREVO_API_KEY, EMAIL_FROM_ADDRESS, and EMAIL_FROM_NAME."""

    name = "brevo"
    SEND_URL = "https://api.brevo.com/v3/smtp/email"
    ACCOUNT_URL = "https://api.brevo.com/v3/account"

    def __init__(self, api_key: str, from_email: str, from_name: str, reply_to: str | None = None):
        if not api_key:
            raise ValueError("BREVO_API_KEY is required to use BrevoEmailProvider")
        if not from_email:
            raise ValueError("EMAIL_FROM_ADDRESS is required to use BrevoEmailProvider")
        self.api_key = api_key
        self.from_email = from_email
        self.from_name = from_name or "GrantSetu"
        self.reply_to = reply_to

    def _headers(self) -> dict:
        return {"api-key": self.api_key, "Content-Type": "application/json", "Accept": "application/json"}

    def send(self, to_email: str, subject: str, body_text: str, body_html: str | None = None) -> EmailSendResult:
        payload = {
            "sender": {"name": self.from_name, "email": self.from_email},
            "to": [{"email": to_email}],
            "subject": subject,
            "htmlContent": body_html or f"<pre>{body_text}</pre>",
            "textContent": body_text,
        }
        if self.reply_to:
            payload["replyTo"] = {"email": self.reply_to}

        try:
            response = httpx.post(self.SEND_URL, headers=self._headers(), json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json() if response.content else {}
            return EmailSendResult(status="sent", provider=self.name, provider_message_id=data.get("messageId"))
        except httpx.TimeoutException as exc:
            logger.error("Brevo send timed out for %s: %s", to_email, exc)
            return EmailSendResult(status="failed", provider=self.name, error="Provider request timed out")
        except httpx.HTTPStatusError as exc:
            # Never log the payload (contains the recipient's email + message
            # body) -- log only the provider's status/response for diagnosis.
            logger.error("Brevo send failed for a recipient: HTTP %s -- %s", exc.response.status_code, exc.response.text[:300])
            return EmailSendResult(status="failed", provider=self.name, error=f"Brevo API error (HTTP {exc.response.status_code})")
        except Exception as exc:  # noqa: BLE001 -- an email failure must never crash the caller
            logger.error("Brevo send failed for a recipient: %s", exc)
            return EmailSendResult(status="failed", provider=self.name, error=str(exc))

    def health_check(self) -> ProviderHealth:
        try:
            response = httpx.get(self.ACCOUNT_URL, headers=self._headers(), timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return ProviderHealth(ok=True, provider=self.name, message="Brevo API key is valid and reachable.")
            return ProviderHealth(ok=False, provider=self.name, message=f"Brevo rejected the request (HTTP {response.status_code}).")
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(ok=False, provider=self.name, message=f"Could not reach Brevo: {exc}")


class ResendEmailProvider(EmailProvider):
    """Real integration with Resend's transactional email API. Requires
    RESEND_API_KEY and RESEND_FROM_EMAIL."""

    name = "resend"
    API_URL = "https://api.resend.com/emails"

    def __init__(self, api_key: str, from_email: str):
        if not api_key:
            raise ValueError("RESEND_API_KEY is required to use ResendEmailProvider")
        if not from_email:
            raise ValueError("RESEND_FROM_EMAIL is required to use ResendEmailProvider")
        self.api_key = api_key
        self.from_email = from_email

    def send(self, to_email: str, subject: str, body_text: str, body_html: str | None = None) -> EmailSendResult:
        try:
            response = httpx.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "from": self.from_email,
                    "to": [to_email],
                    "subject": subject,
                    "text": body_text,
                    "html": body_html or f"<pre>{body_text}</pre>",
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            return EmailSendResult(status="sent", provider=self.name, provider_message_id=data.get("id"))
        except Exception as exc:  # noqa: BLE001 -- an email failure must never crash the caller
            logger.error("Resend send failed for a recipient: %s", exc)
            return EmailSendResult(status="failed", provider=self.name, error=str(exc))

    def health_check(self) -> ProviderHealth:
        # Resend has no dedicated lightweight health endpoint documented;
        # report configured-but-unverified rather than fabricate a check.
        return ProviderHealth(ok=True, provider=self.name, message="Resend API key is configured (not actively verified by a health check call).")


def get_email_provider() -> EmailProvider:
    """Factory used everywhere an email needs to be sent. Reads
    EMAIL_PROVIDER from the environment (default: "mock"). Falls back to
    mock with a logged warning if a real provider is requested but not
    fully configured -- it never silently pretends a real provider is
    active."""
    provider_name = os.environ.get("EMAIL_PROVIDER", "mock").lower()

    if provider_name == "brevo":
        api_key = os.environ.get("BREVO_API_KEY", "")
        from_email = os.environ.get("EMAIL_FROM_ADDRESS", "")
        from_name = os.environ.get("EMAIL_FROM_NAME", "GrantSetu")
        reply_to = os.environ.get("EMAIL_REPLY_TO") or None
        if api_key and from_email:
            return BrevoEmailProvider(api_key=api_key, from_email=from_email, from_name=from_name, reply_to=reply_to)
        logger.warning(
            "EMAIL_PROVIDER=brevo but BREVO_API_KEY/EMAIL_FROM_ADDRESS are not both set; "
            "falling back to MockEmailProvider. No real email will be sent."
        )

    elif provider_name == "resend":
        api_key = os.environ.get("RESEND_API_KEY", "")
        from_email = os.environ.get("RESEND_FROM_EMAIL", "") or os.environ.get("EMAIL_FROM_ADDRESS", "")
        if api_key and from_email:
            return ResendEmailProvider(api_key=api_key, from_email=from_email)
        logger.warning(
            "EMAIL_PROVIDER=resend but RESEND_API_KEY/RESEND_FROM_EMAIL (or EMAIL_FROM_ADDRESS) "
            "are not both set; falling back to MockEmailProvider. No real email will be sent."
        )

    return MockEmailProvider()
