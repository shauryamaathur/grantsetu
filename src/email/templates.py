"""Email templates. No templating engine dependency -- these are a small,
fixed set of messages, so f-strings are enough and avoid adding Jinja2 (or
similar) for what amounts to one shared HTML wrapper plus three call sites.

Every email is a real, table-based, inline-CSS HTML document (see
_wrap_email) -- the layout Gmail/Outlook/Apple Mail/mobile clients actually
support, since none of them run a page's <style> block reliably and none
run JavaScript at all. A plain-text body is always generated alongside the
HTML one (every email provider in src/email/service.py sends both), both
for accessibility/spam-filtering and as the fallback a client with images/
HTML disabled will show.
"""
import html
import os
from datetime import datetime

_BRAND = "#1f3b57"
_INK = "#1c2430"
_MUTED = "#667085"
_BORDER = "#e4e7ec"
_BG = "#f4f5f7"
_SURFACE = "#ffffff"
_ACCENT = "#b5541f"
_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _frontend_base_url() -> str:
    # FRONTEND_BASE_URL takes precedence for backward compatibility, but
    # falls back to FRONTEND_URL (the same variable api/routers/auth.py and
    # api/config.py already read) so a deployment only has to set ONE
    # frontend-URL variable, not two that must be kept in sync.
    return os.environ.get("FRONTEND_BASE_URL") or os.environ.get("FRONTEND_URL", "http://localhost:5500")


def _e(value) -> str:
    """HTML-escape, tolerating None (renders as empty string) -- every
    piece of user/source-provided text (grant titles, funder names,
    descriptions...) goes through this before landing in an f-string, so a
    funder name containing `<`/`&` can't break the layout."""
    return html.escape(str(value)) if value else ""


def _wrap_email(preheader: str, body_html: str, unsubscribe_link: str) -> str:
    """The one shared branded shell every email in this module renders
    through -- header, content slot, footer. Table-based layout with every
    style inline (no <style> block relied upon) because that's what
    actually survives Gmail/Outlook's HTML sanitizers; a `max-width:600px`
    center-aligned table is the de facto standard that behaves correctly
    down to phone-width clients without any @media query needed."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="color-scheme" content="light">
<title>GrantSetu</title>
</head>
<body style="margin:0; padding:0; background-color:{_BG};">
<div style="display:none; max-height:0; max-width:0; overflow:hidden; opacity:0; mso-hide:all;">{_e(preheader)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:{_BG};">
<tr><td align="center" style="padding:32px 16px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="width:100%; max-width:600px; background-color:{_SURFACE}; border-radius:12px; overflow:hidden; border:1px solid {_BORDER};">
<tr>
<td style="background-color:{_BRAND}; padding:24px 32px;">
<span style="font-family:{_FONT}; font-size:19px; font-weight:700; color:#ffffff; letter-spacing:0.2px;">GrantSetu</span>
</td>
</tr>
<tr>
<td style="padding:32px; font-family:{_FONT};">
{body_html}
</td>
</tr>
<tr>
<td style="padding:22px 32px; background-color:{_BG}; border-top:1px solid {_BORDER};">
<p style="margin:0 0 10px; font-family:{_FONT}; font-size:12px; line-height:1.6; color:{_MUTED};">GrantSetu helps NGOs discover and track funding opportunities relevant to their work.</p>
<p style="margin:0; font-family:{_FONT}; font-size:12px; color:{_MUTED};"><a href="{unsubscribe_link}" style="color:{_MUTED}; text-decoration:underline;">Unsubscribe</a></p>
</td>
</tr>
</table>
</td></tr>
</table>
</body>
</html>"""


def _button(label: str, url: str) -> str:
    return (
        f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin:20px 0 4px;">'
        f'<tr><td style="border-radius:7px; background-color:{_BRAND};">'
        f'<a href="{url}" style="display:inline-block; padding:11px 22px; font-family:{_FONT}; '
        f'font-size:14px; font-weight:600; color:#ffffff; text-decoration:none; border-radius:7px;">'
        f"{_e(label)}</a></td></tr></table>"
    )


def verification_email(subscriber_email: str, verify_token: str, unsubscribe_token: str) -> tuple[str, str, str]:
    """The double-opt-in confirmation email sent on every new subscription
    (and every resend request). Returns (subject, text_body, html_body).
    The verify link points at the FRONTEND (not the raw API) so a clicking
    user gets a real page/confirmation rather than raw JSON -- the frontend
    then calls the verify API endpoint itself. See
    frontend/index.html's #view-verify handling.

    NOTE for tests (tests/conftest.py::extract_token): the raw token must
    keep appearing as a literal `verify_token=<value>` / `unsubscribe_token=
    <value>` query string in the PLAIN-TEXT body -- that's what the test
    suite's token extraction relies on, so text_body's link lines are
    intentionally left as plain URLs rather than shortened/styled."""
    verify_link = f"{_frontend_base_url()}/index.html?verify_token={verify_token}"
    unsubscribe_link = f"{_frontend_base_url()}/index.html?unsubscribe_token={unsubscribe_token}"

    subject = "Confirm your GrantSetu subscription"
    text_body = (
        f"Hi,\n\n"
        f"Please confirm the email address {subscriber_email} to start receiving GrantSetu "
        f"funding-opportunity alerts.\n\n"
        f"Confirm your email (link expires in 24 hours): {verify_link}\n\n"
        f"If you didn't request this, you can ignore this email -- no alerts will be sent until "
        f"the link above is used, and it stops working automatically after 24 hours.\n\n"
        f"Already sure you don't want this? Unsubscribe any time: {unsubscribe_link}\n"
    )

    body_html = f"""
<h1 style="margin:0 0 12px; font-family:{_FONT}; font-size:20px; font-weight:700; color:{_INK};">Confirm your subscription</h1>
<p style="margin:0 0 14px; font-family:{_FONT}; font-size:14px; line-height:1.6; color:{_INK};">Hi,</p>
<p style="margin:0 0 14px; font-family:{_FONT}; font-size:14px; line-height:1.6; color:{_INK};">
Please confirm <strong>{_e(subscriber_email)}</strong> to start receiving GrantSetu funding-opportunity alerts.
</p>
{_button("Confirm your email", verify_link)}
<p style="margin:18px 0 0; font-family:{_FONT}; font-size:12.5px; line-height:1.6; color:{_MUTED};">This link expires in 24 hours. If you didn't request this, you can safely ignore this email -- no alerts will be sent until it's used.</p>
"""
    html_body = _wrap_email("Confirm your GrantSetu subscription", body_html, unsubscribe_link)
    return subject, text_body, html_body


def _format_deadline(deadline: datetime | None) -> str | None:
    if deadline is None:
        return None
    return deadline.strftime("%-d %b %Y") if os.name != "nt" else deadline.strftime("%d %b %Y")


def _opportunity_card_html(grant: dict) -> str:
    """One opportunity as a self-contained, bordered card -- title, then
    only the meta facts that are actually present (never 'N/A'/'null'/an
    empty label), a short description, an optional 'why this matches'
    line, and a single CTA. `grant` keys: title, funder, deadline
    (datetime|None), country (str|None), description (str|None),
    eligibility_text (str|None), url (str|None), detail_link (str, always
    present -- the GrantSetu page for this opportunity), is_sample_data
    (bool), matched_focus_areas (list[str])."""
    meta_bits = []
    if grant.get("funder"):
        meta_bits.append(_e(grant["funder"]))
    deadline_str = _format_deadline(grant.get("deadline"))
    if deadline_str:
        meta_bits.append(f"Deadline {_e(deadline_str)}")
    country = grant.get("country")
    if country and country != "unknown":
        meta_bits.append(_e(country))
    meta_line = (
        f'<p style="margin:2px 0 10px; font-family:{_FONT}; font-size:13px; color:{_MUTED};">{" &nbsp;·&nbsp; ".join(meta_bits)}</p>'
        if meta_bits
        else ""
    )

    description = (grant.get("description") or "").strip()
    if len(description) > 220:
        description = description[:217].rsplit(" ", 1)[0] + "…"
    description_html = (
        f'<p style="margin:0 0 10px; font-family:{_FONT}; font-size:13.5px; line-height:1.55; color:{_INK};">{_e(description)}</p>'
        if description
        else ""
    )

    eligibility = (grant.get("eligibility_text") or "").strip()
    if len(eligibility) > 140:
        eligibility = eligibility[:137].rsplit(" ", 1)[0] + "…"
    eligibility_html = (
        f'<p style="margin:0 0 10px; font-family:{_FONT}; font-size:13px; color:{_MUTED};"><strong style="color:{_INK};">Who it\'s for:</strong> {_e(eligibility)}</p>'
        if eligibility
        else ""
    )

    matched = grant.get("matched_focus_areas") or []
    match_html = (
        f'<p style="margin:0 0 10px; font-family:{_FONT}; font-size:13px; color:{_ACCENT};">Matches your focus on {_e(", ".join(matched))}</p>'
        if matched
        else ""
    )

    sample_html = (
        f'<p style="margin:0 0 10px; font-family:{_FONT}; font-size:12px; font-weight:600; color:{_ACCENT};">SAMPLE DATA -- for demonstration only, not a real opportunity</p>'
        if grant.get("is_sample_data")
        else ""
    )

    cta_url = grant.get("url") or grant.get("detail_link")
    cta_label = "View Opportunity" if grant.get("url") else "View on GrantSetu"

    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin:0 0 16px; border:1px solid {_BORDER}; border-radius:10px;">
<tr><td style="padding:18px 20px;">
<p style="margin:0 0 4px; font-family:{_FONT}; font-size:15.5px; font-weight:700; color:{_INK}; line-height:1.4;">{_e(grant["title"])}</p>
{meta_line}
{sample_html}
{description_html}
{eligibility_html}
{match_html}
<a href="{cta_url}" style="display:inline-block; margin-top:4px; font-family:{_FONT}; font-size:13.5px; font-weight:600; color:{_BRAND}; text-decoration:none; border-bottom:1.5px solid {_BRAND};">{cta_label} &rarr;</a>
</td></tr>
</table>"""


def _opportunity_card_text(grant: dict) -> str:
    lines = [f"- {grant['title']}"]
    if grant.get("funder"):
        lines.append(f"  Funder: {grant['funder']}")
    deadline_str = _format_deadline(grant.get("deadline"))
    if deadline_str:
        lines.append(f"  Deadline: {deadline_str}")
    country = grant.get("country")
    if country and country != "unknown":
        lines.append(f"  Location: {country}")
    matched = grant.get("matched_focus_areas") or []
    if matched:
        lines.append(f"  Matches your focus on: {', '.join(matched)}")
    if grant.get("is_sample_data"):
        lines.append("  [SAMPLE DATA -- for demonstration only, not a real opportunity]")
    link = grant.get("url") or grant.get("detail_link")
    lines.append(f"  Link: {link}")
    return "\n".join(lines)


def new_grant_digest_email(
    subscriber_email: str,
    grants: list[dict],
    unsubscribe_token: str,
) -> tuple[str, str, str]:
    """New-opportunity notification -- ONE email per subscriber per
    notification run, covering every new grant that matched them (never
    one email per grant; see src/email/notify.py::notify_subscribers_of_new_grants
    for why that used to spam multiple emails for a single sync). `grants`
    is a list of dicts shaped per _opportunity_card_html's docstring, in
    the order they should be shown (most relevant/soonest-first, decided
    by the caller)."""
    unsubscribe_link = f"{_frontend_base_url()}/index.html?unsubscribe_token={unsubscribe_token}"
    n = len(grants)

    if n == 1:
        subject = f"New funding opportunity: {grants[0]['title']}"
        intro = "A new opportunity matching your NGO's focus areas was just added to GrantSetu:"
    else:
        subject = f"{n} new funding opportunities on GrantSetu"
        intro = f"{n} new opportunities matching your NGO's focus areas were just added to GrantSetu:"

    text_body = (
        f"Hi,\n\n{intro}\n\n"
        + "\n\n".join(_opportunity_card_text(g) for g in grants)
        + f"\n\nUnsubscribe any time: {unsubscribe_link}\n"
    )

    cards_html = "".join(_opportunity_card_html(g) for g in grants)
    body_html = f"""
<h1 style="margin:0 0 6px; font-family:{_FONT}; font-size:20px; font-weight:700; color:{_INK};">{"New funding opportunity" if n == 1 else f"{n} new funding opportunities"}</h1>
<p style="margin:0 0 20px; font-family:{_FONT}; font-size:14px; line-height:1.6; color:{_MUTED};">{_e(intro)}</p>
{cards_html}
"""
    html_body = _wrap_email(subject, body_html, unsubscribe_link)
    return subject, text_body, html_body
