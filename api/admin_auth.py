"""Operator-only endpoint protection. Separate from api/routers/auth.py's
Google-sign-in User model on purpose -- that system authenticates NGO users;
this is a single shared operator secret for infra-facing endpoints (adding a
grant source, forcing a raw single-source sync, rebuilding the historical
index), the kind of thing an external cron/EventBridge rule or an operator's
terminal calls, not something an NGO user ever should."""
from fastapi import Header, HTTPException

from api.config import ADMIN_API_TOKEN, IS_PRODUCTION


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if ADMIN_API_TOKEN is None:
        if IS_PRODUCTION:
            # _check_production_config() in api/main.py already logs this
            # loudly at startup; still fail open rather than locking every
            # operator out of a server that was deployed without the token
            # set -- consistent with this app's existing philosophy of
            # degrading, not crashing, on optional-but-recommended config.
            return
        return
    if x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token header")
