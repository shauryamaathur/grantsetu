"""Date parsing and derived date features for grant opportunity records."""
from datetime import datetime

import pandas as pd

DATE_COLUMNS = ["post_date", "close_date", "last_updated_date", "archive_date"]


def parse_date_column(series: pd.Series) -> pd.Series:
    """Coerce a column to datetime, turning unparseable values into NaT rather
    than raising, so a single malformed row does not break the whole pipeline."""
    return pd.to_datetime(series, errors="coerce")


def parse_all_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = parse_date_column(df[col])
    return df


def add_recency_and_expiry_features(df: pd.DataFrame, reference_date: datetime | None = None) -> pd.DataFrame:
    """Derive is_expired and days_until_close/days_since_posted, relative to
    `reference_date` (defaults to now). Rows with unparseable close_date are
    treated as unknown (not expired, not active) rather than guessed."""
    df = df.copy()
    ref = pd.Timestamp(reference_date or datetime.utcnow())

    df["is_expired"] = df["close_date"].notna() & (df["close_date"] < ref)
    df["days_until_close"] = (df["close_date"] - ref).dt.days
    df["days_since_posted"] = (ref - df["post_date"]).dt.days

    return df
