"""Currency/monetary field cleaning for USD-denominated award amounts.

All amounts in this dataset are US dollars from US federal/foundation programs
(see docs/data_audit_report.md). They are cleaned to numeric form for relative
bucketing (small/medium/large) only -- never presented as INR-equivalent or as
a funding expectation for Indian NGOs.
"""
import numpy as np
import pandas as pd

_PLACEHOLDER_VALUES = {"", "n/a", "na", "none", "null", "-", "tbd"}


def clean_money_value(value):
    """Parse a single monetary field: strip $ and commas, coerce to float.

    Unparseable values become NaN and are flagged, not silently zeroed --
    a missing award ceiling must never be displayed or treated as $0.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in _PLACEHOLDER_VALUES or pd.isna(value):
        return np.nan
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_money_column(series: pd.Series) -> pd.Series:
    return series.apply(clean_money_value)


def clean_all_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["award_ceiling", "award_floor", "estimated_total_program_funding"]:
        if col in df.columns:
            df[f"{col}_parse_failed"] = df[col].notna() & clean_money_column(df[col]).isna()
            df[col] = clean_money_column(df[col])
    return df


def award_size_bucket(df: pd.DataFrame, column: str = "award_ceiling") -> pd.Series:
    """Bucket award amounts into small/medium/large RELATIVE to this corpus's
    own distribution (quartiles). This is explicitly a relative signal for
    ranking/filtering -- it does not imply any INR equivalence (see Section C.4
    of the project blueprint)."""
    values = df[column]
    valid = values.dropna()
    if valid.empty:
        return pd.Series(["unknown"] * len(df), index=df.index)

    q1, q2 = valid.quantile([1 / 3, 2 / 3])

    def bucket(v):
        if pd.isna(v):
            return "unknown"
        if v <= q1:
            return "small"
        if v <= q2:
            return "medium"
        return "large"

    return values.apply(bucket)
