"""Text normalization utilities for free-text grant fields."""
import re

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value) -> str | None:
    """Trim whitespace, collapse internal whitespace, and fix casing artifacts.

    Returns None for missing/empty values instead of the string "nan", so
    downstream code can distinguish "no value" from a real empty string.

    BUG FOUND (live-verified against the real 75,640-row corpus, not just a
    unit test): the old guard here was `isinstance(value, float) and
    pd.isna(value)`, which only catches a missing float (NaN). Columns are
    loaded with pandas nullable "string" dtype (see DTYPE_MAP in
    src/ingestion/load_data.py), whose missing marker is pandas.NA -- NOT a
    float -- so it slipped through, `str(pd.NA)` produced the literal text
    "<NA>", and that literal string then failed every subsequent "looks
    empty" check too (the set below has "na", not "<na>"). The result: real
    API responses returning the string "<NA>" as if it were a genuine value
    -- e.g. additional_information_url for 25,665 of 75,640 rows (34%) and
    agency_name for 44 rows. `pd.isna()` (unlike the old isinstance check)
    correctly handles NaN, None, pandas.NA, and pandas.NaT all in one call.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if text.strip().lower() in {"", "nan", "none", "n/a", "na", "<na>"}:
        return None
    text = text.replace(" ", " ")  # non-breaking space encoding artifact
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def normalize_text_column(series: pd.Series) -> pd.Series:
    return series.apply(normalize_text)


def combine_text_fields(row: pd.Series, columns: list[str]) -> str:
    """Concatenate several text columns into one field for TF-IDF vectorization."""
    parts = [str(row[col]) for col in columns if pd.notna(row.get(col))]
    return " ".join(parts)
