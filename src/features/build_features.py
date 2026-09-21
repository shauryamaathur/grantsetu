"""Structured + text feature engineering on the cleaned grants dataframe."""
import pandas as pd

from src.cleaning.clean_currency import award_size_bucket
from src.cleaning.clean_text import combine_text_fields

RECENCY_WINDOW_YEARS = 5

TEXT_FIELDS_FOR_MATCHING = [
    "opportunity_title",
    "opportunity_category",
    "category_of_funding_activity",
    "agency_name",
]


def add_award_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["award_size_bucket"] = award_size_bucket(df, "award_ceiling")
    return df


def add_recency_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cutoff_days = RECENCY_WINDOW_YEARS * 365
    df["is_recent"] = df["days_since_posted"].notna() & (
        df["days_since_posted"] <= cutoff_days
    )
    return df


def add_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = df.apply(
        lambda row: combine_text_fields(row, TEXT_FIELDS_FOR_MATCHING), axis=1
    )
    return df


def add_eligibility_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize eligibility fields into a single lowercase blob used by the
    rule-based eligibility filter (src/ranking/eligibility_rules.py)."""
    df = df.copy()
    df["eligibility_text"] = (
        df["eligible_applicants"].fillna("") + " " + df["eligible_applicants_type"].fillna("")
    ).str.lower().str.strip()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps. Expects a cleaned dataframe from
    src.cleaning.pipeline.clean_dataset."""
    df = add_award_bucket(df)
    df = add_recency_flag(df)
    df = add_combined_text(df)
    df = add_eligibility_text(df)
    return df


if __name__ == "__main__":
    from src.cleaning.pipeline import clean_dataset
    from src.ingestion.load_data import load_raw_grants

    raw = load_raw_grants()
    cleaned, _ = clean_dataset(raw)
    featured = build_features(cleaned)
    print(featured[["opportunity_title", "award_size_bucket", "is_recent", "combined_text"]].head())
