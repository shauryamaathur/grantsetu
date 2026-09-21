"""End-to-end cleaning pipeline: raw dataframe -> analysis-ready dataframe."""
import pandas as pd

from src.cleaning.clean_currency import clean_all_money_columns
from src.cleaning.clean_dates import add_recency_and_expiry_features, parse_all_dates
from src.cleaning.clean_text import normalize_text_column

TEXT_COLUMNS_TO_NORMALIZE = [
    "opportunity_title",
    "opportunity_category",
    "funding_instrument_type",
    "category_of_funding_activity",
    "eligible_applicants",
    "eligible_applicants_type",
    "agency_name",
    "additional_information_url",
]


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resolve duplicate/near-duplicate opportunity records.

    Exact duplicate rows are dropped outright. Duplicate opportunity_id values
    are treated as amendments/reposts (documented, common in grant listings
    per the blueprint) -- we keep the most recently updated version of each id.
    """
    before = len(df)
    df = df.drop_duplicates()
    exact_dupes_dropped = before - len(df)

    dup_id_count = int(df["opportunity_id"].duplicated(keep=False).sum())
    if "last_updated_date" in df.columns:
        df = df.sort_values("last_updated_date").drop_duplicates(
            subset="opportunity_id", keep="last"
        )
    else:
        df = df.drop_duplicates(subset="opportunity_id", keep="last")

    stats = {
        "exact_duplicate_rows_dropped": exact_dupes_dropped,
        "duplicate_opportunity_id_rows_found": dup_id_count,
        "rows_after_dedup": len(df),
    }
    return df, stats


def clean_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline and return the cleaned dataframe plus a
    stats dict documenting what was done (for the data audit report)."""
    stats = {"rows_raw": len(raw_df)}

    df = raw_df.copy()

    for col in TEXT_COLUMNS_TO_NORMALIZE:
        if col in df.columns:
            df[col] = normalize_text_column(df[col])

    df = parse_all_dates(df)
    df = clean_all_money_columns(df)

    df, dedup_stats = deduplicate(df)
    stats.update(dedup_stats)

    df = add_recency_and_expiry_features(df)

    df = df.dropna(subset=["opportunity_id", "opportunity_title"])
    stats["rows_after_dropping_missing_id_or_title"] = len(df)

    for col in ["award_ceiling", "award_floor", "estimated_total_program_funding"]:
        stats[f"{col}_missing_pct"] = round(100 * df[col].isna().mean(), 2)

    for col in ["eligible_applicants", "eligible_applicants_type", "additional_information_url"]:
        if col in df.columns:
            stats[f"{col}_missing_pct"] = round(100 * df[col].isna().mean(), 2)

    stats["rows_expired"] = int(df["is_expired"].sum())
    stats["rows_expired_pct"] = round(100 * df["is_expired"].mean(), 2)

    return df.reset_index(drop=True), stats


if __name__ == "__main__":
    import json

    from src.ingestion.load_data import load_raw_grants

    raw = load_raw_grants()
    cleaned, run_stats = clean_dataset(raw)
    print(json.dumps(run_stats, indent=2, default=str))
