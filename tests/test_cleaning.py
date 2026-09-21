import pandas as pd

from src.cleaning.clean_currency import clean_money_value, award_size_bucket
from src.cleaning.clean_dates import add_recency_and_expiry_features, parse_date_column
from src.cleaning.clean_text import normalize_text
from src.cleaning.pipeline import clean_dataset, deduplicate


def test_clean_money_value_strips_symbols_and_commas():
    assert clean_money_value("$1,200,000") == 1200000.0


def test_clean_money_value_handles_placeholders():
    assert pd.isna(clean_money_value("N/A"))
    assert pd.isna(clean_money_value(""))
    assert pd.isna(clean_money_value(None))


def test_clean_money_value_handles_unparseable_text():
    assert pd.isna(clean_money_value("varies by award"))


def test_normalize_text_collapses_whitespace_and_trims():
    assert normalize_text("  Environmental   Research  ") == "Environmental Research"


def test_normalize_text_returns_none_for_missing():
    assert normalize_text(None) is None
    assert normalize_text("nan") is None
    assert normalize_text(float("nan")) is None


def test_normalize_text_returns_none_for_pandas_na():
    """Regression test for a real bug found via a live smoke test against the
    75,640-row corpus: pandas.NA (the missing marker for nullable "string"-
    dtype columns, used for every text column per DTYPE_MAP in
    src/ingestion/load_data.py) is not a float, so the old
    `isinstance(value, float) and pd.isna(value)` guard let it through,
    str()'d it to the literal text "<NA>", and that literal then leaked into
    real API responses as if it were genuine data (34% of
    additional_information_url values in the real dataset)."""
    assert normalize_text(pd.NA) is None
    assert normalize_text(pd.NaT) is None


def test_normalize_text_column_handles_pandas_string_dtype_with_na():
    """End-to-end version of the above: a real pandas 'string'-dtype Series
    (as produced by DTYPE_MAP) with a missing value must normalize to None,
    not the literal string "<NA>"."""
    from src.cleaning.clean_text import normalize_text_column

    s = pd.array(["Some Agency", pd.NA, "  Another One  "], dtype="string")
    result = normalize_text_column(pd.Series(s))
    assert result.tolist() == ["Some Agency", None, "Another One"]


def test_parse_date_column_coerces_invalid_to_nat():
    s = pd.Series(["2020-01-01", "not-a-date", None])
    parsed = parse_date_column(s)
    assert parsed.iloc[0] == pd.Timestamp("2020-01-01")
    assert pd.isna(parsed.iloc[1])
    assert pd.isna(parsed.iloc[2])


def test_is_expired_flag_reflects_close_date():
    df = pd.DataFrame({
        "post_date": pd.to_datetime(["2020-01-01", "2099-01-01"]),
        "close_date": pd.to_datetime(["2020-06-01", "2099-06-01"]),
    })
    result = add_recency_and_expiry_features(df, reference_date=pd.Timestamp("2026-01-01"))
    assert result["is_expired"].tolist() == [True, False]


def test_award_size_bucket_relative_quantiles():
    df = pd.DataFrame({"award_ceiling": [10, 20, 30, 40, 50, 60, 70, 80, 90]})
    buckets = award_size_bucket(df, "award_ceiling")
    assert set(buckets.unique()) <= {"small", "medium", "large"}
    assert buckets.iloc[0] == "small"
    assert buckets.iloc[-1] == "large"


def test_award_size_bucket_missing_value_is_unknown():
    df = pd.DataFrame({"award_ceiling": [10, 20, None]})
    buckets = award_size_bucket(df, "award_ceiling")
    assert buckets.iloc[2] == "unknown"


def test_deduplicate_removes_exact_duplicate_rows():
    df = pd.DataFrame({
        "opportunity_id": [1, 1, 2],
        "opportunity_title": ["A", "A", "B"],
        "last_updated_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
    })
    deduped, stats = deduplicate(df)
    assert len(deduped) == 2
    assert stats["exact_duplicate_rows_dropped"] == 1


def test_clean_dataset_drops_rows_missing_id_or_title():
    raw = pd.DataFrame({
        "opportunity_id": [1, None, 3],
        "opportunity_title": ["Valid title", "Missing id", None],
        "opportunity_category": ["Education", "Health", "Health"],
        "funding_instrument_type": [None, None, None],
        "category_of_funding_activity": [None, None, None],
        "eligible_applicants": [None, None, None],
        "eligible_applicants_type": [None, None, None],
        "agency_name": [None, None, None],
        "additional_information_url": [None, None, None],
        "post_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "close_date": ["2020-06-01", "2020-06-01", "2020-06-01"],
        "last_updated_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "archive_date": ["2020-06-01", "2020-06-01", "2020-06-01"],
        "award_ceiling": ["$1,000", None, None],
        "award_floor": [None, None, None],
        "estimated_total_program_funding": [None, None, None],
    })
    cleaned, stats = clean_dataset(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["opportunity_title"] == "Valid title"
