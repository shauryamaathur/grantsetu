"""Load the raw grants dataset into a pandas DataFrame with explicit dtypes."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "grants_raw.csv"

DATE_COLUMNS = ["post_date", "close_date", "last_updated_date", "archive_date"]

# Monetary columns are loaded as strings first because the source CSV can contain
# currency symbols, commas, or "N/A" placeholders that pandas cannot parse directly.
MONEY_COLUMNS = ["award_ceiling", "award_floor", "estimated_total_program_funding"]

TEXT_COLUMNS = [
    "opportunity_title",
    "opportunity_number",
    "opportunity_category",
    "funding_instrument_type",
    "category_of_funding_activity",
    "cfda_numbers",
    "eligible_applicants",
    "eligible_applicants_type",
    "agency_code",
    "agency_name",
    "additional_information_url",
]

DTYPE_MAP = {col: "string" for col in TEXT_COLUMNS + MONEY_COLUMNS}
DTYPE_MAP["opportunity_id"] = "Int64"
DTYPE_MAP["expected_number_of_awards"] = "float64"


def load_raw_grants(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw grants CSV, preserving original values (no cleaning here).

    Raises FileNotFoundError with a clear message if the dataset is missing,
    rather than silently generating placeholder data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place the grants CSV at this path "
            "before running ingestion. No synthetic data will be generated."
        )

    df = pd.read_csv(
        path,
        index_col=0,
        dtype=DTYPE_MAP,
        parse_dates=False,  # dates handled explicitly in src/cleaning/clean_dates.py
        keep_default_na=True,
    )
    return df


if __name__ == "__main__":
    frame = load_raw_grants()
    print(f"Loaded {len(frame):,} rows and {len(frame.columns)} columns from {RAW_DATA_PATH}")
    print(frame.dtypes)
