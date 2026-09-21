"""Schema and structural validation for the raw grants dataset."""
from dataclasses import dataclass, field

import pandas as pd

EXPECTED_COLUMNS = [
    "opportunity_id",
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
    "post_date",
    "close_date",
    "last_updated_date",
    "archive_date",
    "award_ceiling",
    "award_floor",
    "estimated_total_program_funding",
    "expected_number_of_awards",
    "cost_sharing_or_matching_requirement",
    "additional_information_url",
]


@dataclass
class ValidationReport:
    row_count: int
    column_count: int
    missing_columns: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)
    duplicate_opportunity_ids: int = 0
    exact_duplicate_rows: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.missing_columns

    def summary(self) -> str:
        lines = [
            f"Rows: {self.row_count:,}",
            f"Columns: {self.column_count}",
            f"Missing expected columns: {self.missing_columns or 'none'}",
            f"Extra/unexpected columns: {self.extra_columns or 'none'}",
            f"Duplicate opportunity_id values: {self.duplicate_opportunity_ids:,}",
            f"Exact duplicate rows: {self.exact_duplicate_rows:,}",
        ]
        return "\n".join(lines)


def validate_schema(df: pd.DataFrame) -> ValidationReport:
    """Check the loaded dataframe against the documented schema.

    This does not raise on soft issues (duplicates); it reports them so the
    caller can decide whether to proceed, consistent with the blueprint's
    "audit rather than assume" instruction.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    dup_ids = 0
    if "opportunity_id" in df.columns:
        dup_ids = int(df["opportunity_id"].duplicated(keep=False).sum())

    exact_dupes = int(df.duplicated(keep=False).sum())

    return ValidationReport(
        row_count=len(df),
        column_count=len(df.columns),
        missing_columns=missing,
        extra_columns=extra,
        duplicate_opportunity_ids=dup_ids,
        exact_duplicate_rows=exact_dupes,
    )


if __name__ == "__main__":
    from src.ingestion.load_data import load_raw_grants

    report = validate_schema(load_raw_grants())
    print(report.summary())
