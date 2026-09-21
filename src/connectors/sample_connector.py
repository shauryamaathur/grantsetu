"""A SAMPLE/TEST connector that reads a local JSON fixture
(data/connectors/sample_source.json) instead of calling any real external
API. It exists to prove the sync -> dedup -> new-grant-detection ->
notification pipeline actually works end-to-end, WITHOUT pretending to have
a real, live Indian (or any) grant-data source connected -- which the
project has explicitly and repeatedly committed not to fake.

Every record this connector returns is marked `is_sample_data = True` at
the connector level, and `sync_grants.py` copies that flag onto the
resulting `ImportedGrant` rows, so the API and frontend can never present
these as real opportunities.
"""
import json
from pathlib import Path

from src.connectors.base import GrantSourceConnector, RawGrantRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "data" / "connectors" / "sample_source.json"


class SampleFixtureConnector(GrantSourceConnector):
    connector_type = "sample_fixture"
    is_sample_data = True

    def is_configured(self) -> bool:
        return FIXTURE_PATH.exists()

    def fetch_records(self, provider=None) -> list[RawGrantRecord]:
        # `provider` is accepted for interface compatibility (see
        # GrantSourceConnector.fetch_records) but unused -- fixture data is
        # already structured, nothing to extract.
        if not self.is_configured():
            from src.connectors.base import ConnectorNotConfiguredError

            raise ConnectorNotConfiguredError(f"Sample fixture not found at {FIXTURE_PATH}")

        payload = json.loads(FIXTURE_PATH.read_text())
        return [
            RawGrantRecord(
                external_id=rec["external_id"],
                title=rec["title"],
                funder=rec.get("funder"),
                description=rec.get("description"),
                eligibility_text=rec.get("eligibility_text"),
                category=rec.get("category"),
                deadline=rec.get("deadline"),
                url=rec.get("url"),
            )
            for rec in payload["records"]
        ]
