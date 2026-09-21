"""Lightweight schema creation for the MVP (no Alembic -- see docs/architecture.md
for why a full migration framework isn't justified at this scale).

`create_all_tables()` (SQLAlchemy's `Base.metadata.create_all`) only creates
tables that don't exist yet -- it never alters an existing table, so adding a
column to a model that already shipped needs a manual, idempotent
ALTER TABLE step. `_add_missing_columns()` is that step: it inspects each
table's real columns and adds any that are missing, one at a time, ignoring
"duplicate column" errors so it's safe to call on every startup."""
from sqlalchemy import inspect, text

from api.db.models import (  # noqa: F401
    AIProviderConfig,
    EmailLog,
    GrantSource,
    ImportedGrant,
    NGOProfile,
    OpportunityChange,
    OpportunityFeedback,
    RefreshJob,
    SavedOpportunity,
    SearchHistory,
    Subscriber,
    User,
    UserSession,
    WebsiteAnalysis,
)
from api.db.session import Base, engine

# (table, column, DDL type + default) -- append here whenever a column is
# added to an existing model instead of a brand-new table.
_ADDED_COLUMNS = [
    ("saved_opportunities", "opportunity_source", "VARCHAR DEFAULT 'historical'"),
    ("saved_opportunities", "opportunity_ref", "VARCHAR"),
    ("grant_sources", "category", "VARCHAR"),
    ("grant_sources", "country", "VARCHAR"),
    ("imported_grants", "country", "VARCHAR DEFAULT 'unknown'"),
    ("imported_grants", "updated_at", "DATETIME"),
    ("ngo_profiles", "user_id", "VARCHAR"),
]


def _dialect_ddl_type(ddl_type: str) -> str:
    """`_ADDED_COLUMNS` above was written against SQLite's type-affinity
    system, where "DATETIME" is understood but isn't a real column type on
    PostgreSQL (RDS/managed Postgres) -- only its own "TIMESTAMP". Every
    other type used in `_ADDED_COLUMNS` (VARCHAR, and VARCHAR with a quoted
    DEFAULT) is valid SQL on both engines, so this is the one substitution
    needed for the existing list to run unmodified against either
    database."""
    if engine.dialect.name != "sqlite" and ddl_type.upper().startswith("DATETIME"):
        return "TIMESTAMP" + ddl_type[len("DATETIME"):]
    return ddl_type


def _add_missing_columns():
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl_type in _ADDED_COLUMNS:
            if table not in existing_tables:
                continue  # brand-new table, create_all already gave it every column
            existing_columns = {c["name"] for c in inspector.get_columns(table)}
            if column in existing_columns:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {_dialect_ddl_type(ddl_type)}"))
        # Backfill opportunity_ref for pre-existing historical saves so old
        # rows are queryable the same way as new ones.
        if "saved_opportunities" in existing_tables:
            conn.execute(text(
                "UPDATE saved_opportunities SET opportunity_ref = CAST(opportunity_id AS VARCHAR) "
                "WHERE opportunity_ref IS NULL AND opportunity_source = 'historical'"
            ))
        if "imported_grants" in existing_tables:
            # Backfill pre-existing rows (created before `updated_at`/`country`
            # existed) so they aren't NULL: last_verified_at is the best
            # available approximation of "when this row's data was last
            # touched" for anything imported before this column existed.
            conn.execute(text(
                "UPDATE imported_grants SET updated_at = last_verified_at WHERE updated_at IS NULL"
            ))
            conn.execute(text(
                "UPDATE imported_grants SET country = 'unknown' WHERE country IS NULL"
            ))


def create_all_tables():
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


if __name__ == "__main__":
    create_all_tables()
    print("Tables created.")
