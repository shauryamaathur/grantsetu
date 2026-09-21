"""SQLAlchemy engine/session setup. Uses SQLite for the MVP (see blueprint
Section J.1: SQLite is fully sufficient for a portfolio demo; PostgreSQL is
a straightforward swap via DATABASE_URL for a public deployment)."""
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "processed" / "grantsetu.db"

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

connect_args = {"check_same_thread": False} if _IS_SQLITE else {}
# pool_pre_ping: issues a cheap "is this connection still alive" check
# before handing it out, so a connection RDS/a managed Postgres closed for
# being idle (or during a maintenance failover) is transparently replaced
# instead of surfacing as a random request-time OperationalError. SQLite
# has no such server-side idle-connection behavior, so this is skipped
# there (and pool_size/max_overflow don't apply to SQLite's default
# SingleThreadPool the way they do to a real client/server database).
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=not _IS_SQLITE,
    **({} if _IS_SQLITE else {"pool_size": 5, "max_overflow": 10, "pool_recycle": 1800}),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
