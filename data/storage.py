"""SQLite persistence layer via SQLAlchemy. All database access goes through this module."""

from datetime import datetime, date as date_type
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, text
from sqlalchemy.orm import Session

from config import DB_PATH

try:
    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        pass

except ImportError:
    from sqlalchemy.orm import declarative_base  # type: ignore[attr-defined]

    Base = declarative_base()

Path(__file__).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})


class Watchlist(Base):
    __tablename__ = "watchlist"
    id       = Column(Integer, primary_key=True, autoincrement=True)
    ticker   = Column(String, nullable=False, unique=True)
    added_at = Column(DateTime, default=datetime.utcnow)


class Portfolio(Base):
    __tablename__ = "portfolio"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    ticker       = Column(String, nullable=False)          # no unique — multiple lots allowed
    shares       = Column(Float,  nullable=False, default=1.0)
    price_bought = Column(Float,  nullable=False)
    date_bought  = Column(Date,   nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)


def _migrate() -> None:
    """Recreate the portfolio table if the legacy single-lot schema is detected.

    SQLite cannot drop inline UNIQUE constraints (they become autoindexes that
    are undeletable). The only fix is DROP + recreate, preserving existing rows.
    This runs BEFORE create_all so a fresh DB gets the right schema immediately.
    """
    with engine.connect() as conn:
        table_sql = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='portfolio'")
        ).scalar()

        if not table_sql:
            return  # Table doesn't exist yet — create_all will make it correctly

        needs_rebuild = (
            "UNIQUE" in table_sql.upper()       # old unique-per-ticker constraint
            or "shares" not in table_sql.lower() # old schema missing shares column
        )
        if not needs_rebuild:
            return

        # Preserve existing rows (shares defaults to 1.0)
        try:
            old_rows = conn.execute(
                text("SELECT id, ticker, price_bought, date_bought, created_at FROM portfolio")
            ).fetchall()
        except Exception:
            old_rows = []

        conn.execute(text("DROP TABLE portfolio"))
        conn.execute(text("""
            CREATE TABLE portfolio (
                id           INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
                ticker       VARCHAR  NOT NULL,
                shares       FLOAT    NOT NULL DEFAULT 1.0,
                price_bought FLOAT    NOT NULL,
                date_bought  DATE     NOT NULL,
                created_at   DATETIME
            )
        """))
        for row in old_rows:
            conn.execute(
                text("INSERT INTO portfolio (id, ticker, shares, price_bought, date_bought, created_at) "
                     "VALUES (:id, :ticker, 1.0, :pb, :db, :ca)"),
                {"id": row[0], "ticker": row[1], "pb": row[2],
                 "db": str(row[3]), "ca": str(row[4]) if row[4] else None},
            )
        conn.commit()


_migrate()                          # fix schema on existing DBs before create_all
Base.metadata.create_all(engine)    # create any missing tables (no-op if already correct)


# ---------------------------------------------------------------------------
# Watchlist CRUD
# ---------------------------------------------------------------------------

def get_watchlist() -> list[dict]:
    with Session(engine) as session:
        rows = session.query(Watchlist).order_by(Watchlist.added_at.desc()).all()
        return [{"id": r.id, "ticker": r.ticker, "added_at": r.added_at} for r in rows]


def add_to_watchlist(ticker: str) -> bool:
    """Returns True if added, False if the ticker already exists."""
    ticker = ticker.upper().strip()
    with Session(engine) as session:
        if session.query(Watchlist).filter_by(ticker=ticker).first():
            return False
        session.add(Watchlist(ticker=ticker))
        session.commit()
        return True


def remove_from_watchlist(ticker: str) -> None:
    """Remove from watchlist and cascade-delete ALL portfolio lots for this ticker."""
    ticker = ticker.upper().strip()
    with Session(engine) as session:
        session.query(Watchlist).filter_by(ticker=ticker).delete()
        session.query(Portfolio).filter_by(ticker=ticker).delete()
        session.commit()


# ---------------------------------------------------------------------------
# Portfolio CRUD  (each row = one purchase lot)
# ---------------------------------------------------------------------------

def get_portfolio() -> list[dict]:
    """Return all lots ordered by ticker then date."""
    with Session(engine) as session:
        rows = (
            session.query(Portfolio)
            .order_by(Portfolio.ticker, Portfolio.date_bought)
            .all()
        )
        return [
            {
                "id":           r.id,
                "ticker":       r.ticker,
                "shares":       r.shares,
                "price_bought": r.price_bought,
                "date_bought":  r.date_bought,
                "created_at":   r.created_at,
            }
            for r in rows
        ]


def add_portfolio_lot(
    ticker: str,
    shares: float,
    price_bought: float,
    date_bought: date_type,
) -> None:
    """Insert a new purchase lot. Multiple lots per ticker are fully supported."""
    ticker = ticker.upper().strip()
    with Session(engine) as session:
        session.add(
            Portfolio(ticker=ticker, shares=shares,
                      price_bought=price_bought, date_bought=date_bought)
        )
        session.commit()


def remove_portfolio_lot(lot_id: int) -> None:
    """Delete one specific lot by its primary key."""
    with Session(engine) as session:
        session.query(Portfolio).filter_by(id=lot_id).delete()
        session.commit()


def remove_from_portfolio(ticker: str) -> None:
    """Delete ALL lots for a ticker."""
    with Session(engine) as session:
        session.query(Portfolio).filter_by(ticker=ticker.upper().strip()).delete()
        session.commit()


def is_in_portfolio(ticker: str) -> bool:
    with Session(engine) as session:
        return (
            session.query(Portfolio).filter_by(ticker=ticker.upper().strip()).first()
            is not None
        )


def get_lots_for_ticker(ticker: str) -> list[dict]:
    """Return all lots for a single ticker ordered by purchase date."""
    with Session(engine) as session:
        rows = (
            session.query(Portfolio)
            .filter_by(ticker=ticker.upper().strip())
            .order_by(Portfolio.date_bought)
            .all()
        )
        return [
            {
                "id":           r.id,
                "shares":       r.shares,
                "price_bought": r.price_bought,
                "date_bought":  r.date_bought,
            }
            for r in rows
        ]
