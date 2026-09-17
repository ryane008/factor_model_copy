"""
TimescaleDB/Postgres access for the sample prototype.

Host resolution is a single env lookup: docker-compose sets PGHOST=timescaledb
inside the jupyter container, and outside the container the default is localhost
(where compose publishes 5432). That covers Linux, macOS and Windows with no
branching, which is why there is no is_docker_environment() heuristic here --
the old one sniffed JUPYTER_SERVER_ROOT, which a *local* Jupyter also sets, so
running a notebook locally under Jupyter would try to reach host "timescaledb"
and fail.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_engine(echo: bool = False) -> Engine:
    """Build a SQLAlchemy engine from PG* environment variables."""
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE", "factor_db")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    # pre_ping recycles connections the server has already closed, which is what
    # produced the "server closed the connection unexpectedly" failure on the
    # long to_sql write in clean_characteristics.ipynb.
    return create_engine(url, future=True, echo=echo, pool_pre_ping=True)


def check_connection(engine: Engine) -> None:
    """Raise with a readable message if the database is not reachable."""
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1;"))
    except Exception as exc:  # noqa: BLE001 - we re-raise with context
        host = os.getenv("PGHOST", "localhost")
        raise RuntimeError(
            f"Could not reach Postgres at {host}. "
            "If you are outside the container, run `docker compose up -d` first."
        ) from exc
    print(f"✅ Connected to Postgres/TimescaleDB at {os.getenv('PGHOST', 'localhost')}")


def table_exists(engine: Engine, table: str) -> bool:
    """Whether ``table`` exists in the current schema search path."""
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text("SELECT to_regclass(:t) IS NOT NULL;"), {"t": table}
            ).scalar()
        )


def write_table(
    df: pd.DataFrame,
    table: str,
    engine: Engine,
    unique_on: list[str] | None = None,
    chunksize: int = 10_000,
) -> None:
    """
    Replace ``table`` with ``df`` and optionally enforce a uniqueness constraint.

    Always replaces rather than appends. The full-universe ``jkp_characteristics``
    table ended up with every row duplicated because the loader used
    ``if_exists="append"`` and was run twice; a unique index makes that mistake
    fail loudly instead of silently doubling the panel.
    """
    df.to_sql(
        table,
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=chunksize,
    )

    if unique_on:
        cols = ", ".join(f'"{c}"' for c in unique_on)
        with engine.begin() as conn:
            conn.execute(
                text(f'CREATE UNIQUE INDEX "{table}_pk" ON "{table}" ({cols});')
            )

    print(f"✅ Wrote {len(df):,} rows to {table}" + (f" (unique on {unique_on})" if unique_on else ""))


def read_table(table: str, engine: Engine) -> pd.DataFrame:
    """Read a whole table back into a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM "{table}";'), conn)
