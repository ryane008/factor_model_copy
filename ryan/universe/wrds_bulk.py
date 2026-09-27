"""
Universe-scale WRDS pulls.

The sample's queries are ``WHERE id IN (220 ids)``. Here the universe is defined
by the data, so the panel is pulled by DATE RANGE -- every JKP US row passing the
universe screen -- one calendar year per query. Connection, CCM links and GICS
come straight from ryan/sample/wrds_io.py; only what has to change for scale is
here.

Everything is read-only. The window is anchored on explicit dates, never
``datetime.now() - months*30`` (the arithmetic that produced the 3-month panel).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

import config
import wrds_io


def latest_eom(conn) -> date:
    """Last month contrib.global_factor has for the US universe."""
    df = conn.raw_sql(f"SELECT MAX(eom) AS eom FROM contrib.global_factor WHERE {wrds_io._jkp_filter_sql()}")
    return pd.Timestamp(df["eom"].iloc[0]).date()


def pull_start(start_eom=None) -> date:
    """
    First month to pull. Earlier than START_EOM when the membership rule needs a
    rank date before it: a June-effective Russell reconstitution that is in force
    at START_EOM was ranked in the preceding May.
    """
    start = pd.Timestamp(start_eom or config.START_EOM)
    if getattr(config, "ESTU_RULE", "screens") != "rank":
        return start.date()
    year = start.year if start.month >= config.EFFECTIVE_MONTH else start.year - 1
    return (pd.Timestamp(year=year, month=config.RANK_MONTH, day=1) + pd.offsets.MonthEnd(0)).date()


def fetch_panel_range(conn, start, end, chars: list[str] | None = None) -> pd.DataFrame:
    """
    Every JKP US row passing the universe screen with ``start <= eom <= end``.

    One query per calendar year. Rows are kept even when ret_exc is null: a row
    with a cap and exposures but no return still supplies X(t-1) for the next
    month's regression, and step 4 drops it from the left-hand side itself.
    """
    chars = chars or config.characteristics_to_pull()
    cols = ", ".join(chars)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for y in range(start.year, end.year + 1, config.PULL_CHUNK_YEARS):
        lo = max(start, pd.Timestamp(year=y, month=1, day=1))
        hi = min(end, pd.Timestamp(year=y + config.PULL_CHUNK_YEARS - 1, month=12, day=31))
        sql = f"""
            SELECT id, eom, gvkey, permno, size_grp, {cols}
            FROM contrib.global_factor
            WHERE {wrds_io._jkp_filter_sql()}
              AND eom BETWEEN '{lo.date()}' AND '{hi.date()}'
        """
        df = conn.raw_sql(sql)
        print(f"   {lo.date()} → {hi.date()}: {len(df):>9,} rows, {df['id'].nunique():>6,} securities")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["eom"] = pd.to_datetime(out["eom"])
    dup = out.duplicated(["id", "eom"]).sum()
    if dup:
        raise AssertionError(f"{dup} duplicate (id, eom) rows from contrib.global_factor — "
                             "the JKP screen should make it unique; investigate before deduping")
    print(f"📈 {len(out):,} rows, {out['id'].nunique():,} securities, "
          f"{out['eom'].min().date()} → {out['eom'].max().date()}")
    return out


def fetch_ticker_history(conn, permnos) -> pd.DataFrame:
    """crsp.stocknames for many permnos, batched so no single IN list is enormous."""
    permnos = sorted({int(p) for p in permnos if pd.notna(p)})
    frames = []
    for i in range(0, len(permnos), config.PERMNO_BATCH):
        frames.append(wrds_io.fetch_ticker_history(conn, permnos[i:i + config.PERMNO_BATCH]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["permno", "ticker", "comnam", "namedt", "nameenddt"])
