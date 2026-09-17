"""
WRDS queries for the sample prototype.

Everything here is read-only and parameterized by the roster, so once
``01_build_sample_universe`` has picked 220 securities, every later pull is a
small ``WHERE id IN (...)`` query instead of a universe scan.
"""

from __future__ import annotations

import os

import pandas as pd

import config


def connect():
    """
    Open a WRDS connection from WRDS_USERNAME / WRDS_PASSWORD.

    Deliberately does not echo the credentials. The equivalent cell in
    charactersitics_loader.ipynb printed the password, and that output was
    committed to git.
    """
    import wrds  # imported lazily so the module can be used offline

    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")
    if not (username and password):
        raise RuntimeError(
            "WRDS_USERNAME / WRDS_PASSWORD not found in the environment. "
            "Copy .env.template to .env and fill them in."
        )

    print(f"🔐 Connecting to WRDS as {username}...")
    conn = wrds.Connection(wrds_username=username, wrds_password=password)
    print("🔗 WRDS connection established")
    return conn


def _jkp_filter_sql() -> str:
    """The shared JKP universe screen, as a SQL fragment."""
    clauses = [f"{k} = {v}" for k, v in config.JKP_FILTERS.items()]
    clauses.append(f"excntry = '{config.COUNTRY}'")
    return " AND ".join(clauses)


# ---------------------------------------------------------------------------
# Sampling frame
# ---------------------------------------------------------------------------


def fetch_anchor_cross_section(conn, anchor_eom=None) -> pd.DataFrame:
    """
    Every security in the JKP US universe alive at the anchor date.

    One row per ``id``, which is JKP's security-level key and the grain the whole
    pipeline uses. We do *not* sample on ticker: tickers get reused across firms
    (the old loader found BCS mapping to both LIFESTYLE RESTAURANT and BARCLAYS
    PLC), change over time, and are missing for a large share of crsp.stocknames.
    """
    anchor = anchor_eom or config.ANCHOR_EOM

    sql = f"""
        SELECT id, gvkey, permno, eom
        FROM contrib.global_factor
        WHERE {_jkp_filter_sql()}
          AND eom = '{anchor}'
    """
    df = conn.raw_sql(sql)
    print(f"📅 {len(df):,} securities alive at {anchor}")
    return df


def fetch_return_coverage(conn, anchor_eom=None) -> pd.DataFrame:
    """
    Count of non-null returns per security, from the anchor date forward.

    Aggregated server-side, so this returns ~tens of thousands of rows rather
    than the full panel. Counting ``ret_exc`` specifically (not ``*``) is the
    point: a month without a return is unusable in the cross-sectional
    regression, so only those months count toward the minimum-history screen.

    This is also what protects us from the failure that killed the full-universe
    pull -- rows sourced from Compustat with no CRSP link carry no ``ret_exc``,
    and firms made up mostly of those rows get screened out here.
    """
    anchor = anchor_eom or config.ANCHOR_EOM

    sql = f"""
        SELECT id,
               COUNT({config.RETURN_COL}) AS n_ret,
               COUNT(*)                   AS n_months,
               MIN(eom)                   AS first_eom,
               MAX(eom)                   AS last_eom
        FROM contrib.global_factor
        WHERE {_jkp_filter_sql()}
          AND eom >= '{anchor}'
        GROUP BY id
    """
    df = conn.raw_sql(sql)
    print(f"📊 Return coverage computed for {len(df):,} securities")
    return df


def fetch_gics(conn) -> pd.DataFrame:
    """
    Current GICS classification and company name, one row per gvkey.

    ``comp.company`` is the *current* header file, so this is as-of-today for
    every date in the panel. That is fine for a prototype but it is a real
    look-ahead over a 20-year panel: a firm that changed sector in 2015 is
    labelled with its 2026 sector back to 2005. ``comp.co_hgic`` carries the
    point-in-time history when we want to remove that.
    """
    sql = """
        SELECT gvkey, conm, gsector, ggroup, gind, gsubind
        FROM comp.company
        WHERE gsector IS NOT NULL
    """
    df = conn.raw_sql(sql)
    # comp.company is already one row per gvkey; assert rather than dedup blindly.
    dupes = df["gvkey"].duplicated().sum()
    if dupes:
        raise AssertionError(f"comp.company returned {dupes} duplicate gvkeys — investigate before deduping")
    print(f"🏭 GICS classifications for {len(df):,} gvkeys")
    return df


# ---------------------------------------------------------------------------
# Identifier links
# ---------------------------------------------------------------------------


def fetch_ccm_links(conn, require_usedflag: bool = True) -> pd.DataFrame:
    """
    CRSP/Compustat Merged links, *with their validity windows preserved*.

    The old query filtered only on ``linktype IN ('LU','LC')`` and called the
    result "active links" — but linktype is a link *quality* flag, not a date
    filter. With ``linkdt``/``linkenddt`` dropped, a gvkey that mapped to permno
    X in 1990 and permno Y in 2010 contributed both rows at every date, which is
    what made the merge fan 14,813 rows out to 47,448.

    Two things fix it:
      - ``linkprim IN ('P','C')`` keeps only the primary link, collapsing
        multi-share-class firms to one permno.
      - Returning ``linkdt``/``linkenddt`` lets the caller do an interval join on
        ``eom`` instead of a blind many-to-many merge on gvkey.

    A null ``linkenddt`` means the link is still open. It is deliberately left
    NULL here rather than coalesced to a sentinel date in SQL: '9999-12-31' is a
    perfectly valid Postgres date but overflows pandas' datetime64[ns] range,
    which tops out at 2262-04-11, and ``pd.to_datetime`` raises
    ``OutOfBoundsDatetime`` on it. ``sampling.attach_permno`` fills the null with
    an in-range sentinel instead.

    ``usedflag`` is selected but filtered in pandas rather than in SQL, and its
    distribution is printed. Two reasons: the column is spelled ``usedflag`` (an
    earlier version of this query said ``usedflg`` and failed), and its encoding
    varies between CRSP vintages -- so filtering here, after seeing the values,
    beats a SQL predicate that silently returns zero rows. Correctness does not
    actually rest on it: ``attach_permno`` restricts to the link window covering
    each ``eom``, ranks linkprim P over C and linktype LC over LU, and asserts one
    link per ``(id, eom)``.
    """
    sql = """
        SELECT gvkey,
               lpermno AS permno,
               lpermco AS permco,
               linkprim,
               linktype,
               usedflag,
               linkdt,
               linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """
    df = conn.raw_sql(sql)
    print(f"🔗 {len(df):,} links after the linktype/linkprim filter")

    if "usedflag" in df.columns:
        dist = df["usedflag"].value_counts(dropna=False).sort_index()
        print("   usedflag: " + ", ".join(f"{k!r}→{v:,}" for k, v in dist.items()))

        if require_usedflag:
            keep = pd.to_numeric(df["usedflag"], errors="coerce") == 1
            if keep.any():
                df = df[keep].copy()
                print(f"   kept {len(df):,} rows with usedflag == 1")
            else:
                print("   ⚠️  no rows have usedflag == 1 — leaving it unfiltered. "
                      "Check the encoding above before trusting the link set.")

    return df.drop(columns=["usedflag"], errors="ignore")


def fetch_ticker_history(conn, permnos: list[float]) -> pd.DataFrame:
    """
    Ticker history for the roster's permnos, with name-validity windows.

    ``crsp.stocknames`` has one row per (permno, name interval), so this is
    intentionally many rows per permno -- the caller interval-joins on eom. Used
    for display only; nothing keys on ticker.
    """
    if not permnos:
        return pd.DataFrame(columns=["permno", "ticker", "comnam", "namedt", "nameenddt"])

    in_list = ", ".join(str(int(p)) for p in permnos)
    sql = f"""
        SELECT permno, ticker, comnam, namedt, nameenddt
        FROM crsp.stocknames
        WHERE permno IN ({in_list})
    """
    df = conn.raw_sql(sql)
    print(f"🏷️  {len(df):,} ticker-history rows for {len(permnos):,} permnos")
    return df


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def fetch_panel(conn, ids: list[str], anchor_eom=None) -> pd.DataFrame:
    """
    Full monthly history for the roster, from the anchor date forward.

    Unlike the original loader, the window is anchored on an explicit date rather
    than ``datetime.now() - months_back*30``. That arithmetic is what silently
    produced a 3-month panel: run in Dec 2025 with months_back=12 it asked for
    ``eom >= 2024-12-10``, and contrib.global_factor ends 2025-02-28.
    """
    anchor = anchor_eom or config.ANCHOR_EOM
    chars = ", ".join(config.characteristics_to_pull())
    in_list = ", ".join(f"'{i}'" for i in ids)

    sql = f"""
        SELECT id, eom, gvkey, permno, size_grp, {chars}
        FROM contrib.global_factor
        WHERE {_jkp_filter_sql()}
          AND eom >= '{anchor}'
          AND id IN ({in_list})
        ORDER BY id, eom
    """
    df = conn.raw_sql(sql)
    print(f"📈 {len(df):,} monthly rows for {df['id'].nunique():,} securities")
    if len(df):
        print(f"   {df['eom'].min()} → {df['eom'].max()}")
    return df
