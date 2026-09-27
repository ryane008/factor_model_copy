"""
Point-in-time universe membership.

The sample was a fixed cohort drawn once. A real universe is dynamic -- names
list, delist, grow into the index and shrink out of it -- and membership at date
t must be decidable from information available at t. Everything here takes a
panel of (id, eom, market_equity, ...) and returns per-(id, eom) flags, so the
rule is data, not code scattered through the pipeline.

Two rules:

``screens``  every JKP US common stock that passes the screens that month.
``rank``     Russell-style annual reconstitution: rank by cap on the rank date,
             hold the membership from the effective date to the next one.

The Russell rule is a PROXY. What it reproduces: annual reconstitution on a May
rank date effective after June, top-N by total market cap, and the +/-2.5%
cumulative-cap band at the boundary. What it does not: float adjustment, the
$1 price and 5% float screens, quarterly IPO additions, and Russell's own
country/share-class rules (JKP's primary_sec screen keeps one class per firm).
Expect a few percent of names to differ from the real index at the boundary; the
large names that carry the cap are the same. A constituents file, if you have
one, replaces it (``from_constituents``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def screen(panel: pd.DataFrame, exclude_size_grp: list[str] | None = None,
           cap_col: str = "market_equity") -> pd.Series:
    """
    Boolean per row: usable in the estimation universe this month.

    Positive cap (the regression weight and the standardization weight both need
    it) and not in an excluded JKP size group. The JKP universe filters (common,
    main exchange, primary security, obs_main) are applied in SQL already.
    """
    cap = pd.to_numeric(panel[cap_col], errors="coerce")
    ok = cap.notna() & (cap > 0)
    if exclude_size_grp and "size_grp" in panel:
        ok &= ~panel["size_grp"].isin(exclude_size_grp)
    return ok


# ---------------------------------------------------------------------------
# Rank-based reconstitution
# ---------------------------------------------------------------------------

def reconstitute(
    caps: pd.DataFrame,
    n: int,
    band: float | None = None,
    id_col: str = "id",
    date_col: str = "rank_date",
    cap_col: str = "market_equity",
) -> pd.DataFrame:
    """
    Membership decided on each rank date. Returns (id, rank_date, rank, cum_share, member).

    Without a band: top ``n`` by cap.

    With a band (Russell since 2007): compute each name's cumulative share of
    total cap, largest first, and the breakpoint = cumulative share at rank n.
    Then:
        cum_share <= breakpoint - band           member
        cum_share >  breakpoint + band           not a member
        in between                               keeps LAST year's status
    so an incumbent drifting just below the line stays, and a newcomer just above
    it waits a year. The first rank date has no incumbents and is a hard top-n.
    The member count therefore floats around n rather than hitting it exactly --
    as the real index does.
    """
    out, prev = [], set()
    for d, g in caps.sort_values(date_col).groupby(date_col, sort=True):
        g = g[pd.to_numeric(g[cap_col], errors="coerce") > 0].copy()
        g = g.sort_values([cap_col, id_col], ascending=[False, True]).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        g["cum_share"] = g[cap_col].cumsum() / g[cap_col].sum()
        if band is None or not prev:
            member = g["rank"] <= n
        else:
            bp = g.loc[min(n, len(g)) - 1, "cum_share"]
            inside = g["cum_share"] <= bp - band
            outside = g["cum_share"] > bp + band
            member = inside | (~outside & g[id_col].isin(prev))
        g["member"] = member.to_numpy()
        prev = set(g.loc[g["member"], id_col])
        out.append(g[[id_col, date_col, "rank", "cum_share", "member"]].rename(columns={date_col: "rank_date"}))
    return pd.concat(out, ignore_index=True)


def effective_rank_date(eoms: pd.Series, rank_month: int, effective_month: int) -> pd.Series:
    """
    For each eom, the rank date whose membership is in force.

    A June reconstitution uses May's ranks and takes effect at the end of June,
    so eom June Y .. May Y+1 map to rank date May Y. Everything before the first
    effective month maps to the prior year's rank date -- the caller must have
    pulled caps far enough back to have it.
    """
    e = pd.to_datetime(eoms)
    year = np.where(e.dt.month >= effective_month, e.dt.year, e.dt.year - 1)
    return pd.to_datetime(pd.DataFrame({"year": year, "month": rank_month, "day": 1})) + pd.offsets.MonthEnd(0)


def rank_membership(
    panel: pd.DataFrame,
    n: int,
    band: float | None,
    rank_month: int,
    effective_month: int,
    eligible: pd.Series | None = None,
    cap_col: str = "market_equity",
) -> pd.Series:
    """
    Boolean per panel row: member of the top-``n`` index in force at that eom.

    Ranks only among ``eligible`` rows on each rank date (the screened universe),
    using caps observed ON the rank date. A name that was a member but has
    delisted simply has no row afterwards; a name with a row but no rank-date
    observation (listed after the rank date) is not a member until the next
    reconstitution -- which is the real index's rule, not an artifact.
    """
    p = panel if eligible is None else panel[eligible]
    p = p.assign(eom=pd.to_datetime(p["eom"]))
    ranks = p[p["eom"].dt.month == rank_month][["id", "eom", cap_col]].rename(columns={"eom": "rank_date"})
    recon = reconstitute(ranks, n, band, cap_col=cap_col)
    members = recon.loc[recon["member"], ["id", "rank_date"]].assign(_m=True)

    key = pd.DataFrame({"id": panel["id"].to_numpy(),
                        "rank_date": effective_rank_date(panel["eom"], rank_month, effective_month).to_numpy()},
                       index=panel.index)
    flag = key.merge(members, on=["id", "rank_date"], how="left")["_m"].eq(True).to_numpy()
    out = pd.Series(flag, index=panel.index, dtype=bool)
    if eligible is not None:
        out &= eligible.reindex(panel.index).fillna(False).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Actual constituents, when available
# ---------------------------------------------------------------------------

def from_constituents(panel: pd.DataFrame, path: str, permno_col: str = "permno") -> pd.Series:
    """
    Membership from a constituents file: columns permno,start,end (end blank = current).

    Interval-joined on eom, same resolve-then-left-join shape as the CCM link
    join, so a panel row survives whether or not it matches.
    """
    c = pd.read_csv(path)
    c.columns = [x.strip().lower() for x in c.columns]
    missing = {"permno", "start", "end"} - set(c.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    c["start"] = pd.to_datetime(c["start"])
    c["end"] = pd.to_datetime(c["end"], errors="coerce").fillna(pd.Timestamp("2262-04-10"))
    c["permno"] = pd.to_numeric(c["permno"], errors="coerce")

    p = panel[["id", "eom", permno_col]].copy()
    p["eom"] = pd.to_datetime(p["eom"])
    p["_permno"] = pd.to_numeric(p[permno_col], errors="coerce")
    cand = p.reset_index().merge(c, left_on="_permno", right_on="permno", how="inner")
    cand = cand[(cand["start"] <= cand["eom"]) & (cand["eom"] <= cand["end"])]
    out = pd.Series(False, index=panel.index)
    out.loc[cand["index"].unique()] = True
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def turnover(panel: pd.DataFrame, flag: pd.Series) -> pd.DataFrame:
    """Per eom: members, joiners and leavers versus the previous month."""
    m = panel.loc[flag, ["id", "eom"]].copy()
    m["eom"] = pd.to_datetime(m["eom"])
    by = m.groupby("eom")["id"].apply(set).sort_index()
    rows, prev = [], None
    for d, s in by.items():
        rows.append({"eom": d, "members": len(s),
                     "joined": len(s - prev) if prev is not None else np.nan,
                     "left": len(prev - s) if prev is not None else np.nan})
        prev = s
    return pd.DataFrame(rows).set_index("eom")


# ---------------------------------------------------------------------------
# Profile dispatch and the industry gate
# ---------------------------------------------------------------------------

def apply_profile(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Add ``in_estu`` (estimation universe) and, under the R1000 profile, the
    coverage flag, following ``cfg`` (the active config module).

    Rows outside the estimation universe are kept here and dropped by the
    caller, so the counts can be reported before anything disappears.
    """
    out = panel.copy()
    ok = screen(out, cfg.EXCLUDE_SIZE_GRP, cfg.CAP_COL)
    if cfg.ESTU_RULE == "screens":
        out["in_estu"] = ok
    elif cfg.ESTU_RULE == "rank":
        out["in_estu"] = rank_membership(out, cfg.ESTU_N, None, cfg.RANK_MONTH, cfg.EFFECTIVE_MONTH,
                                         eligible=ok, cap_col=cfg.CAP_COL)
        if cfg.CONSTITUENTS_CSV:
            cov = from_constituents(out, cfg.CONSTITUENTS_CSV)
        else:
            cov = rank_membership(out, cfg.COVERAGE_N, cfg.COVERAGE_BAND, cfg.RANK_MONTH,
                                  cfg.EFFECTIVE_MONTH, eligible=ok, cap_col=cfg.CAP_COL)
        # Coverage must sit inside estimation: a covered name nobody estimated
        # would have no residual history and no place in the regression.
        n_outside = int((cov & ~out["in_estu"]).sum())
        if n_outside:
            print(f"⚠️  {n_outside:,} coverage rows fall outside the estimation universe; "
                  "they are added to it so they can be priced")
            out["in_estu"] |= cov
        out[cfg.COVERAGE_COL] = cov
    else:
        raise ValueError(f"unknown ESTU_RULE {cfg.ESTU_RULE!r}")
    return out


def industry_gate(exposures: pd.DataFrame, dummy_cols: list[str], min_names: int,
                  by: str = "eom") -> pd.DataFrame:
    """
    Per-month member count of every industry; raises if any falls below ``min_names``.

    The sample plotted the thinnest industry and relied on a human to look. At
    universe scale with 300 months that is a gate, not a chart: an industry with
    one member fits it exactly, zero residual, zero specific variance, singular
    Sigma -- and an EMPTY industry makes that month's factor return NaN, which
    03 then refuses. Returns the count table so the notebook can plot it either way.
    """
    counts = exposures.groupby(by)[dummy_cols].sum()
    worst = counts.min()
    thin = worst[worst < min_names]
    if len(thin):
        detail = ", ".join(f"{c} min {int(v)} ({counts[c].idxmin().date()})" for c, v in thin.items())
        raise ValueError(f"industries below {min_names} names in some month: {detail}. "
                         "Use a coarser INDUSTRY_CODE_COL or a later START_EOM.")
    return counts
