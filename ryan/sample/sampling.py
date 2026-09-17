"""
The stratified draw, and date-aware identifier joins.

Two separable jobs live here: picking the roster (once, reproducibly), and
attaching permno/ticker to panel rows *as of each eom* rather than by a blind
many-to-many merge on gvkey.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import gics

FAR_FUTURE = pd.Timestamp("2262-04-10")  # pandas datetime64[ns] max, for open links


# ---------------------------------------------------------------------------
# Date-aware identifier joins
# ---------------------------------------------------------------------------


def attach_permno(panel: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """
    Attach permno/permco to each panel row using the CCM link valid at that eom.

    This is the fix for the join that fanned 14,813 rows out to 47,448. The old
    code merged on gvkey alone, producing every historical link for every date,
    then tried to undo the damage with ``drop_duplicates`` plus
    ``drop_duplicates(subset=['gvkey','eom'], keep='first')``. That happened to
    recover the right row count only because ``ticker`` was the sole column that
    varied -- ``permno`` and ``permco`` were being picked arbitrarily, and the
    cleanup was verified on three rows of one gvkey.

    Here the merge is restricted to links whose ``[linkdt, linkenddt]`` window
    contains ``eom``, so there is exactly one answer per row and no cleanup step
    to get wrong.
    """
    out = panel.copy()
    out["eom"] = pd.to_datetime(out["eom"])

    lk = links.copy()
    lk["linkdt"] = pd.to_datetime(lk["linkdt"])
    # An open link comes back as NULL. errors="coerce" is load-bearing: it also
    # absorbs any out-of-range sentinel date (e.g. '9999-12-31', which is a valid
    # Postgres date but overflows datetime64[ns]) into NaT so the fillna catches it.
    lk["linkenddt"] = pd.to_datetime(lk["linkenddt"], errors="coerce").fillna(FAR_FUTURE)

    # Resolve candidates on the inner join, collapse to one link per (id, eom),
    # then left-join the answer back onto the original rows. Structuring it this
    # way makes row preservation automatic: filtering an outer join in place would
    # silently drop any row whose gvkey matched a link but whose eom falls in a
    # *gap* between two link windows (which happens around mergers and
    # re-listings), because every candidate for that row fails the window test.
    cand = out[["id", "eom", "gvkey"]].merge(lk, on="gvkey", how="inner")
    cand = cand[(cand["linkdt"] <= cand["eom"]) & (cand["eom"] <= cand["linkenddt"])].copy()

    # A gvkey can carry both a primary ('P') and a co-primary ('C') link in the
    # same window. Prefer 'P', then the more reliable linktype, deterministically.
    cand["_prim_rank"] = cand["linkprim"].map({"P": 0, "C": 1}).fillna(9)
    cand["_type_rank"] = cand["linktype"].map({"LC": 0, "LU": 1}).fillna(9)
    cand = cand.sort_values(["id", "eom", "_prim_rank", "_type_rank", "permno"])
    cand = cand.drop_duplicates(subset=["id", "eom"], keep="first")

    resolved = cand[["id", "eom", "permno", "permco"]].rename(
        columns={"permno": "permno_ccm", "permco": "permco_ccm"}
    )
    merged = out.merge(resolved, on=["id", "eom"], how="left")

    if len(merged) != len(out):
        raise AssertionError(
            f"attach_permno changed the row count ({len(out):,} -> {len(merged):,}); "
            "resolved links must be unique per (id, eom)"
        )

    matched = merged["permno_ccm"].notna().mean()
    print(f"🔗 CCM permno attached for {matched:.1%} of panel rows (date-aware, 1:1 per (id, eom))")
    if "permno" in merged.columns:
        # JKP carries its own permno; where both exist they should agree. A
        # mismatch means the link filter is picking a different share class.
        both = merged["permno"].notna() & merged["permno_ccm"].notna()
        if both.any():
            agree = (
                pd.to_numeric(merged.loc[both, "permno"], errors="coerce")
                == pd.to_numeric(merged.loc[both, "permno_ccm"], errors="coerce")
            ).mean()
            print(f"   JKP permno vs CCM permno agree on {agree:.1%} of the {both.sum():,} rows carrying both")
    return merged


def attach_ticker(panel: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the ticker in force at each eom. Display only — nothing keys on it.
    """
    if names.empty:
        out = panel.copy()
        out["ticker"] = pd.NA
        return out

    out = panel.copy()
    out["eom"] = pd.to_datetime(out["eom"])

    nm = names.copy()
    nm["namedt"] = pd.to_datetime(nm["namedt"])
    nm["nameenddt"] = pd.to_datetime(nm["nameenddt"], errors="coerce").fillna(FAR_FUTURE)

    # Same resolve-then-left-join shape as attach_permno, for the same reason: a
    # permno with a gap in its stocknames coverage must keep its panel row.
    key = "permno_ccm" if "permno_ccm" in out.columns else "permno"
    cand = out[["id", "eom", key]].merge(nm, left_on=key, right_on="permno", how="inner")
    cand = cand[(cand["namedt"] <= cand["eom"]) & (cand["eom"] <= cand["nameenddt"])].copy()
    cand = cand.sort_values(["id", "eom", "namedt"]).drop_duplicates(subset=["id", "eom"], keep="last")

    merged = out.merge(cand[["id", "eom", "ticker"]], on=["id", "eom"], how="left")

    if len(merged) != len(out):
        raise AssertionError(f"attach_ticker changed the row count ({len(out):,} -> {len(merged):,})")

    print(f"🏷️  ticker attached for {merged['ticker'].notna().mean():.1%} of panel rows (display only)")
    return merged


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def build_sampling_frame(
    anchor: pd.DataFrame,
    coverage: pd.DataFrame,
    gics_df: pd.DataFrame,
    min_return_months: int | None = None,
) -> pd.DataFrame:
    """
    Assemble the eligible population: alive at the anchor, classified, and with
    enough return history to support a specific-risk estimate.

    Returns one row per eligible ``id`` with its ``sector_code``.
    """
    min_months = min_return_months or config.MIN_RETURN_MONTHS

    frame = anchor.merge(coverage, on="id", how="left")
    frame = frame.merge(gics_df, on="gvkey", how="left")
    frame = gics.attach_names(frame)

    n0 = len(frame)
    frame = frame[frame["sector_code"].notna()]
    n1 = len(frame)
    frame = frame[frame["n_ret"].fillna(0) >= min_months]
    n2 = len(frame)

    print(f"🔎 Sampling frame: {n0:,} alive at anchor")
    print(f"   → {n1:,} with a GICS sector      (dropped {n0 - n1:,})")
    print(f"   → {n2:,} with ≥{min_months} months of returns (dropped {n1 - n2:,})")
    print("\n   Eligible names per sector:")
    counts = frame.groupby("sector_name", dropna=False).size().sort_values(ascending=False)
    for name, n in counts.items():
        print(f"     {name:<26} {n:>5,}")

    return frame


def draw_stratified_sample(
    frame: pd.DataFrame,
    per_sector: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Draw ``per_sector`` securities from each GICS sector without replacement.

    Uniform sampling is not used on purpose. Utilities is ~3% of the US listed
    universe, so a uniform draw of ~220 has a meaningful chance of landing 0-2
    Utilities names, and an industry dummy with one member perfectly fits that
    asset's return -- forcing its residual and therefore its specific variance to
    zero, putting a zero on the diagonal of Delta, and telling the optimizer that
    a portfolio concentrated in that single stock is riskless.

    Sectors with fewer than ``per_sector`` eligible names are taken whole and
    reported, so the roster size is a result rather than an assumption.

    **The seed alone does not make the roster stable across sessions.** The draw
    is reproducible only for a fixed ``frame``, and the frame is built from live
    WRDS data: next month ``fetch_return_coverage`` returns larger ``n_ret``
    counts, so different securities clear the ``MIN_RETURN_MONTHS`` screen, the
    eligible pool changes size, and the same seed selects different names. To
    keep a roster, read it back from ``ryan_sample_universe`` rather than
    re-drawing it -- which is what ``run_pipeline.ipynb`` does by default.
    """
    n_per = per_sector or config.NAMES_PER_SECTOR
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)

    picks = []
    for code in sorted(frame["sector_code"].unique()):
        # Sort before drawing so the result depends only on the *set* of eligible
        # ids, not on the row order WRDS happened to return them in. Without this
        # the same frame can yield a different roster between sessions.
        pool = frame[frame["sector_code"] == code].sort_values("id")
        take = min(n_per, len(pool))
        if take < n_per:
            print(f"⚠️  sector {code} ({gics.SECTORS.get(code, '?')}) has only {len(pool)} eligible names; taking all")
        idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        picks.append(pool.loc[idx])

    roster = pd.concat(picks).sort_values(["sector_code", "id"]).reset_index(drop=True)

    roster["sample_seed"] = seed if seed is not None else config.RANDOM_SEED
    roster["anchor_eom"] = pd.Timestamp(config.ANCHOR_EOM)
    roster["min_return_months"] = config.MIN_RETURN_MONTHS

    print(f"\n🎲 Drew {len(roster):,} securities ({n_per} per sector, seed={roster['sample_seed'].iloc[0]})")
    print(roster.groupby("sector_name").size().to_string())
    return roster
