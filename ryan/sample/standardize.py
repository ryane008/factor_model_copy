"""
Cross-sectional standardization of style descriptors, and industry dummies.

Follows USE4's convention rather than a plain z-score. Three differences from
clean_characteristics.ipynb, each of which changes the numbers:

1. **Cap-weighted mean, equal-weighted std.** A plain ``(x - x.mean()) / x.std()``
   makes the *equal-weighted* mean zero. USE4 sets the *cap-weighted* mean to
   zero, which is what makes the market portfolio have zero exposure to every
   style factor -- without it, the market factor and the style factors overlap
   and the factor returns are not separately interpretable.

2. **Winsorization.** With raw z-scores a single observation dominates a whole
   cross-section: the sample panel carried ret_exc values of 19.5 and 13.8.
   USE4 trims at 3 sigma and re-standardizes.

3. **log size.** Raw ``market_equity`` is so skewed that z-scoring it puts a
   handful of mega-caps at +40 sigma. USE4's Size descriptor is log(market cap).

Also: the return column is never standardized. It is the dependent variable in
the cross-sectional regression, not an exposure.

Everything is vectorized with ``groupby.transform`` rather than
``groupby.apply``, both for speed and to avoid depending on the ``include_groups``
argument, which only exists in pandas >= 2.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _grouped_capw_mean(x: pd.Series, w: pd.Series, g: pd.Series) -> pd.Series:
    """
    Cap-weighted mean of ``x`` within each group of ``g``, broadcast back to rows.

    Only rows where both the value and a positive weight are present contribute.
    Groups with no usable weights fall back to the equal-weighted mean, so a
    month with missing market caps degrades rather than going entirely null.
    """
    usable = x.notna() & w.notna() & (w > 0)
    wv = w.where(usable)
    num = (x * wv).groupby(g).transform("sum", min_count=1)
    den = wv.groupby(g).transform("sum", min_count=1)
    mu = num / den
    return mu.fillna(x.groupby(g).transform("mean"))


def _standardize_once(x: pd.Series, w: pd.Series, g: pd.Series) -> pd.Series:
    """One pass: subtract the cap-weighted mean, divide by the equal-weighted std."""
    mu = _grouped_capw_mean(x, w, g)
    sd = x.groupby(g).transform("std")  # ddof=1
    centered = x - mu
    # Zero (or undefined) dispersion means the descriptor carries no information
    # this month: every present name sits at the mean, i.e. exposure 0.
    sd = sd.where(np.isfinite(sd) & (sd > 0))
    return (centered / sd).where(sd.notna(), centered.where(x.isna(), 0.0))


def robust_winsorize(
    x: pd.Series,
    groups: pd.Series,
    n_sigma: float | None = None,
) -> pd.Series:
    """
    Clip each cross-section to median +/- n_sigma * robust_sigma, where
    ``robust_sigma = 1.4826 * MAD`` (the MAD-to-sigma constant for a normal).

    The robust scale is the whole point. Trimming at n_sigma of the *ordinary*
    standard deviation does not work, because the outlier you are trying to
    remove is itself inflating that standard deviation: with 60 names drawn from
    N(0,1) plus one value at 40, the sample sd is ~5.2, so the outlier sits at
    only 7.3 sd and every genuine observation is compressed into +/-0.2. Median
    and MAD are unaffected by the outlier, so the clip lands where it should.

    Falls back to the ordinary std where MAD is zero, which happens when more
    than half the cross-section shares one value.
    """
    sigma_cap = n_sigma if n_sigma is not None else config.WINSOR_SIGMA

    med = x.groupby(groups).transform("median")
    mad = (x - med).abs().groupby(groups).transform("median")
    sigma = 1.4826 * mad
    sigma = sigma.where(np.isfinite(sigma) & (sigma > 0), x.groupby(groups).transform("std"))

    lo, hi = med - sigma_cap * sigma, med + sigma_cap * sigma
    return x.clip(lower=lo, upper=hi)


def standardize_grouped(
    x: pd.Series,
    weights: pd.Series,
    groups: pd.Series,
    winsor_sigma: float | None = None,
) -> pd.Series:
    """
    USE4-style standardization of one descriptor, within each cross-section.

    Winsorize on the robust scale *first*, then standardize once. The order is
    deliberate, and the tempting alternative (standardize, clip at +/-3,
    re-standardize) is wrong: the second standardization divides by the standard
    deviation of the *clipped* series, which is smaller than 1, so it scales the
    clipped values straight back out past the bound. In testing that returned a
    max |z| of 7.3 for a nominal 3-sigma trim.

    The result has, exactly, a cap-weighted mean of 0 and an equal-weighted
    standard deviation of 1 within each cross-section. Max |z| then lands near
    ``winsor_sigma`` but is not hard-bounded by it -- the trim is applied on the
    robust scale and the divisor is the post-trim sd, so a small overshoot is
    expected. Exact moments matter more here than an exact bound: the
    cap-weighted zero is what makes the market portfolio style-neutral.
    """
    trimmed = robust_winsorize(x, groups, winsor_sigma)
    return _standardize_once(trimmed, weights, groups)


def standardize_panel(
    panel: pd.DataFrame,
    descriptors: list[str] | None = None,
    cap_col: str | None = None,
    log_cols: list[str] | None = None,
    by: str = "eom",
) -> pd.DataFrame:
    """
    Standardize every style descriptor within each cross-section.

    Returns a copy with ``z_<descriptor>`` columns added. Raw columns are kept so
    the transform stays auditable.
    """
    cols = descriptors if descriptors is not None else config.all_descriptors()
    cap = cap_col or config.CAP_COL
    logs = log_cols if log_cols is not None else config.LOG_DESCRIPTORS

    out = panel.copy()

    # Log-transform before standardizing. Non-positive caps cannot be logged;
    # they become null rather than -inf.
    for col in logs:
        if col in out.columns:
            src = pd.to_numeric(out[col], errors="coerce")
            out[f"log_{col}"] = np.log(src.where(src > 0))

    weights = pd.to_numeric(out[cap], errors="coerce")
    groups = out[by]

    for col in cols:
        src_col = f"log_{col}" if col in logs else col
        if src_col not in out.columns:
            print(f"⚠️  descriptor {src_col} not in panel; skipped")
            continue
        src = pd.to_numeric(out[src_col], errors="coerce")
        out[f"z_{col}"] = standardize_grouped(src, weights, groups)

    return out


def combine_descriptors(
    panel: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    cap_col: str | None = None,
    by: str = "eom",
) -> pd.DataFrame:
    """
    Average each factor's standardized descriptors into a single exposure.

    ``be_me``, ``ni_me`` and ``ocf_me`` are near-collinear; entering all three in
    the cross-sectional regression makes the individual factor returns unstable
    even though the overall fit is fine. USE4 collapses descriptors into one
    factor per concept, so Value becomes one column, not three.

    Equal weights across whichever descriptors are present, then re-standardized
    so each combined factor is back on a unit-dispersion scale.
    """
    grp = groups if groups is not None else config.STYLE_DESCRIPTORS
    cap = cap_col or config.CAP_COL

    out = panel.copy()
    weights = pd.to_numeric(out[cap], errors="coerce")
    by_col = out[by]

    for factor, descriptors in grp.items():
        zcols = [f"z_{d}" for d in descriptors if f"z_{d}" in out.columns]
        if not zcols:
            print(f"⚠️  factor {factor} has no available descriptors; skipped")
            continue
        raw = out[zcols].mean(axis=1, skipna=True)
        out[f"f_{factor}"] = standardize_grouped(raw, weights, by_col)
        print(f"   {factor:<15} ← {', '.join(zcols)}")

    return out


def build_industry_dummies(
    panel: pd.DataFrame,
    code_col: str = "sector_code",
    prefix: str = "ind",
) -> tuple[pd.DataFrame, list[str]]:
    """
    One 0/1 column per industry.

    Every industry is kept -- none is dropped as a reference category. The
    resulting design matrix is deliberately rank-deficient (the dummies sum to
    the market factor's column of ones), and that is resolved in the regression
    step by USE4's cap-weighted constraint ``sum_i (w_i * f_i) = 0``, which keeps
    all industry factor returns interpretable relative to the market instead of
    relative to an arbitrary omitted industry.

    Worth being explicit about, because pre-orthogonalizing the style factors
    against these dummies does *not* fix the rank deficiency -- the constraint is
    required either way.
    """
    out = panel.copy()
    codes = out[code_col].astype("string")

    dummy_cols = []
    for code in sorted(codes.dropna().unique()):
        col = f"{prefix}_{code}"
        out[col] = (codes == code).astype(float)
        dummy_cols.append(col)

    # An asset with no classification gets zeros everywhere, which would silently
    # place it outside every industry. Flag rather than guess.
    unclassified = int(codes.isna().sum())
    if unclassified:
        print(f"⚠️  {unclassified:,} rows have no {code_col} and zero in every dummy — exclude them from the regression")

    print(f"🏭 {len(dummy_cols)} industry dummies: {', '.join(dummy_cols)}")
    return out, dummy_cols


def coverage_report(panel: pd.DataFrame, by: str = "eom") -> pd.DataFrame:
    """
    Per-cross-section diagnostics: names, return coverage, and the thinnest
    industry. That last column is the one to watch -- any month where an industry
    has fewer than ~3 members will produce a degenerate specific variance.
    """
    dummies = [c for c in panel.columns if c.startswith("ind_")]
    rows = []
    for period, grp in panel.groupby(by):
        counts = grp[dummies].sum() if dummies else pd.Series(dtype=float)
        rows.append(
            {
                by: period,
                "n_assets": len(grp),
                "n_with_return": int(grp[config.RETURN_COL].notna().sum()),
                "min_industry_n": int(counts.min()) if len(counts) else 0,
                "thinnest_industry": counts.idxmin() if len(counts) else None,
            }
        )
    return pd.DataFrame(rows)
