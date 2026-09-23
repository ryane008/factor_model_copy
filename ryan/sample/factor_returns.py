"""
Step 4: the cross-sectional regression that produces factor returns and
specific returns.

One weighted least-squares fit per month:

    r_i = f_market + sum_p X_ip f_p^ind + sum_k X_ik f_k^sty + u_i

`f(t)` are the fitted coefficients, `u(i,t)` the residuals. They come out of the
*same* regression and cannot be computed separately -- the specific return is
defined as what the factor returns fail to explain.

Two things make this not a plain OLS call.

**The design matrix is rank-deficient by construction.** The 11 industry dummies
sum, row by row, to the market factor's column of ones, so `[market | industries]`
has rank 11, not 12. `lstsq` would happily return *a* minimum-norm solution, but
it would be an arbitrary point on a one-dimensional family of equally good fits,
and it would move around from month to month for numerical reasons alone --
making the factor return time series, and therefore F, garbage. USE4 pins it down
with a cap-weighted constraint on the industry returns (eq. 4-2):

    sum_p CAP_p * f_p^ind = 0

which says the cap-weighted industry portfolio *is* the market, so industry
returns are read relative to the market rather than relative to an arbitrary
omitted industry. Note that pre-orthogonalizing styles against industries does
not help here: the deficiency is inside the industry block itself.

**The weights are not uniform.** USE4 fits with weights proportional to the
square root of market cap, which treats specific variance as roughly inversely
proportional to sqrt(cap) -- large caps are less noisy, so they get more say,
but not the quadratic say that cap weighting would give them. This is also the
step that handles cross-sectional heteroskedasticity, which is why Newey-West
later on is only about serial correlation.

Implementation: the constraint is imposed by *substitution*, not by a penalty or
by dropping a column. Writing `f = U @ theta` for a K x (K-1) matrix U whose
columns span the constrained subspace turns the problem into an unconstrained
WLS in 17 free parameters, and `U @ theta` gives back all 18 factor returns with
the constraint satisfied to machine precision. A penalty would only satisfy it
approximately; dropping a column would silently re-baseline every industry
against the dropped one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

MARKET_FACTOR = "market"


def align_returns_to_lagged_exposures(
    exposures: pd.DataFrame,
    style_cols: list[str],
    dummy_cols: list[str],
    by: str = "eom",
    id_col: str = "id",
    return_col: str | None = None,
    cap_col: str | None = None,
) -> pd.DataFrame:
    """
    Attach month-t returns to the exposures observed at the end of month t-1.

    Everything on the right-hand side -- style exposures, industry membership,
    and the market cap that supplies both the regression weight and the
    constraint -- must be knowable before the return happens. The panel as built
    is contemporaneous: ``ret_exc`` at eom t is the return *during* month t, but
    ``market_equity`` at eom t is measured after it. Since be_me, ni_me, ocf_me
    and log size all divide by (or are) that same market equity, the exposures
    carry the very return they are meant to explain.

    The merge is on an explicit month period, ``t-1 -> t``, not ``groupby.shift(1)``.
    Shift would silently pair an exposure from 2008-03 with a return from 2009-07
    wherever a security has a gap in the panel, and this cohort has gaps by
    design -- names delist and some re-appear. Pairing across a gap would use a
    year-old exposure as though it were current.

    The first month of each security necessarily drops: there is no prior
    observation to supply its exposures.
    """
    ret = return_col or config.RETURN_COL
    cap = cap_col or config.CAP_COL

    x_cols = [id_col, by] + style_cols + dummy_cols + [cap]
    keep_meta = [c for c in ("sector_code", "sector_name", "conm", "ticker") if c in exposures.columns]

    left = exposures[x_cols + keep_meta].copy()
    left["_period"] = pd.PeriodIndex(pd.to_datetime(left[by]), freq="M")
    # The exposure row observed at t-1 is used to explain the return at t.
    left["_target"] = left["_period"] + 1

    right = exposures[[id_col, by, ret]].copy()
    right["_target"] = pd.PeriodIndex(pd.to_datetime(right[by]), freq="M")
    right = right.rename(columns={by: "_return_eom"})

    merged = left.merge(right, on=[id_col, "_target"], how="inner", validate="one_to_one")

    # The regression is dated by the month whose RETURN is being explained, so
    # f(t) lines up with the period it describes rather than with the exposure
    # date. Getting this backwards would shift the whole factor return series by
    # a month relative to F's forecast dates.
    merged = merged.rename(columns={by: "exposure_eom", "_return_eom": by})
    return merged.drop(columns=["_period", "_target"])


def winsorize_returns(
    r: pd.Series,
    groups: pd.Series,
    n_sigma: float | None = None,
) -> tuple[pd.Series, int]:
    """
    Clip the dependent variable to median +/- n_sigma robust sigmas per month.

    Deliberately *not* ``standardize.standardize_grouped``: that would divide by
    the cross-sectional standard deviation and subtract a mean, putting factor
    returns in z-units per month and stripping out exactly the cross-sectional
    level the market factor exists to capture. Winsorizing only moves the few
    points that dominate the weighted fit; units, sign and the bulk of the
    distribution are untouched.

    The scale is MAD-based for the same reason it is in ``standardize``: a
    +2225% observation inflates the ordinary standard deviation enough to hide
    itself behind it.

    ``n_sigma=None`` means no trimming, full stop -- there is no fallback to the
    config default here. ``run_panel`` resolves that default before calling, so
    None reaching this function can only mean the caller switched trimming off,
    and quietly reinstating 8 sigma would be the opposite of what they asked.
    """
    if n_sigma is None:
        return r, 0
    sigma_cap = float(n_sigma)

    med = r.groupby(groups).transform("median")
    mad = (r - med).abs().groupby(groups).transform("median")
    sigma = 1.4826 * mad
    sigma = sigma.where(np.isfinite(sigma) & (sigma > 0), r.groupby(groups).transform("std"))

    lo, hi = med - sigma_cap * sigma, med + sigma_cap * sigma
    clipped = r.clip(lower=lo, upper=hi)
    return clipped, int((clipped != r).sum())


def build_constraint_basis(industry_caps: np.ndarray) -> np.ndarray:
    """
    Basis ``R`` (P x (P-1)) for the industry returns satisfying ``caps @ f = 0``.

    The free parameters are the first P-1 industry returns; the last is whatever
    the constraint forces it to be:

        f_last = -(1 / cap_last) * sum_{p < last} cap_p * f_p

    The industry used for the substitution is the one with the **largest** cap,
    not the last one in column order. That choice is numerically load-bearing:
    dividing by the cap of a thin industry inflates every other coefficient's
    contribution to it, and in a 220-name sample the thinnest industry can be a
    handful of small names. The fitted values and the constrained ``f`` are
    mathematically identical whichever industry is eliminated -- only the
    conditioning differs.
    """
    caps = np.asarray(industry_caps, dtype=float)
    p = caps.size
    if p < 2:
        raise ValueError(f"need at least 2 industries to constrain, got {p}")
    if not np.all(np.isfinite(caps)) or np.any(caps <= 0):
        raise ValueError("industry caps must all be finite and positive")

    pivot = int(np.argmax(caps))
    free = [i for i in range(p) if i != pivot]

    r = np.zeros((p, p - 1))
    for col, row in enumerate(free):
        r[row, col] = 1.0
    r[pivot, :] = -caps[free] / caps[pivot]
    return r


def _restriction_matrix(n_industries: int, n_styles: int, industry_caps: np.ndarray) -> np.ndarray:
    """
    Full K x (K-1) basis: market and styles unrestricted, industries constrained.

    Column order throughout this module is [market, industries..., styles...].
    """
    r_ind = build_constraint_basis(industry_caps)
    k = 1 + n_industries + n_styles
    u = np.zeros((k, k - 1))
    u[0, 0] = 1.0                                              # market
    u[1 : 1 + n_industries, 1 : n_industries] = r_ind          # industries
    for s in range(n_styles):                                  # styles
        u[1 + n_industries + s, n_industries + s] = 1.0
    return u


def solve_cross_section(
    x: np.ndarray,
    r: np.ndarray,
    cap: np.ndarray,
    n_industries: int,
    n_styles: int,
) -> dict:
    """
    One month's constrained WLS fit.

    ``x`` is N x K with columns [market, industries..., styles...], ``r`` the
    excess returns, ``cap`` the market caps used for both the regression weights
    and the constraint. Returns the factor returns, the residuals, and the
    diagnostics needed to tell a good fit from a degenerate one.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    cap = np.asarray(cap, dtype=float)

    # Regression weights ~ sqrt(cap), normalized so R^2 and the weighted residual
    # sums are comparable across months with different numbers of assets.
    v = np.sqrt(cap)
    v = v / v.sum()

    # Constraint weights are the *cap* totals per industry, not the sqrt-cap
    # regression weights. Different jobs: the constraint asserts that the
    # cap-weighted industry portfolio equals the market, which is a statement
    # about the market portfolio, so it uses market-cap weights.
    ind_block = x[:, 1 : 1 + n_industries]
    industry_caps = ind_block.T @ cap

    u_basis = _restriction_matrix(n_industries, n_styles, industry_caps)

    # Whiten, then solve unconstrained in the reduced space. Whitening by
    # sqrt(v) rather than forming (X'WX)^-1 keeps the condition number squared
    # out of the solve.
    s = np.sqrt(v)
    a = s[:, None] * (x @ u_basis)
    b = s * r

    theta, _, rank, sing = np.linalg.lstsq(a, b, rcond=None)
    f = u_basis @ theta

    fitted = x @ f
    resid = r - fitted

    ss_res = float(np.sum(v * resid**2))
    # Total sum of squares about the *weighted* mean, so R^2 is the fraction of
    # weighted return dispersion explained.
    r_bar = float(np.sum(v * r))
    ss_tot = float(np.sum(v * (r - r_bar) ** 2))

    return {
        "f": f,
        "resid": resid,
        "fitted": fitted,
        "weights": v,
        "n_assets": int(x.shape[0]),
        "rank": int(rank),
        "full_rank": bool(rank == u_basis.shape[1]),
        "cond": float(sing.max() / sing.min()) if sing.size and sing.min() > 0 else np.inf,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "constraint_residual": float(industry_caps @ f[1 : 1 + n_industries]),
        "industry_caps": industry_caps,
    }


def build_regression_panel(
    exposures: pd.DataFrame,
    style_cols: list[str],
    dummy_cols: list[str],
    by: str = "eom",
    return_col: str | None = None,
    cap_col: str | None = None,
    lag_exposures: bool | None = None,
    winsor_return_sigma: float | None = -1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    The exact frame the regression is run on: one row per (asset, return month).

    This is the only place the two input transforms happen, and it is worth
    having as a named artifact rather than an intermediate hidden inside
    ``run_panel``.

    ``ryan_sample_exposures`` is keyed (id, eom) and means "the exposures
    observed at eom" -- a true, self-consistent statement, which is why the lag
    is NOT applied there. But it makes that table a trap for anything that reads
    it: regressing its ``ret_exc`` on its own ``f_*`` columns is the
    contemporaneous fit, and it reports a value factor return of -18.1%/yr at
    t = -4.7 without a single diagnostic complaining.

    The frame returned here closes that hole by carrying **both** dates in every
    row -- ``exposure_eom`` (when the right-hand side was measured) and ``eom``
    (the month whose return is on the left) -- so the alignment is legible in the
    data instead of implicit in code. ``ret_exc_raw`` keeps the untrimmed return
    alongside the winsorized one so the trim is auditable too.
    """
    ret = return_col or config.RETURN_COL
    do_lag = config.LAG_EXPOSURES if lag_exposures is None else lag_exposures
    sigma = config.WINSOR_RETURN_SIGMA if winsor_return_sigma == -1.0 else winsor_return_sigma

    if do_lag:
        n_before = len(exposures)
        exposures = align_returns_to_lagged_exposures(
            exposures, style_cols, dummy_cols, by=by, return_col=ret, cap_col=cap_col
        )
        if verbose:
            print(f"⏮  exposures lagged one month: {n_before:,} → {len(exposures):,} rows "
                  f"(each security's first month has no prior exposure)")
    else:
        exposures = exposures.copy()
        # Stamped anyway, so the column means the same thing in both modes and a
        # stored panel always says what it was built from.
        exposures["exposure_eom"] = exposures[by]
        if verbose:
            print("⚠️  LAG_EXPOSURES is off — regressing month-t returns on exposures "
                  "measured at the END of month t. The value factor return this produces "
                  "is look-ahead, not a premium.")

    exposures["ret_exc_raw"] = pd.to_numeric(exposures[ret], errors="coerce")
    if sigma is not None:
        trimmed, n_clipped = winsorize_returns(exposures["ret_exc_raw"], exposures[by], sigma)
        exposures[ret] = trimmed
        if verbose:
            print(f"✂️  dependent variable winsorized at {sigma} robust sigma: "
                  f"{n_clipped:,} of {len(exposures):,} observations clipped "
                  f"({n_clipped / len(exposures):.3%})")

    return exposures


def prepare_cross_section(
    frame: pd.DataFrame,
    style_cols: list[str],
    dummy_cols: list[str],
    return_col: str | None = None,
    cap_col: str | None = None,
    fill_missing_exposure: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, list[str], dict]:
    """
    Turn one month of the exposures table into ``(X, r, cap, ids, factor_names)``.

    Rows are dropped only for the three things that make an asset unusable: no
    return (nothing to explain), no positive market cap (no regression weight),
    or no industry (a row of zeros across every dummy, which would quietly place
    the asset outside all industries and bias every industry return).

    A *missing style exposure* is not a reason to drop the asset. The exposures
    are standardized to a cap-weighted mean of zero, so zero is precisely the
    "no information, average name" value, and that is what USE4 substitutes.
    Dropping instead would remove the asset from Delta for that month as well,
    and roughly 2% of rows are missing at least one fundamental descriptor.

    **Industries with no members this month are dropped from the design.** An
    all-zero dummy column is exactly collinear with nothing at all -- it adds a
    free parameter the data cannot identify, and it would make the constraint
    basis divide by a zero cap. Their factor return is reported as NaN rather
    than 0, because "no estimate" and "estimated to be zero" are different
    statements and F must not treat them alike.
    """
    ret = return_col or config.RETURN_COL
    cap_name = cap_col or config.CAP_COL

    g = frame.copy()
    r = pd.to_numeric(g[ret], errors="coerce")
    cap = pd.to_numeric(g[cap_name], errors="coerce")
    in_some_industry = pd.Series(
        g[dummy_cols].fillna(0.0).to_numpy(dtype=float).sum(axis=1) > 0, index=g.index
    )

    has_return = r.notna()
    has_cap = cap.notna() & (cap > 0)
    keep = has_return & has_cap & in_some_industry

    # Tallied before the filter is applied, and each reason counted independently,
    # so a row failing two tests shows up under both rather than under whichever
    # test happened to run first.
    dropped = {
        "no_return": int((~has_return).sum()),
        "no_cap": int((~has_cap).sum()),
        "no_industry": int((~in_some_industry).sum()),
        "kept": int(keep.sum()),
    }

    g, r, cap = g[keep], r[keep], cap[keep]

    ind = g[dummy_cols].fillna(0.0).to_numpy(dtype=float)
    present = ind.sum(axis=0) > 0
    live_dummies = [c for c, ok in zip(dummy_cols, present) if ok]
    ind = ind[:, present]

    styles = g[style_cols].astype(float).fillna(fill_missing_exposure).to_numpy()
    n_filled = int(g[style_cols].isna().to_numpy().sum())

    x = np.column_stack([np.ones(len(g)), ind, styles])
    names = [MARKET_FACTOR] + live_dummies + [c[2:] if c.startswith("f_") else c for c in style_cols]

    dropped["exposures_filled"] = n_filled
    dropped["empty_industries"] = [c for c, ok in zip(dummy_cols, present) if not ok]
    return x, r.to_numpy(), cap.to_numpy(), g["id"], names, dropped


def run_panel(
    exposures: pd.DataFrame,
    style_cols: list[str],
    dummy_cols: list[str],
    by: str = "eom",
    return_col: str | None = None,
    cap_col: str | None = None,
    lag_exposures: bool | None = None,
    winsor_return_sigma: float | None = -1.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit every cross-section and stack the results.

    Returns ``(factor_returns, specific_returns, diagnostics, regression_panel)``:

    - ``factor_returns``  wide, indexed by date, one column per factor. Wide
      rather than long because every downstream consumer (the covariance in 03,
      the plots here) wants a T x K matrix; the long format required for the
      database is produced at write time.
    - ``specific_returns`` long, (id, eom) -- the grain Delta is estimated on.
    - ``diagnostics``     one row per cross-section.
    - ``regression_panel`` exactly the rows that were fitted, with the exposure
      date and the return date both present. Returned rather than rebuilt by the
      caller so that what gets stored is, by construction, what was regressed --
      a caller that re-derived it could drift from this function's parameters and
      no test would catch it.

    ``winsor_return_sigma`` defaults to the sentinel -1.0 meaning "use
    config.WINSOR_RETURN_SIGMA"; pass ``None`` to explicitly disable trimming,
    which None alone could not express.
    """
    ret = return_col or config.RETURN_COL
    exposures = build_regression_panel(
        exposures, style_cols, dummy_cols, by=by, return_col=ret, cap_col=cap_col,
        lag_exposures=lag_exposures, winsor_return_sigma=winsor_return_sigma,
        verbose=verbose,
    )

    all_names = [MARKET_FACTOR] + dummy_cols + [c[2:] if c.startswith("f_") else c for c in style_cols]

    f_rows, u_frames, diag_rows = {}, [], []

    for period, grp in exposures.groupby(by, sort=True):
        x, r, cap, ids, names, dropped = prepare_cross_section(
            grp, style_cols, dummy_cols, return_col=ret, cap_col=cap_col
        )
        n_ind = sum(1 for n in names if n.startswith("ind_"))
        n_sty = len(style_cols)

        if x.shape[0] <= x.shape[1]:
            # Fewer assets than parameters: the fit would interpolate the returns
            # exactly, forcing every residual (and every specific variance) to
            # zero. Refuse rather than emit a perfect-looking month.
            if verbose:
                print(f"⚠️  {pd.Timestamp(period).date()}: {x.shape[0]} assets vs "
                      f"{x.shape[1]} parameters — skipped")
            continue

        res = solve_cross_section(x, r, cap, n_ind, n_sty)

        # NaN, not 0, for industries absent this month -- see prepare_cross_section.
        row = dict.fromkeys(all_names, np.nan)
        row.update(dict(zip(names, res["f"])))
        f_rows[period] = row

        u_frames.append(pd.DataFrame({
            "id": ids.to_numpy(),
            by: period,
            "specific_return": res["resid"],
            "reg_weight": res["weights"],
        }))

        diag_rows.append({
            by: period,
            "n_assets": res["n_assets"],
            "n_params": x.shape[1],
            "resid_df": res["n_assets"] - (x.shape[1] - 1),
            "r2": res["r2"],
            "cond": res["cond"],
            "full_rank": res["full_rank"],
            "constraint_residual": res["constraint_residual"],
            "exposures_filled": dropped["exposures_filled"],
            "n_dropped_no_industry": dropped["no_industry"],
            "n_empty_industries": len(dropped["empty_industries"]),
        })

    factor_returns = pd.DataFrame.from_dict(f_rows, orient="index")[all_names]
    factor_returns.index.name = by
    specific = pd.concat(u_frames, ignore_index=True) if u_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diag_rows)

    if verbose:
        print(f"📐 fitted {len(factor_returns):,} cross-sections, "
              f"{len(all_names)} factors ({len(dummy_cols)} industries + "
              f"{len(style_cols)} styles + market)")
        print(f"   median R² = {diagnostics['r2'].median():.3f}, "
              f"median assets = {diagnostics['n_assets'].median():.0f}, "
              f"median residual df = {diagnostics['resid_df'].median():.0f}")

    # Only the months that were actually fitted. A month skipped for too-few
    # assets contributes no f(t) and no u(i,t), so storing its rows would put
    # observations in the regression panel that no regression ever saw.
    panel = exposures[exposures[by].isin(factor_returns.index)].reset_index(drop=True)

    return factor_returns, specific, diagnostics, panel


def to_long(factor_returns: pd.DataFrame, by: str = "eom") -> pd.DataFrame:
    """Wide T x K factor returns to the long (eom, factor, ret) storage format."""
    out = factor_returns.reset_index().melt(id_vars=by, var_name="factor", value_name="factor_return")
    out["factor_type"] = np.where(
        out["factor"] == MARKET_FACTOR, "market",
        np.where(out["factor"].str.startswith("ind_"), "industry", "style"),
    )
    return out.sort_values([by, "factor"]).reset_index(drop=True)
