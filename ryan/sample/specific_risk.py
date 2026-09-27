"""
Step 6: specific risk Delta(t), one variance per (asset, forecast date).

Same forecast-date convention as factor_cov: Delta(t) uses u(i,s) for s <= t
and forecasts the variance of u(i,t+1). The assets that get a forecast at t
are those with an exposure row at t -- the names you could hold at the end of
month t, priced with X(t).

Three stages, each kept so 05 can show what it bought:

    raw     decay-weighted (optionally Newey-West) variance of the asset's own u
    shrunk  Bayesian shrinkage of the VOL toward its size decile's cap-weighted
            mean (USE4 eq. 5-9)
    final   shrunk x lambda_S^2 (specific volatility regime adjustment)

The shrinkage targets the over-dispersion of sample specific vols, which is
distinct from what the eigenfactor adjustment does for F: the extreme vol
estimates are extreme partly because of estimation error, so the highest are
biased up and the lowest biased down. An optimizer takes the lowest at face
value and piles into them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import riskmath


def shrink_to_size_buckets(
    sigma: np.ndarray,
    cap: np.ndarray,
    n_buckets: int,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    USE4 Bayesian shrinkage of specific VOLS toward their size bucket.

        prior_s = cap-weighted mean of sigma in bucket s
        disp_s  = sqrt(mean over bucket of (sigma - prior_s)^2)
        v_n     = q|sigma_n - prior| / (disp_s + q|sigma_n - prior|)
        out     = v_n prior + (1 - v_n) sigma_n

    The weight grows with distance from the prior, so a name sitting at its
    bucket mean is left alone and an outlier is pulled in hardest.

    NaN sigma (not enough history) takes the prior outright, v = 1. Buckets are
    formed on cap RANK among all assets, including the NaN ones, so a thin-history
    name is compared with names of its own size.

    Returns ``(shrunk, v, bucket)``.
    """
    sigma = np.asarray(sigma, float)
    cap = np.asarray(cap, float)
    n = sigma.size
    rank = pd.Series(cap).rank(method="first").to_numpy()
    bucket = np.minimum((rank - 1) * n_buckets // n, n_buckets - 1).astype(int)

    prior = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    for b in range(n_buckets):
        m = bucket == b
        ok = m & np.isfinite(sigma)
        if not ok.any():
            continue
        p = np.average(sigma[ok], weights=cap[ok])
        prior[m] = p
        disp[m] = np.sqrt(np.mean((sigma[ok] - p) ** 2))

    # A bucket with no usable sigma at all falls back to the cross-sectional
    # cap-weighted mean -- a thin-history name still needs SOME risk.
    ok_all = np.isfinite(sigma)
    if np.isnan(prior).any() and ok_all.any():
        overall = np.average(sigma[ok_all], weights=cap[ok_all])
        prior = np.where(np.isnan(prior), overall, prior)
        disp = np.where(np.isnan(disp), 0.0, disp)

    dist = q * np.abs(sigma - prior)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(disp + dist > 0, dist / (disp + dist), 0.0)
    v = np.where(np.isfinite(sigma), v, 1.0)
    shrunk = np.where(np.isfinite(sigma), v * prior + (1 - v) * sigma, prior)
    return shrunk, v, bucket


def estimate(
    specific_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    window: int | None = None,
    half_life: float | None = None,
    lags: int | None = -1,
    min_obs: int | None = None,
    q: float | None = None,
    n_buckets: int | None = None,
    vra_half_life: float | None = None,
    vra_min_obs: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Delta(t) for every forecast date, long: one row per (id, forecast_date).

    ``specific_returns`` long (id, eom, specific_return); ``exposures`` supplies
    who is live at each forecast date and their market cap AT that date (used
    for the size buckets and the cap-weighted VRA). ``forecast_dates`` should be
    F(t)'s dates, so the two halves of Sigma always exist together.

    ``lags=-1`` means config (None there = rule of thumb on the window).
    """
    window = config.SPECIFIC_WINDOW if window is None else window
    hl = config.SPECIFIC_HALFLIFE if half_life is None else half_life
    if lags == -1:
        lags = config.SPECIFIC_NW_LAGS
    if lags is None:
        lags = riskmath.newey_west_lags(window)
    min_obs = config.SPECIFIC_MIN_OBS if min_obs is None else min_obs
    q = config.SHRINK_Q if q is None else q
    n_buckets = config.SHRINK_BUCKETS if n_buckets is None else n_buckets
    vra_hl = config.SPECIFIC_VRA_HALFLIFE if vra_half_life is None else vra_half_life
    vra_min = config.SPECIFIC_VRA_MIN_OBS if vra_min_obs is None else vra_min_obs

    u = specific_returns.pivot(index="eom", columns="id", values="specific_return").sort_index()
    u.index = pd.to_datetime(u.index)
    months = u.index
    w_full = riskmath.exp_weights(window, hl)

    ex = exposures[["id", "eom", config.CAP_COL]].copy()
    ex["eom"] = pd.to_datetime(ex["eom"])
    ex = ex[pd.to_numeric(ex[config.CAP_COL], errors="coerce") > 0]
    live_by_date = {d: g.set_index("id")[config.CAP_COL].astype(float) for d, g in ex.groupby("eom")}

    frames = []
    for date in pd.DatetimeIndex(forecast_dates):
        if date not in live_by_date:
            continue
        live = live_by_date[date]
        t = months.get_loc(date)
        lo = max(0, t - window + 1)
        block = u.iloc[lo : t + 1].reindex(columns=live.index)
        w = w_full[-block.shape[0]:]
        var, n_obs = riskmath.nw_var_panel(block.to_numpy(), w, lags, min_obs)
        # Univariate NW can in principle push a variance to ~0 when the lag
        # terms are strongly negative; a zero on Delta's diagonal is exactly the
        # singular-Sigma failure the stratified sample exists to prevent.
        var = np.where(np.isfinite(var), np.maximum(var, 1e-8), np.nan)
        sigma = np.sqrt(var)
        shrunk, v, bucket = shrink_to_size_buckets(sigma, live.to_numpy(), n_buckets, q)
        frames.append(pd.DataFrame({
            "id": live.index.to_numpy(),
            "forecast_date": date,
            "cap": live.to_numpy(),
            "n_obs": n_obs,
            "size_bucket": bucket,
            "shrink_weight": v,
            "specific_var_raw": var,
            "specific_var_shrunk": shrunk**2,
        }))

    out = pd.concat(frames, ignore_index=True)

    # VRA. B_t^2 = cap-weighted mean of (u(t) / sigma_shrunk(t-1))^2 over names
    # that had a forecast at t-1 AND a residual at t. Strictly out of sample.
    dates = pd.DatetimeIndex(sorted(out["forecast_date"].unique()))
    b2 = np.full(len(dates), np.nan)
    for i in range(1, len(dates)):
        prev = out[out["forecast_date"] == dates[i - 1]].set_index("id")
        if dates[i] not in u.index:
            continue
        u_now = u.loc[dates[i]].reindex(prev.index)
        ok = u_now.notna()
        if ok.sum() < 10:
            continue
        z2 = u_now[ok] ** 2 / prev.loc[ok, "specific_var_shrunk"]
        b2[i] = float(np.average(z2, weights=prev.loc[ok, "cap"]))
    lam2 = np.array([riskmath.vra_multiplier(b2[: i + 1], vra_hl, vra_min) for i in range(len(dates))])
    vra = pd.DataFrame({"forecast_date": dates, "bias_sq": b2, "vra_lambda": np.sqrt(lam2)})
    out = out.merge(vra[["forecast_date", "vra_lambda"]], on="forecast_date", how="left")
    out["specific_var"] = out["specific_var_shrunk"] * out["vra_lambda"] ** 2
    out.attrs["vra"] = vra

    if verbose:
        print(f"🎯 Δ(t) at {len(dates)} forecast dates, {len(out):,} (asset, date) rows")
        print(f"   window {window}, half-life {hl}, NW lags {lags}, min obs {min_obs}, "
              f"shrink q {q} over {n_buckets} size buckets")
        thin = out["specific_var_raw"].isna().mean()
        print(f"   {thin:.1%} of rows had < {min_obs} obs and took the size-bucket prior")
        print(f"   specific VRA λ range {vra['vra_lambda'].min():.2f}–{vra['vra_lambda'].max():.2f}")
    return out
