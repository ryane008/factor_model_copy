"""
Step 5: the factor covariance F(t), one K x K matrix per forecast date.

Forecast-date convention, used everywhere downstream: F(t) is built from factor
returns f(s) for s <= t -- the window ENDS at t -- and it forecasts the
covariance of f(t+1). So a bias test pairs F(t) with f(t+1), and a portfolio
held at the end of month t is priced with X(t), F(t), Delta(t).

Every stage is kept rather than only the final matrix, because the whole point
of 05's bias tests is to show what each correction bought:

    raw     equal-weighted sample covariance on the window (no decay, no NW)
    nw      decay + Newey-West, USE4 vol/corr half-life split
    eigen   nw + eigenfactor adjustment
    final   eigen x lambda_F^2 (volatility regime adjustment)

F is ONE K x K matrix over all factors jointly -- never an industry block plus a
style block, which would silently zero every industry-style covariance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import riskmath

STAGES = ("raw", "nw", "eigen", "final")


@dataclass
class CovPath:
    """F(t) at every forecast date, for every stage. Arrays are D x K x K."""
    dates: pd.DatetimeIndex
    factors: list[str]
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    diag: pd.DataFrame | None = None

    def at(self, date, stage: str = "final") -> pd.DataFrame:
        i = self.dates.get_loc(pd.Timestamp(date))
        return pd.DataFrame(self.stages[stage][i], index=self.factors, columns=self.factors)


def cross_sectional_bias_sq(f_next: np.ndarray, cov: np.ndarray) -> float:
    """
    B_t^2 = mean_k (f_k,t+1 / sigma_k(t))^2, USE4's factor cross-sectional bias.

    Averaged over factors in the same month, so it measures the LEVEL of risk
    across the whole model at one point in time -- which is what the regime
    adjustment scales. A value of 1 is calibrated.
    """
    sd = np.sqrt(np.diag(cov))
    return float(np.mean((f_next / sd) ** 2))


def estimate(
    factor_returns: pd.DataFrame,
    window: int | None = -1,
    vol_half_life: float | None = None,
    corr_half_life: float | None = None,
    lags: int | None = -1,
    eigen_sims: int | None = None,
    eigen_scale: float | None = None,
    vra_half_life: float | None = None,
    vra_min_obs: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> CovPath:
    """
    Rolling F(t) from the wide T x K factor-return frame.

    Every argument defaults to config. ``window`` and ``lags`` use -1 as the
    "use config" sentinel, the same convention as run_panel, because None has a
    meaning of its own for both: ``window=None`` is the full-sample look-ahead
    diagnostic (one F fitted on all months, used at every date), and
    ``lags=None`` is the rule of thumb on the window length.
    ``eigen_sims=0`` skips the eigenfactor adjustment (eigen == nw), which is the
    fast path while iterating on anything else.

    VRA is computed strictly out of sample: lambda at date t uses B_s^2 for
    s <= t, and each B_s^2 compares f(s) with the pre-VRA forecast made at s-1.
    It is applied to the eigen-adjusted matrix, the thing it is correcting.
    """
    window = config.COV_WINDOW if window == -1 else window
    vol_hl = config.FACTOR_VOL_HALFLIFE if vol_half_life is None else vol_half_life
    corr_hl = config.FACTOR_CORR_HALFLIFE if corr_half_life is None else corr_half_life
    sims = config.EIGEN_SIMS if eigen_sims is None else eigen_sims
    scale = config.EIGEN_SCALE if eigen_scale is None else eigen_scale
    vra_hl = config.VRA_HALFLIFE if vra_half_life is None else vra_half_life
    vra_min = config.VRA_MIN_OBS if vra_min_obs is None else vra_min_obs
    seed = config.EIGEN_SEED if seed is None else seed

    fr = factor_returns.sort_index()
    if fr.isna().any().any():
        bad = fr.columns[fr.isna().any()].tolist()
        raise ValueError(f"factor returns have gaps in {bad}; an empty industry upstream?")
    r = fr.to_numpy(dtype=float)
    n, k = r.shape

    win_len = window if window is not None else n
    if lags == -1:
        lags = config.FACTOR_NW_LAGS
    if lags is None:
        lags = riskmath.newey_west_lags(win_len)

    positions = list(riskmath.rolling_windows(
        n, window, min_periods=config.COV_WINDOW if window is None else None))
    d = len(positions)
    out = {s: np.empty((d, k, k)) for s in STAGES}
    rows = []

    for i, (t, sl) in enumerate(positions):
        if window is None and i > 0:
            # Full-sample diagnostic: the identical matrix at every date.
            for s in ("raw", "nw", "eigen"):
                out[s][i] = out[s][0]
            rows.append({**rows[0], "forecast_date": fr.index[t]})
            continue
        rw = r[sl]
        out["raw"][i] = riskmath.nw_cov(rw, None, 0)
        nw = riskmath.vol_corr_cov(rw, vol_hl, corr_hl, lags, lags)
        out["nw"][i] = nw

        if sims:
            adj, info = riskmath.eigenfactor_adjust(
                nw, riskmath.exp_weights(rw.shape[0], corr_hl), n_sims=sims,
                scale=scale, seed=None if seed is None else seed + t,
            )
            lam = info["lam"]
        else:
            adj, lam = nw, np.ones(k)
        out["eigen"][i] = adj

        ev = np.linalg.eigvalsh(adj)
        rows.append({
            "forecast_date": fr.index[t],
            "n_obs": rw.shape[0],
            "nw_lags": lags,
            "eigen_bias_smallest": float(lam[0]),
            "eigen_bias_largest": float(lam[-1]),
            "cond_raw": _cond(out["raw"][i]),
            "cond_final": float(ev.max() / ev.min()),
        })

    # VRA pass. Needs the whole sequence of pre-VRA forecasts, so it runs after.
    b2 = np.full(d, np.nan)
    for i, (t, _) in enumerate(positions):
        if i > 0 and window is not None:
            # f(t) against the forecast made one month earlier.
            b2[i] = cross_sectional_bias_sq(r[t], out["eigen"][i - 1])
    for i in range(d):
        lam2 = riskmath.vra_multiplier(b2[: i + 1], vra_hl, vra_min) if window is not None else 1.0
        out["final"][i] = lam2 * out["eigen"][i]
        rows[i]["bias_sq"] = b2[i]
        rows[i]["vra_lambda"] = float(np.sqrt(lam2))

    diag = pd.DataFrame(rows)
    path = CovPath(pd.DatetimeIndex(diag["forecast_date"]), list(fr.columns), out, diag)

    if verbose:
        print(f"🧮 F(t) at {d} forecast dates  ({path.dates[0].date()} → {path.dates[-1].date()})")
        print(f"   window {window if window is not None else 'FULL SAMPLE (look-ahead diagnostic)'}, "
              f"vol/corr half-life {vol_hl}/{corr_hl}, NW lags {lags}, "
              f"eigen sims {sims}, VRA half-life {vra_hl}")
        print(f"   eigen bias of smallest eigenvalue: median {diag['eigen_bias_smallest'].median():.3f}; "
              f"VRA λ range {diag['vra_lambda'].min():.2f}–{diag['vra_lambda'].max():.2f}")
    return path


def canonical_order(names) -> list[str]:
    """market, then industries by GICS code, then styles in config order.

    Database reads come back in no guaranteed order, and a covariance whose rows
    and columns are permuted differently from X's columns is a silent disaster,
    so every reader re-imposes this one order.
    """
    styles = list(config.STYLE_DESCRIPTORS)
    names = list(names)
    def key(n):
        if n == "market":
            return (0, 0, n)
        if n.startswith("ind_"):
            return (1, 0, n)
        return (2, styles.index(n) if n in styles else len(styles), n)
    return sorted(names, key=key)


def _cond(m: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(m)
    return float(ev.max() / ev.min()) if ev.min() > 0 else np.inf


def to_long(path: CovPath) -> pd.DataFrame:
    """
    Long storage format: (forecast_date, factor_i, factor_j, cov, cov_raw, cov_nw, cov_eigen).

    Both triangles are stored. It doubles 153 rows to 324 per date, and it means
    every reader can pivot straight back to a full matrix without having to know
    to mirror it -- a reader that forgot would get a lower-triangular "F" with
    zero upper covariances and still a plausible-looking diagonal.
    """
    k = len(path.factors)
    ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    frames = []
    for d, date in enumerate(path.dates):
        frames.append(pd.DataFrame({
            "forecast_date": date,
            "factor_i": np.asarray(path.factors)[ii.ravel()],
            "factor_j": np.asarray(path.factors)[jj.ravel()],
            "cov": path.stages["final"][d].ravel(),
            "cov_raw": path.stages["raw"][d].ravel(),
            "cov_nw": path.stages["nw"][d].ravel(),
            "cov_eigen": path.stages["eigen"][d].ravel(),
        }))
    return pd.concat(frames, ignore_index=True)


def from_long(long: pd.DataFrame, col: str = "cov") -> dict[pd.Timestamp, pd.DataFrame]:
    """Inverse of to_long for one stage: {forecast_date: K x K DataFrame}."""
    long = long.copy()
    long["forecast_date"] = pd.to_datetime(long["forecast_date"])
    order = canonical_order(long["factor_i"].unique())
    out = {}
    for date, g in long.groupby("forecast_date", sort=True):
        m = g.pivot(index="factor_i", columns="factor_j", values=col).loc[order, order]
        out[pd.Timestamp(date)] = m
    return out


def path_from_long(long: pd.DataFrame, diag: pd.DataFrame | None = None) -> CovPath:
    """Rebuild a CovPath (every stage) from the stored table -- the resume path in 03."""
    cols = {"raw": "cov_raw", "nw": "cov_nw", "eigen": "cov_eigen", "final": "cov"}
    per = {s: from_long(long, c) for s, c in cols.items()}
    dates = pd.DatetimeIndex(sorted(per["final"]))
    factors = list(per["final"][dates[0]].columns)
    stages = {s: np.stack([per[s][d].to_numpy() for d in dates]) for s in cols}
    if diag is not None:
        diag = diag.copy()
        diag["forecast_date"] = pd.to_datetime(diag["forecast_date"])
    return CovPath(dates, factors, stages, diag)
