"""
Estimator math shared by the factor covariance (03) and specific risk (04).

Pure numpy, no database, no config lookups -- every knob is an argument, so the
notebooks and test_logic.py exercise exactly the same code.

Three ideas live here.

**Exponential decay.** Risk levels move; a flat 60-month average answers "how
risky was this over five years" when the question is "how risky is it next
month". Weights halve every ``half_life`` observations, most recent heaviest.

**Newey-West is a long-run VARIANCE estimator here, not a standard-error fix.**
It addresses the AC in HAC and nothing else: cross-sectional
heteroskedasticity was already handled by the sqrt(cap) weights in step 4, and
the time-varying level is handled by decay + VRA. What it corrects is that the
variance of a sum of serially correlated monthly returns is not the sum of the
monthly variances. Measured lag-1 autocorrelation of u is mean -0.047, 66%
negative, so for specific risk NW nudges variance *down*, not up.

**Bartlett weights keep it PSD -- but only if the weighting is done right.**
The lag-l cross-covariance is built from sqrt(w)-scaled deviations,
``z_t = sqrt(w_t)(r_t - mu)``, so that

    S = sum_{|l|<=L} (1 - |l|/(L+1)) Gamma_l = 1/(L+1) sum_s b_s b_s'

with ``b_s = sum_{j=0..L} z_{s-j}``. That is a sum of outer products, PSD by
construction. Weighting each lag term by ``w_t`` alone (the obvious-looking
choice) breaks the identity and can go indefinite on a 60-month window.

The eigenfactor adjustment (Menchero, Wang & Orr 2011) is the fix for the fact
that at T/K ~ 3.3 the sample covariance systematically UNDERSTATES variance
along its smallest eigenvectors -- exactly the directions a min-variance
optimizer goes looking for.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Weights and lags
# ---------------------------------------------------------------------------

def exp_weights(n: int, half_life: float | None) -> np.ndarray:
    """
    Length-``n`` weights, OLDEST FIRST, summing to one.

    The last observation gets the largest weight and each step back multiplies
    by ``0.5 ** (1 / half_life)``. ``half_life=None`` gives equal weights, which
    is the switch back to a plain rolling sample covariance.
    """
    if n <= 0:
        raise ValueError(f"need n > 0, got {n}")
    if half_life is None:
        w = np.ones(n)
    else:
        if half_life <= 0:
            raise ValueError(f"half_life must be positive, got {half_life}")
        age = np.arange(n - 1, -1, -1, dtype=float)   # oldest has the largest age
        w = 0.5 ** (age / half_life)
    return w / w.sum()


def effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size, 1 / sum(w^2) for weights summing to one."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))


def newey_west_lags(t: int) -> int:
    """Rule-of-thumb Bartlett lag, floor(4 (T/100)^(2/9)): 3 at T=60, 5 at T=240."""
    return int(np.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def bartlett(lags: int) -> np.ndarray:
    """Kernel weights for lags 1..L: 1 - l/(L+1)."""
    return 1.0 - np.arange(1, lags + 1) / (lags + 1.0)


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

def nw_cov(
    r: np.ndarray,
    weights: np.ndarray | None = None,
    lags: int = 0,
    demean: bool = True,
) -> np.ndarray:
    """
    Exponentially weighted Newey-West covariance of a complete T x K panel.

    ``weights`` oldest first; None means equal. ``lags=0`` is the plain
    weighted covariance. Output is per-period (monthly) covariance.

    No NaNs allowed: the factor-return panel is complete by construction (every
    factor is estimated in every month), and a NaN here would mean an industry
    went empty -- which should fail loudly rather than be imputed.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if not np.all(np.isfinite(r)):
        raise ValueError("nw_cov needs a complete panel; got NaN/inf")
    t = r.shape[0]
    w = exp_weights(t, None) if weights is None else np.asarray(weights, float) / np.sum(weights)
    if w.shape[0] != t:
        raise ValueError(f"weights length {w.shape[0]} != T {t}")

    mu = w @ r if demean else np.zeros(r.shape[1])
    z = np.sqrt(w)[:, None] * (r - mu)

    s = z.T @ z
    for l, b in enumerate(bartlett(lags), start=1):
        if l >= t:
            break
        g = z[l:].T @ z[:-l]
        s += b * (g + g.T)
    return 0.5 * (s + s.T)


def nw_var_panel(
    u: np.ndarray,
    weights: np.ndarray,
    lags: int = 0,
    min_obs: int = 1,
    demean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-column Newey-West variance of a T x N panel WITH gaps.

    This is the specific-risk case: each asset is its own univariate series and
    assets drop in and out of the window. Missing observations get zero weight
    and the remaining weights are renormalized per asset, so an asset observed
    for 40 of 60 months is estimated from its own 40 months rather than being
    diluted by 20 zeros. A gap also zeroes every lag product that straddles it,
    which is the right answer: there is no observed autocovariance across a gap.

    Returns ``(var, n_obs)``; ``var`` is NaN where ``n_obs < min_obs``.
    Vectorized over assets -- this runs once per forecast date on ~170 series.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    mask = np.isfinite(u)
    n_obs = mask.sum(axis=0)

    w = np.asarray(weights, dtype=float)[:, None] * mask
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / np.where(wsum > 0, wsum, np.nan)
    u0 = np.where(mask, u, 0.0)
    mu = np.nansum(w * u0, axis=0) if demean else np.zeros(u.shape[1])
    z = np.sqrt(np.nan_to_num(w)) * np.where(mask, u0 - mu, 0.0)

    var = np.sum(z**2, axis=0)
    for l, b in enumerate(bartlett(lags), start=1):
        if l >= u.shape[0]:
            break
        var = var + 2.0 * b * np.sum(z[l:] * z[:-l], axis=0)

    var = np.where(n_obs >= min_obs, var, np.nan)
    return var, n_obs


def vol_corr_cov(
    r: np.ndarray,
    vol_half_life: float | None,
    corr_half_life: float | None,
    vol_lags: int,
    corr_lags: int,
) -> np.ndarray:
    """
    USE4's split: correlations from a long half-life, volatilities from a short one.

    Correlations are slow-moving and need many observations to pin down K(K-1)/2
    numbers; volatilities move fast and are one number each. One half-life
    cannot serve both. The combination sigma * C * sigma is PSD because it is a
    congruence of the (PSD) correlation matrix.
    """
    t = r.shape[0]
    c = nw_cov(r, exp_weights(t, corr_half_life), corr_lags)
    v = np.diag(nw_cov(r, exp_weights(t, vol_half_life), vol_lags))
    sd_c = np.sqrt(np.diag(c))
    corr = c / np.outer(sd_c, sd_c)
    sd = np.sqrt(v)
    return corr * np.outer(sd, sd)


def is_psd(m: np.ndarray, tol: float = 1e-12) -> bool:
    """Symmetric and no eigenvalue below -tol * largest."""
    m = np.asarray(m, dtype=float)
    if not np.allclose(m, m.T, atol=1e-12 * max(1.0, np.abs(m).max())):
        return False
    ev = np.linalg.eigvalsh(m)
    return bool(ev.min() >= -tol * max(ev.max(), 1e-300))


# ---------------------------------------------------------------------------
# Eigenfactor risk adjustment
# ---------------------------------------------------------------------------

def eigenfactor_adjust(
    f0: np.ndarray,
    weights: np.ndarray,
    n_sims: int = 1000,
    scale: float = 1.2,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """
    Menchero-Wang-Orr eigenfactor adjustment of a factor covariance ``f0``.

    Treat ``f0`` as the truth, simulate T-period histories from it, re-estimate,
    and measure how badly each re-estimated eigenvalue understates the true
    variance of its own eigenportfolio. The ratio is a function of eigenvalue
    RANK -- the smallest eigenvalues are the most understated, because sampling
    noise pushes the spectrum apart and an eigen-decomposition then labels the
    noise-deflated direction as "low risk". Rescale ``f0``'s eigenvalues by the
    simulated bias and rotate back.

    Simulation is done in ``f0``'s eigenbasis, where the truth is simply
    diag(d0): the result is identical to simulating in factor space and
    rotating, and skips two K x K matmuls per draw. The simulated histories use
    the SAME weights as the estimator, because what matters is the effective
    number of observations, not the nominal window length.

    ``scale`` is USE4's empirical a in gamma = a (lambda - 1) + 1; the
    simulation assumes normal, stationary returns and a > 1 compensates for the
    real world being less kind than that.

    Returns ``(adjusted, info)`` where info carries the per-eigenvalue bias
    ``lam`` (>= 1 means understated), the applied multiplier ``gamma`` and the
    raw eigen-decomposition, all ascending by eigenvalue.
    """
    f0 = 0.5 * (np.asarray(f0, float) + np.asarray(f0, float).T)
    d0, u0 = np.linalg.eigh(f0)
    if d0.min() <= 0:
        raise ValueError(f"eigenfactor adjustment needs a positive definite input; min eig {d0.min():.3e}")

    w = np.asarray(weights, float)
    w = w / w.sum()
    t, k = w.shape[0], d0.shape[0]
    sw = np.sqrt(w)[None, :, None]
    rng = np.random.default_rng(seed)

    # Batched so memory stays bounded at 1000 sims x 60 x 18.
    ratio_sum = np.zeros(k)
    done, batch = 0, 250
    while done < n_sims:
        m = min(batch, n_sims - done)
        b = rng.standard_normal((m, t, k)) * np.sqrt(d0)[None, None, :]
        mu = np.einsum("t,mtk->mk", w, b)
        z = sw * (b - mu[:, None, :])
        fm = np.einsum("mti,mtj->mij", z, z)
        dm, um = np.linalg.eigh(fm)
        # True variance of each simulated eigenportfolio under diag(d0).
        dtrue = np.einsum("mik,i,mik->mk", um, d0, um)
        ratio_sum += (dtrue / dm).sum(axis=0)
        done += m

    lam = np.sqrt(ratio_sum / n_sims)
    gamma = scale * (lam - 1.0) + 1.0
    adjusted = (u0 * (gamma**2 * d0)) @ u0.T
    return 0.5 * (adjusted + adjusted.T), {"lam": lam, "gamma": gamma, "d0": d0, "u0": u0}


# ---------------------------------------------------------------------------
# Volatility regime adjustment and bias statistics
# ---------------------------------------------------------------------------

def vra_multiplier(
    b2: np.ndarray,
    half_life: float,
    min_obs: int,
) -> float:
    """
    lambda^2 from a history of squared cross-sectional bias statistics.

    ``b2`` is oldest first and must contain only OUT-OF-SAMPLE values: each
    B_t^2 compares the month-t realization with the forecast made at t-1. The
    return value multiplies a variance (so its square root multiplies a vol).
    Returns 1.0 -- no adjustment -- until ``min_obs`` values exist, because an
    exponentially weighted mean of three noisy bias statistics would do more
    harm than good.
    """
    b2 = np.asarray(b2, float)
    b2 = b2[np.isfinite(b2)]
    if b2.size < min_obs:
        return 1.0
    w = exp_weights(b2.size, half_life)
    return float(w @ b2)


def bias_stat(z: np.ndarray) -> float:
    """Std dev of standardized outcomes r / sigma_forecast. 1 means calibrated."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    return float(np.std(z, ddof=1)) if z.size > 1 else np.nan


def bias_ci(n: int, level_z: float = 1.96) -> tuple[float, float]:
    """Approximate CI for a bias statistic on n normal draws: 1 +/- z sqrt(1/(2n))."""
    half = level_z * np.sqrt(1.0 / (2.0 * n))
    return 1.0 - half, 1.0 + half


def mrad(z: np.ndarray) -> float:
    """Mean rolling absolute deviation of the bias statistic from one, USE4's summary."""
    z = np.asarray(z, float)
    return float(np.nanmean(np.abs(z - 1.0)))


# ---------------------------------------------------------------------------
# Rolling driver
# ---------------------------------------------------------------------------

def rolling_windows(n: int, window: int | None, min_periods: int | None = None) -> Iterator[tuple[int, slice]]:
    """
    Yield ``(t, slice)`` for each forecast position t = the LAST observation used.

    ``window=None`` is the full-sample diagnostic: EVERY forecast position gets
    the whole panel, future included. That is deliberate look-ahead, kept as a
    switch because it reproduces the single-F behaviour whose in-sample bias
    statistic (~1.0) looks like success while out-of-sample it is 1.43.
    """
    if window is None:
        start = (min_periods or 1) - 1
        for t in range(start, n):
            yield t, slice(0, n)
        return
    first = (min_periods or window) - 1
    for t in range(first, n):
        yield t, slice(max(0, t - window + 1), t + 1)
