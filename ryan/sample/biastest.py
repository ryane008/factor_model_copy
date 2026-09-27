"""
Out-of-sample bias tests: does the forecast made at t describe what happened at t+1?

The bias statistic is the standard deviation of standardized outcomes
z = r(t+1) / sigma_forecast(t). One means calibrated; above one means risk was
UNDERSTATED. Over the 180 testable dates the 95% band for a calibrated forecast
is roughly 1 +/- 0.10.

Every function pairs F(t) with f(t+1) -- never F(t) with f(t). A forecast is
only tested on a month it did not see. The one deliberate exception is the
full-sample path from ``factor_cov.estimate(window=None)``, which has seen every
month; testing it reproduces the in-sample illusion decision #2 is about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import factor_cov
import riskmath


def _pairs(path: factor_cov.CovPath, fr: pd.DataFrame):
    """(index into path, realized f(t+1)) for every forecast date with a next month."""
    pos = {d: i for i, d in enumerate(fr.index)}
    r = fr[path.factors].to_numpy()
    for i, d in enumerate(path.dates):
        t = pos[d]
        if t + 1 < len(r):
            yield i, r[t + 1]


def factor_z(path, fr, stage="final") -> pd.DataFrame:
    """z_k(t+1) = f_k(t+1) / sigma_k(t), dates x factors."""
    rows, idx = [], []
    for i, f1 in _pairs(path, fr):
        rows.append(f1 / np.sqrt(np.diag(path.stages[stage][i])))
        idx.append(path.dates[i])
    return pd.DataFrame(rows, index=idx, columns=path.factors)


def eigen_z(path, fr, stage="final", basis_stage="nw") -> np.ndarray:
    """
    Standardized returns of the eigenportfolios of ``basis_stage``'s F(t), priced
    under ``stage``'s F(t). Dates x K, columns ascending by eigenvalue.

    Using a FIXED basis (the pre-adjustment NW matrix) for every stage means
    rank k refers to the same portfolio across stages, so the curves are
    comparable -- which is the whole point of the plot.
    """
    out = []
    for i, f1 in _pairs(path, fr):
        _, u = np.linalg.eigh(path.stages[basis_stage][i])
        c = path.stages[stage][i]
        out.append((u.T @ f1) / np.sqrt(np.einsum("ik,ij,jk->k", u, c, u)))
    return np.array(out)


def minvar_factor_z(path, fr, stage="final", pin: str = "market") -> np.ndarray:
    """
    z of the minimum-variance FACTOR portfolio with unit exposure to ``pin``,
    rebuilt at every date from that date's F. This is what an optimizer does:
    it goes looking for the direction F(t) thinks is safest.
    """
    k = path.factors.index(pin)
    out = []
    for i, f1 in _pairs(path, fr):
        c = path.stages[stage][i]
        a = np.zeros(len(f1)); a[k] = 1.0
        y = np.linalg.solve(c, a)
        y /= a @ y
        out.append(y @ f1 / np.sqrt(y @ c @ y))
    return np.array(out)


def random_factor_z(path, fr, stage="final", n: int = 200, seed: int = 0) -> np.ndarray:
    """z for ``n`` fixed random factor portfolios (dates x n). The 'typical' portfolio."""
    rng = np.random.default_rng(seed)
    ys = rng.normal(size=(len(path.factors), n))
    out = []
    for i, f1 in _pairs(path, fr):
        c = path.stages[stage][i]
        out.append((f1 @ ys) / np.sqrt(np.einsum("kn,kl,ln->n", ys, c, ys)))
    return np.array(out)


def summarize(path, fr, stages=factor_cov.STAGES) -> pd.DataFrame:
    """One row per stage: the headline statistics 03 and 05 compare."""
    rows = []
    for s in stages:
        fz = factor_z(path, fr, s)
        ez = eigen_z(path, fr, s)
        rows.append({
            "stage": s,
            "factor bias (mean over K)": fz.std(ddof=1).mean(),
            "smallest-eigvec bias": riskmath.bias_stat(ez[:, 0]),
            "largest-eigvec bias": riskmath.bias_stat(ez[:, -1]),
            "min-var ptf bias": riskmath.bias_stat(minvar_factor_z(path, fr, s)),
            "random ptf bias (median)": float(np.median(random_factor_z(path, fr, s).std(axis=0, ddof=1))),
        })
    return pd.DataFrame(rows).set_index("stage")


def rolling_bias(z: pd.Series | np.ndarray, window: int = 12) -> pd.Series:
    """12-month rolling bias statistic of a z series -- the input to MRAD."""
    return pd.Series(z).rolling(window).std(ddof=1)
