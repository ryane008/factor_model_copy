"""
Mean-variance optimization against the factor risk model.

    maximize   alpha'w - (lambda / 2) w' Sigma w
    subject to sum(w) = 1,  lo <= w_i <= hi

with Sigma never formed: the objective and its gradient go through factor
space, ``Sigma w = X (F (X'w)) + Delta w``.

Two solvers, picked by size (``method="auto"``):

- **SLSQP** with an analytic gradient for small books (<= 250 names). It forms
  dense n x n matrices internally, which is fine at 20-220 names and hopeless at
  3,000.
- **Accelerated projected gradient** (FISTA with adaptive restart) above that.
  Each iteration is one factor-space gradient, O(NK), plus an exact projection
  onto {sum w = 1, lo <= w <= hi} by bisection. Nothing n x n ever exists, so a
  full-universe or Russell 3000 book solves in well under a second. The problem
  is strictly convex (Delta > 0), so both solvers reach the same unique optimum;
  test_logic checks that they agree.

Everything is in ANNUALIZED units (alpha annual, Sigma x 12), so lambda means
the same thing as in any textbook: at lambda = 4 an extra 1% of variance must
buy 2% of expected return.

A min-variance optimizer is the harshest consumer a covariance estimate has --
it searches out whichever direction F-hat understates most. That is why the
eigenfactor adjustment in 03 is a correctness requirement for this notebook,
not a refinement, and why 07 reports the optimized portfolio's risk under the
UNADJUSTED model too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import riskmodel


SLSQP_MAX_N = 250


def project_box_simplex(v: np.ndarray, lo: float, hi: float, total: float = 1.0) -> np.ndarray:
    """
    Euclidean projection onto {w : sum w = total, lo <= w_i <= hi}.

    The projection is clip(v - tau, lo, hi) for the scalar tau that makes the sum
    come out right; the sum is monotone in tau, so bisection finds it to machine
    precision in ~60 steps. O(n) per step.
    """
    n = v.size
    if not (n * lo - 1e-12 <= total <= n * hi + 1e-12):
        raise ValueError(f"infeasible: {n} names in [{lo}, {hi}] cannot sum to {total}")
    a, b = v.min() - hi, v.max() - lo          # sum(clip(v - a)) = n*hi, sum(clip(v - b)) = n*lo
    for _ in range(200):
        tau = 0.5 * (a + b)
        s = np.clip(v - tau, lo, hi).sum()
        if s > total:
            a = tau
        else:
            b = tau
        if b - a < 1e-15 * max(1.0, abs(tau)):
            break
    return np.clip(v - 0.5 * (a + b), lo, hi)


def _solve_pg(x, f, d, a, lam, lo, hi, w0, max_iter: int = 20000, tol: float = 1e-9) -> np.ndarray:
    """
    FISTA with adaptive restart for  min -a'w + lam/2 w'Sigma w  over the box-simplex.

    Step size 1/L with L = lam * lambda_max(Sigma), found by power iteration in
    factor space. Restart (O'Donoghue & Candes) whenever momentum points uphill,
    which is what keeps FISTA fast on the badly conditioned Sigma a single-name
    specific variance of 1e-4 next to a market variance of 3e-2 produces.

    ``tol`` is on the largest weight change per step. Measured at N=4000:
    1e-9 takes ~2 s and lands within 5e-6 of the exact weights (vol error 4e-11);
    1e-10 buys one more digit for 9x the time, and nothing here reports weights
    past 1e-4.
    """
    def sigma_mul(w):
        return x @ (f @ (x.T @ w)) + d * w

    v = np.random.default_rng(0).normal(size=len(d))
    for _ in range(100):
        v = sigma_mul(v)
        v /= np.linalg.norm(v)
    step = 1.0 / (1.02 * lam * float(v @ sigma_mul(v)))

    w = project_box_simplex(w0, lo, hi)
    y, t = w.copy(), 1.0
    for _ in range(max_iter):
        g = -a + lam * sigma_mul(y)
        w_new = project_box_simplex(y - step * g, lo, hi)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        if g @ (w_new - w) > 0:            # momentum going uphill: restart
            t, y = 1.0, w_new
        else:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = w_new + ((t - 1) / t_new) * (w_new - w)
            t = t_new
        w = w_new
    return w


def _parts(model: riskmodel.RiskModel, ids):
    sub = model.subset(ids)
    return (sub.X.to_numpy(), riskmodel.ANN * sub.F.to_numpy(),
            riskmodel.ANN * sub.delta.to_numpy(), sub)


def optimize(
    model: riskmodel.RiskModel,
    ids,
    alpha: pd.Series | None = None,
    risk_aversion: float = 4.0,
    long_only: bool = True,
    max_weight: float | None = None,
    w0: pd.Series | None = None,
    method: str = "auto",
) -> pd.Series:
    """
    Optimal weights over ``ids``. ``alpha=None`` gives the minimum-variance portfolio.

    ``method``: "auto" (SLSQP up to SLSQP_MAX_N names, projected gradient above),
    "slsqp" or "pg".

    Bounds: [0, max_weight] long-only, [-max_weight, max_weight] otherwise
    (max_weight None = 1 long-only / 2 long-short). Raises when the bounds make
    full investment impossible rather than returning a portfolio that silently
    violates the budget.
    """
    ids = pd.Index(ids)
    x, f, d, _ = _parts(model, ids)
    n = len(ids)
    hi = max_weight if max_weight is not None else (1.0 if long_only else 2.0)
    lo = 0.0 if long_only else -hi
    if hi * n < 1 - 1e-12:
        raise ValueError(f"max_weight {hi} x {n} names < 1: full investment is infeasible")

    a = np.zeros(n) if alpha is None else alpha.reindex(ids).fillna(0.0).to_numpy()
    lam = 1.0 if alpha is None else risk_aversion

    def obj(w):
        y = x.T @ w
        sw = x @ (f @ y) + d * w
        return -(a @ w) + 0.5 * lam * (w @ sw), -a + lam * sw

    start = np.full(n, 1.0 / n) if w0 is None else w0.reindex(ids).fillna(0.0).to_numpy()
    if method == "pg" or (method == "auto" and n > SLSQP_MAX_N):
        wv = _solve_pg(x, f, d, a, lam, lo, hi, start)
        w = pd.Series(wv, index=ids, name="weight")
        w[w.abs() < 1e-7] = 0.0
        return w / w.sum()
    start = np.clip(start, lo, hi)
    res = minimize(
        obj, start, jac=True, method="SLSQP",
        bounds=[(lo, hi)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    w = pd.Series(res.x, index=ids, name="weight")
    w[w.abs() < 1e-7] = 0.0
    return w / w.sum()


def frontier(
    model: riskmodel.RiskModel,
    ids,
    alpha: pd.Series,
    n_points: int = 25,
    long_only: bool = True,
    max_weight: float | None = None,
    method: str = "auto",
) -> pd.DataFrame:
    """
    Efficient frontier by sweeping risk aversion from ~risk-neutral to min-variance.

    Each point is re-solved from the previous one's weights, which keeps SLSQP on
    the same branch and makes the sweep several times faster.
    """
    rows, w_prev = [], None
    for lam in np.geomspace(200.0, 0.2, n_points):
        w = optimize(model, ids, alpha, lam, long_only, max_weight, w0=w_prev, method=method)
        w_prev = w
        rows.append({"risk_aversion": lam, "vol": model.vol(w),
                     "expected_return": float(alpha.reindex(w.index).fillna(0) @ w),
                     "n_names": int((w.abs() > 1e-4).sum()), "weights": w})
    out = pd.DataFrame(rows)
    out["sharpe"] = out["expected_return"] / out["vol"]
    return out.sort_values("vol").reset_index(drop=True)


def min_variance_closed_form(model: riskmodel.RiskModel) -> pd.Series:
    """
    Fully-invested, unconstrained (long-short) minimum-variance weights,
    Sigma^-1 1 / 1' Sigma^-1 1, via Woodbury so Sigma is never formed or inverted:

        Sigma^-1 = D^-1 - D^-1 X (F^-1 + X' D^-1 X)^-1 X' D^-1

    Only a K x K system is solved. This is the most aggressive consumer of the
    covariance there is -- no bounds to stop it -- which makes it the sharpest
    bias test of what the optimizer in 07 will do.
    """
    x = model.X.to_numpy()
    dinv = 1.0 / model.delta.to_numpy()
    fm = model.F.to_numpy()
    inner = np.linalg.inv(fm) + x.T @ (dinv[:, None] * x)
    b = dinv * np.ones(len(dinv))
    s1 = b - dinv * (x @ np.linalg.solve(inner, x.T @ b))
    return pd.Series(s1 / s1.sum(), index=model.ids, name="weight")
