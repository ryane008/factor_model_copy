"""
Portfolio CSV input, shared by 06 (risk) and 07 (optimization).

CSV format -- header row required, column names case-insensitive:

    ticker   or  id     identifier; `id` is the CRSP permno and is unambiguous,
                        a ticker is resolved as-of the forecast date
    weight              optional; equal weight if the column is absent
    expected_return     optional, annualized excess return; used by 07 only

Weights may be negative (shorts). They are normalized to sum to one when
``normalize=True`` and the net is positive; a dollar-neutral book (net ~ 0) is
left as given, since dividing by ~0 would be meaningless.

Rows that cannot be priced are NEVER dropped silently: they come back in the
report with a reason, and the notebook prints them before anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import riskmodel


def load_portfolio(
    path: str | Path,
    store: riskmodel.ModelStore,
    as_of=None,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a portfolio CSV and resolve it against the model at ``as_of``.

    Returns ``(holdings, report)``. ``holdings`` is indexed by model id with
    columns ticker, conm, weight (and expected_return if supplied); ``report``
    has one row per CSV line with its resolution status.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "permno" in raw.columns and "id" not in raw.columns:
        raw = raw.rename(columns={"permno": "id"})
    key = "id" if "id" in raw.columns else "ticker" if "ticker" in raw.columns else None
    if key is None:
        raise ValueError(f"{path}: need a `ticker` or `id` column, got {list(raw.columns)}")

    if "weight" not in raw.columns:
        raw["weight"] = 1.0 / len(raw)
        print(f"ℹ️  no `weight` column — equal-weighting {len(raw)} names")
    raw["weight"] = pd.to_numeric(raw["weight"], errors="coerce")
    if raw["weight"].isna().any():
        raise ValueError(f"{path}: non-numeric weight on rows {list(raw.index[raw['weight'].isna()])}")

    report = store.resolve(raw[key].tolist(), as_of)
    report["weight"] = raw["weight"].to_numpy()
    if "expected_return" in raw.columns:
        report["expected_return"] = pd.to_numeric(raw["expected_return"], errors="coerce").to_numpy()

    ok = report[report["status"] == "ok"]
    agg = {"weight": "sum", "ticker": "first", "conm": "first"}
    if "expected_return" in ok.columns:
        agg["expected_return"] = "first"
    holdings = ok.groupby("id").agg(agg)
    if len(holdings) < len(ok):
        print(f"ℹ️  {len(ok) - len(holdings)} duplicate rows merged (weights summed)")

    input_net = report["weight"].sum()
    kept_net = holdings["weight"].sum()
    if normalize and kept_net > 1e-9:
        holdings["weight"] = holdings["weight"] / kept_net

    bad = report[report["status"] != "ok"]
    print(f"📄 {len(raw)} rows read, {len(holdings)} priced, {len(bad)} not priced")
    if len(bad):
        print(f"   ⚠️  {bad['weight'].sum() / input_net:.1%} of input weight dropped:")
        for _, r in bad.iterrows():
            print(f"      {str(r['input']):>10s}  {r['status']:<14s} {r.get('note', '') or ''}")
    if normalize and kept_net > 1e-9 and abs(kept_net - 1) > 1e-9:
        print(f"   weights rescaled by 1/{kept_net:.4f} to sum to 1")
    # Priced but outside the coverage universe (e.g. an R2000 name under the R1000
    # profile): the risk numbers are valid -- the model is estimated on it -- but
    # say so, because it is not in the benchmark the notebooks compare against.
    outside = ok[~ok["in_coverage"].fillna(True).astype(bool)] if "in_coverage" in ok else ok.iloc[:0]
    if len(outside):
        print(f"   ℹ️  {len(outside)} holdings are priced but outside the coverage universe: "
              f"{', '.join(outside['input'].astype(str))}")
    return holdings, report


def default_alpha(model: riskmodel.RiskModel, factor_returns: pd.DataFrame,
                  shrink: float, ids=None) -> pd.Series:
    """
    Illustrative expected returns: shrink x historical mean STYLE factor returns x exposures.

    Uses factor returns up to and including the forecast date only. Market and
    industries are excluded on purpose: every name has market exposure 1, so it
    is a constant under a fully-invested budget and cannot change the solution,
    and industry means over this sample are sector luck, not premia. The mean
    style returns themselves are barely significant (value t = 2.1 is the best),
    hence the shrink. This exists to exercise the optimizer, not to forecast.
    """
    hist = factor_returns.loc[:model.date]
    styles = [f for f in model.factors if riskmodel._group(f) == "style"]
    premia = hist[styles].mean() * riskmodel.ANN * shrink
    x = model.X if ids is None else model.X.loc[ids]
    return (x[styles] @ premia).rename("expected_return")
