"""
Single-asset query: everything the risk model says about one security.

    report = query.asset_report(store, model, "XOM")

returns a dict of small DataFrames that 08_asset_query.ipynb renders. Kept
separate from the notebook so it can be called from anywhere -- another
notebook, a script, a test -- with the same answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import riskmodel


def find(store: riskmodel.ModelStore, model: riskmodel.RiskModel, query) -> float:
    """Resolve a ticker, permno or company-name fragment to one model id, or raise."""
    res = store.resolve([query], model.date).iloc[0]
    if res["status"] == "ok" and res["id"] in model.ids:
        return res["id"]
    if res["status"] == "ambiguous":
        raise KeyError(f"{query!r} is ambiguous at {model.date.date()}: {res['note']} — query by id")
    # Name fragment fallback, e.g. "chevron".
    hit = model.meta[model.meta["conm"].str.contains(str(query), case=False, na=False)]
    if len(hit) == 1:
        return hit.index[0]
    if len(hit) > 1:
        raise KeyError(f"{query!r} matches {len(hit)} names: "
                       f"{hit[['ticker', 'conm']].to_dict('index')} — be more specific or use the id")
    raise KeyError(f"{query!r}: {res['status']}. It must be one of the {len(model.ids)} names "
                   f"priced at {model.date.date()} — see query.universe(model).")


def universe(model: riskmodel.RiskModel, coverage_only: bool = True) -> pd.DataFrame:
    """
    Every priced name at the model date, with its headline risk -- the menu.

    ``coverage_only`` hides names the model is estimated on but does not cover
    (the R3000-not-R1000 names under the R1000 profile). No effect elsewhere.
    """
    v = model.asset_vols()
    out = model.meta[["ticker", "conm", "sector_name", "cap"]].join(v[["total_vol", "specific_vol"]])
    out["beta"] = model.betas()
    if coverage_only and "in_coverage" in model.meta:
        out = out[model.meta["in_coverage"].reindex(out.index).fillna(False).astype(bool)]
    return out.sort_values(["sector_name", "cap"], ascending=[True, False])


def asset_report(store: riskmodel.ModelStore, model: riskmodel.RiskModel, query, top_n: int = 10) -> dict:
    """
    Profile, exposures, risk decomposition, covariance neighbours, and history.

    Covariances come from one row of Sigma, ``X @ (F @ x_i)``; Sigma itself is
    never formed. Between two DIFFERENT assets the specific terms are zero by
    the model's assumption, so every off-diagonal covariance is purely a factor
    covariance -- which is why two names in the same sector with similar style
    exposures always top the list, and why the model cannot see a pair
    relationship (a merger arb, a supplier) that no factor captures.
    """
    aid = find(store, model, query)
    x = model.X.loc[aid]
    fm = model.F.to_numpy()
    fx = fm @ x.to_numpy()
    fac_var = float(x.to_numpy() @ fx)
    spec_var = float(model.delta.loc[aid])
    tot_var = fac_var + spec_var
    ann = riskmodel.ANN
    meta = model.meta.loc[aid]
    betas = model.betas()

    # Realized history for context: trailing vols and a realized beta to the
    # cap-weighted sample market, from monthly excess returns up to the date.
    rets = store.realized_returns()
    rets = rets.loc[:model.date]
    ri = rets[aid].dropna() if aid in rets else pd.Series(dtype=float)
    mkt = _cap_weighted_market(store, model.date)
    joined = pd.concat([ri.rename("a"), mkt.rename("m")], axis=1).dropna().tail(60)
    real_beta = (joined.cov().loc["a", "m"] / joined["m"].var()) if len(joined) > 24 else np.nan

    profile = pd.Series({
        "id": aid,
        "ticker": meta["ticker"],
        "company": meta["conm"],
        "sector": meta["sector_name"],
        "market cap ($M)": meta["cap"],
        "forecast date": model.date.date(),
        "ticker as of": store.ticker_table(model.date).loc[aid, "ticker_asof"].date()
        if pd.notna(store.ticker_table(model.date).loc[aid, "ticker_asof"]) else None,
        "predicted total vol": np.sqrt(ann * tot_var),
        "predicted factor vol": np.sqrt(ann * fac_var),
        "predicted specific vol": np.sqrt(ann * spec_var),
        "% variance specific": spec_var / tot_var,
        "predicted beta (sample mkt)": betas.loc[aid],
        "realized vol, trailing 12m": ri.tail(12).std() * np.sqrt(ann) if len(ri) >= 12 else np.nan,
        "realized vol, trailing 60m": ri.tail(60).std() * np.sqrt(ann) if len(ri) >= 24 else np.nan,
        "realized beta, trailing 60m": real_beta,
        "vol rank in universe": f"{int((model.asset_vols()['total_vol'] <= np.sqrt(ann * tot_var)).sum())}"
                                f" of {len(model.ids)}",
    }, name="value")

    styles = [f for f in model.factors if riskmodel._group(f) == "style"]
    exposures = pd.DataFrame({
        "exposure": x[styles],
        "universe percentile": [(model.X[f] <= x[f]).mean() for f in styles],
        "factor vol (ann.)": np.sqrt(ann * np.diag(model.F.loc[styles, styles])),
    })

    contrib = pd.DataFrame({
        "exposure": x,
        "var_contrib": ann * x.to_numpy() * fx,
    }, index=model.factors)
    contrib["pct_of_total_var"] = contrib["var_contrib"] / (ann * tot_var)
    contrib = contrib[contrib["exposure"] != 0]
    contrib.loc["specific"] = [np.nan, ann * spec_var, spec_var / tot_var]

    cov = model.cov_row(aid).drop(aid)
    corr = model.corr_row(aid).drop(aid)
    peers = pd.DataFrame({"cov": cov, "corr": corr}).join(
        model.meta[["ticker", "conm", "sector_name"]])
    peers["same_sector"] = peers["sector_name"] == meta["sector_name"]
    peers = peers[["ticker", "conm", "sector_name", "same_sector", "cov", "corr"]]

    return {
        "id": aid,
        "profile": profile,
        "exposures": exposures,
        "risk_contrib": contrib.sort_values("var_contrib", ascending=False),
        "highest_cov": peers.nlargest(top_n, "cov"),
        "lowest_cov": peers.nsmallest(top_n, "cov"),
        "highest_corr": peers.nlargest(top_n, "corr"),
        "lowest_corr": peers.nsmallest(top_n, "corr"),
        "all_peers": peers,
        "history": store.asset_history(aid),
        "realized": ri,
    }


def _cap_weighted_market(store: riskmodel.ModelStore, until) -> pd.Series:
    """Monthly return of the cap-weighted sample, weights from the PRIOR month's cap."""
    ex = store.exposures[store.exposures["eom"] <= until][["id", "eom", "ret_exc", "market_equity"]].copy()
    ex["period"] = ex["eom"].dt.to_period("M")
    cap_prev = ex[["id", "period", "market_equity"]].copy()
    cap_prev["period"] = cap_prev["period"] + 1
    m = ex[["id", "eom", "period", "ret_exc"]].merge(
        cap_prev.rename(columns={"market_equity": "w"}), on=["id", "period"])
    m = m.dropna(subset=["ret_exc", "w"])
    m = m[m["w"] > 0]
    return m.groupby("eom").apply(lambda g: np.average(g["ret_exc"], weights=g["w"]), include_groups=False)


def pair_report(store: riskmodel.ModelStore, model: riskmodel.RiskModel, a, b) -> dict:
    """
    Why two names co-move, factor by factor.

    cov(a, b) = sum_k x_a,k (F x_b)_k exactly -- specific risk contributes nothing
    between distinct names -- so the table below adds up to the model covariance
    and shows which factors carry it. Realized correlation over the trailing 60
    months is alongside for a sanity check.
    """
    ia, ib = find(store, model, a), find(store, model, b)
    if ia == ib:
        raise ValueError("pick two different names")
    xa, xb = model.X.loc[ia], model.X.loc[ib]
    fxb = model.F.to_numpy() @ xb.to_numpy()
    parts = pd.DataFrame({"exposure a": xa, "exposure b": xb,
                          "cov contrib": riskmodel.ANN * xa.to_numpy() * fxb}, index=model.factors)
    parts = parts[(parts["exposure a"] != 0) | (parts["exposure b"] != 0)]
    cov = float(parts["cov contrib"].sum())
    vols = model.asset_vols()["total_vol"]
    rets = store.realized_returns().loc[:model.date]
    both = rets[[ia, ib]].dropna().tail(60) if {ia, ib} <= set(rets.columns) else pd.DataFrame()
    summary = pd.Series({
        "a": f"{model.meta.loc[ia, 'ticker']} — {model.meta.loc[ia, 'conm']}",
        "b": f"{model.meta.loc[ib, 'ticker']} — {model.meta.loc[ib, 'conm']}",
        "predicted cov (ann.)": cov,
        "predicted corr": cov / (vols.loc[ia] * vols.loc[ib]),
        "realized corr, trailing 60m": both.corr().iloc[0, 1] if len(both) > 24 else np.nan,
        "months in realized": len(both),
    }, name="value")
    return {"summary": summary, "by_factor": parts.sort_values("cov contrib", ascending=False)}
