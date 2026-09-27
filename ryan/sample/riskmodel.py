"""
Step 7: the risk model Sigma(t) = X(t) F(t) X(t)' + Delta(t), assembled lazily.

Sigma is NEVER materialized. Every quantity goes through factor space:

    y   = X' w                       K-vector, the portfolio's factor exposures
    var = y' F y + sum_i w_i^2 d_i   O(NK + K^2) instead of O(N^2 K)

and a single row of Sigma, which is all the asset query needs, is
``X @ (F @ x_i)`` plus d_i on the diagonal element.

Units: F and Delta are stored per MONTH. Everything this module returns to a
user -- vols, risk contributions, covariances -- is ANNUALIZED (variance x 12,
vol x sqrt 12), and says so in its column names.

Timing: the model at forecast date t prices a portfolio held from the end of
month t through month t+1, using X(t), F(t) and Delta(t). X(t) is the exposure
row observed at eom t -- the same one-month lag the regression uses, by
construction, because F(t)'s latest input is f(t) = regression of r(t) on X(t-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import factor_cov
import factor_returns

ANN = 12.0


def exposure_matrix(rows: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """
    N x K exposure matrix for one date, columns in ``factors`` order, index = id.

    Mirrors prepare_cross_section exactly -- same zero-fill for missing styles,
    same drop for no industry / no cap -- so the X that prices a portfolio is the
    X that the factor returns were estimated against. A mismatch here (say,
    dropping a name with a missing descriptor instead of zero-filling it) would
    price a portfolio on a different universe than the one F describes.
    """
    rows = rows.drop_duplicates("id").set_index("id")
    x = pd.DataFrame(0.0, index=rows.index, columns=factors)
    for f in factors:
        if f == factor_returns.MARKET_FACTOR:
            x[f] = 1.0
        elif f.startswith("ind_"):
            x[f] = rows[f].fillna(0.0).astype(float) if f in rows else 0.0
        else:
            col = f"f_{f}"
            x[f] = rows[col].astype(float).fillna(0.0) if col in rows else 0.0
    ind_cols = [f for f in factors if f.startswith("ind_")]
    cap = pd.to_numeric(rows[config.CAP_COL], errors="coerce")
    ok = (x[ind_cols].sum(axis=1) > 0) & (cap > 0)
    return x[ok]


@dataclass
class RiskModel:
    """Sigma at one forecast date. Built by ModelStore.at(); cheap to create."""
    date: pd.Timestamp
    F: pd.DataFrame           # K x K, monthly
    X: pd.DataFrame           # N x K, index id
    delta: pd.Series          # N, monthly specific variance, index id
    meta: pd.DataFrame        # N, index id: ticker, conm, sector_name, cap

    # ------------------------------------------------------------ basics
    @property
    def ids(self) -> pd.Index:
        return self.X.index

    @property
    def factors(self) -> list[str]:
        return list(self.F.columns)

    def _w(self, w: pd.Series) -> np.ndarray:
        """Align a weight Series (index id) to the model's assets; unknown ids fail."""
        w = pd.Series(w, dtype=float)
        missing = w.index.difference(self.ids)
        if len(missing):
            raise KeyError(f"{len(missing)} ids not priced by the model at {self.date.date()}: "
                           f"{list(missing)[:10]}")
        return w.reindex(self.ids).fillna(0.0).to_numpy()

    def exposures_of(self, w: pd.Series) -> pd.Series:
        """y = X'w -- the portfolio's factor exposures."""
        return pd.Series(self.X.to_numpy().T @ self._w(w), index=self.factors)

    def variance(self, w: pd.Series) -> float:
        """Annualized variance, w' Sigma w, computed in factor space."""
        wv = self._w(w)
        y = self.X.to_numpy().T @ wv
        return ANN * float(y @ self.F.to_numpy() @ y + np.sum(wv**2 * self.delta.to_numpy()))

    def vol(self, w: pd.Series) -> float:
        return float(np.sqrt(self.variance(w)))

    # ------------------------------------------------------------ assets
    def asset_factor_var(self) -> pd.Series:
        """diag(X F X') per asset, monthly."""
        xv = self.X.to_numpy()
        return pd.Series(np.einsum("ik,kl,il->i", xv, self.F.to_numpy(), xv), index=self.ids)

    def asset_vols(self) -> pd.DataFrame:
        """Total / factor / specific annualized vol for every asset."""
        fv = self.asset_factor_var()
        tv = fv + self.delta
        return pd.DataFrame({
            "total_vol": np.sqrt(ANN * tv),
            "factor_vol": np.sqrt(ANN * fv),
            "specific_vol": np.sqrt(ANN * self.delta),
            "pct_specific_var": self.delta / tv,
        })

    def cov_row(self, asset_id) -> pd.Series:
        """Row i of Sigma (annualized) against every asset, without forming Sigma."""
        xi = self.X.loc[asset_id].to_numpy()
        row = self.X.to_numpy() @ (self.F.to_numpy() @ xi)
        row = pd.Series(row, index=self.ids)
        row.loc[asset_id] += self.delta.loc[asset_id]
        return ANN * row

    def corr_row(self, asset_id) -> pd.Series:
        tv = ANN * (self.asset_factor_var() + self.delta)
        return self.cov_row(asset_id) / np.sqrt(tv * tv.loc[asset_id])

    def market_weights(self, coverage_only: bool = False) -> pd.Series:
        """
        Cap-weighted portfolio of the priced universe -- the model's 'market'.

        ``coverage_only`` restricts it to the coverage universe (e.g. the R1000
        inside a model estimated on a Russell 3000 proxy), which is the natural
        benchmark there. Identical to the default when every name is covered.
        """
        cap = self.meta["cap"].reindex(self.ids)
        if coverage_only and "in_coverage" in self.meta:
            cap = cap.where(self.meta["in_coverage"].reindex(self.ids).fillna(False).astype(bool), 0.0)
        return cap / cap.sum()

    def betas(self, bench: pd.Series | None = None) -> pd.Series:
        """Predicted beta of every asset to ``bench`` (default the cap-weighted market)."""
        b = self.market_weights() if bench is None else bench
        bv = self._w(b)
        yb = self.X.to_numpy().T @ bv
        cov = self.X.to_numpy() @ (self.F.to_numpy() @ yb) + self.delta.to_numpy() * bv
        return pd.Series(cov / (yb @ self.F.to_numpy() @ yb + np.sum(bv**2 * self.delta.to_numpy())),
                         index=self.ids)

    # ------------------------------------------------------------ decomposition
    def decompose(self, w: pd.Series) -> dict:
        """
        Euler decomposition of portfolio risk.

        Factor k contributes y_k (F y)_k to variance -- these sum exactly to the
        factor variance, and signs are meaningful: a negative contribution is a
        hedge. Asset i contributes w_i * MCTR_i to VOLATILITY, where
        MCTR = (Sigma w)_i / sigma_p; those sum exactly to sigma_p.
        """
        wv = self._w(w)
        xv, fm, d = self.X.to_numpy(), self.F.to_numpy(), self.delta.to_numpy()
        y = xv.T @ wv
        fy = fm @ y
        fac_var = float(y @ fy)
        spec_var = float(np.sum(wv**2 * d))
        tot_var = fac_var + spec_var
        sigma = np.sqrt(tot_var)

        by_factor = pd.DataFrame({
            "exposure": y,
            "factor_vol": np.sqrt(ANN * np.diag(fm)),
            "var_contrib": ANN * y * fy,
        }, index=self.factors)
        by_factor["pct_of_total_var"] = by_factor["var_contrib"] / (ANN * tot_var)
        by_factor["group"] = [_group(f) for f in self.factors]

        sigma_w = xv @ fy + d * wv                     # (Sigma w), monthly
        mctr = sigma_w / sigma
        by_asset = pd.DataFrame({
            "weight": wv,
            "mctr": np.sqrt(ANN) * mctr,
            "vol_contrib": np.sqrt(ANN) * wv * mctr,
            "specific_var_contrib": ANN * wv**2 * d,
        }, index=self.ids)
        by_asset["pct_of_vol"] = by_asset["vol_contrib"] / (np.sqrt(ANN) * sigma)
        by_asset = by_asset[by_asset["weight"] != 0].join(self.meta[["ticker", "conm", "sector_name"]])

        groups = by_factor.groupby("group")["var_contrib"].sum()
        groups["specific"] = ANN * spec_var
        groups = groups.reindex(["market", "industry", "style", "specific"]).fillna(0.0)

        return {
            "total_vol": float(np.sqrt(ANN * tot_var)),
            "factor_vol": float(np.sqrt(ANN * fac_var)),
            "specific_vol": float(np.sqrt(ANN * spec_var)),
            "pct_factor": fac_var / tot_var,
            "by_group": pd.DataFrame({"var_contrib": groups, "pct_of_total_var": groups / (ANN * tot_var)}),
            "by_factor": by_factor,
            "by_asset": by_asset.sort_values("vol_contrib", ascending=False),
        }

    def subset(self, ids) -> "RiskModel":
        ids = pd.Index(ids)
        return RiskModel(self.date, self.F, self.X.loc[ids], self.delta.loc[ids], self.meta.loc[ids])


def _group(f: str) -> str:
    if f == factor_returns.MARKET_FACTOR:
        return "market"
    return "industry" if f.startswith("ind_") else "style"


class ModelStore:
    """
    Every stored input, loaded once; ``at(date)`` hands back a RiskModel.

    Built from DataFrames so tests need no database; ``from_db`` is the normal
    entry point.
    """

    def __init__(self, fcov_long: pd.DataFrame, specific: pd.DataFrame,
                 exposures: pd.DataFrame, factor_returns_long: pd.DataFrame | None = None,
                 cov_col: str = "cov"):
        self._fcov_long = fcov_long
        self._F_by_stage = {cov_col: factor_cov.from_long(fcov_long, cov_col)}
        self.F = self._F_by_stage[cov_col]
        self.dates = pd.DatetimeIndex(sorted(self.F))
        self.factors = list(next(iter(self.F.values())).columns)

        sp = specific.copy()
        sp["forecast_date"] = pd.to_datetime(sp["forecast_date"])
        self.specific = sp

        ex = exposures.copy()
        ex["eom"] = pd.to_datetime(ex["eom"])
        self.exposures = ex

        if factor_returns_long is not None:
            fr = factor_returns_long.copy()
            fr["eom"] = pd.to_datetime(fr["eom"])
            fr = fr.pivot(index="eom", columns="factor", values="factor_return")
            self.factor_returns = fr[factor_cov.canonical_order(fr.columns)]
        else:
            self.factor_returns = None

        self._by_eom = {d: g for d, g in ex.groupby("eom")}
        self._sp_by_date = {d: g.set_index("id")["specific_var"] for d, g in sp.groupby("forecast_date")}
        # Computed once: at full-universe scale the exposures table is ~1M rows,
        # and rescanning it for every ticker lookup made each query take seconds.
        tick = ex.loc[ex["ticker"].notna(), ["id", "eom", "ticker"]].sort_values("eom")
        self._ticker_hist = tick
        self._ever_tickers = set(tick["ticker"].astype(str).str.upper())
        self._tt_cache: dict = {}

    @classmethod
    def from_db(cls, engine, cov_col: str = "cov") -> "ModelStore":
        import db
        for t in (config.TBL_FACTOR_COV, config.TBL_SPECIFIC_RISK, config.TBL_EXPOSURES):
            if not db.table_exists(engine, t):
                raise RuntimeError(f"{t} is missing — run notebooks 03 and 04 first.")
        return cls(
            db.read_table(config.TBL_FACTOR_COV, engine),
            db.read_table(config.TBL_SPECIFIC_RISK, engine),
            db.read_table(config.TBL_EXPOSURES, engine),
            db.read_table(config.TBL_FACTOR_RETURNS, engine),
            cov_col=cov_col,
        )

    @property
    def latest(self) -> pd.Timestamp:
        return self.dates[-1]

    def F_stage(self, cov_col: str) -> dict:
        """F(t) for a stored stage: 'cov' (final), 'cov_eigen', 'cov_nw' or 'cov_raw'."""
        if cov_col not in self._F_by_stage:
            self._F_by_stage[cov_col] = factor_cov.from_long(self._fcov_long, cov_col)
        return self._F_by_stage[cov_col]

    def at(self, date=None, cov_col: str = "cov") -> RiskModel:
        """
        The model at ``date`` (default: latest). ``cov_col`` swaps in an earlier
        F stage with everything else held fixed -- the way to ask "what would the
        unadjusted covariance have said about this portfolio".
        """
        date = self.latest if date is None else pd.Timestamp(date)
        if date not in self.F:
            raise KeyError(f"no F(t) at {date.date()}; forecast dates run "
                           f"{self.dates[0].date()} → {self.dates[-1].date()}")
        rows = self._by_eom[date]
        x = exposure_matrix(rows, self.factors)
        delta = self._sp_by_date[date]
        ids = x.index.intersection(delta.index)
        meta = self.ticker_table(date).reindex(ids)
        return RiskModel(date, self.F_stage(cov_col)[date], x.loc[ids], delta.loc[ids], meta)

    # ------------------------------------------------------------ identifiers
    def ticker_table(self, as_of=None) -> pd.DataFrame:
        """
        Securities live at ``as_of`` with the ticker most recently known for them.

        Tickers are resolved as-of, never globally: they get reused across firms
        (BCS was both LIFESTYLE RESTAURANT and BARCLAYS). CRSP's name history in
        this pull ends at 2024-12, so 2025 rows carry no ticker at all; carrying
        the last known one forward is right for names that did not change
        ticker, and `ticker_asof` says how stale each one is.
        """
        as_of = self.latest if as_of is None else pd.Timestamp(as_of)
        if as_of in self._tt_cache:
            return self._tt_cache[as_of]
        live = self._by_eom[as_of].drop_duplicates("id").set_index("id")
        hist = self._ticker_hist
        last = hist[hist["eom"] <= as_of].groupby("id").tail(1).set_index("id")
        out = pd.DataFrame({
            "ticker": last["ticker"].reindex(live.index),
            "ticker_asof": last["eom"].reindex(live.index),
            "conm": live["conm"],
            "sector_name": live["sector_name"],
            "cap": pd.to_numeric(live[config.CAP_COL], errors="coerce"),
        })
        cov_col = getattr(config, "COVERAGE_COL", None)
        out["in_coverage"] = (live[cov_col].fillna(False).astype(bool)
                              if cov_col and cov_col in live else True)
        out.index.name = "id"
        self._tt_cache[as_of] = out
        return out

    def resolve(self, identifiers, as_of=None) -> pd.DataFrame:
        """
        Map tickers or ids to model ids at ``as_of``.

        Returns one row per input with ``id`` and ``status`` in
        {ok, ambiguous, not_in_sample, not_live}. Ambiguity is reported, never
        guessed: if two live securities last traded under the same ticker the
        caller must disambiguate by id.
        """
        as_of = self.latest if as_of is None else pd.Timestamp(as_of)
        tt = self.ticker_table(as_of)
        rows = []
        for raw in identifiers:
            key = str(raw).strip()
            num = pd.to_numeric(key, errors="coerce")
            if pd.notna(num):
                if float(num) in tt.index:
                    rows.append({"input": raw, "id": float(num), "status": "ok"})
                elif float(num) in set(self.exposures["id"]):
                    rows.append({"input": raw, "id": float(num), "status": "not_live"})
                else:
                    rows.append({"input": raw, "id": np.nan, "status": "not_in_sample"})
                continue
            hit = tt.index[tt["ticker"].str.upper() == key.upper()]
            if len(hit) == 1:
                rows.append({"input": raw, "id": hit[0], "status": "ok"})
            elif len(hit) > 1:
                rows.append({"input": raw, "id": np.nan, "status": "ambiguous",
                             "note": f"ids {list(hit)}"})
            elif key.upper() in self._ever_tickers:
                rows.append({"input": raw, "id": np.nan, "status": "not_live",
                             "note": "in the sample but not live at the forecast date"})
            else:
                rows.append({"input": raw, "id": np.nan, "status": "not_in_sample"})
        out = pd.DataFrame(rows)
        if "note" not in out:
            out["note"] = ""
        return out.join(tt[["ticker", "conm", "in_coverage"]], on="id")

    # ------------------------------------------------------------ history
    def asset_history(self, asset_id) -> pd.DataFrame:
        """Predicted vol and exposures of one asset at every forecast date it is priced."""
        rows = []
        for d in self.dates:
            g = self._by_eom.get(d)
            if g is None or asset_id not in set(g["id"]):
                continue
            x = exposure_matrix(g[g["id"] == asset_id], self.factors)
            dv = self._sp_by_date[d].get(asset_id, np.nan)
            if x.empty or not np.isfinite(dv):
                continue
            xi = x.iloc[0].to_numpy()
            fv = float(xi @ self.F[d].to_numpy() @ xi)
            rows.append({"forecast_date": d, "total_vol": np.sqrt(ANN * (fv + dv)),
                         "factor_vol": np.sqrt(ANN * fv), "specific_vol": np.sqrt(ANN * dv),
                         **{f: xi[k] for k, f in enumerate(self.factors)
                            if not f.startswith("ind_") and f != "market"}})
        return pd.DataFrame(rows).set_index("forecast_date")

    def realized_returns(self, ids=None) -> pd.DataFrame:
        """Wide monthly excess returns (eom x id) from the exposures panel, untrimmed."""
        ex = self.exposures if ids is None else self.exposures[self.exposures["id"].isin(ids)]
        return ex.pivot_table(index="eom", columns="id", values=config.RETURN_COL)
