"""
End-to-end smoke test on a SYNTHETIC universe. No WRDS, no database.

    python ryan/universe/smoke_synthetic.py

Generates a JKP-shaped panel -- listings, delistings, drifting caps, nano caps,
a known factor structure in the returns -- and pushes it through every stage the
notebooks run: membership → exposures → industry gate → factor returns → F(t) →
Δ(t) → ModelStore → portfolio CSV → decomposition → projected-gradient optimizer
→ asset query. It proves the pieces fit together at universe scale before any
WRDS time is spent; it does not validate the model on real data (05 does that).

Scale is reduced (1,800 names, 108 months, window 36) so it runs in seconds.
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import config
import factor_cov, factor_returns, gics, membership, optimizer, portfolio, query
import riskmath, riskmodel, specific_risk, standardize

# Shrink the problem; the logic is identical at full size.
config.ESTU_RULE, config.ESTU_N, config.COVERAGE_N = "rank", 1200, 400
config.COVERAGE_COL, config.COVERAGE_BAND, config.CONSTITUENTS_CSV = "in_r1000", 0.025, None
config.COV_WINDOW, config.SPECIFIC_WINDOW, config.SPECIFIC_MIN_OBS = 36, 36, 12
config.VRA_MIN_OBS = config.SPECIFIC_VRA_MIN_OBS = 6

FAIL = []
T0 = time.time()


def check(name, cond, extra=""):
    print(f"{'✅' if cond else '❌'} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- synthetic JKP
rng = np.random.default_rng(7)
n_sec, months = 1800, pd.date_range("2000-05-31", periods=108, freq="ME")
sectors = list(gics.SECTORS)
groups = {s: [g for g in gics.INDUSTRY_GROUPS if g.startswith(s)] for s in sectors}
sec_of = rng.choice(sectors, n_sec)
start = rng.integers(0, 60, n_sec) * (rng.random(n_sec) < 0.35)        # 35% list later
life = rng.integers(30, 200, n_sec)
logcap0 = rng.normal(6.5, 1.8, n_sec)
beta_true = rng.normal(1, .3, n_sec)
sty_true = rng.normal(size=(n_sec, 3))
ind_ret = rng.normal(0, .04, (len(months), len(sectors)))
mkt = rng.normal(.006, .045, len(months))
sty_ret = rng.normal(0, .015, (len(months), 3))

rows = []
logcap = logcap0.copy()
for t, eom in enumerate(months):
    alive = (t >= start) & (t < start + life)
    idio = rng.normal(0, np.exp(-0.15 * (logcap - 6)) * .08)
    r = (beta_true * mkt[t] + ind_ret[t, [sectors.index(s) for s in sec_of]]
         + sty_true @ sty_ret[t] + idio)
    logcap = logcap + r
    for i in np.flatnonzero(alive):
        rows.append((float(10000 + i), eom, f"{i:06d}", 10000 + i, sec_of[i], rng.choice(groups[sec_of[i]]),
                     float(np.exp(logcap[i])), float(r[i]), float(sty_true[i, 0] + rng.normal(0, .3)),
                     float(sty_true[i, 1]), float(sty_true[i, 2] * .5 + rng.normal(0, .3)),
                     float(rng.normal()), float(rng.normal()), float(rng.normal()), float(beta_true[i])))
raw = pd.DataFrame(rows, columns=["id", "eom", "gvkey", "permno", "gsector", "ggroup", "market_equity",
                                  "ret_exc", "be_me", "ret_12_1", "gp_at", "ni_at", "at_gr1",
                                  "oaccruals_at", "beta_60m"])
raw["size_grp"] = np.where(raw["market_equity"] < raw.groupby("eom")["market_equity"].transform(
    lambda s: s.quantile(.05)), "nano", "large")
raw["permno_ccm"] = raw["permno"].astype(float)
raw["ticker"] = "T" + raw["id"].astype(int).astype(str)
raw["conm"] = "CO " + raw["id"].astype(int).astype(str)
raw = gics.attach_names(raw)
print(f"synthetic JKP panel: {len(raw):,} rows, {raw['id'].nunique():,} securities, {len(months)} months")

# ---------------------------------------------------------------- 01: membership + exposures
panel = membership.apply_profile(raw, config)
cnt = panel.groupby("eom")[["in_estu", "in_r1000"]].sum()
first_eff = cnt.index >= "2000-06-30"
check("estimation universe ≈ ESTU_N once reconstituted",
      cnt.loc[first_eff, "in_estu"].between(900, 1200).all(),
      f"{int(cnt.loc[first_eff, 'in_estu'].min())}–{int(cnt.loc[first_eff, 'in_estu'].max())}")
check("coverage floats around COVERAGE_N (band)",
      cnt.loc[first_eff, "in_r1000"].between(300, 470).all(),
      f"{int(cnt.loc[first_eff, 'in_r1000'].min())}–{int(cnt.loc[first_eff, 'in_r1000'].max())}")
check("nano caps never in the estimation universe", not (panel["in_estu"] & (panel["size_grp"] == "nano")).any())
panel = panel[panel["in_estu"] & (panel["eom"] >= "2000-06-30")].reset_index(drop=True)

ex = standardize.standardize_panel(panel)
ex = standardize.combine_descriptors(ex)
ex, dummies = standardize.build_industry_dummies(ex, code_col=config.INDUSTRY_CODE_COL)
membership.industry_gate(ex, dummies, config.MIN_INDUSTRY_NAMES)
styles = [c for c in ex.columns if c.startswith("f_")]
check("11 industry dummies + 6 styles", len(dummies) == 11 and len(styles) == 6)

# ---------------------------------------------------------------- 02: factor returns
F, U, diag, reg = factor_returns.run_panel(ex, styles, dummies, verbose=False)
check("every factor estimated every month (dynamic universe, no empty industry)", F.notna().all().all())
check("constraint holds to machine precision", diag["constraint_residual"].abs().max() < 1e-6)
capw = reg.groupby("eom").apply(lambda g: np.average(g["ret_exc"], weights=g[config.CAP_COL]),
                                include_groups=False)
check("market factor tracks the cap-weighted universe return",
      F["market"].corr(capw.reindex(F.index)) > 0.97, f"corr {F['market'].corr(capw.reindex(F.index)):.3f}")

# ---------------------------------------------------------------- 03: F(t)
F = F[factor_cov.canonical_order(F.columns)]
path = factor_cov.estimate(F, eigen_sims=50, verbose=False)
check("F(t) PSD at every date, every stage",
      all(riskmath.is_psd(m) for s in factor_cov.STAGES for m in path.stages[s]))
check("forecast dates = months − window + 1", len(path.dates) == len(F) - config.COV_WINDOW + 1)

# ---------------------------------------------------------------- 04: Δ(t)
D = specific_risk.estimate(U, ex, path.dates, verbose=False)
check("Δ positive for every (asset, date)", (D["specific_var"] > 0).all())
prior = D["specific_var_raw"].isna().mean()
check("entrants on the size prior are a minority", prior < 0.25, f"{prior:.1%}")

# ---------------------------------------------------------------- 05-08: tools
store = riskmodel.ModelStore(factor_cov.to_long(path), D, ex, factor_returns.to_long(F))
model = store.at()
check("model prices the estimation universe; coverage flag carried",
      len(model.ids) > 800 and model.meta["in_coverage"].sum() < len(model.ids))
cov_ids = model.ids[model.meta["in_coverage"].to_numpy()]
mw = model.market_weights(coverage_only=True)
check("coverage-only benchmark has weight only on covered names", mw.drop(cov_ids).abs().sum() == 0)

tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
pick = model.meta.loc[cov_ids].nlargest(25, "cap")
pd.DataFrame({"ticker": list(pick["ticker"]) + ["NOPE"], "weight": [0.04] * 25 + [0.0]}).to_csv(tmp.name, index=False)
tmp.close()
hold, rep = portfolio.load_portfolio(tmp.name, store, model.date)
os.unlink(tmp.name)
check("CSV resolves; unknown ticker reported, not dropped silently",
      len(hold) == 25 and (rep["status"] == "not_in_sample").sum() == 1)
dec = model.decompose(hold["weight"])
check("Euler decomposition sums to total vol", np.isclose(dec["by_asset"]["vol_contrib"].sum(), dec["total_vol"]))

t = time.time()
w = optimizer.optimize(model, cov_ids, max_weight=0.02)          # > SLSQP_MAX_N → projected gradient
t_pg = time.time() - t
check("coverage-wide min-variance solves with projected gradient",
      np.isclose(w.sum(), 1) and w.max() <= 0.02 + 1e-9 and model.vol(w) < model.vol(mw),
      f"{len(cov_ids)} names, {t_pg:.1f}s, vol {model.vol(w):.3f} vs benchmark {model.vol(mw):.3f}")
w_ls = optimizer.min_variance_closed_form(model)
check("Woodbury min-var at universe scale", np.isclose(w_ls.sum(), 1) and model.vol(w_ls) <= model.vol(w) + 1e-9)

rpt = query.asset_report(store, model, pick["ticker"].iloc[0])
check("asset query works; menu limited to coverage",
      len(rpt["highest_corr"]) == 10 and len(query.universe(model)) == len(cov_ids))

print("\n" + "=" * 60)
print(f"{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else " — all checks passed")
      + f"   ({time.time() - T0:.0f}s)")
sys.exit(1 if FAIL else 0)
