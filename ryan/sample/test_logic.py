"""
Offline validation of the pure logic in this package.

Touches neither WRDS nor the database, so it runs anywhere:

    python ryan/sample/test_logic.py

Every check encodes a specific failure this pipeline is built to avoid, so a
failure here means a real defect rather than a stale expectation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config, factor_returns, gics, sampling, standardize

rng = np.random.default_rng(0)
FAIL = []

def check(name, cond, extra=""):
    print(f"{'✅' if cond else '❌'} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)

def _raises(fn, *args):
    """True if ``fn`` rejects the input. Used where failing loudly is the feature."""
    try:
        fn(*args)
    except Exception:
        return True
    return False

# ---------------------------------------------------------------- gics
print("\n=== gics ===")
check("11 sectors", len(gics.SECTORS) == 11)
check("25 industry groups", len(gics.INDUSTRY_GROUPS) == 25)
check("gsector 20 -> Industrials", gics.SECTORS["20"] == "Industrials",
      "the Kaggle map said Materials")
raw = pd.DataFrame({"gsector": [20.0, "35", np.nan, 99.0], "ggroup": [2030.0, "3520", np.nan, 9999.0]})
named = gics.attach_names(raw)
check("float 20.0 normalizes to '20'", named.loc[0, "sector_code"] == "20")
check("AAL sector resolves to Industrials", named.loc[0, "sector_name"] == "Industrials")
check("ggroup 2030 -> Transportation", named.loc[0, "group_name"] == "Transportation")
check("unknown code keeps code, drops name",
      named.loc[3, "sector_code"] == "99" and pd.isna(named.loc[3, "sector_name"]))

# ------------------------------------------------- standardization
print("\n=== standardization ===")
n_per, n_months = 60, 6
eoms = pd.date_range("2006-01-31", periods=n_months, freq="ME")
rows = []
for eom in eoms:
    cap = np.exp(rng.normal(7, 2, n_per))          # heavily skewed, like real caps
    x = rng.normal(0, 1, n_per)
    x[0] = 40.0                                     # one wild outlier per month
    rows.append(pd.DataFrame({"eom": eom, "market_equity": cap, "be_me": x,
                              "ret_exc": rng.normal(0, 0.08, n_per)}))
panel = pd.concat(rows, ignore_index=True)

z = standardize.standardize_grouped(panel["be_me"], panel["market_equity"], panel["eom"])
check("outlier pulled in near the 3-sigma trim", z.abs().max() < 4.0,
      f"max |z| = {z.abs().max():.4f} (raw outlier was 40)")
# The failure mode robust winsorization prevents: an uncorrupted scale must leave
# ordinary observations with real dispersion, not compress them toward zero.
ordinary = z[panel["be_me"] < 10]
check("ordinary observations keep their dispersion", ordinary.std() > 0.8,
      f"sd of non-outlier z = {ordinary.std():.3f}")
naive_sd = panel.groupby("eom")["be_me"].transform("std")
check("naive sd really is corrupted by the outlier", naive_sd.mean() > 3.0,
      f"mean naive sd = {naive_sd.mean():.2f} vs true 1.0")

# the defining USE4 property: cap-weighted mean is zero, not equal-weighted
cw = [np.average(g["z"], weights=g["w"]) for _, g in
      pd.DataFrame({"z": z, "w": panel["market_equity"], "e": panel["eom"]}).groupby("e")]
check("cap-weighted mean is ~0 per cross-section", max(abs(v) for v in cw) < 1e-9,
      f"max |cap-wtd mean| = {max(abs(v) for v in cw):.2e}")
ew = z.groupby(panel["eom"]).mean()
check("equal-weighted mean is NOT 0 (proves cap weighting bit)", ew.abs().max() > 1e-4,
      f"max |eq-wtd mean| = {ew.abs().max():.4f}")
sd = z.groupby(panel["eom"]).std()
check("unit dispersion per cross-section", (sd - 1).abs().max() < 1e-9,
      f"max |sd-1| = {(sd - 1).abs().max():.2e}")

# log size
out = standardize.standardize_panel(panel, descriptors=["market_equity", "be_me"])
check("log_market_equity created", "log_market_equity" in out.columns)
# The real question is which scale fed the standardization, so compare against
# both candidates rather than checking a bound.
z_from_log = standardize.standardize_grouped(
    out["log_market_equity"], out["market_equity"], out["eom"])
z_from_raw = standardize.standardize_grouped(
    out["market_equity"], out["market_equity"], out["eom"])
check("size exposure comes from log(cap), not raw cap",
      out["z_market_equity"].corr(z_from_log) > 0.9999
      and out["z_market_equity"].corr(z_from_raw) < 0.99,
      f"corr(log)={out['z_market_equity'].corr(z_from_log):.4f}, "
      f"corr(raw)={out['z_market_equity'].corr(z_from_raw):.4f}")
skew_raw = pd.Series(panel["market_equity"]).skew()
skew_log = out["log_market_equity"].skew()
check("log transform removes the cap skew", abs(skew_log) < abs(skew_raw),
      f"skew {skew_raw:.2f} -> {skew_log:.2f}")
check("ret_exc is never standardized", "z_ret_exc" not in out.columns)
check("ret_exc absent from descriptor list", "ret_exc" not in config.all_descriptors())

# all-NaN and zero-dispersion cross sections must not blow up
edge = pd.DataFrame({"eom": ["a"]*4 + ["b"]*4 + ["c"]*4,
                     "market_equity": [1., 2, 3, 4]*3,
                     "be_me": [np.nan]*4 + [5., 5, 5, 5] + [1., 2, 3, 4]})
ze = standardize.standardize_grouped(edge["be_me"], edge["market_equity"], edge["eom"])
check("all-NaN cross-section stays NaN", ze[:4].isna().all())
check("zero-dispersion cross-section -> 0 exposure", (ze[4:8] == 0).all(),
      f"got {ze[4:8].tolist()}")
check("normal cross-section still standardizes", ze[8:].notna().all())

# descriptor combination
panel2 = panel.copy()
panel2["ni_me"] = panel2["be_me"] * 0.9 + rng.normal(0, .2, len(panel2))
panel2["ocf_me"] = panel2["be_me"] * 0.8 + rng.normal(0, .3, len(panel2))
st = standardize.standardize_panel(panel2, descriptors=["be_me", "ni_me", "ocf_me", "market_equity"])
cb = standardize.combine_descriptors(st, groups={"value": ["be_me", "ni_me", "ocf_me"],
                                                 "size": ["market_equity"]})
check("3 collinear descriptors collapse to one f_value column", "f_value" in cb.columns)
sdv = cb["f_value"].groupby(cb["eom"]).std()
check("combined factor re-standardized to unit dispersion", (sdv - 1).abs().max() < 1e-9)

# ------------------------------------------------- industry dummies
print("\n=== industry dummies ===")
dpanel = pd.DataFrame({"eom": ["a"]*11, "sector_code": list(gics.SECTORS.keys()),
                       "ret_exc": rng.normal(0, .05, 11)})
dd, cols = standardize.build_industry_dummies(dpanel)
check("one dummy per sector present", len(cols) == 11)
check("dummies are 0/1", set(np.unique(dd[cols].to_numpy())) <= {0.0, 1.0})
check("each row in exactly one industry", (dd[cols].sum(axis=1) == 1).all())
check("dummies sum to the ones vector (rank deficient by design)",
      np.allclose(dd[cols].sum(axis=1).to_numpy(), 1.0))

# ------------------------------------------------- stratified draw
print("\n=== stratified draw ===")
# Realistic sector skew: Utilities ~3% of the universe.
shares = {"10": .06, "15": .05, "20": .12, "25": .11, "30": .04, "35": .15,
          "40": .15, "45": .13, "50": .03, "55": .03, "60": .08}
N = 4000
codes = rng.choice(list(shares), size=N, p=np.array(list(shares.values())) / sum(shares.values()))
frame = pd.DataFrame({
    "id": [f"s{i:05d}" for i in range(N)],
    "gvkey": [f"g{i:05d}" for i in range(N)],
    "sector_code": codes,
})
frame["sector_name"] = frame["sector_code"].map(gics.SECTORS)

roster = sampling.draw_stratified_sample(frame, per_sector=20, seed=config.RANDOM_SEED)
check("roster is 220 names", len(roster) == 220, f"got {len(roster)}")
per = roster.groupby("sector_code").size()
check("exactly 20 per sector, all 11 sectors", (per == 20).all() and len(per) == 11)
check("no duplicate ids", not roster["id"].duplicated().any())
r2 = sampling.draw_stratified_sample(frame, per_sector=20, seed=config.RANDOM_SEED)
check("draw is reproducible under the same seed", set(r2["id"]) == set(roster["id"]))
r3 = sampling.draw_stratified_sample(frame, per_sector=20, seed=config.RANDOM_SEED + 1)
check("a different seed gives a different roster", set(r3["id"]) != set(roster["id"]))

# quantify the failure mode stratification prevents
mins = [pd.Series(rng.choice(list(shares), size=220,
                            p=np.array(list(shares.values())) / sum(shares.values()))
                  ).value_counts().reindex(list(shares)).fillna(0).min() for _ in range(4000)]
p_degenerate = np.mean([m <= 1 for m in mins])
print(f"   ℹ️  uniform 220-name draw: P(some sector has ≤1 name) = {p_degenerate:.1%}; "
      f"P(≤2) = {np.mean([m <= 2 for m in mins]):.1%}")
check("stratification is load-bearing (uniform draw is materially risky)", p_degenerate > 0.01)

# sector with too few eligible names is taken whole, not padded
thin = frame[~((frame.sector_code == "55") & (frame.index % 4 != 0))]
rt = sampling.draw_stratified_sample(thin, per_sector=20, seed=1)
check("thin sector handled without crashing", len(rt) > 0)

# ------------------------------------------------- date-aware link join
print("\n=== date-aware permno join ===")
# gvkey G maps to permno 111 up to 2009, then 222 -- the exact pattern that made
# the old gvkey-only merge fan out.
pan = pd.DataFrame({
    "id": ["A"] * 4, "gvkey": ["G"] * 4,
    "eom": pd.to_datetime(["2006-06-30", "2008-06-30", "2012-06-30", "2015-06-30"]),
    "ret_exc": [.01, .02, .03, .04],
})
links = pd.DataFrame({
    "gvkey": ["G", "G"], "permno": [111.0, 222.0], "permco": [1.0, 2.0],
    "linkprim": ["P", "P"], "linktype": ["LC", "LC"],
    "linkdt": pd.to_datetime(["2000-01-01", "2010-01-01"]),
    # NULL = still-open link, exactly as WRDS returns it. Note this is a NaT and
    # NOT '9999-12-31': that sentinel overflows datetime64[ns] (max 2262-04-11).
    "linkenddt": [pd.Timestamp("2009-12-31"), pd.NaT],
})
joined = sampling.attach_permno(pan, links)
check("no row explosion on a multi-link gvkey", len(joined) == 4, f"got {len(joined)}")
check("pre-2010 rows get permno 111", (joined.iloc[:2]["permno_ccm"] == 111).all())
check("post-2010 rows get permno 222", (joined.iloc[2:]["permno_ccm"] == 222).all())

# and confirm the old approach is what blew up
naive = pan.merge(links, on="gvkey", how="left")
print(f"   ℹ️  gvkey-only merge: {len(pan)} rows -> {len(naive)} rows "
      f"({len(naive)//len(pan)}x); interval join keeps {len(joined)}")

# unmatched gvkey must survive as a left join
pan2 = pd.concat([pan, pd.DataFrame({"id": ["B"], "gvkey": ["Z"],
                                     "eom": [pd.Timestamp("2006-06-30")], "ret_exc": [.05]})],
                 ignore_index=True)
j2 = sampling.attach_permno(pan2, links)
check("unmatched gvkey kept with null permno",
      len(j2) == 5 and pd.isna(j2.loc[j2.id == "B", "permno_ccm"]).all())

# a gap in link coverage must not drop the row either
links_gap = links.copy()
links_gap.loc[0, "linkenddt"] = pd.Timestamp("2007-01-01")
j3 = sampling.attach_permno(pan, links_gap)
check("row in a link-coverage gap kept, with null permno",
      len(j3) == 4 and pd.isna(j3.loc[j3.eom == pd.Timestamp("2008-06-30"), "permno_ccm"]).all())

# ------------------------------------------------- cross-sectional regression
print("\n=== cross-sectional regression (step 4) ===")

# A cross-section with a KNOWN answer: build returns from factor returns we
# choose, with no noise, and check the regression recovers them. Anything less
# than exact recovery here is a bug in the solver, not sampling error.
n_ind, n_sty, n_per_ind = 11, 6, 20
n = n_ind * n_per_ind
ind_of = np.repeat(np.arange(n_ind), n_per_ind)
cap_t = np.exp(rng.normal(7, 1.5, n))
ind_mat = np.zeros((n, n_ind))
ind_mat[np.arange(n), ind_of] = 1.0
sty = rng.normal(0, 1, (n, n_sty))
X = np.column_stack([np.ones(n), ind_mat, sty])

ind_caps = ind_mat.T @ cap_t
f_ind_true = rng.normal(0, 0.02, n_ind)
# Make the truth satisfy the constraint, otherwise no estimator could recover it:
# the constraint is an identifying restriction, not an approximation.
f_ind_true -= (ind_caps @ f_ind_true) / ind_caps.sum()
f_true = np.concatenate([[0.01], f_ind_true, rng.normal(0, 0.01, n_sty)])
r_t = X @ f_true

res = factor_returns.solve_cross_section(X, r_t, cap_t, n_ind, n_sty)
check("noiseless fit recovers the true factor returns",
      np.allclose(res["f"], f_true, atol=1e-10),
      f"max |f_hat - f| = {np.abs(res['f'] - f_true).max():.2e}")
check("noiseless fit has zero residuals", np.abs(res["resid"]).max() < 1e-10)
check("noiseless R^2 is 1", abs(res["r2"] - 1.0) < 1e-10)
check("reduced design is full rank (K-1 = 17 free params)",
      res["full_rank"] and res["rank"] == 17, f"rank {res['rank']}")

# The constraint is the whole reason this is not an lstsq one-liner.
check("cap-weighted industry constraint holds to machine precision",
      abs(res["constraint_residual"]) < 1e-12,
      f"sum_p cap_p f_p = {res['constraint_residual']:.2e}")

# ...and demonstrate the failure it prevents. Unconstrained lstsq on the same
# rank-deficient design returns a minimum-norm solution that is a *different*
# point, so the industry returns it reports are not USE4's.
naive, *_ = np.linalg.lstsq(X, r_t, rcond=None)
check("unconstrained lstsq fits equally well but reports different factor returns",
      np.allclose(X @ naive, r_t, atol=1e-8) and not np.allclose(naive, f_true, atol=1e-4),
      f"same fit, max |f_naive - f_true| = {np.abs(naive - f_true).max():.4f}")
check("unconstrained solution violates the cap-weighted constraint",
      abs(ind_caps @ naive[1:1 + n_ind]) > 1e-6,
      f"sum_p cap_p f_p = {ind_caps @ naive[1:1 + n_ind]:.2e}")

# With noise, the identities must still hold exactly even though f no longer does.
r_noisy = X @ f_true + rng.normal(0, 0.05, n)
resn = factor_returns.solve_cross_section(X, r_noisy, cap_t, n_ind, n_sty)
check("fitted + residual reconstructs the return exactly",
      np.allclose(resn["fitted"] + resn["resid"], r_noisy, atol=1e-12))
# Relative, not absolute: the constraint sums cap*return terms of order 1e5*1e-2,
# so cancellation to an *absolute* 1e-12 is below the floating-point floor of the
# quantities involved and would fail for reasons that have nothing to do with the
# solver. Scale by the size of the terms being cancelled.
scale_n = float(np.abs(resn["industry_caps"]).sum()
                * np.abs(resn["f"][1:1 + n_ind]).max())
check("constraint still exact under noise",
      abs(resn["constraint_residual"]) / scale_n < 1e-12,
      f"relative residual = {abs(resn['constraint_residual']) / scale_n:.2e}")
check("residuals are weight-orthogonal to every free direction",
      np.abs((X @ factor_returns._restriction_matrix(n_ind, n_sty, ind_caps)).T
             @ (resn["weights"] * resn["resid"])).max() < 1e-12)
check("noisy R^2 is in (0, 1)", 0 < resn["r2"] < 1, f"R² = {resn['r2']:.3f}")

# WLS, not OLS: big names must move the answer more than small ones.
ols = np.linalg.lstsq(X @ factor_returns._restriction_matrix(n_ind, n_sty, ind_caps),
                      r_noisy, rcond=None)[0]
check("sqrt-cap weighting actually changes the estimate (it is not OLS)",
      not np.allclose(ols, np.linalg.lstsq(
          np.sqrt(np.sqrt(cap_t))[:, None] * (X @ factor_returns._restriction_matrix(n_ind, n_sty, ind_caps)),
          np.sqrt(np.sqrt(cap_t)) * r_noisy, rcond=None)[0], atol=1e-6))

# The pivot must be the largest industry, not the last column -- dividing by a
# thin industry's cap is what makes the basis badly conditioned.
caps_skew = np.array([1e6] + [1.0] * 10)
basis = factor_returns.build_constraint_basis(caps_skew)
# The pivot row is the one that is not a unit vector -- it carries the
# -cap_p/cap_pivot multipliers. Dividing by the LARGEST cap makes those
# multipliers small, which is the entire point.
pivot_row = int(np.argmax([not np.isclose(np.abs(basis[i]).sum(), 1.0) for i in range(len(caps_skew))]))
check("constraint basis pivots on the largest industry",
      pivot_row == int(np.argmax(caps_skew)) and np.abs(basis[pivot_row]).max() < 1e-3,
      f"pivot row {pivot_row}, max multiplier {np.abs(basis[pivot_row]).max():.1e}")
# Quantify the conditioning that choice buys, against the naive "drop the last
# column" alternative, which on this cap distribution divides by the smallest.
p_ = len(caps_skew)
alt = np.zeros((p_, p_ - 1))
for c_, r_ in enumerate(range(p_ - 1)):
    alt[r_, c_] = 1.0
alt[p_ - 1, :] = -caps_skew[: p_ - 1] / caps_skew[p_ - 1]
check("largest-cap pivot is far better conditioned than dropping the last column",
      np.linalg.cond(basis) < np.linalg.cond(alt) / 1e3,
      f"cond {np.linalg.cond(basis):.1f} vs {np.linalg.cond(alt):.3g}")
check("any vector in the basis satisfies the constraint",
      abs(caps_skew @ (basis @ rng.normal(0, 1, 10))) < 1e-6)
check("degenerate industry caps are refused, not silently divided by",
      _raises(factor_returns.build_constraint_basis, np.array([1.0, 0.0])))

# Frame-level preparation: the three drop rules and the fill rule.
fr = pd.DataFrame({
    "id": [f"a{i}" for i in range(6)],
    "ret_exc": [0.01, 0.02, np.nan, 0.03, 0.04, 0.05],
    "market_equity": [100.0, 200.0, 300.0, np.nan, 500.0, 600.0],
    "f_value": [1.0, np.nan, 1.0, 1.0, 1.0, 1.0],
    "f_size": [0.5] * 6,
})
fr["ind_10"] = [1.0, 1, 1, 1, 0, 1]
fr["ind_20"] = [0.0, 0, 0, 0, 0, 0]          # empty industry, must be dropped
Xp, rp, capp, idsp, namesp, dp = factor_returns.prepare_cross_section(
    fr, ["f_value", "f_size"], ["ind_10", "ind_20"])
check("row with no return dropped", dp["no_return"] == 1)
check("row with no cap dropped", dp["no_cap"] == 1)
check("row in no industry dropped", dp["no_industry"] == 1)
check("3 usable rows survive", dp["kept"] == 3 and Xp.shape[0] == 3)
check("empty industry column dropped from the design",
      dp["empty_industries"] == ["ind_20"] and "ind_20" not in namesp,
      f"names = {namesp}")
check("missing style exposure filled with 0, not dropped", dp["exposures_filled"] == 1)
check("design columns are [market, live industries, styles]",
      namesp == ["market", "ind_10", "value", "size"])

# ------------------------------------------------- return/exposure alignment
print("\n=== exposure lag (step 4 timing) ===")

# Security A is continuous; security B has a 4-year hole in the middle -- the
# exact shape that makes groupby.shift(1) pair a 2008 exposure with a 2012
# return and call it lagged by one month.
lagf = pd.DataFrame({
    "id": ["A"] * 4 + ["B"] * 3,
    "eom": pd.to_datetime(["2008-01-31", "2008-02-29", "2008-03-31", "2008-04-30",
                           "2008-01-31", "2008-02-29", "2012-06-30"]),
    "f_value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
    "market_equity": [100.0, 110, 120, 130, 200, 210, 220],
    "ret_exc": [0.01, 0.02, 0.03, 0.04, 0.10, 0.20, 0.30],
})
lagf["ind_10"] = 1.0
al = factor_returns.align_returns_to_lagged_exposures(lagf, ["f_value"], ["ind_10"])

check("first month of each security drops (no prior exposure)",
      len(al) == 3 + 1, f"got {len(al)} rows")
a_rows = al[al.id == "A"].sort_values("eom")
check("month-t return paired with month t-1 exposure",
      a_rows["f_value"].tolist() == [1.0, 2.0, 3.0]
      and np.allclose(a_rows["ret_exc"], [0.02, 0.03, 0.04]),
      f"exposures {a_rows['f_value'].tolist()} vs returns {a_rows['ret_exc'].tolist()}")
check("regression row is dated by the RETURN month, not the exposure month",
      a_rows["eom"].tolist() == pd.to_datetime(["2008-02-29", "2008-03-31", "2008-04-30"]).tolist())
check("exposure_eom kept for audit and is one month earlier",
      (a_rows["eom"].dt.to_period("M") - a_rows["exposure_eom"].dt.to_period("M")
       ).apply(lambda x: x.n).eq(1).all())
b_rows = al[al.id == "B"]
check("no pairing across a 4-year gap", len(b_rows) == 1 and b_rows["eom"].iloc[0] == pd.Timestamp("2008-02-29"),
      "the 2012 return has no 2012-05 exposure, so it must not borrow 2008's")
# Demonstrate the bug this avoids.
naive_shift = lagf.sort_values(["id", "eom"]).groupby("id")["f_value"].shift(1)
print(f"   ℹ️  groupby.shift(1) would give security B's 2012-06 return an exposure of "
      f"{naive_shift.iloc[-1]:.0f}, observed 4 years earlier")

# The cap used for the weight and the constraint must travel with the exposures.
check("market cap comes from the exposure month, not the return month",
      a_rows["market_equity"].tolist() == [100.0, 110.0, 120.0])

print("\n=== stored regression panel ===")

# The panel that gets written must BE the thing that was regressed, not a
# re-derivation of it — that is the whole point of returning it from run_panel
# rather than rebuilding it in the notebook.
rp_n, rp_m = 40, 8
rp = pd.concat([
    pd.DataFrame({
        "id": [f"s{i}" for i in range(rp_n)],
        "eom": eom,
        "ret_exc": rng.normal(0, .06, rp_n),
        "market_equity": np.exp(rng.normal(7, 1.2, rp_n)),
        "f_value": rng.normal(0, 1, rp_n),
        "f_size": rng.normal(0, 1, rp_n),
    })
    for eom in pd.date_range("2006-01-31", periods=rp_m, freq="ME")
], ignore_index=True)
rp["sector_code"] = np.tile(np.repeat(["10", "20"], rp_n // 2), rp_m)
rp, rp_dummies = standardize.build_industry_dummies(rp)

Fp, Up, Dp, panel = factor_returns.run_panel(
    rp, ["f_value", "f_size"], rp_dummies, verbose=False)

check("run_panel returns the regression panel as a 4th value", isinstance(panel, pd.DataFrame))
check("panel carries both dates",
      {"eom", "exposure_eom"} <= set(panel.columns))
gap = (panel["eom"].dt.to_period("M") - panel["exposure_eom"].dt.to_period("M")).apply(lambda x: x.n)
check("exposure_eom is exactly one month before eom, every row", (gap == 1).all(),
      f"gaps seen: {sorted(gap.unique())}")
check("panel holds only months that were actually fitted",
      set(panel["eom"]) == set(Fp.index),
      "a skipped month must not leave rows behind")
check("panel row count matches the rows the fit consumed", len(panel) == len(Up))
check("raw return retained alongside the winsorized one", "ret_exc_raw" in panel.columns)

# And the trap it exists to prevent: the UNLAGGED frame must be a different join.
panel_nolag = factor_returns.build_regression_panel(
    rp, ["f_value", "f_size"], rp_dummies, lag_exposures=False, verbose=False)
check("with the lag off, exposure_eom equals eom (and is still stamped)",
      (panel_nolag["exposure_eom"] == panel_nolag["eom"]).all(),
      "so a stored panel always says which alignment built it")

print("\n=== dependent-variable winsorization ===")
wr = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 22.0, -0.03, 0.015, -0.025, 0.005])
wg = pd.Series(["m"] * 10)
cl, n_cl = factor_returns.winsorize_returns(wr, wg, n_sigma=8.0)
check("the +2200% observation is pulled in", cl.max() < 1.0, f"max went {wr.max():.1f} → {cl.max():.4f}")
check("only the outlier moved", n_cl == 1)
check("ordinary returns are untouched", np.allclose(cl[:5], wr[:5]))
# The distinction the handover insists on: winsorizing is not standardizing.
check("returns keep their units (not z-scored)", abs(cl[0] - 0.01) < 1e-12)
check("cross-sectional mean is NOT zeroed — the market factor still has a level",
      abs(cl.mean()) > 1e-6, f"mean = {cl.mean():.4f}")
check("disabling the trim returns the series untouched",
      factor_returns.winsorize_returns(wr, wg, n_sigma=None)[0].equals(wr))
# Robust scale, same argument as for the descriptors.
naive_sigma = wr.std()
check("ordinary sd is corrupted by the outlier, MAD is not", naive_sigma > 6.0,
      f"sd = {naive_sigma:.2f} vs MAD-sigma = {1.4826 * (wr - wr.median()).abs().median():.4f}")


# ================================================================ riskmath
import riskmath, factor_cov, specific_risk, riskmodel, optimizer

print("\n=== riskmath: weights and Newey-West ===")
w = riskmath.exp_weights(60, 12)
check("exp weights sum to one, newest heaviest", abs(w.sum() - 1) < 1e-12 and w.argmax() == 59)
check("weight halves every half-life", abs(w[59 - 12] / w[59] - 0.5) < 1e-12)
check("half_life=None is equal weighting", np.allclose(riskmath.exp_weights(10, None), 0.1))
check("NW rule of thumb floors 4(T/100)^(2/9): 3 at T=60, 4 at T=240 (4.86)",
      riskmath.newey_west_lags(60) == 3 and riskmath.newey_west_lags(240) == 4)

r_iid = rng.normal(0, 1, (60, 18))
check("lags=0, equal weights == ML sample covariance",
      np.allclose(riskmath.nw_cov(r_iid, None, 0), np.cov(r_iid.T, bias=True)))

# The PSD guarantee is the reason for Bartlett weights, so prove the alternative
# fails on the SAME data: a flat (truncated) kernel goes indefinite at T/K = 3.3.
def truncated_kernel(r, lags):
    z = (r - r.mean(0)) / np.sqrt(len(r))
    s = z.T @ z
    for l in range(1, lags + 1):
        g = z[l:].T @ z[:-l]
        s += g + g.T
    return s
check("Bartlett NW is PSD at T=60, K=18, L=5", riskmath.is_psd(riskmath.nw_cov(r_iid, None, 5)))
check("...and the flat-kernel version on the same data is NOT (why Bartlett)",
      not riskmath.is_psd(truncated_kernel(r_iid, 5)),
      f"min eig {np.linalg.eigvalsh(truncated_kernel(r_iid, 5)).min():.3f}")
check("Bartlett NW stays PSD with exponential weights (sqrt-weight construction)",
      riskmath.is_psd(riskmath.nw_cov(r_iid, riskmath.exp_weights(60, 12), 5)))

# NW's actual job: the long-run variance of an autocorrelated series.
rho, n_ar = 0.5, 20_000
e_ar = rng.normal(0, 1, n_ar)
ar = np.empty(n_ar); ar[0] = e_ar[0]
for i in range(1, n_ar):
    ar[i] = rho * ar[i - 1] + e_ar[i]
lr_true = (1 / (1 - rho**2)) * (1 + rho) / (1 - rho)          # = 4.0 for rho = .5
v0 = riskmath.nw_cov(ar, None, 0)[0, 0]
v30 = riskmath.nw_cov(ar, None, 30)[0, 0]
check("NW recovers AR(1) long-run variance; plain variance does not",
      abs(v30 / lr_true - 1) < 0.1 and v0 / lr_true < 0.4,
      f"plain {v0:.2f}, NW(30) {v30:.2f}, true {lr_true:.2f}")

# Panel version with gaps must equal the complete-data estimator on each column.
panel_u = rng.normal(0, 0.05, (60, 5))
wv = riskmath.exp_weights(60, 24)
var_p, nobs = riskmath.nw_var_panel(panel_u, wv, 3)
check("nw_var_panel == nw_cov column by column on complete data",
      np.allclose(var_p, [riskmath.nw_cov(panel_u[:, [j]], wv, 3)[0, 0] for j in range(5)]))
gappy = panel_u.copy(); gappy[:30, 0] = np.nan
v_gap, n_gap = riskmath.nw_var_panel(gappy, wv, 0, min_obs=24)
check("a gap renormalizes the weights over observed months (no dilution by zeros)",
      np.isclose(v_gap[0], riskmath.nw_cov(panel_u[30:, [0]], wv[30:], 0)[0, 0]) and n_gap[0] == 30)
check("below min_obs the variance is NaN, not a guess",      # rows 20-49: 20 observed
      np.isnan(riskmath.nw_var_panel(gappy[20:50, :1], wv[20:50], 0, min_obs=24)[0][0]))

vc = riskmath.vol_corr_cov(r_iid, 6, 48, 0, 0)
check("vol/corr split: diagonal comes from the SHORT half-life",
      np.allclose(np.diag(vc), np.diag(riskmath.nw_cov(r_iid, riskmath.exp_weights(60, 6), 0))))
c48 = riskmath.nw_cov(r_iid, riskmath.exp_weights(60, 48), 0)
sd48 = np.sqrt(np.diag(c48)); sdv = np.sqrt(np.diag(vc))
check("vol/corr split: correlations come from the LONG half-life, and it is PSD",
      np.allclose(vc / np.outer(sdv, sdv), c48 / np.outer(sd48, sd48)) and riskmath.is_psd(vc))

print("\n=== riskmath: eigenfactor adjustment ===")
# Truth with a spread-out spectrum, T/K = 60/18 like the rolling window.
q_true, _ = np.linalg.qr(rng.normal(size=(18, 18)))
f_true = (q_true * np.geomspace(1e-4, 4e-3, 18)) @ q_true.T
sims_true, sims_est = [], []
for _ in range(200):
    rs = rng.multivariate_normal(np.zeros(18), f_true, size=60)
    d_s, u_s = np.linalg.eigh(np.cov(rs.T, bias=True))
    sims_true.append(u_s[:, 0] @ f_true @ u_s[:, 0]); sims_est.append(d_s[0])
understate = np.mean(np.array(sims_true) / np.array(sims_est))
check("the failure it fixes: sample min-eigenportfolio variance is understated at T/K=3.3",
      understate > 1.5, f"true/estimated variance = {understate:.2f}")
rs = rng.multivariate_normal(np.zeros(18), f_true, size=60)
f_hat = np.cov(rs.T, bias=True)
f_adj, info = riskmath.eigenfactor_adjust(f_hat, riskmath.exp_weights(60, None), n_sims=300, seed=1)
check("simulated bias is largest for the smallest eigenvalue",
      info["lam"][0] > 1.2 and info["lam"][0] > info["lam"][-1],
      f"lambda smallest {info['lam'][0]:.2f}, largest {info['lam'][-1]:.2f}")
d_adj = np.linalg.eigvalsh(f_adj)
rot = info["u0"].T @ f_adj @ info["u0"]
check("adjustment rescales eigenvalues but keeps eigenvectors (U0' F_adj U0 is diagonal)",
      np.allclose(rot - np.diag(np.diag(rot)), 0, atol=1e-12 * np.abs(rot).max()))
check("adjusted matrix is PSD and larger along the smallest eigenvector",
      riskmath.is_psd(f_adj) and d_adj[0] > info["d0"][0])
check("same seed, same answer",
      np.allclose(f_adj, riskmath.eigenfactor_adjust(f_hat, riskmath.exp_weights(60, None), 300, seed=1)[0]))

print("\n=== riskmath: VRA and windows ===")
check("VRA is 1 until min_obs out-of-sample values exist",
      riskmath.vra_multiplier(np.full(5, 4.0), 12, 12) == 1.0)
check("VRA of a constant B^2 = 4 is 4 (vol x2)",
      abs(riskmath.vra_multiplier(np.full(30, 4.0), 12, 12) - 4.0) < 1e-12)
pos = list(riskmath.rolling_windows(240, 60))
check("240 months, 60-month window -> 181 forecast dates, 180 testable", len(pos) == 181)
check("every window ends AT its forecast position and is 60 long",
      all(sl.stop == t + 1 and sl.stop - sl.start == 60 for t, sl in pos))
check("window=None hands every date the FULL panel (the look-ahead diagnostic)",
      all(sl == slice(0, 240) for _, sl in riskmath.rolling_windows(240, None, 60)))

print("\n=== factor_cov ===")
fr_idx = pd.date_range("2006-01-31", periods=90, freq="ME")
fr_syn = pd.DataFrame(rng.multivariate_normal(np.zeros(18), f_true, size=90), index=fr_idx,
                      columns=["market"] + [f"ind_{c}" for c in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)]
                      + list(config.STYLE_DESCRIPTORS))
path_a = factor_cov.estimate(fr_syn, window=60, eigen_sims=50, verbose=False)
fr_bumped = fr_syn.copy(); fr_bumped.iloc[-1] *= 25
path_b = factor_cov.estimate(fr_bumped, window=60, eigen_sims=50, verbose=False)
check("no look-ahead: blowing up the LAST month changes no earlier F(t)",
      all(np.allclose(path_a.stages[s][:-1], path_b.stages[s][:-1]) for s in factor_cov.STAGES))
check("...but does change the last one (the test can fail)",
      not np.allclose(path_a.stages["final"][-1], path_b.stages["final"][-1]))
check("every stage PSD at every date",
      all(riskmath.is_psd(m) for s in factor_cov.STAGES for m in path_a.stages[s]))
long_f = factor_cov.to_long(path_a).sample(frac=1, random_state=0)   # DB order is arbitrary
back = factor_cov.from_long(long_f)
check("long round-trip restores canonical order and values from shuffled rows",
      list(back[path_a.dates[3]].columns) == path_a.factors
      and np.allclose(back[path_a.dates[3]].to_numpy(), path_a.stages["final"][3]))
check("both triangles stored (a reader never has to mirror)",
      len(long_f) == len(path_a.dates) * 18 * 18)

print("\n=== specific_risk: Bayesian shrinkage ===")
sig = np.array([0.02, 0.03, 0.04, 0.30, 0.03, np.nan, 0.05, 0.035, 0.025, 0.045])
caps = np.linspace(1, 10, 10)
sh, vw, bk = specific_risk.shrink_to_size_buckets(sig, caps, 1, 0.1)
prior = np.average(sig[np.isfinite(sig)], weights=caps[np.isfinite(sig)])
check("the extreme vol is pulled toward the bucket mean", sh[3] < sig[3] and sh[3] > prior)
check("shrink weight grows with distance from the prior",
      vw[3] > vw[np.nanargmin(np.abs(sig - prior))])
check("missing history takes the prior outright", vw[5] == 1.0 and np.isclose(sh[5], prior))
check("shrinkage reduces cross-sectional dispersion",
      np.nanstd(sh[np.isfinite(sig)]) < np.nanstd(sig))
_, _, bk10 = specific_risk.shrink_to_size_buckets(np.full(100, .05), np.arange(1, 101.), 10, .1)
check("10 size buckets of 10 on cap rank", np.bincount(bk10).tolist() == [10] * 10)

print("\n=== riskmodel: Sigma never materialized, but equal to it ===")
n_a, facs = 40, ["market", "ind_10", "ind_20", "value", "size"]
x_df = pd.DataFrame({"market": 1.0, "ind_10": np.r_[np.ones(20), np.zeros(20)],
                     "ind_20": np.r_[np.zeros(20), np.ones(20)],
                     "value": rng.normal(size=n_a), "size": rng.normal(size=n_a)},
                    index=np.arange(n_a, dtype=float))
a5 = rng.normal(size=(5, 5)); f5 = a5 @ a5.T * 1e-3
dl = pd.Series(rng.uniform(.002, .01, n_a), index=x_df.index)
meta = pd.DataFrame({"ticker": [f"T{i}" for i in range(n_a)], "conm": "x", "sector_name": "s",
                     "cap": rng.uniform(1, 100, n_a)}, index=x_df.index)
rm_ = riskmodel.RiskModel(pd.Timestamp("2020-01-31"), pd.DataFrame(f5, facs, facs), x_df, dl, meta)
sigma_full = x_df.to_numpy() @ f5 @ x_df.to_numpy().T + np.diag(dl)
wp = pd.Series(rng.dirichlet(np.ones(n_a)), index=x_df.index)
check("factor-space variance == w' Sigma w on the materialized Sigma",
      np.isclose(rm_.variance(wp), 12 * wp.to_numpy() @ sigma_full @ wp.to_numpy()))
check("cov_row == the row of Sigma", np.allclose(rm_.cov_row(3.0), 12 * sigma_full[3]))
dec = rm_.decompose(wp)
check("factor + specific contributions sum to total variance",
      np.isclose(dec["by_group"]["var_contrib"].sum(), dec["total_vol"] ** 2))
check("asset vol contributions sum to total vol (Euler)",
      np.isclose(dec["by_asset"]["vol_contrib"].sum(), dec["total_vol"]))
check("predicted beta of the market to itself is 1",
      np.isclose(rm_.betas() @ rm_.market_weights(), 1.0))

ex_rows = pd.DataFrame({"id": [1., 2., 3.], config.CAP_COL: [10., 20., 30.],
                        "f_value": [np.nan, 1., 2.], "f_size": [0., 1., 2.],
                        "ind_10": [1., 0., 0.], "ind_20": [0., 1., 0.]})
xm = riskmodel.exposure_matrix(ex_rows, facs)
check("missing style exposure is zero-filled, as in the regression",
      1.0 in xm.index and xm.loc[1.0, "value"] == 0.0)
check("a name with no industry is not priced (it was not regressed either)", 3.0 not in xm.index)

# Ticker resolution: reused tickers must surface as ambiguous, not be guessed.
exp_t = pd.DataFrame({
    "id": [1., 1., 2., 2., 3., 3.],
    "eom": pd.to_datetime(["2024-12-31", "2025-01-31"] * 3),
    "ticker": ["AAA", None, "BBB", None, "BBB", None],
    "conm": ["a", "a", "b", "b", "c", "c"], "sector_name": "s", config.CAP_COL: 1.0,
})
store_t = riskmodel.ModelStore.__new__(riskmodel.ModelStore)
store_t.exposures = exp_t
store_t._by_eom = {d: g for d, g in exp_t.groupby("eom")}
store_t._ticker_hist = exp_t.loc[exp_t["ticker"].notna(), ["id", "eom", "ticker"]].sort_values("eom")
store_t._ever_tickers = set(store_t._ticker_hist["ticker"].str.upper())
store_t._tt_cache = {}
store_t.dates = pd.DatetimeIndex(["2025-01-31"])
res_t = store_t.resolve(["aaa", "BBB", "ZZZ", "2"], "2025-01-31")
check("last known ticker carried forward past the end of CRSP names (case-insensitive)",
      res_t.loc[0, "status"] == "ok" and res_t.loc[0, "id"] == 1.0)
check("a ticker shared by two live ids is reported ambiguous, not guessed",
      res_t.loc[1, "status"] == "ambiguous")
check("unknown ticker -> not_in_sample; numeric input resolves as id",
      res_t.loc[2, "status"] == "not_in_sample" and res_t.loc[3, "id"] == 2.0)

print("\n=== optimizer ===")
w_mv = optimizer.optimize(rm_, x_df.index, long_only=False, max_weight=10.0)
inv1 = np.linalg.solve(sigma_full, np.ones(n_a))
check("unconstrained min-variance == closed form Sigma^-1 1 / 1'Sigma^-1 1",
      np.allclose(w_mv.to_numpy(), inv1 / inv1.sum(), atol=1e-5))
check("Woodbury closed form == Sigma^-1 1 / 1'Sigma^-1 1 without inverting Sigma",
      np.allclose(optimizer.min_variance_closed_form(rm_).to_numpy(), inv1 / inv1.sum()))
w_lo = optimizer.optimize(rm_, x_df.index, long_only=True, max_weight=0.05)
check("long-only min-var respects bounds and budget",
      w_lo.min() >= -1e-9 and w_lo.max() <= 0.05 + 1e-6 and np.isclose(w_lo.sum(), 1))
check("min-var beats equal weight", rm_.vol(w_lo) < rm_.vol(pd.Series(1 / n_a, x_df.index)))
check("an infeasible cap raises instead of breaking the budget",
      _raises(lambda: optimizer.optimize(rm_, x_df.index, max_weight=0.01)))
# Projected gradient must land on the same unique optimum as SLSQP -- it is the
# solver the full universe uses, where SLSQP cannot run at all.
check("projected-gradient min-var == SLSQP min-var (long-only, capped)",
      np.allclose(optimizer.optimize(rm_, x_df.index, max_weight=0.05, method="pg").to_numpy(),
                  w_lo.to_numpy(), atol=2e-5))
check("projected-gradient long-short == closed form",
      np.allclose(optimizer.optimize(rm_, x_df.index, long_only=False, max_weight=10.0, method="pg").to_numpy(),
                  inv1 / inv1.sum(), atol=2e-5))
pv = optimizer.project_box_simplex(rng.normal(0, 1, 50), 0.0, 0.05)
check("box-simplex projection: sums to one, respects bounds",
      np.isclose(pv.sum(), 1) and pv.min() >= 0 and pv.max() <= 0.05 + 1e-12)
feas = optimizer.project_box_simplex(np.full(50, 0.02), 0.0, 0.05)
check("projection leaves a feasible point where it is", np.allclose(feas, 0.02))
alpha_t = pd.Series(rng.normal(0.05, 0.03, n_a), index=x_df.index)
check("projected-gradient mean-variance == SLSQP mean-variance",
      np.allclose(optimizer.optimize(rm_, x_df.index, alpha_t, 2.0, max_weight=0.2, method="pg").to_numpy(),
                  optimizer.optimize(rm_, x_df.index, alpha_t, 2.0, max_weight=0.2, method="slsqp").to_numpy(),
                  atol=2e-5))
w_hi = optimizer.optimize(rm_, x_df.index, alpha_t, risk_aversion=1.0, max_weight=0.2)
w_lo2 = optimizer.optimize(rm_, x_df.index, alpha_t, risk_aversion=50.0, max_weight=0.2)
check("lower risk aversion buys more expected return and more risk",
      alpha_t @ w_hi > alpha_t @ w_lo2 and rm_.vol(w_hi) > rm_.vol(w_lo2))

print("\n" + "=" * 60)
print(f"{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else " — all checks passed"))
sys.exit(1 if FAIL else 0)
