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

print("\n" + "=" * 60)
print(f"{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else " — all checks passed"))
sys.exit(1 if FAIL else 0)
