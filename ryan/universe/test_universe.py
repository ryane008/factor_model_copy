"""
Offline checks for the universe layer. No WRDS, no database.

    python ryan/universe/test_universe.py

The shared model logic is covered by ryan/sample/test_logic.py; this file covers
what is new here -- the config shadowing, point-in-time membership, and the
gates -- and, as there, asserts the failure mode too wherever it can.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import config                     # the universe config -- must be first
import factor_returns, riskmodel  # shared modules from ryan/sample
import membership, wrds_bulk

FAIL = []


def check(name, cond, extra=""):
    print(f"{'✅' if cond else '❌'} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


print("\n=== config shadowing ===")
check("this directory's config is the one imported", os.path.dirname(config.__file__) == HERE)
check("shared modules see the SAME config object (not the sample's)",
      factor_returns.config is config and riskmodel.config is config)
check("default profile is r1000", config.PROFILE == "r1000")
tbls = {k: v for k, v in vars(config).items() if k.startswith("TBL_")}
check("every table re-prefixed; none left pointing at the sample",
      all(v.startswith("ryan_r1000_") for v in tbls.values())
      and len(tbls) == sum(k.startswith("TBL_") for k in vars(config._base)),
      f"{len(tbls)} tables")
check("model settings inherited from the sample (descriptors, lag, eigen sims)",
      config.STYLE_DESCRIPTORS["value"] == ["be_me"] and config.LAG_EXPOSURES and config.EIGEN_SIMS == 1000)
check("survivorship screen is off", config.MIN_RETURN_MONTHS is None)
check("all_descriptors reads this module's descriptors",
      config.all_descriptors()[0] == "be_me" and "ret_exc" not in config.all_descriptors())

# The profile switch is an environment variable read at import, so test it in a
# fresh interpreter -- in this one config is already imported.
probe = ("import sys; sys.path.insert(0, %r); import config; "
         "print(config.PROFILE, config.TBL_EXPOSURES, config.COVERAGE_COL)") % HERE
out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                     env={**os.environ, "BARRA_UNIVERSE": "full"}).stdout.split()
check("BARRA_UNIVERSE=full switches profile, tables and coverage",
      out == ["full", "ryan_full_exposures", "None"], " ".join(out))
bad = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                     env={**os.environ, "BARRA_UNIVERSE": "sp500"})
check("an unknown profile refuses to load", bad.returncode != 0)

print("\n=== screens ===")
p = pd.DataFrame({"market_equity": [100., 0., np.nan, 5., 50.],
                  "size_grp": ["large", "large", "small", "nano", "micro"]})
check("drops non-positive / missing cap and excluded size groups",
      membership.screen(p, ["nano"]).tolist() == [True, False, False, False, True])

print("\n=== reconstitution ===")
# 10 names; caps chosen so the cumulative-share band decides the boundary. Each
# name is ~9% of total cap here, so the band has to be wider than Russell's 2.5%
# to contain one -- 10% puts rank 5 and rank 6 both inside it.
ids = [f"s{i}" for i in range(10)]
caps0 = pd.DataFrame({"id": ids, "rank_date": pd.Timestamp("2001-05-31"),
                      "market_equity": [100., 90, 80, 70, 60, 50, 40, 30, 20, 10]})
r = membership.reconstitute(caps0, n=5)
check("no band: exactly the top n", set(r.loc[r.member, "id"]) == set(ids[:5]))

# Year 2: s4 (incumbent) slips to rank 6, s5 (newcomer) rises to rank 5 -- by a hair.
caps1 = caps0.copy()
caps1["rank_date"] = pd.Timestamp("2002-05-31")
caps1.loc[caps1.id == "s4", "market_equity"] = 49.9
caps1.loc[caps1.id == "s5", "market_equity"] = 50.1
both = pd.concat([caps0, caps1])
hard = membership.reconstitute(both, n=5)
banded = membership.reconstitute(both, n=5, band=0.10)
y2h = set(hard.loc[hard.member & (hard.rank_date == "2002-05-31"), "id"])
y2b = set(banded.loc[banded.member & (banded.rank_date == "2002-05-31"), "id"])
check("hard top-n swaps the two boundary names (the turnover banding prevents)",
      "s5" in y2h and "s4" not in y2h)
check("band: incumbent just below the line stays, newcomer just above waits",
      "s4" in y2b and "s5" not in y2b)
caps2 = caps1.copy()
caps2["rank_date"] = pd.Timestamp("2003-05-31")
caps2.loc[caps2.id == "s4", "market_equity"] = 1.0          # collapses far below the band
b3 = membership.reconstitute(pd.concat([caps0, caps1, caps2]), n=5, band=0.10)
check("band: an incumbent that falls OUTSIDE the band leaves",
      "s4" not in set(b3.loc[b3.member & (b3.rank_date == "2003-05-31"), "id"]))

eoms = pd.Series(pd.to_datetime(["2001-05-31", "2001-06-30", "2001-12-31", "2002-05-31", "2002-06-30"]))
eff = membership.effective_rank_date(eoms, 5, 6)
check("June reconstitution: Jun Y..May Y+1 use May Y's ranks",
      [d.date().isoformat() for d in eff] ==
      ["2000-05-31", "2001-05-31", "2001-05-31", "2001-05-31", "2002-05-31"])

# No look-ahead: a name that explodes in August is not a member until next June.
months = pd.date_range("2001-05-31", "2002-07-31", freq="ME")
rows = []
for m in months:
    for i, cap in enumerate([100, 90, 80, 70, 60, 50]):
        c = cap
        if i == 5 and m >= pd.Timestamp("2001-08-31"):
            c = 1000                                          # s5 becomes the biggest
        rows.append({"id": f"s{i}", "eom": m, "market_equity": c})
pan = pd.DataFrame(rows)
flag = membership.rank_membership(pan, n=5, band=None, rank_month=5, effective_month=6)
s5 = pan.assign(f=flag)[pan.id == "s5"].set_index("eom")["f"]
check("no look-ahead: a cap surge after the rank date changes nothing until the next June",
      not s5.loc[:"2002-05-31"].any() and s5.loc["2002-06-30":].all())
late = pd.concat([pan, pd.DataFrame({"id": "new", "eom": months[months >= "2001-09-30"], "market_equity": 5000})])
late = late.reset_index(drop=True)
fl = membership.rank_membership(late, n=5, band=None, rank_month=5, effective_month=6)
nw = late.assign(f=fl)[late.id == "new"].set_index("eom")["f"]
check("a name listed after the rank date waits for the next reconstitution",
      not nw.loc[:"2002-05-31"].any() and nw.loc["2002-06-30":].all())
elig = pan["id"] != "s0"
fe = membership.rank_membership(pan, n=5, band=None, rank_month=5, effective_month=6, eligible=elig)
check("ineligible rows are never members, and do not take a rank slot",
      not fe[pan.id == "s0"].any() and fe[(pan.id == "s5") & (pan.eom == "2001-06-30")].all())

print("\n=== constituents file ===")
tmp = os.path.join(HERE, "_tmp_constituents.csv")
pd.DataFrame({"permno": [1, 2], "start": ["2001-01-01", "2001-06-01"], "end": ["2001-12-31", ""]}).to_csv(tmp, index=False)
cp = pd.DataFrame({"id": ["a", "a", "b", "b", "c"], "permno_ccm": [1, 1, 2, 2, 3],
                   "eom": pd.to_datetime(["2001-03-31", "2002-03-31", "2001-03-31", "2005-03-31", "2001-03-31"])})
try:
    fc = membership.from_constituents(cp, tmp, permno_col="permno_ccm")
finally:
    os.remove(tmp)
check("interval join on permno and date; open end = current; rows preserved",
      fc.tolist() == [True, False, False, True, False] and len(fc) == len(cp))

print("\n=== profile dispatch ===")
pan2 = pan.assign(size_grp="large", permno_ccm=np.nan)
applied = membership.apply_profile(pan2, config)
check("r1000 profile adds in_estu and the coverage flag",
      {"in_estu", config.COVERAGE_COL} <= set(applied.columns))
check("coverage ⊆ estimation universe", not (applied[config.COVERAGE_COL] & ~applied["in_estu"]).any())

print("\n=== industry gate ===")
ex = pd.DataFrame({"eom": pd.to_datetime(["2001-01-31"] * 4 + ["2001-02-28"] * 4),
                   "ind_10": [1, 1, 1, 0, 1, 1, 0, 0], "ind_20": [0, 0, 0, 1, 0, 0, 1, 1]})
check("gate raises when an industry falls below the floor",
      _raises(lambda: membership.industry_gate(ex, ["ind_10", "ind_20"], 2)))
check("gate passes when every industry clears it",
      not _raises(lambda: membership.industry_gate(ex, ["ind_10", "ind_20"], 1)))

print("\n=== pull window ===")
check("R1000 pull starts at the May rank date behind START_EOM (2000-12 → 2000-05)",
      wrds_bulk.pull_start() == pd.Timestamp("2000-05-31").date(), str(wrds_bulk.pull_start()))

print("\n=== turnover ===")
tv = membership.turnover(pan, flag)
check("turnover counts joiners/leavers only at reconstitution",
      tv.loc["2002-06-30", "joined"] == 1 and tv.loc["2002-06-30", "left"] == 1
      and tv.loc["2001-09-30", "joined"] == 0)

print("\n" + "=" * 60)
print(f"{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else " — all checks passed"))
sys.exit(1 if FAIL else 0)
