"""
Configuration for the full-universe and Russell 1000 builds.

This file SHADOWS ryan/sample/config.py. Notebooks here put this directory first
on sys.path and ryan/sample second, so ``import config`` -- in the notebooks and
inside every shared module (factor_returns, factor_cov, riskmodel, ...) --
resolves to this file, while the model logic itself is imported from
ryan/sample. One codebase; a fix there fixes every universe.

Everything not overridden below is inherited from the sample config verbatim:
the style descriptors, winsorization, exposure lag, covariance half-lives,
eigenfactor settings, shrinkage, VRA. Those were validated out of sample on the
sample and there is no reason to believe the universe changes them -- 03-05 will
say if it does.

Pick the universe with the BARRA_UNIVERSE environment variable or by editing
DEFAULT_PROFILE. Tables are prefixed per profile (ryan_full_*, ryan_r1000_*), so
the two builds and the sample never overwrite each other.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAMPLE = os.path.abspath(os.path.join(_HERE, "..", "sample"))

# The shared modules live in ryan/sample. Appended, not inserted, so this
# directory -- and therefore this config -- stays ahead of the sample's config.
if _SAMPLE not in sys.path:
    sys.path.append(_SAMPLE)

_spec = importlib.util.spec_from_file_location("sample_config", os.path.join(_SAMPLE, "config.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
globals().update({k: v for k, v in vars(_base).items() if not k.startswith("_") and k.isupper()})

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = "r1000"
PROFILE = os.environ.get("BARRA_UNIVERSE", DEFAULT_PROFILE).strip().lower()
PROFILES = ("full", "r1000")
if PROFILE not in PROFILES:
    raise ValueError(f"BARRA_UNIVERSE={PROFILE!r}; expected one of {PROFILES}")

TABLE_PREFIX = f"ryan_{PROFILE}_"

# Every TBL_* name from the sample, re-prefixed. Done by rewriting the prefix
# rather than listing tables, so a table added to the sample later is picked up
# here automatically instead of silently pointing at the sample's copy.
for _k in [k for k in list(globals()) if k.startswith("TBL_")]:
    globals()[_k] = globals()[_k].replace("ryan_sample_", TABLE_PREFIX)
if any(str(v).startswith("ryan_sample_") for k, v in list(globals().items()) if k.startswith("TBL_")):
    raise AssertionError("a table name still points at the sample")

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

# First exposure month. Factor returns start one month later, and F(t) needs
# COV_WINDOW months of them, so forecasts start ~5 years after this.
#
# 2000-12 rather than the sample's 2005-12: T is the binding constraint on F
# (handover decision #8 -- more assets cannot fix it), and five more years is
# 60 more forecast dates to test on. Earlier is possible (JKP US goes back much
# further) but GICS itself dates from 1999; before that comp.company's codes are
# back-filled onto firms that never carried them.
START_EOM = date(2000, 12, 31)

# Last month pulled. None = whatever contrib.global_factor has.
END_EOM = None

# Kept for code paths that read the sample's name for the first date.
ANCHOR_EOM = START_EOM

# The sample's 60-month survival screen is a survivorship filter: it excludes
# exactly the names that died early, biasing specific risk low. It existed only
# because 220 names could not afford thin histories. Off -- a name with too few
# residuals for its own specific variance takes its size-decile prior (04).
MIN_RETURN_MONTHS = None

# ---------------------------------------------------------------------------
# Industries
# ---------------------------------------------------------------------------

# "sector_code" (11 GICS sectors, K = 18) or "group_code" (25 industry groups,
# K = 32). Sectors by default, deliberately.
#
# The universe has plenty of names per group; the constraint is TIME. With a
# 60-month window, K = 32 puts T/K at 1.9 against 3.3 today, and at T/K = 3.3 the
# smallest eigenvalues were already understated by ~55% before the eigenfactor
# adjustment. Groups are the obvious next experiment: switch this, set
# COV_WINDOW to 120 (START_EOM 2000 still leaves 181 forecast dates), re-run
# 01-05, and let the bias tests in 05 decide.
INDUSTRY_CODE_COL = "sector_code"

# Refuse to build if any industry ever has fewer names than this in a month. An
# industry dummy with one member fits it exactly: zero residual, zero specific
# variance, singular Sigma. The sample watched this by eye; here it is a gate.
MIN_INDUSTRY_NAMES = 10

# ---------------------------------------------------------------------------
# Universe membership
# ---------------------------------------------------------------------------

# JKP size_grp values dropped from the estimation universe. 'nano' is below the
# 1st percentile of NYSE market cap: untradeable, stale-priced, and returns
# dominated by bid-ask bounce. They would get tiny sqrt(cap) weights in the
# regression anyway, but they would still pollute the size-decile priors in 04.
EXCLUDE_SIZE_GRP = ["nano"]

if PROFILE == "full":
    # Estimation universe = coverage = every JKP US common stock passing the
    # screens, month by month. Dynamic: names enter at listing and leave at
    # delisting, with no survival requirement.
    ESTU_RULE = "screens"
    COVERAGE_COL = None
else:
    # Russell 1000 proxy, ESTIMATED on a Russell 3000 proxy.
    #
    # Why not estimate on the R1000 alone: a name joining the index would arrive
    # with no specific-return history (it was never in the regression), so every
    # new member would sit on the size-bucket prior for its first 24 months --
    # roughly a fifth of the index at any time. Estimating on the top 3000 gives
    # them a history, adds ~2000 names of cross-sectional information, and costs
    # little: the R1000 is ~93% of R3000 cap, so cap-weighted standardization
    # barely moves. This is Barra's estimation-universe / coverage split.
    ESTU_RULE = "rank"
    ESTU_N = 3000
    COVERAGE_N = 1000
    COVERAGE_COL = "in_r1000"

    # Reconstitution, FTSE Russell style: rank on market cap at the end of
    # RANK_MONTH, effective from the end of EFFECTIVE_MONTH (the late-June
    # reconstitution) through the following RANK_MONTH. Membership at any date is
    # decided only from caps known on the rank date -- no look-ahead.
    RANK_MONTH = 5
    EFFECTIVE_MONTH = 6

    # Banding at the R1000/R2000 boundary: +/-2.5% of cumulative market cap
    # around the 1000th name, as Russell has done since 2007. Incumbents inside
    # the band stay; newcomers inside it wait. Reduces turnover at the boundary,
    # which is what the real index holds. None = hard top-N.
    COVERAGE_BAND = 0.025

    # If you have actual constituents (FTSE Russell via WRDS or elsewhere),
    # point this at a CSV with columns permno,start,end (end blank = current) and
    # it replaces the proxy for the coverage flag. The R3000 estimation proxy is
    # kept either way.
    CONSTITUENTS_CSV = None

# ---------------------------------------------------------------------------
# Pull mechanics
# ---------------------------------------------------------------------------

# contrib.global_factor is the whole world; pulled one calendar year per query
# so a timeout costs a year, not the build.
PULL_CHUNK_YEARS = 1
# crsp.stocknames lookups batched by permno.
PERMNO_BATCH = 2000

# 03's anchor check compares against the sample's handover numbers; those do not
# apply to a different universe. Set after a first successful build to guard
# later rebuilds the way the sample's anchors do.
ANCHORS = None

PORTFOLIO_CSV = "portfolios/example_portfolio.csv"


def all_descriptors() -> list[str]:
    """Flat list of every style descriptor (redefined so it reads THIS module's STYLE_DESCRIPTORS)."""
    seen: dict[str, None] = {}
    for cols in STYLE_DESCRIPTORS.values():
        for col in cols:
            seen.setdefault(col, None)
    return list(seen)


def characteristics_to_pull() -> list[str]:
    return all_descriptors() + [RETURN_COL]


def describe() -> str:
    """One line naming the active profile -- every notebook prints it first."""
    rule = ("all JKP US common stocks passing screens" if ESTU_RULE == "screens"
            else f"top {ESTU_N} by cap (R3000 proxy), coverage = top {COVERAGE_N} "
                 f"({'actual constituents' if CONSTITUENTS_CSV else f'proxy, band ±{COVERAGE_BAND:.1%}'})")
    return (f"PROFILE={PROFILE}  tables={TABLE_PREFIX}*  {START_EOM}→{END_EOM or 'latest'}  "
            f"industries={INDUSTRY_CODE_COL}  estimation universe: {rule}")
