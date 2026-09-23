"""
Configuration for the stratified-sample Barra prototype.

Everything that defines the experiment lives here so a run is reproducible from
one file. Nothing in this module touches the network or the database.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Sampling design
# ---------------------------------------------------------------------------

# Cohort anchor. We draw the sample from securities alive at this date and then
# follow that fixed cohort forward, so delistings and data gaps show up in the
# panel the way they would in a live model. Drawing from the *current* universe
# instead would keep only survivors and bias specific risk low.
ANCHOR_EOM = date(2005, 12, 31)

# Names kept per GICS sector. 11 sectors x 20 = 220 securities.
#
# Why stratify: Utilities is ~3% of the US listed universe, so a uniform draw of
# ~100-220 names can easily land 0-2 Utilities names. An industry dummy with one
# member perfectly fits that asset's return, forcing its residual (and therefore
# its specific variance) to zero -- which puts a zero on the diagonal of Delta,
# makes Sigma singular, and tells the optimizer a portfolio concentrated in that
# one stock is riskless. Stratifying makes that failure mode impossible.
NAMES_PER_SECTOR = 20

# An asset needs at least this many months of non-null returns (from the anchor
# date forward) to support a specific-risk estimate. Applied *before* the draw.
MIN_RETURN_MONTHS = 60

# Fixed so the draw is reproducible across machines and reruns.
RANDOM_SEED = 20260912

# ---------------------------------------------------------------------------
# Characteristics to pull
# ---------------------------------------------------------------------------

# The dependent variable in the cross-sectional regression. Deliberately kept
# out of STYLE_DESCRIPTORS: standardizing the return would put factor returns in
# z-units/month instead of returns, and subtracting the cross-sectional mean
# removes exactly what the market factor is supposed to capture.
RETURN_COL = "ret_exc"

# Style descriptors, grouped by the factor they feed. Grouping matters because
# collinear descriptors entered separately make the individual factor returns
# unstable -- USE4 combines descriptors into one factor rather than letting each
# compete in the regression.
#
# Value is book-to-price ALONE, which is USE4's definition. It previously blended
# be_me with ni_me and ocf_me, and that blend did not mean what its name said.
# The two earnings yields share a market-equity denominator with be_me, but their
# NUMERATORS carry far more cross-sectional dispersion -- losers are firms with
# collapsing or negative earnings -- so the numerator channel swamps the price
# channel and they behave as profitability proxies. Measured against momentum
# exposure: be_me -0.33, ocf_me -0.01, ni_me +0.14. Blended, Value came out at
# -0.09 against momentum and +0.19 against profitability, the latter being
# backwards for a value factor.
#
# On be_me alone: -0.333 against momentum, -0.326 against profitability. Cheap
# stocks are the ones that fell and the ones with weak businesses; both signs
# are right, and the momentum figure is stable across five-year eras
# (-0.37, -0.33, -0.33, -0.30, -0.29).
#
# Nothing measurable in Sigma turns on this. Design condition number 8.58 vs
# 8.60, median R^2 0.274 either way, condition number of F 44.7 vs 43.6. It is
# an interpretability choice, not a numerical one.
#
# Splitting the earnings yields into their own USE4-style "Earnings Yield"
# factor was tested and is worse HERE: it lands +0.408 against profitability on
# exposures and -0.420 on factor returns -- the most collinear pair in the model
# -- and splits the premium so neither factor is significant (t = 1.35, 1.40).
# USE4 can afford that nineteenth factor on ~3,000 names; at N ~ 170 it spends a
# degree of freedom this sample cannot resolve. Revisit if the universe grows.
STYLE_DESCRIPTORS = {
    "value": ["be_me"],
    "size": ["market_equity"],
    "momentum": ["ret_12_1"],
    "profitability": ["gp_at", "ni_at"],
    "investment": ["at_gr1", "oaccruals_at"],
    "beta": ["beta_60m"],
}

# Descriptors that are standardized on a log scale. Raw market cap is so skewed
# that z-scoring it puts a handful of mega-caps at +40 sigma; USE4's Size
# descriptor is log(market cap).
LOG_DESCRIPTORS = ["market_equity"]

# Cross-sectional weight for cap-weighted standardization and (later) the WLS
# regression weights.
CAP_COL = "market_equity"

# Outlier trim, in cross-sectional standard deviations.
WINSOR_SIGMA = 3.0

# ---------------------------------------------------------------------------
# Step 4: cross-sectional regression
# ---------------------------------------------------------------------------

# Regress month-t returns on exposures known at the END OF MONTH t-1.
#
# This is not a refinement, it is a correctness requirement, and the panel as
# built violates it: `ret_exc` at eom t is the return *during* month t, while
# `market_equity` at eom t is the cap *after* that month's price move. So
# be_me = book / market_equity at eom t is mechanically higher for a stock that
# fell during month t. Measured on this panel, the cross-sectional correlation
# between z_be_me and the same-month return is -0.116, and the contemporaneous
# fit reports a value factor return of -18.1%/yr with t = -4.7 -- a "premium"
# that is pure look-ahead. Lagging one month moves the value exposure's
# correlation with the return from -0.067 to +0.031 and size from +0.045 to
# -0.005.
LAG_EXPOSURES = True

# Cross-sectional trim of the DEPENDENT variable, in robust (MAD) sigmas.
#
# This is winsorization, not standardization -- ret_exc keeps its units and its
# cross-sectional mean, so factor returns stay in returns/month. What it removes
# is leverage, not level.
#
# Secondary to LAG_EXPOSURES, and worth being precise about why, because the
# obvious reading of the evidence is wrong. Measured on this panel:
#
#     lag    winsor   corr(value,momentum)   value %/yr    t      max style vol
#     off    off              +0.924           -18.1     -4.69        22.8%
#     off     8 sigma         +0.230           -14.6    -12.43        12.0%
#     on     off              +0.059            +2.3      2.06         7.9%
#     on      8 sigma         +0.018            +2.6      2.44         7.8%
#
# The pathological untrimmed numbers were caused by SHARPLINK INC, which
# returned +2225% in 2025-05. Its market cap went from $2.0M to $4,575M during
# that same month, so a CONTEMPORANEOUS fit weights it by the post-spike cap and
# hands one observation the whole covariance matrix. Lagging takes the weight
# from the pre-spike $2.0M and the leverage disappears on its own -- which is
# why row 3 is already sane with no trimming at all.
#
# Note row 2: winsorizing WITHOUT fixing the timing makes the look-ahead look
# *more* convincing, not less (t goes from -4.7 to -12.4), because it strips the
# noise while leaving the bias. Treating the outlier as the disease rather than
# the symptom would have produced a confidently wrong model.
#
# Kept on as cheap insurance against genuine extreme returns at stable caps,
# where the lag does not help. Set to None to disable.
WINSOR_RETURN_SIGMA = 8.0


def all_descriptors() -> list[str]:
    """Flat list of every style descriptor, order-stable and de-duplicated."""
    seen: dict[str, None] = {}
    for cols in STYLE_DESCRIPTORS.values():
        for col in cols:
            seen.setdefault(col, None)
    return list(seen)


def characteristics_to_pull() -> list[str]:
    """Every numeric column we request from ``contrib.global_factor``."""
    return all_descriptors() + [RETURN_COL]


# ---------------------------------------------------------------------------
# JKP universe filters (applied on the WRDS side)
# ---------------------------------------------------------------------------

# JKP's recommended screens for a clean, tradeable, one-row-per-security panel.
JKP_FILTERS = {
    "common": 1,  # common stock only
    "exch_main": 1,  # main exchange listing
    "primary_sec": 1,  # primary security for the firm
    "obs_main": 1,  # JKP's main observation flag
}
COUNTRY = "USA"

# ---------------------------------------------------------------------------
# Industry factor granularity
# ---------------------------------------------------------------------------

# Which GICS level becomes the set of industry dummies. "gsector" gives 11
# dummies with 20 names each by construction. "ggroup" would give 25 dummies
# averaging ~9 names, with thin ones -- too sparse at N=220.
INDUSTRY_CODE_COL = "gsector"

# ---------------------------------------------------------------------------
# TimescaleDB tables (all prefixed so they never collide with the full-universe
# tables in the same database)
# ---------------------------------------------------------------------------

TBL_UNIVERSE = "ryan_sample_universe"  # the 220-name roster + how it was drawn
TBL_PANEL = "ryan_sample_panel"  # raw monthly characteristics for the roster
TBL_EXPOSURES = "ryan_sample_exposures"  # standardized exposures + industry dummies

# Step 4 outputs. Two tables rather than one because they are at different
# grains -- f(t) is per (eom, factor), u(i,t) is per (id, eom) -- but they come
# out of a single regression and are always written together.
# The exact rows the regression was run on, carrying BOTH dates: exposure_eom
# (right-hand side) and eom (the month whose return is on the left). Stored so
# the lag is visible in the data rather than implicit in code -- reading
# TBL_EXPOSURES and regressing its ret_exc on its own f_* columns is the
# contemporaneous fit, and nothing in the output warns you.
TBL_REGRESSION_PANEL = "ryan_sample_regression_panel"  # (id, eom) + exposure_eom

TBL_FACTOR_RETURNS = "ryan_sample_factor_returns"  # f(t), long: (eom, factor)
TBL_SPECIFIC_RETURNS = "ryan_sample_specific_returns"  # u(i,t), (id, eom)
TBL_REGRESSION_DIAG = "ryan_sample_regression_diag"  # per-cross-section fit quality
