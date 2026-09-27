# Barra prototype — 220-name stratified sample

Builds the inputs for a USE4-style risk model on a deliberately small universe:
**20 securities per GICS sector × 11 sectors = 220**, drawn once from the
securities alive at **2005-12-31** and followed forward.

> **Full universe / Russell 1000:** `ryan/universe/` builds the same model on a
> real universe, reusing every module here. See its README.

## Quick start

```bash
cp .env.template .env           # from the repo root; fill in WRDS_USERNAME / WRDS_PASSWORD
docker compose up -d
```

Open http://localhost:8888 (token `qtab_jupyter_2024`) → `ryan/sample/run_pipeline.ipynb`
and run top to bottom. Nothing needs installing: the container carries every
dependency, and `db.py` resolves the database hostname whether you run inside the
container or against the published port from the host.

The pure logic has no WRDS or database dependency, so it can be checked anywhere:

```bash
python ryan/sample/test_logic.py      # 137 checks, no network
```

## Re-running it

Run the notebook **top to bottom every time** — don't start from a middle cell.
Two flags in the *Run mode* cell decide what actually recomputes, and by default
a second run resumes from TimescaleDB without touching WRDS:

| Want to… | `REDRAW_UNIVERSE` | `RELOAD_PANEL` | Hits WRDS | Roughly |
|---|---|---|---|---|
| Re-standardize, change descriptors, tune winsorization | `False` | `False` | no | seconds |
| Pull newer months for the same 220 names | `False` | `True` | yes | one small query |
| Draw a different sample | `True` | `True` | yes | full rebuild |

The notebook checks `to_regclass` for each table, so a missing table rebuilds
itself automatically and rebuilding the universe always forces a panel rebuild
(a new roster invalidates the stored panel). Stage 3 always recomputes — it's
pure pandas over a panel already in memory, and it's the stage you're most likely
iterating on.

> ⚠️ **`REDRAW_UNIVERSE = True` changes which 220 companies you are modelling,
> and the fixed seed does not protect you.** The draw is reproducible only for a
> fixed sampling frame, and the frame is built from live WRDS data: next month
> `fetch_return_coverage` returns larger `n_ret` counts, so different securities
> clear the `MIN_RETURN_MONTHS` screen, the eligible pool changes size, and the
> same seed selects different names. Anything computed against the old roster
> stops being comparable. `ryan_sample_universe` is the durable record of which
> sample a result belongs to — it stores the seed, anchor, and screen alongside
> the roster.

## Why a sample

The full-universe pull produced **three months** of data with returns in only one
of them. The factor covariance matrix `F` is `K×K` over ~17 factors and needs
`T >> K` monthly cross-sections to be non-singular, so one usable month makes it
rank-1 and the model unbuildable. A small universe with a *long* history is the
opposite trade, and it is the right one: `F` needs large `T`, not large `N`.

With 220 assets you can still build the full `220×220` `Σ` and run real bias tests
— compare `wᵀΣw` against realized out-of-sample portfolio variance. The prototype
is a genuine validation, not a smoke test.

## Why stratified rather than uniform

Utilities is ~3% of the US listed universe. Simulated over 4,000 draws, a uniform
220-name sample lands **≤1 name in some sector 1.4% of the time and ≤2 about 5.9%
of the time**.

An industry dummy with a single member is fatal, not merely noisy. That column
fits its one asset's return exactly, so the residual — and therefore the specific
variance — is forced to zero. That puts a zero on the diagonal of `Δ`, makes `Σ`
singular, and reports that a portfolio concentrated in that one stock has **zero
risk**. An optimizer will find it. Stratifying removes the failure mode by
construction; `run_pipeline.ipynb` also charts the thinnest industry per month so
you can see it staying healthy.

## Files

| File | Role |
|---|---|
| `run_pipeline.ipynb` | **Main driver.** Three stages with sanity checks and plots at each. |
| `config.py` | Every knob that defines the experiment: anchor, seed, sector count, descriptors, table names. |
| `db.py` | Engine construction, connection check, table writes with a uniqueness constraint. |
| `wrds_io.py` | All WRDS queries. Read-only. |
| `sampling.py` | The stratified draw and the date-aware identifier joins. |
| `standardize.py` | Robust winsorization, cap-weighted standardization, descriptor combination, industry dummies. |
| `gics.py` | GICS code → name tables, hardcoded. |
| `02_factor_returns.ipynb` | Step 4: constrained WLS → factor returns f(t), specific returns u(i,t). |
| `03_factor_covariance.ipynb` | Step 5: rolling F(t) — decay, eigenfactor adjustment, VRA — with each component scored out of sample. |
| `04_specific_risk.ipynb` | Step 6: rolling Δ(t) — decay, Bayesian size-decile shrinkage, specific VRA. |
| `05_risk_model.ipynb` | Step 7: Σ = XFXᵀ + Δ assembled lazily; out-of-sample bias tests at factor, asset and portfolio level. |
| `06_portfolio_risk.ipynb` | Risk of a portfolio CSV: vol, beta, tracking error, factor/asset attribution, forecast-vs-realized history. |
| `07_mean_variance.ipynb` | Mean-variance / min-variance optimization of a portfolio CSV, frontier, trades, walk-forward check. |
| `08_asset_query.ipynb` | One stock: exposures, risk decomposition, most/least covarying names, history; dropdown + `pair(a, b)`. |
| `factor_returns.py` | Step 4 logic: exposure lag, return winsorization, constrained WLS. |
| `riskmath.py` | Shared estimator math: exponential weights, Newey-West (Bartlett), eigenfactor adjustment, VRA, bias stats. |
| `factor_cov.py` | Rolling F(t) driver; every stage kept; long-format I/O. |
| `specific_risk.py` | Rolling Δ(t) driver and USE4 Bayesian shrinkage. |
| `riskmodel.py` | `ModelStore` / `RiskModel`: Σ in factor space, decomposition, cov rows, ticker resolution. |
| `biastest.py` | Out-of-sample bias statistics shared by 03 and 05. |
| `portfolio.py` | Portfolio CSV loading/resolution; illustrative default α. |
| `optimizer.py` | SLSQP mean-variance in factor space; Woodbury closed-form min-variance. |
| `query.py` | Single-asset report and pairwise covariance breakdown used by 08. |
| `portfolios/` | Portfolio CSVs. `example_portfolio.csv` is the default input; 07 writes optimized weights next to it. |
| `test_logic.py` | 137 offline checks over the pure logic. |

## Output tables

All prefixed `ryan_sample_` so they never collide with the full-universe tables
in the same database. Each is written with `if_exists="replace"` plus a unique
index, so a double run fails loudly instead of silently doubling the panel.

| Table | Grain | Contents |
|---|---|---|
| `ryan_sample_universe` | one row per `id` | The roster, plus the seed and rules that produced it |
| `ryan_sample_panel` | `(id, eom)` | Raw monthly characteristics, returns, resolved identifiers, sector |
| `ryan_sample_exposures` | `(id, eom)` | Standardized style factors `f_*` and industry dummies `ind_*` |
| `ryan_sample_regression_panel` | `(id, eom)` | The rows actually regressed, carrying both `exposure_eom` and `eom` |
| `ryan_sample_factor_returns` | `(eom, factor)` | f(t) |
| `ryan_sample_specific_returns` | `(id, eom)` | u(i,t) |
| `ryan_sample_regression_diag` | `eom` | Per-month fit quality |
| `ryan_sample_factor_covariance` | `(forecast_date, factor_i, factor_j)` | F(t): `cov` (final) plus `cov_raw`, `cov_nw`, `cov_eigen` |
| `ryan_sample_factor_cov_diag` | `forecast_date` | VRA λ, eigen bias, condition numbers |
| `ryan_sample_specific_risk` | `(id, forecast_date)` | Δ(t): `specific_var` (final) plus raw / shrunk stages |
| `ryan_sample_bias_tests` | `(level, test, stage)` | 05's out-of-sample bias summary |

## What changed from the full-universe pipeline

Each of these was a real defect, not a style preference.

**Sample on `id`, never ticker.** Tickers get reused across firms — the old
loader found `BCS` mapping to both `LIFESTYLE RESTAURANT` and `BARCLAYS PLC` —
change over time, and are missing for a large share of `crsp.stocknames`. `id` is
JKP's security identifier and the grain of the whole pipeline. Tickers are
carried for display only.

**Anchor the window on an explicit date.** The old loader computed
`datetime.now() - months_back*30`. Run in Dec 2025 with `months_back=12` that
asked for `eom >= 2024-12-10`, and `contrib.global_factor` ends 2025-02-28 —
which is the entire reason only three months came back.

**Date-aware CCM link join.** The old query filtered on `linktype IN ('LU','LC')`
and called the result "active links", but linktype is a link *quality* flag, not
a date filter. With `linkdt`/`linkenddt` dropped and no `linkprim` filter, a
gvkey that mapped to permno X in 1990 and Y in 2010 contributed both rows at
every date — which is what fanned 14,813 rows out to 47,448. The fan-out was then
papered over with two `drop_duplicates` calls that happened to recover the right
row count only because `ticker` was the sole column that varied; `permno` and
`permco` were being picked arbitrarily. Here the join is restricted to the link
whose window contains `eom`, so there is exactly one answer per row and no
cleanup step to get wrong.

**No GICS mapping file.** The Kaggle `gics-map-2023.csv` join had two independent
defects: ~6.6% of rows carrying valid WRDS codes failed the four-key merge (WRDS
ships current post-March-2023 GICS; that file predates the revision for at least
some codes), and the labels were wrong even where the merge *succeeded* —
American Airlines, `gsector 20`, came back as Sector "Materials". There are only
11 sectors and 25 industry groups, so they live in `gics.py` as literals. The
model keys on the numeric **code** and treats names as display labels, so a name
drifting between vintages cannot affect any number.

**Robust winsorization before standardizing.** Trimming at 3σ of the *ordinary*
standard deviation does not work, because the outlier you are removing is itself
inflating that standard deviation: with 60 names from N(0,1) plus one value at 40,
the sample sd is ~5.2, so the outlier sits at only 7.3σ and every genuine
observation is compressed into ±0.2. Median and MAD are immune to it. And the
tempting alternative — standardize, clip at ±3, re-standardize — is worse than
useless: the second pass divides by the *clipped* standard deviation, which is
below 1, scaling the clipped values straight back out. That returned a max |z| of
7.3 for a nominal 3σ trim.

**Cap-weighted mean, equal-weighted std.** A plain `(x - x.mean()) / x.std()`
zeroes the *equal-weighted* mean. USE4 zeroes the *cap-weighted* mean, which is
what makes the market portfolio have zero exposure to every style factor —
without it the market factor and the style factors overlap and the factor returns
are not separately interpretable.

**log(market cap) for size.** Raw `market_equity` is skewed enough (skew ≈ 12 in
testing) that z-scoring it puts a handful of mega-caps at +40σ. USE4's Size
descriptor is the log.

**`ret_exc` is never standardized.** It is the dependent variable in the
cross-sectional regression, not an exposure. Standardizing it would put factor
returns in z-units per month instead of returns, and subtracting the
cross-sectional mean removes exactly what the market factor is supposed to
capture. The old `fama_french_5_columns` list included it.

**Collinear descriptors collapse to one factor.** `be_me`, `ni_me` and `ocf_me`
entered as three separate factors make the individual factor returns unstable
even where the overall fit is fine. USE4 combines descriptors into one factor per
concept, so Value becomes one column.

**Simplified host resolution.** The old `is_docker_environment()` sniffed
`JUPYTER_SERVER_ROOT`, which a *local* Jupyter also sets — so running a notebook
locally under Jupyter would try to reach host `timescaledb` and fail. compose
already sets `PGHOST=timescaledb` for the container, so
`os.getenv("PGHOST", "localhost")` does the whole job.

## Known limitations, deliberately accepted

**Point-in-time GICS.** `comp.company` is the *current* header file, so a firm
that changed sector in 2015 is labelled with its 2026 sector back to 2005. That
is real look-ahead over a 20-year panel. `comp.co_hgic` carries the history when
we want to remove it.

**The `MIN_RETURN_MONTHS` screen is a survivorship-flavoured filter.** Requiring
60 months of returns from the anchor forward excludes firms that delisted in
2006–2010. It is applied on purpose — every sampled asset must be able to support
a specific-risk estimate — but it does bias specific risk low, and it should come
off before any result is treated as an out-of-sample claim.

**Descriptor weights are equal.** USE4 uses fitted weights within each factor.
Equal weighting is a placeholder.

## The risk model — steps 5–7

Run order: `run_pipeline` → `02` → `03` → `04` → `05`. Then `06`–`08` are tools
that read the stored model and can be run in any order, as often as you like.

**Forecast-date convention.** F(t) and Δ(t) use data through eom *t* and forecast
month *t + 1*. A portfolio held at the end of month *t* is priced with X(t), F(t),
Δ(t), and every bias test pairs the forecast at *t* with the outcome at *t + 1*.
240 months of factor returns on a 60-month window give 181 forecast dates, 180
of them testable.

**Σ is never materialized.** `RiskModel.variance(w)` computes
`y = Xᵀw; yᵀFy + Σ w²δ`; the asset query uses one row of Σ, `X @ (F @ x_i)`; the
optimizer's gradient is `X(F(Xᵀw)) + Δw`. 05 builds Σ once, only to verify these
against it.

### Out-of-sample results (bias statistic: 1 = calibrated; 95% band ≈ 1 ± 0.10)

| | raw sample F | final F |
|---|---|---|
| min-variance, long-short (assets) | 1.18 | **1.07** |
| min-variance, long-only ≤5% (assets) | 1.07 | **1.04** |
| cap-weighted / equal-weighted / random portfolios | 0.98–1.02 | 1.00–1.03 |
| smallest eigenportfolio (factors) | 2.04 | 1.35 |
| single stocks, median monthly B_t | — | 0.88 (pooled 1.26; 1.007 excluding 9 of 28,072 \|z\|>10 months) |

A single full-sample F scores 0.98 on the min-variance factor portfolio in
sample, against 1.63 for the rolling raw F out of sample. That is handover
decision #2 reproduced on real data.

### Newey-West is implemented and OFF (`FACTOR_NW_LAGS = SPECIFIC_NW_LAGS = 0`)

This departs from the original plan, based on measurement. Every added lag made F
worse out of sample (smallest-eigenvector bias 2.15 → 2.55 → 2.99 → 3.52 for lags
0–3, min-variance 1.65 → 3.12), and made Δ worse too (1.04 → 1.14). On 60 monthly
observations the 18×18 lag cross-covariances are almost pure noise (factor lag-1
|ρ| median 0.04), and adding noise to a covariance spreads its spectrum. Bartlett
keeps the result PSD; it does not keep it right. USE4 affords NW because it
estimates on daily data. 03 and 04 re-run the comparison on every execution.

### Portfolio CSVs

```
ticker,weight          # or `id` (permno); weight optional (equal weight);
COST,0.08              # optional expected_return (annualized) for 07
```

Only names in the sample can be priced: ~117 are alive at the latest date, and
08 lists them. Tickers are resolved as-of the forecast date and carried forward
from CRSP's name history, which ends at 2024-12 in this pull. Ambiguous or
unknown tickers are reported, never guessed or silently dropped.
