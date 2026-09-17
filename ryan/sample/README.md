# Barra prototype — 220-name stratified sample

Builds the inputs for a USE4-style risk model on a deliberately small universe:
**20 securities per GICS sector × 11 sectors = 220**, drawn once from the
securities alive at **2005-12-31** and followed forward.

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
python ryan/sample/test_logic.py      # 40 checks, no network
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
| `test_logic.py` | 40 offline checks over the pure logic. |

## Output tables

All prefixed `ryan_sample_` so they never collide with the full-universe tables
in the same database. Each is written with `if_exists="replace"` plus a unique
index, so a double run fails loudly instead of silently doubling the panel.

| Table | Grain | Contents |
|---|---|---|
| `ryan_sample_universe` | one row per `id` | The roster, plus the seed and rules that produced it |
| `ryan_sample_panel` | `(id, eom)` | Raw monthly characteristics, returns, resolved identifiers, sector |
| `ryan_sample_exposures` | `(id, eom)` | Standardized style factors `f_*` and industry dummies `ind_*` |

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

## Not done yet — stages 4 and 5

**Step 4 — factor and specific returns.** One cross-sectional WLS regression per
month, weights ∝ √(market cap):

```
r_i = Σ_k X_ik f_k + u_i
```

Industry dummies and style factors go into the *same* regression. The design
matrix `[market | industries | styles]` is rank-deficient by construction — the
industry dummies sum to the market's column of ones — so it needs USE4's
cap-weighted constraint `Σ_i w_i f_i^ind = 0`. Pre-orthogonalizing the styles
against the industry dummies does **not** fix this; the constraint is required
either way.

**Step 5 — the covariance matrix.**

```
Σ = X F Xᵀ + Δ
```

`F` is the covariance of **all** factor returns jointly — one `K×K` matrix, not
separate industry and style matrices added together, which would drop every
industry–style cross-covariance. `Δ` is diagonal specific variance, Newey-West
adjusted for serial correlation in each asset's specific-return series and then
Bayes-shrunk toward its size-decile mean. For an `Σ` destined for mean-variance
optimization, add the Eigenfactor Risk Adjustment: the optimizer systematically
loads into exactly the eigenportfolios whose risk the sample covariance
understates.
