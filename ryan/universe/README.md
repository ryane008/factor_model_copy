# Barra USE4 risk model — full universe and Russell 1000

The same model as `ryan/sample`, built on a real universe instead of a 220-name
cohort. **Not yet run against WRDS.** Everything below that says "measured" was
measured on the sample or on the synthetic smoke test, not on this universe.

## How it relates to `ryan/sample`

One codebase. All model logic stays in `ryan/sample/*.py`; this folder adds only
what is universe-specific:

| File | Role |
|---|---|
| `config.py` | **Shadows** `sample/config.py`: loads it, then overrides the universe, history, table prefix and industry level. |
| `membership.py` | Point-in-time membership: screens, Russell-style reconstitution with banding, constituents-file override, industry gate. |
| `wrds_bulk.py` | Date-range pulls one year per query, and batched ticker lookups. Connection, CCM links and GICS reuse `sample/wrds_io.py`. |
| `01_build_universe.ipynb` | The counterpart of `run_pipeline`: pull → identifiers → membership → exposures. |
| `02`–`08` | The sample's notebooks, derived with the sample-specific findings replaced by checks. |
| `test_universe.py` | 30 offline checks on config shadowing and membership. |
| `smoke_synthetic.py` | End-to-end run on a synthetic JKP-shaped universe, from membership through the tools. No WRDS or DB; takes seconds. |

Notebooks here put this folder first on `sys.path`, so `import config` resolves
to this file **inside every shared module too**. They assert
`config.PROFILE != "sample"` on their first line, so a misconfigured path fails
at once instead of writing into the sample's tables.

A fix in `sample/` fixes every universe. A change to a *model* setting (half-lives,
descriptors, eigen sims) belongs in `sample/config.py`, and a *universe* setting
belongs here.

## Profiles

Selected by the `BARRA_UNIVERSE` environment variable (default `r1000`). Tables
are prefixed `ryan_full_*` / `ryan_r1000_*`, so the two builds and the sample never
collide.

| | `full` | `r1000` |
|---|---|---|
| estimation universe (the regression, Δ) | every JKP US common stock passing screens | top 3000 by cap: Russell 3000 proxy |
| coverage universe (what the tools list and benchmark to) | same | top 1000: Russell 1000 proxy, `in_r1000` |
| typical N per month | ~3,500–5,000 | ~3,000 |

To run the full universe: `BARRA_UNIVERSE=full` in the container environment
(e.g. `docker exec -e BARRA_UNIVERSE=full …`), or change `DEFAULT_PROFILE`.

## Decisions, and why

**Estimate the R1000 on an R3000 proxy.** If the regression only saw R1000
members, a name joining the index would arrive with no specific-return history
and sit on its size-bucket prior for two years, which is roughly a fifth of the
index at any time. Estimating on the top 3000 gives entrants a history and adds
~2,000 names of cross-sectional information. Cap-weighted standardization barely
moves, because the R1000 is ~93% of R3000 cap. This is Barra's split between the
estimation universe and the coverage universe.

**The Russell membership is a proxy**, unless `CONSTITUENTS_CSV` points at real
constituents (`permno,start,end`). It reproduces: May rank date, reconstitution
effective after June, top-N by total market cap, and the ±2.5% cumulative-cap
band at the 1000 boundary (incumbents in the band stay, newcomers in it wait).
It does not reproduce: float adjustment, the $1 / 5%-float screens, quarterly IPO
additions, or Russell's share-class rules. Expect a few percent of names to differ
at the boundary. The names that carry the cap are the same. Membership at *t*
uses only caps known on the rank date; `test_universe.py` checks that a cap
surge after the rank date changes nothing until the next June.

**GICS sectors, not industry groups, by default.** Your decision #8 applies more
here, not less: more assets cannot fix T. Groups make K = 32, which puts T/K at
1.9 on a 60-month window against the sample's 3.3, where the smallest eigenvalues
were already understated ~55% before adjustment. Groups are the obvious next
experiment: set `INDUSTRY_CODE_COL = "group_code"` and `COV_WINDOW = 120`,
re-run 01–05, and let 05's bias tests decide.

**History starts 2000-12**, not 2005-12. T is the binding constraint, and five more
years is 60 more forecast dates. GICS dates from 1999; before that its codes are
back-filled.

**No survival screen.** `MIN_RETURN_MONTHS = 60` biased the sample's specific risk
low. Here thin histories take the size-bucket prior instead (04 plots the share).

**Nano caps excluded** (JKP `size_grp`). They are untradeable, stale-priced and
dominated by bid-ask bounce. √cap weighting would mute them in the regression,
but they would still distort the size-decile priors in Δ.

**Still current GICS, not point-in-time.** `comp.co_hgic` would remove the
look-ahead, but GICS has been restructured under us: Real Estate (60) was carved
out of Financials in 2016, and Communication Services (50) was rebuilt in 2018.
Point-in-time, those factors are **empty** before their creation dates, and a
rolling F needs every factor observed in every month of its window. Doing it
properly means USE4-style nascent-industry handling in F. Until then the
industry gate in 01 would stop the build rather than produce NaN factor returns.

**Newey-West stays off, as in the sample.** More assets don't add months, so
the reason it hurt is unchanged. 03 and 04 re-run the comparison anyway.

## Changes made to the shared code for scale

All backward-compatible. The sample re-prices identically (17.10% vol, 15.84%
min-var) and `sample/test_logic.py` passes 137 checks.

- **Optimizer**: SLSQP forms n×n matrices and cannot run at 3,000 names.
  Above 250 names it switches to accelerated projected gradient (FISTA with
  restart). Each iteration is one factor-space gradient plus an exact projection
  onto the capped simplex. Measured ~2s at N=4,000; tests check it matches SLSQP
  and the closed form.
- **`ModelStore`**: the as-of ticker table is cached and the ticker-history scan
  happens once. At ~1M exposure rows, each lookup had been rescanning the table.
- **Coverage flag**: `COVERAGE_COL` in config, carried as `meta["in_coverage"]`.
  `market_weights(coverage_only=True)` is the R1000 benchmark,
  `query.universe()` lists covered names only, and `load_portfolio` flags holdings
  that are priced but outside coverage.

## Running it

```bash
docker exec factor_jupyter python /home/jovyan/work/ryan/universe/test_universe.py    # offline
docker exec factor_jupyter python /home/jovyan/work/ryan/universe/smoke_synthetic.py  # offline
```

Then `01 → 02 → 03 → 04 → 05`, then `06`–`08` in any order. **01 hits WRDS** on
its first run (one query per year of `contrib.global_factor`, plus CCM links,
names and GICS). Afterwards it resumes from the database unless
`RELOAD_PANEL = True`.

Runtime and memory at full scale are **not yet measured**. Expected hotspots:
the WRDS pull; writing ~1M-row tables with `to_sql` (minutes); and 05's long-only
test (~2s per solve × 180 dates × 2 stages, ~12 min, switch off with
`RUN_LONG_ONLY_TEST`). Everything else is vectorized or O(NK) per month.

## First-run checklist

1. **01**: return coverage > 90%; members per month near 3000 / 1000 (r1000);
   R1000 annual turnover in the real index's ~5–10% range; the industry gate
   passes; value ↔ momentum and value ↔ profitability still negative (sample
   −0.33 / −0.33).
2. **02**: market factor vs cap-weighted return ~0.99+; style vols in 2–8%/yr
   (17–23% means the lag broke); median R² will be lower than the sample's 0.27
   with more small names, which is expected.
3. **03**: the NW table still favours zero lags; eigen still does the heavy lifting.
4. **05**: min-variance bias raw vs final. That number says whether 07 can be
   trusted at this scale.
5. Record 03's factor means and vols in `config.ANCHORS`, so later rebuilds are
   checked against them.

## Next steps, in order of value

1. **Daily factor returns.** USE4 estimates F on daily data, and that is what
   makes T/K comfortable, allows industry groups, and would let Newey-West help.
   This needs CRSP daily returns (`crsp.dsf`) and a daily version of step 4. It is
   the largest remaining gap between this and USE4.
2. **Point-in-time GICS with nascent-factor handling in F.**
3. **Industry groups with a 120-month window**, decided by 05's bias tests.
4. **A structural model for thin-history specific risk** (USE4 regresses specific
   vol on exposures) instead of the size-bucket prior.
