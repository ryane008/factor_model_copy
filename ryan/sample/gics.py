"""
GICS code -> name mappings, hardcoded.

This replaces the Kaggle ``gics-map-2023.csv`` join in charactersitics_loader.ipynb,
which had two independent defects:

1. Coverage. ~6.6% of rows carrying valid WRDS GICS codes failed the four-key
   merge, because WRDS ``comp.company`` uses the current (post-March-2023) GICS
   structure and that file predates the 2023 revision for at least some codes.
   CECO Environmental (gsubind 20106020, "Industrial Machinery & Supplies &
   Components") came back all-NaN.

2. Correctness. Labels were wrong even where the merge *succeeded*: American
   Airlines (gsector 20, Industrials) came back as Sector "Materials".

The fix is not a better mapping file — it is to stop needing one. There are only
11 sectors and 25 industry groups, so they live here as literals.

The model keys on the numeric **code**, never the name. Names are display labels
only, so a name drifting between GICS vintages cannot affect any number.
"""

from __future__ import annotations

import pandas as pd

# 2-digit GICS sectors. Stable across every GICS revision to date.
SECTORS: dict[str, str] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# 4-digit GICS industry groups, current (post-March-2023) names. Several were
# renamed in that revision -- 4020 was "Diversified Financials", 2550 was
# "Retailing", 3010 was "Food & Staples Retailing", and sector 60 was split into
# 6010/6020. Only the labels changed; the codes below are what WRDS carries.
INDUSTRY_GROUPS: dict[str, str] = {
    "1010": "Energy",
    "1510": "Materials",
    "2010": "Capital Goods",
    "2020": "Commercial & Professional Services",
    "2030": "Transportation",
    "2510": "Automobiles & Components",
    "2520": "Consumer Durables & Apparel",
    "2530": "Consumer Services",
    "2550": "Consumer Discretionary Distribution & Retail",
    "3010": "Consumer Staples Distribution & Retail",
    "3020": "Food, Beverage & Tobacco",
    "3030": "Household & Personal Products",
    "3510": "Health Care Equipment & Services",
    "3520": "Pharmaceuticals, Biotechnology & Life Sciences",
    "4010": "Banks",
    "4020": "Financial Services",
    "4030": "Insurance",
    "4510": "Software & Services",
    "4520": "Technology Hardware & Equipment",
    "4530": "Semiconductors & Semiconductor Equipment",
    "5010": "Telecommunication Services",
    "5020": "Media & Entertainment",
    "5510": "Utilities",
    "6010": "Equity Real Estate Investment Trusts (REITs)",
    "6020": "Real Estate Management & Development",
}


def normalize_code(series: pd.Series) -> pd.Series:
    """
    Coerce a GICS code column to a clean string.

    WRDS returns these as floats or strings depending on the driver, so 10.0 and
    "10" both have to land on "10" or the mapping silently misses.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def attach_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``sector_code``/``sector_name`` and ``group_code``/``group_name``.

    Expects the raw WRDS ``gsector`` and ``ggroup`` columns. Unlike the four-key
    merge this replaced, an unrecognized code yields a null *name* while the
    *code* survives -- so the industry dummies stay intact even if a future GICS
    revision adds a group this file has not caught up with.
    """
    out = df.copy()

    out["sector_code"] = normalize_code(out["gsector"])
    out["sector_name"] = out["sector_code"].map(SECTORS).astype("string")

    out["group_code"] = normalize_code(out["ggroup"])
    out["group_name"] = out["group_code"].map(INDUSTRY_GROUPS).astype("string")

    unmapped = out.loc[out["sector_code"].notna() & out["sector_name"].isna(), "sector_code"]
    if len(unmapped):
        print(f"⚠️  {len(unmapped)} rows have an unrecognized gsector: {sorted(unmapped.unique())}")

    unmapped_grp = out.loc[out["group_code"].notna() & out["group_name"].isna(), "group_code"]
    if len(unmapped_grp):
        print(f"⚠️  {len(unmapped_grp)} rows have an unrecognized ggroup: {sorted(unmapped_grp.unique())}")

    return out
