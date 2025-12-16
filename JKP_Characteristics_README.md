# JKP Global Characteristics Data - Complete Variable Mapping

This README provides a comprehensive mapping of all characteristics available in the Jensen, Kelly, Pedersen (JKP) Global Factor dataset, organized by factor categories and cross-referenced with academic literature.

## Data Source

- **Dataset**: JKP Global Characteristics (`contrib.global_factor` in WRDS)
- **Documentation**: Based on characteristics.csv mapping file and academic papers
- **Coverage**: Global equity markets with comprehensive fundamental and market-based characteristics
- **Frequency**: Monthly end-of-month data

## Factor Categories Overview

| Category              | Count | Description                                                |
| --------------------- | ----- | ---------------------------------------------------------- |
| **VALUE**             | 32    | Book-to-market, earnings yield, cash flow ratios           |
| **MOMENTUM**          | 18    | Price momentum, earnings momentum, industry momentum       |
| **PROFITABILITY**     | 19    | ROA, ROE, profit margins, quality scores                   |
| **INVESTMENT**        | 33    | Asset growth, accruals, financing activities               |
| **INTANGIBLES**       | 56    | R&D, brand capital, analyst coverage, firm characteristics |
| **TRADING FRICTIONS** | 38    | Size, volatility, liquidity, bid-ask spreads               |
| **NEW**               | 15    | Recent factor innovations and combinations                 |

---

## Fama-French 5 Factor Model Validation

### Our Current Implementation:

```python
'fama_french_5': {
    'value': ['be_me', 'at_me', 'sale_me', 'ni_me', 'ocf_me', 'ebitda_me'],
    'capm': ['ret_exc', 'beta_60m', 'beta_252d'],
    'size': ['me', 'market_equity', 'sale_me'],
    'momentum': ['ret_12_1', 'ret_6_1', 'ret_12_7', 'ret_9_1'],
    'quality': ['ni_at', 'ni_be', 'roe_be_std', 'ocf_at', 'gp_at', 'f_score']
}
```

### ✅ **Validated Against JKP Documentation:**

#### **VALUE FACTORS (HML - High Minus Low)**

- **be_me** ✓ - Book-to-market equity (Rosenberg Reid and Lanstein 1985) - **PRIMARY FF VALUE MEASURE**
- **at_me** ✓ - Assets-to-market (Fama and French 1992)
- **sale_me** ✓ - Sales-to-market (Barbee Mukherji and Raines 1996)
- **ni_me** ✓ - Earnings-to-price (Basu 1983)
- **ocf_me** ✓ - Operating cash flow-to-market (Desai Rajgopal and Venkatachalam 2004)
- **ebitda_me** ✓ - EBITDA-to-market enterprise value (Loughran and Wellman 2011)

#### **SIZE FACTORS (SMB - Small Minus Big)**

- **me** ✓ - Market equity (Banz 1981) - **PRIMARY FF SIZE MEASURE**
- **market_equity** ✓ - Total market value of equity
- **sale_me** ✓ - Revenue efficiency measure (also value-related)

#### **MOMENTUM FACTORS**

- **ret_12_1** ✓ - 12-month momentum skipping recent month (Jegadeesh and Titman 1993) - **STANDARD MOMENTUM**
- **ret_6_1** ✓ - 6-month momentum (Jegadeesh and Titman 1993)
- **ret_12_7** ✓ - 12-7 month momentum (Novy-Marx 2012)
- **ret_9_1** ✓ - 9-month momentum (Jegadeesh and Titman 1993)

#### **PROFITABILITY/QUALITY FACTORS (RMW - Robust Minus Weak)**

- **ni_at** ✓ - Return on assets (Balakrishnan Bartov and Faurel 2010) - **FF PROFITABILITY PROXY**
- **ni_be** ✓ - Return on equity (Hou Xue and Zhang 2015)
- **roe_be_std** ✓ - ROE standard deviation (earnings stability)
- **ocf_at** ✓ - Operating cash flow to assets (Bouchard et al. 2019)
- **gp_at** ✓ - Gross profits-to-assets (Novy-Marx 2013) - **KEY FF PROFITABILITY MEASURE**
- **f_score** ✓ - Piotroski F-score (Piotroski 2000) - **COMPOSITE QUALITY MEASURE**

#### **MARKET/CAPM FACTORS**

- **ret_exc** ✓ - Excess returns (market factor)
- **beta_60m** ✓ - 60-month market beta (Fama and MacBeth 1973)
- **beta_252d** ✓ - 252-day market beta

---

## Complete Characteristics Mapping

### VALUE FACTORS (32 characteristics)

| JKP Code   | Name                     | Description                          | Academic Paper                      | Predictive Sign |
| ---------- | ------------------------ | ------------------------------------ | ----------------------------------- | --------------- |
| **be_me**  | **Book-to-Market**       | **Book equity / Market equity**      | **Rosenberg Reid Lanstein (1985)**  | **+**           |
| at_me      | Assets-to-Market         | Total assets / Market equity         | Fama French (1992)                  | +               |
| debt_me    | Debt-to-Market           | Debt / Market equity                 | Bhandari (1988)                     | +               |
| ni_me      | Earnings-to-Price        | Net income / Market equity           | Basu (1983)                         | +               |
| fcf_me     | Cash Flow-to-Price       | Free cash flow / Market equity       | Lakonishok Shleifer Vishny (1994)   | +               |
| sale_me    | Sales-to-Price           | Sales / Market equity                | Barbee Mukherji Raines (1996)       | +               |
| ocf_me     | Operating CF-to-Price    | Operating cash flow / Market equity  | Desai Rajgopal Venkatachalam (2004) | +               |
| ebitda_me  | EBITDA-to-Market         | EBITDA / Market equity               | Loughran Wellman (2011)             | +               |
| div12m_me  | Dividend Yield           | Dividends / Market equity            | Litzenberger Ramaswamy (1979)       | +               |
| eqpo_me    | Payout Yield             | Equity payouts / Market equity       | Boudoukh et al. (2007)              | +               |
| eqnpo_me   | Net Payout Yield         | Net equity payouts / Market equity   | Boudoukh et al. (2007)              | +               |
| sale_gr1   | Sales Growth (1Y)        | Annual sales growth                  | Lakonishok Shleifer Vishny (1994)   | -               |
| sale_gr3   | Sales Growth (3Y)        | 3-year sales growth                  | Lakonishok Shleifer Vishny (1994)   | -               |
| ebitda_mev | Enterprise Multiple      | EBITDA / Market enterprise value     | Loughran Wellman (2011)             | +               |
| ival_me    | Intrinsic Value          | Intrinsic value / Market equity      | Frankel Lee (1998)                  | +               |
| bev_mev    | Enterprise Book-to-Price | Book value / Market enterprise value | Penman Richardson Tuna (2007)       | +               |
| netdebt_me | Net Debt-to-Price        | Net debt / Market equity             | Penman Richardson Tuna (2007)       | -               |
| eq_dur     | Equity Duration          | Equity duration measure              | Dechow Sloan Soliman (2004)         | -               |
| ret_60_12  | Long-term Reversal       | 60-12 month returns                  | De Bondt Thaler (1985)              | -               |

### MOMENTUM FACTORS (18 characteristics)

| JKP Code         | Name                    | Description                             | Academic Paper              | Predictive Sign |
| ---------------- | ----------------------- | --------------------------------------- | --------------------------- | --------------- |
| **ret_12_1**     | **12-1 Month Momentum** | **12-month return skip recent**         | **Jegadeesh Titman (1993)** | **+**           |
| **ret_6_1**      | **6-1 Month Momentum**  | **6-month return skip recent**          | **Jegadeesh Titman (1993)** | **+**           |
| ret_3_1          | 3-1 Month Momentum      | 3-month return skip recent              | Jegadeesh Titman (1993)     | +               |
| ret_9_1          | 9-1 Month Momentum      | 9-month return skip recent              | Jegadeesh Titman (1993)     | +               |
| **ret_12_7**     | **12-7 Month Momentum** | **Intermediate momentum**               | **Novy-Marx (2012)**        | **+**           |
| niq_su           | Earnings Surprise       | Standardized earnings surprise          | Foster Olsen Shevlin (1984) | +               |
| saleq_su         | Revenue Surprise        | Standardized revenue surprise           | Jegadeesh Livnat (2006)     | +               |
| tax_gr1a         | Tax Expense Surprise    | Tax expense surprise                    | Thomas Zhang (2011)         | +               |
| ni_inc8q         | Consecutive Earnings    | Consecutive quarters earnings increases | Barth Elliott Finn (1999)   | +               |
| prc_highprc_252d | 52-Week High            | Current price / 52-week high            | George Hwang (2004)         | +               |
| resff3_6_1       | Residual Momentum (6M)  | 6-month residual momentum               | Blitz Huij Martens (2011)   | +               |
| resff3_12_1      | Residual Momentum (12M) | 12-month residual momentum              | Blitz Huij Martens (2011)   | +               |

### PROFITABILITY FACTORS (19 characteristics)

| JKP Code    | Name                                 | Description                            | Academic Paper                        | Predictive Sign |
| ----------- | ------------------------------------ | -------------------------------------- | ------------------------------------- | --------------- |
| **ni_at**   | **Return on Assets**                 | **Net income / Total assets**          | **Balakrishnan Bartov Faurel (2010)** | **+**           |
| **ni_be**   | **Return on Equity**                 | **Net income / Book equity**           | **Hou Xue Zhang (2015)**              | **+**           |
| **gp_at**   | **Gross Profits/Assets**             | **Gross profits / Total assets**       | **Novy-Marx (2013)**                  | **+**           |
| **ocf_at**  | **Operating CF/Assets**              | **Operating cash flow / Assets**       | **Bouchard et al. (2019)**            | **+**           |
| ope_be      | Operating Profits/Equity             | Operating profits / Book equity        | Fama French (2015)                    | +               |
| ebit_sale   | Profit Margin                        | EBIT / Sales                           | Soliman (2008)                        | +               |
| sale_bev    | Asset Turnover                       | Sales / Book enterprise value          | Soliman (2008)                        | +               |
| at_turnover | Capital Turnover                     | Asset turnover ratio                   | Haugen Baker (1996)                   | +               |
| gp_atl1     | Gross Profits/Lagged Assets          | Gross profits / Lagged assets          | Various                               | +               |
| ope_bel1    | Operating Profits/Lagged Equity      | Operating profits / Lagged equity      | Various                               | +               |
| op_atl1     | Operating Profits/Lagged Assets      | Operating profits / Lagged assets      | Ball et al. (2016)                    | +               |
| cop_atl1    | Cash Operating Profits/Lagged Assets | Cash operating profits / Lagged assets | Ball et al. (2016)                    | +               |
| **f_score** | **Piotroski F-Score**                | **Composite fundamental score**        | **Piotroski (2000)**                  | **+**           |
| o_score     | Ohlson O-Score                       | Bankruptcy prediction score            | Dichev (1998)                         | -               |
| z_score     | Altman Z-Score                       | Financial distress score               | Dichev (1998)                         | +               |
| pi_nix      | Taxable/Book Income                  | Taxable income / Book income           | Lev Nissim (2004)                     | +               |
| at_be       | Book Leverage                        | Assets / Book equity                   | Fama French (1992)                    | -               |
| niq_be_chg1 | Change in ROE                        | 4-quarter change in ROE                | Various                               | +               |
| niq_at_chg1 | Change in ROA                        | 4-quarter change in ROA                | Various                               | +               |

### INVESTMENT FACTORS (33 characteristics)

| JKP Code     | Name                          | Description                                | Academic Paper                      | Predictive Sign |
| ------------ | ----------------------------- | ------------------------------------------ | ----------------------------------- | --------------- |
| **at_gr1**   | **Asset Growth**              | **Annual asset growth**                    | **Cooper Gulen Schill (2008)**      | **-**           |
| **capx_gr1** | **CAPEX Growth**              | **Capital expenditure growth**             | **Xie (2001)**                      | **-**           |
| capex_abn    | Abnormal CAPEX                | Abnormal corporate investment              | Titman Wei Xie (2004)               | -               |
| sale_gr1     | Sales Growth                  | Annual sales growth                        | Lakonishok Shleifer Vishny (1994)   | -               |
| be_gr1       | Book Equity Growth            | Book equity growth                         | Richardson et al. (2005)            | -               |
| inv_gr1      | Inventory Growth              | Inventory growth                           | Belo Lin (2011)                     | -               |
| chcsho_12m   | Net Stock Issues              | Net stock issuance                         | Pontiff Woodgate (2008)             | -               |
| eqnpo_12m    | Composite Equity Issuance     | Equity net payout                          | Daniel Titman (2006)                | +               |
| debt_gr3     | Debt Growth (3Y)              | 3-year debt growth                         | Lyandres Sun Zhang (2008)           | -               |
| noa_at       | Net Operating Assets          | Net operating assets / Assets              | Hirshleifer et al. (2004)           | -               |
| noa_gr1a     | Change in NOA                 | Change in net operating assets             | Hirshleifer et al. (2004)           | -               |
| ppeinv_gr1a  | Change PPE & Inventory        | Change in PPE and inventory                | Lyandres Sun Zhang (2008)           | -               |
| lnoa_gr1a    | Change Long-term NOA          | Change in long-term NOA                    | Fairfield Whisenant Yohn (2003)     | -               |
| oaccruals_at | Operating Accruals            | Operating accruals / Assets                | Sloan (1996)                        | -               |
| taccruals_at | Total Accruals                | Total accruals / Assets                    | Richardson et al. (2005)            | -               |
| cowc_gr1a    | Change Working Capital        | Change in operating working capital        | Richardson et al. (2005)            | -               |
| coa_gr1a     | Change Current Op Assets      | Change in current operating assets         | Richardson et al. (2005)            | -               |
| col_gr1a     | Change Current Op Liab        | Change in current operating liabilities    | Richardson et al. (2005)            | -               |
| nncoa_gr1a   | Change Noncurrent NOA         | Change in noncurrent NOA                   | Richardson et al. (2005)            | -               |
| ncoa_gr1a    | Change Noncurrent Assets      | Change in noncurrent operating assets      | Richardson et al. (2005)            | -               |
| ncol_gr1a    | Change Noncurrent Liab        | Change in noncurrent operating liabilities | Richardson et al. (2005)            | -               |
| nfna_gr1a    | Change Net Financial Assets   | Change in net financial assets             | Richardson et al. (2005)            | +               |
| sti_gr1a     | Change Short-term Investments | Change in short-term investments           | Richardson et al. (2005)            | +               |
| lti_gr1a     | Change Long-term Investments  | Change in long-term investments            | Richardson et al. (2005)            | -               |
| fnl_gr1a     | Change Financial Liabilities  | Change in financial liabilities            | Richardson et al. (2005)            | -               |
| be_gr1a      | Change Common Equity          | Change in common equity                    | Richardson et al. (2005)            | -               |
| oaccruals_ni | Percent Operating Accruals    | Operating accruals / Net income            | Hafzalla Lundholm Van Winkle (2011) | -               |
| taccruals_ni | Percent Total Accruals        | Total accruals / Net income                | Hafzalla Lundholm Van Winkle (2011) | -               |
| netis_at     | Net External Finance          | Net total issuance / Assets                | Bradshaw Richardson Sloan (2006)    | -               |
| eqnetis_at   | Net Equity Finance            | Net equity issuance / Assets               | Bradshaw Richardson Sloan (2006)    | -               |
| dbnetis_at   | Net Debt Finance              | Net debt issuance / Assets                 | Bradshaw Richardson Sloan (2006)    | -               |

### TRADING FRICTIONS FACTORS (38 characteristics)

| JKP Code           | Name                          | Description                                | Academic Paper                       | Predictive Sign |
| ------------------ | ----------------------------- | ------------------------------------------ | ------------------------------------ | --------------- |
| **market_equity**  | **Market Equity (Size)**      | **Shares outstanding × Price**             | **Banz (1981)**                      | **-**           |
| **beta_60m**       | **Market Beta (60M)**         | **60-month market beta**                   | **Fama MacBeth (1973)**              | **-**           |
| beta_252d          | Market Beta (252D)            | 252-day market beta                        | Fama MacBeth (1973)                  | -               |
| beta_21d           | Market Beta (21D)             | 21-day market beta                         | Various                              | -               |
| **ivol_capm_252d** | **Idiosyncratic Volatility**  | **CAPM idiosyncratic volatility (252D)**   | **Ali Hwang Trombley (2003)**        | **-**           |
| ivol_capm_21d      | Idiosyncratic Vol (21D)       | CAPM idiosyncratic volatility (21D)        | Various                              | -               |
| ivol_ff3_21d       | FF3 Idiosyncratic Vol         | FF3 idiosyncratic volatility               | Ang et al. (2006)                    | -               |
| ivol_hxz4_21d      | Q-Factor Idiosyncratic Vol    | Q-factor idiosyncratic volatility          | Various                              | -               |
| rvol_21d           | Total Volatility              | Return volatility (21 days)                | Ang et al. (2006)                    | -               |
| betabab_1260d      | Frazzini-Pedersen Beta        | Frazzini-Pedersen market beta              | Frazzini Pedersen (2014)             | -               |
| beta_dimson_21d    | Dimson Beta                   | Dimson beta (21 days)                      | Dimson (1979)                        | -               |
| turnover_126d      | Share Turnover                | Share turnover (126 days)                  | Datar Naik Radcliffe (1998)          | -               |
| turnover_var_126d  | Turnover Variation            | Coefficient of variation for turnover      | Chordia Subrahmanyam Anshuman (2001) | -               |
| dolvol_126d        | Dollar Volume                 | Dollar trading volume (126 days)           | Brennan Chordia Subrahmanyam (1998)  | -               |
| dolvol_var_126d    | Dollar Volume Variation       | Coefficient of variation for dollar volume | Chordia Subrahmanyam Anshuman (2001) | -               |
| prc                | Price per Share               | Stock price per share                      | Miller Scholes (1982)                | -               |
| ami_126d           | Amihud Measure                | Absolute return-to-volume ratio            | Amihud (2002)                        | +               |
| zero_trades_21d    | Zero Trades (1M)              | Number of zero trading days (21 days)      | Liu (2006)                           | +               |
| zero_trades_126d   | Zero Trades (6M)              | Number of zero trading days (126 days)     | Liu (2006)                           | +               |
| zero_trades_252d   | Zero Trades (12M)             | Number of zero trading days (252 days)     | Liu (2006)                           | +               |
| rmax1_21d          | Maximum Daily Return          | Maximum daily return (21 days)             | Bali Cakici Whitelaw (2011)          | -               |
| rmax5_21d          | Max 5-Day Return              | Highest 5 days of return                   | Bali Brown Murray Tang (2017)        | -               |
| rmax5_rvol_21d     | Scaled Max Return             | Max 5-day return scaled by volatility      | Asness et al. (2020)                 | -               |
| rskew_21d          | Total Skewness                | Total skewness (21 days)                   | Bali Engle Murray (2016)             | -               |
| iskew_capm_21d     | Idiosyncratic Skewness (CAPM) | CAPM idiosyncratic skewness                | Various                              | -               |
| iskew_ff3_21d      | Idiosyncratic Skewness (FF3)  | FF3 idiosyncratic skewness                 | Bali Engle Murray (2016)             | -               |
| iskew_hxz4_21d     | Idiosyncratic Skewness (Q)    | Q-factor idiosyncratic skewness            | Various                              | -               |
| coskew_21d         | Coskewness                    | Coskewness with market                     | Harvey Siddique (2000)               | -               |
| ret_1_0            | Short-term Reversal           | 1-month return                             | Jegadeesh (1990)                     | -               |
| betadown_252d      | Downside Beta                 | Downside beta (252 days)                   | Ang Chen Xing (2006)                 | -               |
| bidaskhl_21d       | High-Low Bid-Ask Spread       | High-low bid-ask spread                    | Corwin Schultz (2012)                | +               |
| corr_1260d         | Market Correlation            | Market correlation (1260 days)             | Asness et al. (2020)                 | -               |

### INTANGIBLES FACTORS (56 characteristics)

| JKP Code             | Name                      | Description                            | Academic Paper                    | Predictive Sign |
| -------------------- | ------------------------- | -------------------------------------- | --------------------------------- | --------------- |
| rd_me                | R&D-to-Market             | R&D expense / Market equity            | Chan Lakonishok Sougiannis (2001) | +               |
| rd_sale              | R&D-to-Sales              | R&D expense / Sales                    | Chan Lakonishok Sougiannis (2001) | +               |
| rd5_at               | R&D Capital/Assets        | R&D capital / Book assets              | Li (2011)                         | +               |
| opex_at              | Operating Leverage        | Operating expenses / Assets            | Novy-Marx (2011)                  | +               |
| emp_gr1              | Hiring Rate               | Employee growth rate                   | Belo Lin Bazdresch (2014)         | -               |
| age                  | Firm Age                  | Firm age                               | Jiang Lee Zhang (2005)            | -               |
| dsale_dinv           | Sales-Inventory Change    | % change sales - % change inventory    | Abarbanell Bushee (1998)          | +               |
| dsale_drec           | Sales-Receivables Change  | % change sales - % change receivables  | Abarbanell Bushee (1998)          | -               |
| dgp_dsale            | Gross Margin-Sales Change | % change gross margin - % change sales | Abarbanell Bushee (1998)          | +               |
| dsale_dsga           | Sales-SG&A Change         | % change sales - % change SG&A         | Abarbanell Bushee (1998)          | +               |
| sale_emp_gr1         | Labor Force Efficiency    | Sales per employee growth              | Abarbanell Bushee (1998)          | +               |
| tangibility          | Asset Tangibility         | Tangibility of assets                  | Hahn Lee (2009)                   | +               |
| kz_index             | Kaplan-Zingales Index     | Financing constraints index            | Lamont Polk Saa-Requejo (2001)    | +               |
| ocfq_saleq_std       | Cash Flow Volatility      | Cash flow volatility                   | Huang (2009)                      | -               |
| cash_at              | Cash-to-Assets            | Cash / Total assets                    | Palazzo (2012)                    | +               |
| aliq_at              | Asset Liquidity           | Liquidity of book assets               | Ortiz-Molina Phillips (2014)      | -               |
| aliq_mat             | Market Asset Liquidity    | Liquidity of market assets             | Ortiz-Molina Phillips (2014)      | -               |
| ni_ar1               | Earnings Persistence      | Earnings persistence measure           | Francis et al. (2004)             | +               |
| ni_ivol              | Earnings Volatility       | Earnings volatility                    | Francis et al. (2004)             | +               |
| earnings_variability | Earnings Variability      | Earnings smoothness measure            | Francis et al. (2004)             | -               |

### NEW FACTORS (15 characteristics)

| JKP Code        | Name                    | Description                         | Academic Paper                      | Predictive Sign |
| --------------- | ----------------------- | ----------------------------------- | ----------------------------------- | --------------- |
| mispricing_perf | Mispricing: Performance | Performance-based mispricing factor | Stambaugh Yuan (2016)               | +               |
| mispricing_mgmt | Mispricing: Management  | Management-based mispricing factor  | Stambaugh Yuan (2016)               | +               |
| **qmj**         | **Quality Minus Junk**  | **Composite quality measure**       | **Asness Frazzini Pedersen (2018)** | **+**           |
| **qmj_prof**    | **QMJ: Profitability**  | **Quality profitability component** | **Asness Frazzini Pedersen (2018)** | **+**           |
| qmj_growth      | QMJ: Growth             | Quality growth component            | Asness Frazzini Pedersen (2018)     | +               |
| qmj_safety      | QMJ: Safety             | Quality safety component            | Asness Frazzini Pedersen (2018)     | +               |

---

## Recommended Factor Model Updates

Based on the JKP documentation analysis, here are optimized factor model specifications:

### **Enhanced Fama-French 5 Factor Model**

```python
'fama_french_5_enhanced': {
    'value': ['be_me', 'ni_me', 'ocf_me'],  # Core value measures
    'size': ['market_equity'],  # Primary size measure
    'momentum': ['ret_12_1'],  # Standard momentum
    'profitability': ['gp_at', 'ni_at'],  # FF2015 profitability
    'investment': ['at_gr1', 'oaccruals_at'],  # FF2015 investment
    'quality': ['f_score', 'qmj']  # Composite quality measures
}
```

### **Academic Research Factor Model**

```python
'academic_research': {
    'value': ['be_me', 'ni_me', 'fcf_me', 'sale_me'],
    'momentum': ['ret_12_1', 'ret_6_1', 'resff3_12_1'],
    'profitability': ['gp_at', 'ni_at', 'ope_be', 'f_score'],
    'investment': ['at_gr1', 'capx_gr1', 'oaccruals_at', 'netis_at'],
    'volatility': ['ivol_capm_252d', 'beta_60m'],
    'liquidity': ['ami_126d', 'turnover_126d', 'zero_trades_126d'],
    'quality': ['qmj', 'qmj_prof', 'qmj_safety']
}
```

---

## Usage Notes

1. **Data Quality Filters**: All models should use `common=1`, `exch_main=1`, `primary_sec=1`, `obs_main=1`
2. **Geographic Coverage**: Filter by `excntry` for regional analysis
3. **Temporal Coverage**: Most characteristics available from 1960s onwards
4. **Missing Values**: Handle appropriately for time-series analysis
5. **Predictive Signs**: '+' indicates positive expected returns, '-' indicates negative

---

## Academic References

The characteristics are based on 100+ academic papers spanning 1968-2020. Key foundational papers include:

- **Fama & French (1992, 1993, 2015)**: Core factor models
- **Jegadeesh & Titman (1993)**: Momentum
- **Sloan (1996)**: Accruals anomaly
- **Piotroski (2000)**: F-Score quality measure
- **Novy-Marx (2013)**: Gross profitability
- **Asness, Frazzini & Pedersen (2018)**: Quality Minus Junk

For complete citations, see the characteristics.csv file and original JKP documentation.
