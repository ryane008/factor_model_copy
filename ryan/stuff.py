#!/usr/bin/env python
# coding: utf-8

# In[59]:


import math
from statistics import NormalDist
import numpy as np
import pandas as pd

# Systematic equity book: 5000 stocks. Each position-round is a hit or a miss: a winner earns +payoff
# and a loser earns -payoff, whether or not we traded the name that round.
# Returns depend only on the hit rate; turnover only matters through trading costs.
n_stocks = 5000
rho = 0.1                       # correlation between stocks' outcomes (shared market/sector/style shocks); 0 = independent
stock_annual_vol = 0.2           # vol that mirrors the S&P
cost_per_bet = 0.0005            # 5bp impact/transaction cost per $1 traded
turnover = 0.40                  # share of the book traded each round (1.0 = whole book)
risk_free_rate = 0.03            # annualized; the book is self-funding, so it earns this on top

rounds_per_year = 35             # a round of bets every ~1.5 weeks (52 / 1.5 = 34.7)
n_years = 10
n_paths = 10_000                 # simulated 10-year histories

# Per-round numbers
payoff = stock_annual_vol / np.sqrt(rounds_per_year)                # +/- move of a winner / loser per round (~5%)
cost_per_round = cost_per_bet * turnover                            # cost drag per $1 of book per round
rf_per_round = (1 + risk_free_rate) ** (1 / rounds_per_year) - 1
breakeven_win_prob = 0.5 + cost_per_round / (2 * payoff)            # edge needed just to cover costs
print(f"{n_stocks} positions per round, {rounds_per_year} rounds/year | payoff = +/-{payoff:.4%} per round | "
      f"cost = {cost_per_round:.4%} per round | breakeven win rate = {breakeven_win_prob:.2%}\n")

rng = np.random.default_rng(42)

norm_cdf = np.vectorize(lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))))

def simulate(win_prob):
    # Stock i wins if sqrt(rho)*m + sqrt(1-rho)*e_i < Phi^-1(p), with m shared by all stocks in a round.
    # Given m, each stock wins independently with probability p_cond, so winners ~ Binomial(n, p_cond).
    m = rng.standard_normal((n_paths, n_years, rounds_per_year))
    p_cond = norm_cdf((NormalDist().inv_cdf(win_prob) - np.sqrt(rho) * m) / np.sqrt(1 - rho))
    wins = rng.binomial(n_stocks, p_cond)
    bet_ret = (2 * wins - n_stocks) * payoff / n_stocks - cost_per_round       # equal-weighted across stocks
    total_ret = (1 + rf_per_round) * (1 + bet_ret) - 1                       # book also earns rf
    annual = (1 + total_ret).prod(axis=2) - 1                                # (path, year) calendar-year returns
    cagr = (1 + annual).prod(axis=1) ** (1 / n_years) - 1                    # per 10-year path
    return dict(avg_annual_return=annual.mean(), annual_vol=annual.std(),
                sharpe=(annual.mean() - risk_free_rate) / annual.std(),
                pct_losing_years=(annual < 0).mean(), median_10y_cagr=np.median(cagr))

win_probs = [0.50, 0.505, 0.51, 0.52, 0.55, 0.6]
results = pd.DataFrame({p: simulate(p) for p in win_probs}).T
results.index = [f"{p:.1%}" for p in win_probs]
results.index.name = "share of positions that win"
pct = lambda col: results[col].map("{:.2%}".format)
stock_excess = [(2 * p - 1) * payoff * rounds_per_year for p in win_probs]   # one stock's expected excess return/yr
table = pd.DataFrame({"single-stock excess return": [f"{r:.2%}" for r in stock_excess],
                      "book avg annual return": pct("avg_annual_return"),
                      "annual vol": pct("annual_vol"),
                      "Sharpe": results["sharpe"].map("{:.2f}".format),
                      "losing years": results["pct_losing_years"].map("{:.1%}".format),
                      "median 10y annualized": pct("median_10y_cagr")})
display(table)

