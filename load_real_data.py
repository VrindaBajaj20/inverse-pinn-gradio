# load_real_data.py
"""
Fetch SPY (or SPX if accessible) option chain from yfinance
and produce arrays (S_grid, t_grid, V_grid) for PINN inference.
Note: yfinance returns end-of-day quotes, not full market depth.
"""

import yfinance as yf
import numpy as np
from datetime import datetime
import pandas as pd

def fetch_options(ticker='SPY', date=None):
    # date: expiration date string 'YYYY-MM-DD' or None for next expiry
    T = yf.Ticker(ticker)
    if date is None:
        # pick nearest expiry
        exps = T.options
        if len(exps) == 0:
            raise RuntimeError("No option expiries found")
        date = exps[0]
    opt = T.option_chain(date)
    calls = opt.calls
    return calls

def build_grid_from_yfinance(ticker='SPY', expiries=3, strikes_per_expiry=25):
    T = yf.Ticker(ticker)
    exps = list(T.options)[:expiries]
    S_now = T.history(period='1d')['Close'].values[-1]
    S_grid = np.linspace(max(1, S_now*0.5), S_now*1.8, 80)
    # create t_grid based on days to maturity normalized to year fraction
    t_list = []
    price_matrix = []
    for exp in exps:
        calls = T.option_chain(exp).calls
        # choose strikes near ATM
        mid = S_now
        strikes = calls['strike'].values
        # pick subset of strikes
        mask_idx = np.argsort(np.abs(strikes - mid))[:strikes_per_expiry]
        strikes_sel = strikes[mask_idx]
        # time to maturity in years
        tau = (pd.to_datetime(exp) - pd.Timestamp.today()).days / 252.0
        # collect prices by interpolating call prices across S_grid (use simple linear interp)
        strikes_prices = dict(zip(calls['strike'].values, calls['lastPrice'].values))
        #price_interp = np.interp(S_grid, strikes_sel, np.array([strikes_prices.get(k, np.nan) for k in strikes_sel]))
        strike_prices_array = np.array([strikes_prices.get(k, np.nan) for k in strikes_sel])
        strike_prices_array = np.nan_to_num(strike_prices_array, nan=0.0)  # replace NaN with 0
        price_interp = np.interp(S_grid, strikes_sel, strike_prices_array)
        t_list.append(tau)
        price_matrix.append(price_interp)
    t_grid = np.array(sorted(t_list))
    V_grid = np.vstack(price_matrix)  # shape (nExpiries, nS)
    # reorder by increasing t
    idx = np.argsort(t_grid)
    return S_grid, t_grid[idx], V_grid[idx,:], S_now

if __name__ == '__main__':
    S_grid, t_grid, V_grid, S_now = build_grid_from_yfinance('SPY', expiries=3, strikes_per_expiry=25)
    print("S_now", S_now)
    print("S grid shape", S_grid.shape, "t grid shape", t_grid.shape, "V shape", V_grid.shape)
