# simulate_data.py
"""
Generate synthetic option price datasets for testing inverse PINN.
Saves .npz with keys: S_grid, t_grid, V_true, sigma_true, V_noisy
"""

import numpy as np
from scipy.stats import norm
import os

# Black-Scholes closed-form European call price (constant vol)
def bs_call_price(S, K, r, sigma, tau):
    # S,K,tau can be arrays; use elementwise
    # tau = T - t (time to maturity)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        price = np.where(tau <= 0, np.maximum(S - K, 0.0), price)
    return price

def sigma_surface(S, t, scenario='smooth'):
    # S, t are numpy arrays; returns sigma(S,t)
    if scenario == 'smooth':
        return 0.15 + 0.08 * np.sin(np.pi * t) + 0.03 * np.log1p(S)
    elif scenario == 'regime_switch':
        # Two regimes in time: low vol then high vol then low
        sigma = np.ones_like(S) * 0.12
        sigma = sigma + 0.12 * ((t > 0.35) & (t < 0.6)).astype(float)  # bump in mid-time
        sigma = sigma + 0.06 * np.sin(2 * np.pi * t) * 0  # optional small mod
        return sigma
    elif scenario == 'heston_like':
        return 0.18 + 0.06 * np.exp(-((t-0.5)**2)/0.03) + 0.02 * np.sin(3*np.log1p(S))
    else:
        raise ValueError("unknown scenario")

def generate_dataset(S_min=5, S_max=200, nS=80, nT=50, strikes=None,
                     T=1.0, r=0.01, scenario='regime_switch', noise_std=0.0, out_path='data'):
    os.makedirs(out_path, exist_ok=True)
    S_grid = np.linspace(S_min, S_max, nS)
    t_grid = np.linspace(0.0 + 1e-6, T, nT)  # t in [0,T]
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='xy')  # shape (nT, nS)
    # time-to-maturity tau = T - t
    tau = T - t_mesh

    sigma_true = sigma_surface(S_mesh, t_mesh / T, scenario=scenario)  # normalize t to [0,1]
    # choose strike K as middle of S range or vary
    if strikes is None:
        K = 100.0
    else:
        K = strikes

    V_true = bs_call_price(S_mesh, K, r, sigma_true, tau)
    rng = np.random.default_rng(0)
    V_noisy = V_true + rng.normal(scale=noise_std, size=V_true.shape)

    fname = os.path.join(out_path, f'synthetic_{scenario}.npz')
    np.savez_compressed(fname,
                        S_grid=S_grid, t_grid=t_grid, V_true=V_true,
                        sigma_true=sigma_true, V_noisy=V_noisy, K=K, r=r, T=T)
    print(f"Saved synthetic dataset to {fname}")
    return fname

if __name__ == '__main__':
    generate_dataset(nS=80, nT=50, scenario='regime_switch', noise_std=1e-2)
