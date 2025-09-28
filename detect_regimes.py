# detect_regimes.py
"""
Regime detection utilities.
Provides:
 - compute_time_series_features(sigma) -> features per time slice
 - run_kmeans(features, k)
 - run_hmm(features, n_states)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# optional: pip install hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

def compute_features_from_sigma(sigma_grid, S_grid):
    # sigma_grid shape (nT, nS)
    # features: mean sigma, std sigma, skewness approx, slope across strikes (OLS)
    nT, nS = sigma_grid.shape
    features = []
    for i in range(nT):
        arr = sigma_grid[i,:]
        mean = arr.mean()
        std = arr.std()
        # slope across strikes via linear regression
        coeffs = np.polyfit(S_grid, arr, 1)
        slope = coeffs[0]
        features.append([mean, std, slope])
    return np.array(features)

def run_kmeans(features, k=2):
    scaler = StandardScaler()
    fscaled = scaler.fit_transform(features)
    km = KMeans(k, random_state=0).fit(fscaled)
    labels = km.labels_
    return labels, km

def run_hmm(features, n_states=2):
    if not HMM_AVAILABLE:
        raise RuntimeError("hmmlearn not installed. pip install hmmlearn to use HMM.")
    scaler = StandardScaler()
    fscaled = scaler.fit_transform(features)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=0)
    model.fit(fscaled)
    labels = model.predict(fscaled)
    return labels, model

def change_point_simple(mean_ts, threshold=1.5):
    # naive change point: detect points where rolling z-score exceeds threshold
    import pandas as pd
    s = pd.Series(mean_ts)
    z = (s - s.rolling(20, min_periods=1).mean()) / (s.rolling(20, min_periods=1).std() + 1e-9)
    cps = np.where(np.abs(z) > threshold)[0]
    return cps
