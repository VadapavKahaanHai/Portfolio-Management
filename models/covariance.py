# models/covariance.py — Step 6: Robust Covariance Estimation

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def compute_ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance (annualised).
    Best general-purpose covariance estimator — reduces estimation error
    vs plain sample covariance, especially for large number of stocks.
    """
    lw = LedoitWolf()
    lw.fit(returns.values)
    cov = pd.DataFrame(
        lw.covariance_ * 252,  # annualise
        index=returns.columns,
        columns=returns.columns,
    )
    return cov


def compute_ewma_covariance(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """
    Exponentially Weighted Moving Average covariance (annualised).
    Gives more weight to recent observations — better in trending markets.
    """
    decay = 2.0 / (span + 1)
    weights = np.array([(1 - decay) ** i for i in range(len(returns))][::-1])
    weights /= weights.sum()

    demeaned = returns.values - returns.mean().values
    cov_raw = np.einsum("t,ti,tj->ij", weights, demeaned, demeaned)
    cov = pd.DataFrame(cov_raw * 252, index=returns.columns, columns=returns.columns)
    return cov


def compute_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Dispatch to the chosen covariance estimator.
    method: 'ledoit_wolf' | 'ewma' | 'sample'
    """
    if method == "ledoit_wolf":
        Sigma = compute_ledoit_wolf(returns)
    elif method == "ewma":
        Sigma = compute_ewma_covariance(returns)
    elif method == "sample":
        Sigma = returns.cov() * 252
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure positive semi-definiteness (numerical fix)
    Sigma = _fix_psd(Sigma)
    return Sigma


def _fix_psd(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Clip negative eigenvalues to a small positive number.
    Ensures the matrix is positive semi-definite (required for optimization).
    """
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, 1e-8, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)
