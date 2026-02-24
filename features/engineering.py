# features/engineering.py — Step 3: Feature Engineering (risk, return, correlation features)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

import sys, os
from config import VOL_WINDOWS, MOM_WINDOWS, CORR_WINDOW, RISKFREE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# RISK FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling volatility for each window in VOL_WINDOWS."""
    feats = {}
    for w in VOL_WINDOWS:
        feats[f"vol_{w}d"] = returns.rolling(w).std().iloc[-1] * np.sqrt(252)
    return pd.DataFrame(feats)


def compute_beta(returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    """
    OLS beta of each stock vs market (full history).
    Beta = Cov(stock, market) / Var(market)
    """
    aligned = returns.align(market_returns, join="inner", axis=0)
    r_stocks = aligned[0]
    r_mkt    = aligned[1]

    betas = {}
    mkt_var = r_mkt.var()
    for col in r_stocks.columns:
        cov = r_stocks[col].cov(r_mkt)
        betas[col] = cov / mkt_var if mkt_var != 0 else 1.0
    return pd.Series(betas, name="beta")


def compute_max_drawdown(close: pd.DataFrame) -> pd.Series:
    """Max drawdown per stock over entire history."""
    drawdowns = {}
    for col in close.columns:
        prices = close[col].dropna()
        roll_max = prices.cummax()
        dd = (prices - roll_max) / roll_max
        drawdowns[col] = dd.min()
    return pd.Series(drawdowns, name="max_drawdown")


def compute_downside_deviation(returns: pd.DataFrame) -> pd.Series:
    """Annualised downside deviation (Sortino denominator)."""
    daily_rf = RISKFREE_RATE / 252
    dd = {}
    for col in returns.columns:
        excess = returns[col] - daily_rf
        neg    = excess[excess < 0]
        dd[col] = neg.std() * np.sqrt(252) if len(neg) > 1 else returns[col].std() * np.sqrt(252)
    return pd.Series(dd, name="downside_dev")


def compute_var_cvar(returns: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Historical VaR and CVaR at given confidence level."""
    var_vals, cvar_vals = {}, {}
    for col in returns.columns:
        r = returns[col].dropna()
        var_vals[col]  = np.percentile(r, (1 - confidence) * 100)
        cvar_vals[col] = r[r <= var_vals[col]].mean()
    return pd.DataFrame({"var_95": var_vals, "cvar_95": cvar_vals})


# ─────────────────────────────────────────────────────────────────────────────
# RETURN / MOMENTUM FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum = total return over last N trading days.
    Skip the last 1 day to avoid short-term reversal contamination.
    """
    feats = {}
    for w in MOM_WINDOWS:
        if len(close) > w + 1:
            mom = (close.iloc[-2] - close.iloc[-(w + 1)]) / close.iloc[-(w + 1)]
            feats[f"mom_{w}d"] = mom
    return pd.DataFrame(feats)


def compute_rolling_mean_return(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised mean return over short and medium windows."""
    feats = {}
    for w in [21, 63]:
        feats[f"mean_ret_{w}d"] = returns.rolling(w).mean().iloc[-1] * 252
    return pd.DataFrame(feats)


def compute_trend_strength(close: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    R² of linear regression of log(price) vs time.
    R²=1 → perfect trend, R²=0 → no trend.
    """
    trend_r2 = {}
    for col in close.columns:
        prices = np.log(close[col].dropna().iloc[-window:])
        if len(prices) < 10:
            trend_r2[col] = 0.0
            continue
        x = np.arange(len(prices))
        slope, intercept, r, p, se = stats.linregress(x, prices)
        trend_r2[col] = r ** 2 * np.sign(slope)  # signed R²
    return pd.Series(trend_r2, name="trend_r2")


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation_features(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = CORR_WINDOW,
) -> pd.DataFrame:
    """
    - corr_with_market: rolling correlation to NIFTY50
    - avg_corr_peers: average pairwise correlation with all other stocks
    """
    recent = returns.iloc[-window:]
    mkt    = market_returns.iloc[-window:]

    corr_mkt = recent.corrwith(mkt)
    corr_mkt.name = "corr_market"

    corr_matrix = recent.corr()
    avg_corr = corr_matrix.mean()
    avg_corr.name = "avg_peer_corr"

    return pd.DataFrame({"corr_market": corr_mkt, "avg_peer_corr": avg_corr})


# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTAL-STYLE FEATURES (from price only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_price_ratios(close: pd.DataFrame) -> pd.DataFrame:
    """
    Price location within 52-week range.
    - price_52w_position: 0=52w low, 1=52w high
    - distance_from_ma200: % above/below 200-day MA
    """
    feats = {}
    window_52w = min(252, len(close))
    window_ma  = min(200, len(close))

    for col in close.columns:
        prices    = close[col].dropna()
        hi_52w    = prices.iloc[-window_52w:].max()
        lo_52w    = prices.iloc[-window_52w:].min()
        current   = prices.iloc[-1]
        ma200     = prices.iloc[-window_ma:].mean()

        pos = (current - lo_52w) / (hi_52w - lo_52w) if hi_52w != lo_52w else 0.5
        dist = (current - ma200) / ma200

        feats[col] = {"price_52w_pos": pos, "dist_from_ma200": dist}

    return pd.DataFrame(feats).T


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_table(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.DataFrame:
    """
    Combine all features into a single feature table X[stock, features].
    This is the input for both clustering (risk buckets) and return prediction.
    """
    print("\n  Building feature table…")

    frames = [
        compute_volatility(returns),
        compute_beta(returns, market_returns).to_frame(),
        compute_max_drawdown(close).to_frame(),
        compute_downside_deviation(returns).to_frame(),
        compute_var_cvar(returns),
        compute_momentum(close),
        compute_rolling_mean_return(returns),
        compute_trend_strength(close).to_frame(),
        compute_correlation_features(returns, market_returns),
        compute_price_ratios(close),
    ]

    feature_table = pd.concat(frames, axis=1)

    # Align to common stock set
    feature_table = feature_table.loc[close.columns]
    feature_table = feature_table.fillna(feature_table.median())

    print(f"  ✓ Feature table: {feature_table.shape}  (stocks × features)")
    print(f"  Features: {list(feature_table.columns)}")
    return feature_table


# ─────────────────────────────────────────────────────────────────────────────
# ML TARGET: FORWARD RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_dataset(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    market_returns: pd.Series,
    horizon: int,
    lookback: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a panel dataset for return prediction:
      - For each stock and each date, compute rolling features (X)
      - Target (y) = forward return over 'horizon' days

    Returns X_all, y_all (stacked across all stocks and dates).
    """
    print(f"\n  Building ML dataset (horizon={horizon}d, lookback={lookback}d)…")

    X_list, y_list = [], []

    stocks = list(close.columns)
    dates  = returns.index

    for i in range(lookback, len(dates) - horizon):
        window_close   = close.iloc[i - lookback: i]
        window_returns = returns.iloc[i - lookback: i]
        window_mkt     = market_returns.iloc[i - lookback: i]

        feats = build_feature_table(window_close, window_returns, window_mkt)

        # Forward return for each stock
        fwd_prices = close.iloc[i: i + horizon]
        fwd_return = (fwd_prices.iloc[-1] - fwd_prices.iloc[0]) / fwd_prices.iloc[0]

        feats_common = feats.loc[feats.index.intersection(fwd_return.index)]
        fwd_common   = fwd_return.loc[feats_common.index]

        feats_common = feats_common.copy()
        feats_common["date"]  = dates[i]
        feats_common["stock"] = feats_common.index

        X_list.append(feats_common.reset_index(drop=True))
        y_list.append(fwd_common.rename("fwd_return").reset_index(drop=True))

    X_all = pd.concat(X_list, ignore_index=True)
    y_all = pd.concat(y_list, ignore_index=True)

    print(f"  ✓ ML dataset: X={X_all.shape}, y={y_all.shape}")
    return X_all, y_all


if __name__ == "__main__":
    import sys
    from data.ingestion import run_ingestion

    close, volume, returns, market_returns = run_ingestion()
    features = build_feature_table(close, returns, market_returns)
    print(features)
