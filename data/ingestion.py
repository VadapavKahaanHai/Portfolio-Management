# data/ingestion.py — Step 1 & 2: Fetch, clean, and filter stock data

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Import config ───────────────────────────────────────────────────────────
import sys
from config import (
    STOCK_UNIVERSE, INDEX_TICKER, DATA_START, DATA_END,
    DATA_DIR, MISSING_THRESH, MIN_PRICE, MIN_AVG_VOLUME
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: RAW DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str = DATA_START,
    end: str   = DATA_END,
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download OHLCV from yfinance for all tickers.
    Returns (close_df, volume_df) with dates as index, tickers as columns.
    auto_adjust=True handles splits & dividends automatically.
    """
    print(f"\n{'='*60}")
    print(f"  Fetching {len(tickers)} stocks  [{start} → {end}]")
    print(f"{'='*60}")

    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=True,
                threads=True,
            )
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}. Retrying…")
            time.sleep(5)
    else:
        raise RuntimeError("Failed to download data after retries.")

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        # Single ticker edge case
        close  = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()

    print(f"\n  Raw shape: {close.shape}  ({close.index[0].date()} → {close.index[-1].date()})")
    return close, volume


def fetch_index(
    ticker: str  = INDEX_TICKER,
    start: str   = DATA_START,
    end: str     = DATA_END,
) -> pd.Series:
    """Fetch NIFTY 50 index close prices."""
    print(f"\n  Fetching index: {ticker}")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    series = raw["Close"].squeeze()
    series.name = "NIFTY50"
    return series


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: UNIVERSE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def filter_universe(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    missing_thresh: float   = MISSING_THRESH,
    min_price: float        = MIN_PRICE,
    min_avg_volume: float   = MIN_AVG_VOLUME,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove stocks that will produce nonsense portfolios:
      1. Too much missing data
      2. Penny stocks (low price)
      3. Illiquid stocks (low avg volume)

    Returns filtered (close, volume, surviving_tickers).
    """
    print(f"\n{'─'*60}")
    print(f"  Universe Filtering  (starting: {close.shape[1]} stocks)")
    print(f"{'─'*60}")

    original = set(close.columns)

    # ── Filter 1: Missing data ─────────────────────────────────────────────
    missing_pct = close.isnull().mean()
    keep_missing = missing_pct[missing_pct < missing_thresh].index
    removed_missing = original - set(keep_missing)
    if removed_missing:
        print(f"  Removed (missing data >{ missing_thresh*100:.0f}%): {sorted(removed_missing)}")
    close  = close[keep_missing]
    volume = volume[keep_missing]

    # ── Filter 2: Price floor ──────────────────────────────────────────────
    mean_price = close.mean()
    keep_price = mean_price[mean_price >= min_price].index
    removed_price = set(keep_missing) - set(keep_price)
    if removed_price:
        print(f"  Removed (avg price < ₹{min_price}): {sorted(removed_price)}")
    close  = close[keep_price]
    volume = volume[keep_price]

    # ── Filter 3: Liquidity ────────────────────────────────────────────────
    avg_vol = volume.mean()
    keep_vol = avg_vol[avg_vol >= min_avg_volume].index
    removed_vol = set(keep_price) - set(keep_vol)
    if removed_vol:
        print(f"  Removed (avg volume < {min_avg_volume:,}): {sorted(removed_vol)}")
    close  = close[keep_vol]
    volume = volume[keep_vol]

    # ── Final fill ─────────────────────────────────────────────────────────
    close  = close.ffill().bfill()
    volume = volume.ffill().bfill()

    surviving = list(close.columns)
    print(f"\n  ✓ Surviving stocks: {len(surviving)}")
    print(f"  {surviving}")
    return close, volume, surviving


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(
    close: pd.DataFrame,
    log: bool = True,
) -> pd.DataFrame:
    """
    Compute daily returns from price matrix.
    log=True → log returns (better for ML, additive over time)
    log=False → simple returns
    """
    if log:
        returns = np.log(close / close.shift(1))
    else:
        returns = close.pct_change()
    return returns.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_data(close, volume, returns, market_returns, data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    close.to_parquet(f"{data_dir}/close.parquet")
    volume.to_parquet(f"{data_dir}/volume.parquet")
    returns.to_parquet(f"{data_dir}/returns.parquet")
    market_returns.to_frame().to_parquet(f"{data_dir}/market_returns.parquet")
    print(f"\n  ✓ Data saved to '{data_dir}/'")


def load_data(data_dir: str = DATA_DIR):
    close          = pd.read_parquet(f"{data_dir}/close.parquet")
    volume         = pd.read_parquet(f"{data_dir}/volume.parquet")
    returns        = pd.read_parquet(f"{data_dir}/returns.parquet")
    market_returns = pd.read_parquet(f"{data_dir}/market_returns.parquet").squeeze()
    print(f"  ✓ Data loaded from '{data_dir}/'")
    return close, volume, returns, market_returns


def data_exists(data_dir: str = DATA_DIR) -> bool:
    return all(
        os.path.exists(f"{data_dir}/{f}")
        for f in ["close.parquet", "volume.parquet", "returns.parquet", "market_returns.parquet"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(force_refresh: bool = False) -> tuple:
    """
    Full ingestion pipeline.
    If data already exists on disk, loads from cache unless force_refresh=True.
    Returns (close, volume, returns, market_returns).
    """
    if not force_refresh and data_exists():
        print("  [Cache hit] Loading existing data from disk…")
        return load_data()

    # 1. Fetch raw prices
    close_raw, volume_raw = fetch_prices(STOCK_UNIVERSE)
    market_close = fetch_index()

    # 2. Filter universe
    close, volume, _ = filter_universe(close_raw, volume_raw)

    # 3. Compute returns
    returns = compute_returns(close)
    market_returns = compute_returns(market_close.to_frame()).squeeze()
    market_returns.name = "NIFTY50"

    # Align dates
    idx = returns.index.intersection(market_returns.index)
    returns        = returns.loc[idx]
    market_returns = market_returns.loc[idx]
    close          = close.loc[idx]
    volume         = volume.loc[idx]

    # 4. Save
    save_data(close, volume, returns, market_returns)

    print(f"\n{'='*60}")
    print(f"  Ingestion complete!")
    print(f"  Price matrix : {close.shape}")
    print(f"  Returns      : {returns.shape}")
    print(f"  Date range   : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"{'='*60}\n")

    return close, volume, returns, market_returns


if __name__ == "__main__":
    close, volume, returns, market_returns = run_ingestion(force_refresh=True)
