import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
from config import (
    STOCK_UNIVERSE, INDEX_TICKER, MISSING_THRESH, MIN_PRICE, MIN_AVG_VOLUME
)

def _get_date_range():
    end = date.today()
    start = end - timedelta(days=5 * 365)
    return start.isoformat(), end.isoformat()

def fetch_prices(tickers, retries=3):
    start, end = _get_date_range()
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )
            break
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Download failed after {retries} attempts: {e}")
            time.sleep(5)

    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close  = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()

    # Use available data if less than 5 years
    close  = close.dropna(how="all")
    volume = volume.dropna(how="all")

    return close, volume

def fetch_index(retries=3):
    start, end = _get_date_range()
    for attempt in range(retries):
        try:
            raw = yf.download(
                INDEX_TICKER, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            break
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Index download failed: {e}")
            time.sleep(5)
    series = raw["Close"].squeeze()
    series.name = "NIFTY50"
    return series

def filter_universe(close, volume):
    original = set(close.columns)

    missing_pct = close.isnull().mean()
    keep = missing_pct[missing_pct < MISSING_THRESH].index
    close, volume = close[keep], volume[keep]

    mean_price = close.mean()
    keep = mean_price[mean_price >= MIN_PRICE].index
    close, volume = close[keep], volume[keep]

    avg_vol = volume.mean()
    keep = avg_vol[avg_vol >= MIN_AVG_VOLUME].index
    close, volume = close[keep], volume[keep]

    close  = close.ffill().bfill()
    volume = volume.ffill().bfill()

    return close, volume, list(close.columns)

def compute_returns(close, log=True):
    if log:
        returns = np.log(close / close.shift(1))
    else:
        returns = close.pct_change()
    return returns.dropna()

def run_ingestion():
    close_raw, volume_raw = fetch_prices(STOCK_UNIVERSE)
    market_close = fetch_index()

    close, volume, _ = filter_universe(close_raw, volume_raw)

    returns = compute_returns(close)
    market_returns = compute_returns(market_close.to_frame()).squeeze()
    market_returns.name = "NIFTY50"

    idx = returns.index.intersection(market_returns.index)
    returns        = returns.loc[idx]
    market_returns = market_returns.loc[idx]
    close          = close.loc[idx]
    volume         = volume.loc[idx]

    return close, volume, returns, market_returns