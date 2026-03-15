import sys
import os

# points to portfolio_management/ root where config.py lives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from config import (
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    STOCK_UNIVERSE,
    INDEX_TICKER,
    MISSING_THRESH,
    MIN_PRICE,
    MIN_AVG_VOLUME,
)