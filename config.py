# config.py — Central configuration for the ML Portfolio Pipeline

# ─────────────────────────────────────────────
# UNIVERSE: 50 NSE stocks across major sectors
# ─────────────────────────────────────────────
STOCK_UNIVERSE = [
    # IT (5)
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Banking / Finance (8)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS",
    # FMCG (5)
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    # Auto (5)
    "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    # Pharma / Health (5)
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    # Energy / Oil (5)
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "POWERGRID.NS", "NTPC.NS",
    # Infra / Cement (5)
    "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "LT.NS", "SHREECEM.NS",
    # Metals (4)
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS",
    # Consumer / Lifestyle (5)
    "ASIANPAINT.NS", "TITAN.NS", "PIDILITIND.NS", "DMART.NS", "HAVELLS.NS",
    # Insurance / Others (3)
    "HDFCLIFE.NS", "SBILIFE.NS", "BHARTIARTL.NS",
]

INDEX_TICKER   = "^NSEI"          # NIFTY 50
RISKFREE_RATE  = 0.045            # ~6.5% (current Indian 10yr yield)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
DATA_START      = "2018-01-01"
DATA_END        = "2024-12-31"
DATA_DIR        = "data/"
MISSING_THRESH  = 0.05            # drop stock if >5% data missing
MIN_PRICE       = 10.0            # penny stock filter (₹)
MIN_AVG_VOLUME  = 100_000         # avg daily volume filter

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
VOL_WINDOWS     = [21, 63]        # rolling volatility windows (trading days)
MOM_WINDOWS     = [21, 63, 126, 252]  # momentum windows
CORR_WINDOW     = 63              # rolling correlation window

# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────
RETURN_HORIZON  = 21              # predict next N trading days return
N_RISK_CLUSTERS = 3               # Low / Medium / High
CV_SPLITS       = 5               # time-series cross-validation splits
RANDOM_STATE    = 42

# ─────────────────────────────────────────────
# PORTFOLIO OPTIMIZATION
# ─────────────────────────────────────────────
MAX_WEIGHT      = 0.12           # max 20% in any single stock
MIN_WEIGHT      = 0.04            # min 2% if selected
MAX_SECTOR_WEIGHT = 0.25          # max 35% in any single sector
MIN_STOCKS      = 8
MAX_STOCKS      = 20
N_PORTFOLIOS    = 500             # Monte Carlo random portfolios
N_TOP_PORTFOLIOS = 5              # final portfolios to return

# ─────────────────────────────────────────────
# SECTOR MAP
# ─────────────────────────────────────────────
SECTOR_MAP = {
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking",
    "BAJFINANCE.NS": "Finance", "BAJAJFINSV.NS": "Finance", "INDUSINDBK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "DABUR.NS": "FMCG",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "HEROMOTOCO.NS": "Auto", "EICHERMOT.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "APOLLOHOSP.NS": "Healthcare",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "POWERGRID.NS": "Energy", "NTPC.NS": "Energy",
    "ULTRACEMCO.NS": "Cement", "GRASIM.NS": "Cement", "SHREECEM.NS": "Cement",
    "ADANIPORTS.NS": "Infra", "LT.NS": "Infra",
    "TATASTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "JSWSTEEL.NS": "Metals", "COALINDIA.NS": "Metals",
    "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer", "PIDILITIND.NS": "Consumer",
    "DMART.NS": "Retail", "HAVELLS.NS": "Consumer",
    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "BHARTIARTL.NS": "Telecom",
}
