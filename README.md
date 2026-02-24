# ML Portfolio Pipeline — India Stocks (NSE)

End-to-end ML system that takes ₹ amount + risk appetite and returns
optimised stock portfolios with exact share quantities to buy.

---

## Project Structure

```
ml_portfolio/
│
├── config.py                  ← All settings (universe, params, sector map)
├── main.py                    ← Pipeline entry point (CLI)
├── requirements.txt
│
├── data/
│   └── ingestion.py           ← Steps 1-2: Fetch + filter + clean data
│
├── features/
│   └── engineering.py         ← Step 3: Feature engineering (risk/return/corr)
│
├── models/
│   ├── risk_clustering.py     ← Step 4: ML risk bucket assignment
│   ├── return_predictor.py    ← Step 5: XGBoost/LightGBM return prediction
│   └── covariance.py          ← Step 6: Ledoit-Wolf / EWMA covariance
│
├── portfolio/
│   ├── optimizer.py           ← Steps 7-9: Corr clusters + optimization + ranking
│   └── allocator.py           ← Step 10: Convert weights → integer share quantities
│
└── utils/
    └── visualization.py       ← Charts: frontier, weights, heatmap, dashboard
```

---

## Setup

```bash
# 1. Clone / download the folder
cd ml_portfolio

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Basic

```bash
# ₹5 lakh, medium risk
python main.py --amount 500000 --risk Medium

# ₹10 lakh, low risk (HRP optimization)
python main.py --amount 1000000 --risk Low

# ₹2.5 lakh, high risk, 60-day horizon
python main.py --amount 250000 --risk High --horizon 60
```

### Advanced Options

```bash
python main.py \
  --amount 500000 \
  --risk Medium \
  --horizon 21 \           # predict next 21 trading days return
  --predictor lgbm \       # lgbm | ridge | lasso | gbm
  --cov ledoit_wolf \      # ledoit_wolf | ewma | sample
  --cluster kmeans \       # kmeans | gmm | hierarchical
  --refresh                # force re-download data + retrain models
```

### Programmatic

```python
from main import run_pipeline

results = run_pipeline(
    amount       = 500_000,
    user_risk    = "Medium",
    horizon      = 21,
)

# Best portfolio
best = results["best_portfolio"]
print(best["weights"])          # stock weights
print(best["expected_return"])  # annualised return
print(best["sharpe"])           # Sharpe ratio

# Allocation (exact shares to buy)
print(results["allocation"])    # DataFrame with qty, price, amount
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data/ingestion.py` | Fetch OHLCV from yfinance, auto-adjust for splits/dividends |
| 2 | `data/ingestion.py` | Universe filter: remove illiquid, penny, missing-data stocks |
| 3 | `features/engineering.py` | Compute volatility, beta, momentum, correlation features |
| 4 | `models/risk_clustering.py` | KMeans/GMM cluster stocks into Low/Med/High risk buckets |
| 5 | `models/return_predictor.py` | LightGBM predicts forward returns (time-series CV) |
| 6 | `models/covariance.py` | Ledoit-Wolf shrinkage covariance matrix |
| 7 | `portfolio/optimizer.py` | Hierarchical correlation clustering for diversification |
| 8 | `portfolio/optimizer.py` | Weight optimization: HRP (Low) / Sharpe (Med) / Return (High) |
| 9 | `portfolio/optimizer.py` | Monte Carlo + rank by risk-adjusted score |
| 10 | `portfolio/allocator.py` | Greedy integer rounding to exact share quantities |

---

## Outputs

After running, check the `outputs/` folder:

```
outputs/
├── portfolio_report.txt    ← Full allocation report with metrics
├── weights_best.png        ← Bar chart of portfolio weights
├── correlation.png         ← Stock correlation heatmap
└── dashboard.png           ← 4-panel analysis dashboard
```

---

## Key Design Decisions

**Why LightGBM for returns?**
Tree models handle non-linearity and feature interactions in financial data
better than linear models. IC (Information Coefficient) > 0.05 is considered
useful in quantitative finance.

**Why Ledoit-Wolf covariance?**
Plain sample covariance has high estimation error with 50 stocks and
~1500 observations. Ledoit-Wolf shrinkage reduces noise significantly.

**Why HRP for Low risk?**
Hierarchical Risk Parity doesn't require mean estimates (which are noisy),
making it more robust than Mean-Variance for conservative portfolios.

**Why correlation clusters?**
Ensures diversification at the basket-generation stage, before optimization.
Prevents all-IT or all-Banking portfolios even when those have high momentum.

---

## Notes & Limitations

- **Survivorship bias**: yfinance only covers currently listed stocks.
  Historical performance will look better than it would have been live.
- **Look-ahead bias**: TimeSeriesSplit is used strictly to prevent this in training.
- **Transaction costs**: Not modelled. Add 0.1-0.3% per trade for realism.
- **This is not financial advice.** Use for educational purposes only.
