#!/usr/bin/env python3
"""
main.py — End-to-End ML Portfolio Pipeline (India Stocks)
         with Goal-Based Investing Engine

Usage:
  python main.py --amount 500000 --risk Medium --goal retirement --years 20
  python main.py --amount 1000000 --risk Low --goal education --years 12 --sip 10000
  python main.py --amount 250000 --risk High --goal fire --years 10 --target 10000000
  python main.py --amount 300000 --risk Medium --goal marriage --years 4 --sip 5000
  python main.py --amount 200000 --risk Low --goal savings --years 3
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path fix: works on Windows & Linux regardless of CWD ─────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

from data.ingestion          import run_ingestion
from features.engineering    import build_feature_table, build_ml_dataset
from models.risk_clustering  import assign_risk_clusters, get_risk_subset
from models.return_predictor import train_return_model, predict_expected_returns
from models.covariance       import compute_covariance
from portfolio.optimizer     import build_portfolios, monte_carlo_portfolios
from portfolio.allocator     import allocate_to_shares, format_allocation_report
from portfolio.goals         import (
    GoalConfig, GOAL_PROFILES,
    resolve_effective_risk,
    get_equity_allocation,
    get_goal_constraints,
    project_corpus,
    recommend_sip,
    format_goal_report,
    validate_goal_inputs,
)
from config import RETURN_HORIZON, N_TOP_PORTFOLIOS


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    amount: float,
    user_risk: str,
    goal_type: str           = "savings",
    investment_years: int    = 5,
    monthly_sip: float       = 0.0,
    target_corpus: float     = None,
    current_age: int         = None,
    horizon: int             = RETURN_HORIZON,
    force_refresh: bool      = False,
    predictor_method: str    = "lgbm",
    cov_method: str          = "ledoit_wolf",
    cluster_method: str      = "kmeans",
) -> dict:
    """Full pipeline with goal-based investing layer."""

    # ── Build GoalConfig ───────────────────────────────────────────────────
    goal_config = GoalConfig(
        goal_type                = goal_type,
        investment_horizon_years = investment_years,
        lump_sum_amount          = amount,
        monthly_sip              = monthly_sip,
        target_corpus            = target_corpus,
        current_age              = current_age,
        user_risk                = user_risk,
    )

    # ── Validation warnings ────────────────────────────────────────────────
    warnings_list = validate_goal_inputs(goal_type, investment_years, amount, user_risk)

    # ── Resolve effective risk ─────────────────────────────────────────────
    effective_risk   = resolve_effective_risk(goal_config)
    goal_constraints = get_goal_constraints(goal_config)
    equity_pct       = goal_constraints["equity_allocation"] * 100
    profile          = GOAL_PROFILES[goal_type]

    # ── Banner ─────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       ML PORTFOLIO PIPELINE  —  GOAL-BASED INVESTING     ║
╠══════════════════════════════════════════════════════════╣
║  Goal        : {profile['emoji']} {profile['display_name']:<43}║
║  Horizon     : {investment_years} years                                      ║
║  Amount      : Rs.{amount:>12,.0f}                            ║
║  Monthly SIP : Rs.{monthly_sip:>12,.0f}                            ║
║  Risk (user) : {user_risk:<44}  ║
║  Risk (eff.) : {effective_risk:<44}  ║
║  Equity %    : {equity_pct:.0f}%   Debt % : {100-equity_pct:.0f}%                          ║
╚══════════════════════════════════════════════════════════╝""")

    if warnings_list:
        print("\n  WARNINGS:")
        for w in warnings_list:
            print(f"  * {w}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1-2: Data Ingestion
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STEP 1-2] Data Ingestion & Universe Filtering")
    close, volume, returns, market_returns = run_ingestion(force_refresh=force_refresh)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Features
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] Feature Engineering")
    feature_table = build_feature_table(close, returns, market_returns)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Risk Clustering  — use EFFECTIVE risk (goal-adjusted)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Risk Bucket Assignment (effective risk: {effective_risk})")
    risk_series = assign_risk_clusters(feature_table, method=cluster_method)
    candidate_stocks = get_risk_subset(risk_series, effective_risk)

    if len(candidate_stocks) < 8:
        print(f"  Expanding stock pool to Medium risk...")
        candidate_stocks = get_risk_subset(risk_series, "Medium")

    # Apply goal-based sector tilts
    sector_tilt = goal_constraints.get("sector_tilt", {})
    candidate_stocks = _apply_sector_tilt(candidate_stocks, sector_tilt)
    print(f"  After sector tilt filter: {len(candidate_stocks)} stocks")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Return Prediction
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STEP 5] Expected Return Estimation (ML)")
    model_path = f"models/return_model_{horizon}d.pkl"

    import joblib
    if os.path.exists(model_path) and not force_refresh:
        print(f"  [Cache] Loading model from {model_path}")
        model_bundle = joblib.load(model_path)
    else:
        print("  Building training dataset (this takes a few minutes)...")
        X_all, y_all = build_ml_dataset(
            close=close, returns=returns,
            market_returns=market_returns,
            horizon=horizon, lookback=126,
        )
        y = y_all["fwd_return"] if hasattr(y_all, "columns") and "fwd_return" in y_all.columns else y_all
        X = X_all.drop(columns=["date", "stock", "fwd_return"], errors="ignore")
        model_bundle = train_return_model(X_all=X, y_all=y, method=predictor_method, save_path=model_path)
        model_bundle["feature_cols"] = list(X.columns)

    mu_hat = predict_expected_returns(feature_table, model_bundle)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: Covariance
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STEP 6] Covariance Matrix Estimation")
    ret_sub = returns[candidate_stocks]
    Sigma   = compute_covariance(ret_sub, method=cov_method)
    print(f"  Covariance: {Sigma.shape}  ({cov_method})")

    # ──────────────────────────────────────────────────────────────────────
    # STEPS 7-9: Portfolio Generation + Optimization
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[STEPS 7-9] Portfolio Optimization  (objective: {effective_risk})")
    top_portfolios = build_portfolios(
        candidate_stocks = candidate_stocks,
        mu_hat           = mu_hat,
        Sigma            = Sigma,
        returns          = returns,
        user_risk        = effective_risk,   # goal-adjusted risk drives the objective
        close            = close,
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 10: Allocate to shares (equity portion only)
    # ──────────────────────────────────────────────────────────────────────
    equity_amount = amount * goal_constraints["equity_allocation"]
    debt_amount   = amount * goal_constraints["debt_allocation"]

    print(f"\n[STEP 10] Share Allocation")
    print(f"  Equity portion : Rs.{equity_amount:,.0f}  ({equity_pct:.0f}%)")
    if debt_amount > 0:
        print(f"  Debt portion   : Rs.{debt_amount:,.0f}  ({100-equity_pct:.0f}%)  -> allocate to FD/debt funds manually")

    best_portfolio = top_portfolios[0]
    allocation     = allocate_to_shares(best_portfolio["weights"], close, equity_amount)
    alloc_report   = format_allocation_report(allocation, best_portfolio, effective_risk, equity_amount)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 11: Goal projection
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STEP 11] Goal Projection & Report")
    projection = project_corpus(
        goal_config               = goal_config,
        portfolio_expected_return = best_portfolio["expected_return"],
        portfolio_volatility      = best_portfolio["volatility"],
    )

    sip_needed = None
    if target_corpus and monthly_sip == 0:
        sip_needed = recommend_sip(goal_config, target_corpus, best_portfolio["expected_return"])
        print(f"\n  To reach Rs.{target_corpus:,.0f}, you need Rs.{sip_needed:,.0f}/month SIP")

    goal_report = format_goal_report(
        goal_config   = goal_config,
        projection    = projection,
        portfolio     = best_portfolio,
        constraints   = goal_constraints,
    )
    print(goal_report)

    # ──────────────────────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STEP 12] Generating Charts...")
    try:
        from utils.visualization import (
            plot_portfolio_weights, plot_correlation_heatmap,
            plot_risk_dashboard, plot_goal_projection,
        )
        os.makedirs("outputs", exist_ok=True)
        plot_portfolio_weights(best_portfolio, 1, "outputs/weights_best.png")
        plot_correlation_heatmap(returns, best_portfolio["stocks"][:20], "outputs/correlation.png")
        plot_risk_dashboard(top_portfolios, allocation, "outputs/dashboard.png")
        plot_goal_projection(projection, goal_config, "outputs/goal_projection.png")
        print("  All charts saved to outputs/")
    except Exception as e:
        print(f"  Some charts skipped: {e}")

    # Save reports
    os.makedirs("outputs", exist_ok=True)
    full_report = goal_report + "\n\n" + alloc_report
    with open("outputs/portfolio_report.txt", "w", encoding="utf-8") as f:
        f.write(full_report)
    pd.DataFrame(projection["yearly_projection"]).to_csv("outputs/corpus_projection.csv", index=False)
    print("  Reports saved -> outputs/")

    return {
        "top_portfolios":   top_portfolios,
        "best_portfolio":   best_portfolio,
        "allocation":       allocation,
        "projection":       projection,
        "goal_config":      goal_config,
        "goal_constraints": goal_constraints,
        "risk_series":      risk_series,
        "mu_hat":           mu_hat,
        "sip_needed":       sip_needed,
        "equity_amount":    equity_amount,
        "debt_amount":      debt_amount,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sector_tilt(stocks: list, sector_tilt: dict) -> list:
    """Filter/reorder stocks using sector tilt — penalised sectors dropped if enough stocks."""
    from config import SECTOR_MAP
    if not sector_tilt:
        return stocks
    MIN_KEEP = 15
    scored = [(s, sector_tilt.get(SECTOR_MAP.get(s, "Other"), 1.0)) for s in stocks]
    filtered = [(s, t) for s, t in scored if t >= 0.70]
    if len(filtered) < MIN_KEEP:
        filtered = sorted(scored, key=lambda x: x[1], reverse=True)[:MIN_KEEP]
    return [s for s, _ in filtered]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Portfolio Pipeline — Goal-Based Investing (India / NSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --amount 500000  --risk Medium --goal retirement --years 20 --sip 10000
  python main.py --amount 1000000 --risk High   --goal fire       --years 12 --target 20000000
  python main.py --amount 300000  --risk Medium --goal education  --years 10 --sip 5000
  python main.py --amount 200000  --risk Low    --goal marriage   --years 4
  python main.py --amount 100000  --risk Medium --goal savings    --years 3
        """
    )
    parser.add_argument("--amount",    type=float, required=True,   help="Lump sum investment in Rs.")
    parser.add_argument("--risk",      type=str,   default="Medium", choices=["Low","Medium","High"])
    parser.add_argument("--goal",      type=str,   default="savings", choices=list(GOAL_PROFILES.keys()),
                        help="Goal: retirement | fire | education | marriage | savings")
    parser.add_argument("--years",     type=int,   default=5,        help="Investment horizon in years (1-40)")
    parser.add_argument("--sip",       type=float, default=0.0,      help="Monthly SIP amount in Rs.")
    parser.add_argument("--target",    type=float, default=None,     help="Target corpus in Rs. (optional)")
    parser.add_argument("--age",       type=int,   default=None,     help="Your current age (optional)")
    parser.add_argument("--horizon",   type=int,   default=RETURN_HORIZON, help="ML prediction horizon (trading days)")
    parser.add_argument("--refresh",   action="store_true",          help="Re-download data + retrain models")
    parser.add_argument("--predictor", type=str,   default="lgbm",   choices=["ridge","lasso","lgbm","gbm"])
    parser.add_argument("--cov",       type=str,   default="ledoit_wolf", choices=["ledoit_wolf","ewma","sample"])
    parser.add_argument("--cluster",   type=str,   default="kmeans", choices=["kmeans","gmm","hierarchical"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_pipeline(
        amount           = args.amount,
        user_risk        = args.risk,
        goal_type        = args.goal,
        investment_years = args.years,
        monthly_sip      = args.sip,
        target_corpus    = args.target,
        current_age      = args.age,
        horizon          = args.horizon,
        force_refresh    = args.refresh,
        predictor_method = args.predictor,
        cov_method       = args.cov,
        cluster_method   = args.cluster,
    )
