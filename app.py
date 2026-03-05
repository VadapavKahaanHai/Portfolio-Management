"""
app.py — Flask Web UI for ML Portfolio Management System
Place this file in your project root:
C:/Users/YASH JOSHI/.../Portfolio-Management/app.py
"""

import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib import

import os
import sys
import json
import traceback

from flask import Flask, render_template, request, jsonify

# ── Path fix so Flask can find your modules ───────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Ensure output folders exist
os.makedirs("static/plots", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)


def run_pipeline_for_web(amount: float, risk: str, goal: str, years: int, sip: float = 0.0):
    """
    Wrapper around main.py's run_pipeline().
    Redirects chart output to static/plots/ for Flask to serve.
    Returns a clean dict for the template.
    """
    # Monkey-patch the output path before importing visualization
    import config
    config.PLOT_DIR = "static/plots"
    os.makedirs("static/plots", exist_ok=True)

    from main import run_pipeline

    results = run_pipeline(
        amount           = amount,
        user_risk        = risk,
        goal_type        = goal,
        investment_years = years,
        monthly_sip      = sip,
        force_refresh    = False,
    )

    # ── Save charts to static/plots/ ─────────────────────────────────────────
    try:
        from utils.visualization import (
            plot_portfolio_weights, plot_correlation_heatmap,
            plot_risk_dashboard, plot_goal_projection,
        )
        bp  = results["best_portfolio"]
        ret = results  # full results dict

        plot_portfolio_weights(bp,  1, "static/plots/weights.png")
        plot_correlation_heatmap(
            ret["mu_hat"].to_frame(),           # pass returns if available
            bp["stocks"][:20],
            "static/plots/correlation.png"
        )
        plot_risk_dashboard(ret["top_portfolios"], ret["allocation"], "static/plots/dashboard.png")
        plot_goal_projection(ret["projection"],    ret["goal_config"], "static/plots/goal_projection.png")
    except Exception as e:
        print(f"  Chart generation warning: {e}")

    # ── Build a clean response dict ───────────────────────────────────────────
    bp   = results["best_portfolio"]
    proj = results["projection"]
    gc   = results["goal_constraints"]
    alloc = results["allocation"]

    # Allocation table rows
    alloc_rows = []
    for ticker, row in alloc.iterrows():
        alloc_rows.append({
            "ticker":   ticker.replace(".NS", ""),
            "shares":   int(row["shares"]),
            "price":    f"₹{row['price_inr']:,.1f}",
            "invested": f"₹{row['invested_inr']:,.0f}",
            "weight":   f"{row['weight_actual']*100:.1f}%",
        })

    # Risk contributors
    risk_contrib = [
        {"stock": k.replace(".NS",""), "pct": f"{v*100:.1f}%"}
        for k, v in bp["risk_contrib"].sort_values(ascending=False).head(5).items()
    ]

    return {
        # Key metrics
        "expected_return": f"{bp['expected_return']*100:.1f}",
        "volatility":      f"{bp['volatility']*100:.1f}",
        "sharpe":          f"{bp['sharpe']:.2f}",
        "max_drawdown":    f"{bp['max_drawdown_est']*100:.1f}",
        "sortino":         f"{bp['sortino']:.2f}",

        # Goal info
        "goal_type":       goal,
        "investment_years": years,
        "effective_risk":  gc["effective_risk"],
        "user_risk":       risk,
        "equity_pct":      f"{gc['equity_allocation']*100:.0f}",
        "debt_pct":        f"{gc['debt_allocation']*100:.0f}",
        "equity_amount":   f"₹{results['equity_amount']:,.0f}",
        "debt_amount":     f"₹{results['debt_amount']:,.0f}",

        # Corpus projection
        "total_invested":  f"₹{proj['total_invested']:,.0f}",
        "total_corpus":    f"₹{proj['total_corpus']:,.0f}",
        "total_gain":      f"₹{proj['total_gain']:,.0f}",
        "wealth_multiple": f"{proj['wealth_multiple']:.2f}",
        "cagr":            f"{proj['cagr']:.1f}",
        "p10":  f"₹{proj['p10']:,.0f}",
        "p25":  f"₹{proj['p25']:,.0f}",
        "p50":  f"₹{proj['p50']:,.0f}",
        "p75":  f"₹{proj['p75']:,.0f}",
        "p90":  f"₹{proj['p90']:,.0f}",

        # Yearly data for Chart.js
        "yearly_labels":  [str(r["year"]) for r in proj["yearly_projection"]],
        "yearly_corpus":  [r["corpus"]    for r in proj["yearly_projection"]],
        "yearly_invested":[r["invested"]  for r in proj["yearly_projection"]],

        # Allocation table
        "allocation":     alloc_rows,
        "risk_contrib":   risk_contrib,
        "n_stocks":       len(alloc_rows),

        # Charts
        "chart_weights":     "plots/weights.png",
        "chart_correlation": "plots/correlation.png",
        "chart_dashboard":   "plots/dashboard.png",
        "chart_projection":  "plots/goal_projection.png",

        # Raw for pie chart
        "equity_pct_raw": gc["equity_allocation"] * 100,
        "debt_pct_raw":   gc["debt_allocation"]   * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Show the input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data, run pipeline, return results."""
    try:
        amount = float(request.form.get("amount", 100000))
        risk   = request.form.get("risk",   "Medium")
        goal   = request.form.get("goal",   "savings")
        years  = int(request.form.get("years", 5))
        sip    = float(request.form.get("sip", 0))

        data = run_pipeline_for_web(amount, risk, goal, years, sip)
        return render_template("results.html", **data)

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"Pipeline error:\n{tb}")
        return render_template("index.html", error=error_msg), 500


if __name__ == "__main__":
    print("\n🚀  Portfolio AI — Starting Flask server...")
    print("    Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)