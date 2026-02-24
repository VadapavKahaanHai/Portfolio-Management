# utils/visualization.py — Charts: efficient frontier, weights, correlation heatmap

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

plt.rcParams["figure.facecolor"]  = "#0d1117"
plt.rcParams["axes.facecolor"]    = "#161b22"
plt.rcParams["axes.edgecolor"]    = "#30363d"
plt.rcParams["axes.labelcolor"]   = "#c9d1d9"
plt.rcParams["text.color"]        = "#c9d1d9"
plt.rcParams["xtick.color"]       = "#8b949e"
plt.rcParams["ytick.color"]       = "#8b949e"
plt.rcParams["grid.color"]        = "#21262d"
plt.rcParams["font.family"]       = "monospace"


def plot_efficient_frontier(
    mc_df: pd.DataFrame,
    portfolios: list[dict],
    user_risk: str,
    save_path: str = "outputs/efficient_frontier.png",
):
    """Scatter plot: Monte Carlo portfolios + optimized overlays."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(
        mc_df["vol"] * 100,
        mc_df["ret"] * 100,
        c=mc_df["sharpe"],
        cmap="plasma",
        alpha=0.4,
        s=8,
        label="Random portfolios",
    )
    fig.colorbar(sc, ax=ax, label="Sharpe Ratio")

    colors = ["#00ff88", "#ff6b35", "#4ecdc4", "#ffe66d", "#a8dadc"]
    for i, p in enumerate(portfolios[:5]):
        ax.scatter(
            p["volatility"] * 100,
            p["expected_return"] * 100,
            color=colors[i],
            s=120,
            zorder=5,
            marker="★",
            label=f"Portfolio {i+1} (Sharpe={p['sharpe']:.2f})",
        )

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title(f"Efficient Frontier  |  Risk={user_risk}", fontsize=14, pad=15)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_portfolio_weights(
    portfolio: dict,
    portfolio_num: int = 1,
    save_path: str = "outputs/weights.png",
):
    """Horizontal bar chart of portfolio weights."""
    weights = portfolio["weights"].sort_values(ascending=True)
    tickers = [s.replace(".NS", "") for s in weights.index]

    fig, ax = plt.subplots(figsize=(8, max(4, len(weights) * 0.4)))

    bars = ax.barh(
        tickers,
        weights.values * 100,
        color=["#00ff88" if w > 0.1 else "#4ecdc4" for w in weights.values],
        edgecolor="#30363d",
        linewidth=0.5,
    )

    for bar, val in zip(bars, weights.values * 100):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, color="#c9d1d9")

    ax.set_xlabel("Weight (%)")
    ax.set_title(
        f"Portfolio {portfolio_num} Weights  |  "
        f"Ret={portfolio['expected_return']*100:.1f}%  "
        f"Vol={portfolio['volatility']*100:.1f}%  "
        f"Sharpe={portfolio['sharpe']:.2f}",
        fontsize=11, pad=10
    )
    ax.axvline(x=100/len(weights), color="#ff6b35", linestyle="--", alpha=0.5, label="Equal weight")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    stocks: list[str],
    save_path: str = "outputs/correlation.png",
):
    """Heatmap of stock correlations."""
    ret_sub = returns[stocks]
    corr = ret_sub.corr()
    tickers = [s.replace(".NS", "") for s in corr.index]
    corr.index   = tickers
    corr.columns = tickers

    fig, ax = plt.subplots(figsize=(max(8, len(stocks) * 0.5), max(7, len(stocks) * 0.45)))
    sns.heatmap(
        corr,
        annot=len(stocks) <= 20,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        linewidths=0.3,
        linecolor="#0d1117",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Stock Correlation Matrix", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_risk_dashboard(
    portfolios: list[dict],
    allocation: pd.DataFrame,
    save_path: str = "outputs/dashboard.png",
):
    """4-panel dashboard: weights, risk contrib, metrics comparison, sector."""
    p = portfolios[0]
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Portfolio Weights (pie) ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    labels  = [s.replace(".NS", "") for s in p["weights"].index]
    sizes   = p["weights"].values
    colors  = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, pctdistance=0.8,
        textprops={"color": "#c9d1d9", "fontsize": 7},
    )
    ax1.set_title("Portfolio Weights", fontsize=11)

    # ── Panel 2: Risk Contribution ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rc = p["risk_contrib"].sort_values(ascending=False)
    rc_tickers = [s.replace(".NS", "") for s in rc.index]
    ax2.bar(range(len(rc)), rc.values * 100, color="#ff6b35", edgecolor="#30363d")
    ax2.set_xticks(range(len(rc)))
    ax2.set_xticklabels(rc_tickers, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Risk Contribution (%)")
    ax2.set_title("Risk Contribution by Stock", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Portfolio Metrics Comparison ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ["expected_return", "volatility", "sharpe", "sortino"]
    labels3 = ["Exp. Return", "Volatility", "Sharpe", "Sortino"]
    x = np.arange(len(portfolios[:5]))
    width = 0.18
    pal = ["#00ff88", "#ff6b35", "#4ecdc4", "#ffe66d"]
    for i, (m, lbl) in enumerate(zip(metrics, labels3)):
        vals = [port.get(m, 0) for port in portfolios[:5]]
        ax3.bar(x + i * width, vals, width, label=lbl, color=pal[i], alpha=0.85)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([f"P{j+1}" for j in range(len(portfolios[:5]))], fontsize=9)
    ax3.set_title("Portfolio Metrics Comparison", fontsize=11)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Sector Breakdown ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sector_wt = {}
    for stock, row in allocation.iterrows():
        sec = SECTOR_MAP_LOOKUP(stock)
        sector_wt[sec] = sector_wt.get(sec, 0) + row["weight_actual"]
    sector_s = pd.Series(sector_wt).sort_values(ascending=True)
    ax4.barh(sector_s.index, sector_s.values * 100,
             color="#4ecdc4", edgecolor="#30363d")
    ax4.set_xlabel("Allocation (%)")
    ax4.set_title("Sector Allocation", fontsize=11)
    ax4.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Portfolio Analysis Dashboard", fontsize=15, y=1.01, color="#c9d1d9")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def SECTOR_MAP_LOOKUP(stock):
    from config import SECTOR_MAP
    return SECTOR_MAP.get(stock, "Other")


def plot_goal_projection(
    projection: dict,
    goal_config,
    save_path: str = "outputs/goal_projection.png",
):
    """
    3-panel goal projection chart:
      Panel 1: Corpus growth (invested vs corpus)
      Panel 2: Annual cumulative gain bars
      Panel 3: Percentile fan (p10/p25/p50/p75/p90)
    """
    from portfolio.goals import GOAL_PROFILES
    profile = GOAL_PROFILES[goal_config.goal_type]
    yearly  = projection["yearly_projection"]
    n_years = goal_config.investment_horizon_years
    lump    = goal_config.lump_sum_amount

    years   = [r["year"]     for r in yearly]
    corpus  = [r["corpus"]   for r in yearly]
    invested= [r["invested"] for r in yearly]
    gains   = [r["gain"]     for r in yearly]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")

    # Panel 1: Corpus growth
    ax = axes[0]
    ax.fill_between(years, invested, corpus, alpha=0.25, color="#00ff88")
    ax.plot(years, corpus,    color="#00ff88", lw=2.5, label="Expected Corpus")
    ax.plot(years, invested,  color="#ff6b35", lw=1.8, linestyle="--", label="Amount Invested")
    if goal_config.target_corpus:
        ax.axhline(goal_config.target_corpus, color="#ffe66d", linestyle=":", lw=1.5,
                   label=f"Target")
    ax.set_title(f"{profile['emoji']} {profile['display_name']}\nCorpus Growth", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Rs.")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs.{x/1e5:.0f}L" if x < 1e7 else f"Rs.{x/1e7:.1f}Cr")
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cumulative gain bars
    ax2 = axes[1]
    bar_colors = ["#00ff88" if g >= 0 else "#ff6b35" for g in gains]
    ax2.bar(years, [g / 1e5 for g in gains], color=bar_colors, alpha=0.85, edgecolor="#21262d")
    ax2.set_title("Cumulative Gain Over Time", fontsize=11)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Gain (Rs. Lakh)")
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Percentile fan
    ax3 = axes[2]
    pcts = [("p10","#ff6b35"),("p25","#ffe66d"),("p50","#00ff88"),("p75","#4ecdc4"),("p90","#a8dadc")]
    for key, color in pcts:
        final = projection[key]
        vals  = [lump + (final - lump) * (yr / n_years) for yr in years]
        ax3.plot(years, [v/1e5 for v in vals], color=color, lw=1.8, label=key.upper(), alpha=0.95)
    p10_line = [lump/1e5 + (projection["p10"]-lump)/1e5 * yr/n_years for yr in years]
    p90_line = [lump/1e5 + (projection["p90"]-lump)/1e5 * yr/n_years for yr in years]
    ax3.fill_between(years, p10_line, p90_line, alpha=0.10, color="#00ff88")
    ax3.set_title("Outcome Confidence Range", fontsize=11)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Corpus (Rs. Lakh)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Goal Projection  |  {n_years}yr  |  "
        f"Exp. Corpus: Rs.{projection['total_corpus']/1e5:.0f}L  |  "
        f"{projection['wealth_multiple']:.1f}x wealth multiple",
        fontsize=12, color="#c9d1d9", y=1.02,
    )
    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {save_path}")
