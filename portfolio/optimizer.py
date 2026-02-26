# portfolio/optimizer.py — Steps 7-9: Diversification, Generation, Optimization, Ranking

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

import sys, os
from config import (
    MAX_WEIGHT, MIN_WEIGHT, MAX_SECTOR_WEIGHT, MIN_STOCKS, MAX_STOCKS,
    N_PORTFOLIOS, N_TOP_PORTFOLIOS, SECTOR_MAP, RISKFREE_RATE, RANDOM_STATE
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: DIVERSIFICATION ENGINE — Correlation Clusters
# ─────────────────────────────────────────────────────────────────────────────

def build_correlation_clusters(
    returns: pd.DataFrame,
    n_clusters: int = None,
) -> dict[int, list[str]]:
    """
    Hierarchical clustering on correlation distance matrix.
    Ensures we don't over-concentrate in correlated stocks.

    Returns dict: {cluster_id: [stock1, stock2, ...]}
    """
    corr = returns.corr()
    # Distance = 1 - |correlation|  (correlated stocks are "close")
    dist = (1 - corr.abs()).clip(0, 2)
    dist_arr = dist.values.copy()
    np.fill_diagonal(dist_arr, 0)

    condensed = squareform(dist_arr, checks=False)
    Z = linkage(condensed, method="ward")

    if n_clusters is None:
        # Auto-determine: roughly 1 cluster per 4-5 stocks
        n_clusters = max(3, len(returns.columns) // 4)

    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {}
    for stock, label in zip(returns.columns, cluster_labels):
        clusters.setdefault(int(label), []).append(stock)

    print(f"\n  Correlation clusters ({n_clusters}):")
    for cid, members in sorted(clusters.items()):
        print(f"  C{cid}: {members}")

    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8A: CANDIDATE PORTFOLIO GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_sets(
    stocks: list[str],
    corr_clusters: dict,
    mu_hat: pd.Series,
    n_stocks_per_portfolio: int = 12,
    n_candidates: int = 30,
    seed: int = RANDOM_STATE,
) -> list[list[str]]:
    """
    Generate diverse candidate stock baskets by:
    1. Picking 1-2 stocks per correlation cluster
    2. Preferring high expected-return stocks (softly)
    3. Enforcing minimum coverage of clusters
    """
    rng = np.random.default_rng(seed)
    candidate_sets = []
    seen = set()
    max_attempts = n_candidates * 100

    cluster_stocks = {
        cid: [s for s in members if s in stocks]
        for cid, members in corr_clusters.items()
    }
    cluster_stocks = {k: v for k, v in cluster_stocks.items() if v}
    cluster_ids = list(cluster_stocks.keys())

    for _ in range(max_attempts):
        if len(candidate_sets) >= n_candidates:
            break

        selected = []

        # ── Step 1: pick 1-2 from each cluster ────────────────────────────
        for cid in cluster_ids:
            pool = cluster_stocks[cid]
            if not pool:
                continue
            # Weight selection by predicted return (softmax)
            mu_pool = np.array([mu_hat.get(s, 0.0) for s in pool])
            mu_pool = mu_pool - mu_pool.min()
            probs   = np.exp(mu_pool) / np.exp(mu_pool).sum()
            n_pick  = rng.choice([1, 2], p=[0.6, 0.4])
            n_pick  = min(n_pick, len(pool))
            picks   = rng.choice(pool, size=n_pick, replace=False, p=probs)
            selected.extend(picks)

        # ── Step 2: trim or pad to target size ────────────────────────────
        selected = list(set(selected))
        if len(selected) > n_stocks_per_portfolio:
            # Keep top by mu_hat
            selected = sorted(selected, key=lambda s: mu_hat.get(s, 0), reverse=True)
            selected = selected[:n_stocks_per_portfolio]
        elif len(selected) < MIN_STOCKS:
            continue  # too few, try again

        key = frozenset(selected)
        if key not in seen:
            seen.add(key)
            candidate_sets.append(sorted(selected))

    print(f"\n  Generated {len(candidate_sets)} candidate stock sets")
    return candidate_sets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8B: WEIGHT OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def optimize_weights(
    stocks: list[str],
    mu: pd.Series,
    Sigma: pd.DataFrame,
    user_risk: str,
    close: pd.DataFrame = None,
) -> dict:
    """
    Compute portfolio weights based on user risk appetite.

    Low    → Minimum Variance (minimize vol)
    Medium → Maximum Sharpe (maximize Sharpe ratio)
    High   → Maximum Return / Momentum tilt

    Returns dict with weights + metrics.
    """
    mu_sub    = mu[stocks].values
    Sigma_sub = Sigma.loc[stocks, stocks].values
    n         = len(stocks)
    bounds    = [(MIN_WEIGHT, MAX_WEIGHT)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # ── Sector constraint ─────────────────────────────────────────────────
    sector_groups = {}
    for s in stocks:
        sec = SECTOR_MAP.get(s, "Other")
        sector_groups.setdefault(sec, []).append(stocks.index(s))

    for sec, idxs in sector_groups.items():
        if len(idxs) > 1:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, i=idxs: MAX_SECTOR_WEIGHT - sum(w[j] for j in i),
            })

    w0 = np.ones(n) / n  # equal weight start

    if user_risk == "Low":
        # Minimize portfolio variance
        def objective(w):
            return w @ Sigma_sub @ w
    elif user_risk == "Medium":
        # Maximize Sharpe ratio
        rf = RISKFREE_RATE / 252
        def objective(w):
            port_ret = w @ mu_sub          # mu_sub is already annualised (252d horizon)
            port_vol = np.sqrt(w @ Sigma_sub @ w) * np.sqrt(252)  # annualise vol
            sharpe   = (port_ret - RISKFREE_RATE) / (port_vol + 1e-8)
            return -sharpe
    else:
        # Maximize return with volatility penalty
        def objective(w):
            port_ret = w @ mu_sub
            port_vol = np.sqrt(w @ Sigma_sub @ w)
            return -(port_ret - 0.3 * port_vol)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not res.success:
        # Fallback: equal weights
        w_opt = np.ones(n) / n
    else:
        w_opt = res.x
        w_opt = np.clip(w_opt, 0, MAX_WEIGHT)
        w_opt /= w_opt.sum()

    w_series = pd.Series(w_opt, index=stocks)
    return compute_portfolio_metrics(w_series, mu, Sigma)


def hrp_weights(
    stocks: list[str],
    Sigma: pd.DataFrame,
) -> pd.Series:
    """
    Hierarchical Risk Parity — for Low risk portfolios.
    Allocates inversely proportional to hierarchical cluster volatility.
    No mean estimates needed (robust to estimation error).
    """
    cov   = Sigma.loc[stocks, stocks]
    corr  = pd.DataFrame(
        np.diag(1 / np.sqrt(np.diag(cov.values))) @ cov.values @ np.diag(1 / np.sqrt(np.diag(cov.values))),
        index=stocks, columns=stocks,
    )

    dist  = np.sqrt((1 - corr) / 2)
    dist  = dist.fillna(0)
    link  = linkage(squareform(dist.values, checks=False), method="single")
    order = _get_quasi_diag(link, len(stocks))
    order_stocks = [stocks[i] for i in order]

    weights = _hrp_recursive_bisection(cov, order_stocks)
    return pd.Series(weights, name="weight")


def _get_quasi_diag(link, n):
    """Recursively sort leaf nodes for HRP."""
    sorted_items = [int(link[-1, 0]), int(link[-1, 1])]
    num_items = link.shape[0] + 1
    while max(sorted_items) >= num_items:
        idx = sorted_items.index(max(sorted_items))
        item = sorted_items.pop(idx)
        i = int(link[item - num_items, 0])
        j = int(link[item - num_items, 1])
        sorted_items.insert(idx, j)
        sorted_items.insert(idx, i)
    return sorted_items


def _hrp_recursive_bisection(cov, sorted_stocks):
    w = pd.Series(1.0, index=sorted_stocks)
    clusters = [sorted_stocks]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c)//2), (len(c)//2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0, c1 = clusters[i], clusters[i+1]
            var0 = _cluster_var(cov, c0)
            var1 = _cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1 + 1e-8)
            w[c0] *= alpha
            w[c1] *= (1 - alpha)
    return w / w.sum()


def _cluster_var(cov, cluster):
    cov_sub = cov.loc[cluster, cluster].values
    n = len(cluster)
    ivp = 1 / np.diag(cov_sub)
    ivp /= ivp.sum()
    return float(ivp @ cov_sub @ ivp)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: pd.Series,
    mu: pd.Series,
    Sigma: pd.DataFrame,
) -> dict:
    """Compute all portfolio-level metrics."""
    stocks = weights.index.tolist()
    w = weights.values
    mu_sub    = mu[stocks].values
    Sigma_sub = Sigma.loc[stocks, stocks].values

    port_ret   = float(w @ mu_sub)                          # annualised return
    port_var   = float(w @ Sigma_sub @ w)
    port_vol   = float(np.sqrt(port_var)) * np.sqrt(252)    # annualise volatility
    sharpe     = (port_ret - RISKFREE_RATE) / (port_vol + 1e-8)

    # Sortino
    # Downside deviation: penalise returns below risk-free rate
    neg_excess = np.minimum(mu_sub - RISKFREE_RATE, 0)   # annual scale
    downside_var = float(w @ np.diag(neg_excess ** 2) @ w)
    sortino = (port_ret - RISKFREE_RATE) / (np.sqrt(downside_var) + 1e-8)

    # Risk contribution per stock
    marginal_risk = Sigma_sub @ w
    risk_contrib  = w * marginal_risk / (port_vol + 1e-8)
    risk_contrib_pct = pd.Series(risk_contrib / risk_contrib.sum(), index=stocks)

    # Max drawdown estimate (rough: 2-sigma monthly)
    # est_max_dd = -2.33 * port_vol / np.sqrt(12)

    # After (port_vol is now already annualised):
    est_max_dd = -2.33 * (port_vol / np.sqrt(252)) * np.sqrt(252/12)
    # Simplified:
    est_max_dd = -2.33 * port_vol / np.sqrt(12)   # keep same, it's fine

    return {
        "weights":          pd.Series(w, index=stocks),
        "expected_return":  port_ret,
        "volatility":       port_vol,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "max_drawdown_est": est_max_dd,
        "risk_contrib":     risk_contrib_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8C: MONTE CARLO EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_portfolios(
    stocks: list[str],
    mu: pd.Series,
    Sigma: pd.DataFrame,
    n_sims: int = N_PORTFOLIOS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Randomly generate N portfolios and compute metrics.
    Used to find the efficient frontier boundary.
    Returns DataFrame with one row per simulated portfolio.
    """
    rng = np.random.default_rng(seed)
    n   = len(stocks)
    results = []

    mu_sub    = mu[stocks].values
    Sigma_sub = Sigma.loc[stocks, stocks].values

    for _ in range(n_sims):
        # Dirichlet random weights (sum to 1, all positive)
        w = rng.dirichlet(np.ones(n))
        w = np.clip(w, 0, MAX_WEIGHT)
        w /= w.sum()

        ret  = float(w @ mu_sub)
        vol  = float(np.sqrt(w @ Sigma_sub @ w))
        shp  = (ret - RISKFREE_RATE) / (vol + 1e-8)
        results.append({"ret": ret, "vol": vol, "sharpe": shp, "weights": w.tolist()})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: FULL PORTFOLIO BUILDER + RANKING
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolios(
    candidate_stocks: list[str],
    mu_hat: pd.Series,
    Sigma: pd.DataFrame,
    returns: pd.DataFrame,
    user_risk: str,
    close: pd.DataFrame = None,
) -> list[dict]:
    """
    Master function: takes candidate stocks, runs full optimization pipeline,
    returns top N_TOP_PORTFOLIOS ranked portfolios.
    """
    print(f"\n{'='*60}")
    print(f"  Portfolio Builder  (risk={user_risk}, stocks={len(candidate_stocks)})")
    print(f"{'='*60}")

    # ── 7. Correlation clusters ───────────────────────────────────────────
    ret_sub = returns[candidate_stocks]
    corr_clusters = build_correlation_clusters(ret_sub)

    # ── 8A. Generate candidate sets ───────────────────────────────────────
    candidate_sets = generate_candidate_sets(
        stocks=candidate_stocks,
        corr_clusters=corr_clusters,
        mu_hat=mu_hat,
        n_stocks_per_portfolio=min(15, len(candidate_stocks)),
        n_candidates=50,
    )

    # ── 8B. Optimize weights for each candidate ───────────────────────────
    portfolios = []
    print(f"\n  Optimizing {len(candidate_sets)} candidate portfolios…")

    for i, stock_set in enumerate(candidate_sets):
        try:
            if user_risk == "Low":
                # Use HRP for low risk — more robust
                w = hrp_weights(stock_set, Sigma)
                result = compute_portfolio_metrics(w, mu_hat, Sigma)
            else:
                result = optimize_weights(stock_set, mu_hat, Sigma, user_risk, close)
            result["stocks"] = stock_set
            portfolios.append(result)
        except Exception as e:
            pass  # skip failed optimizations silently

    # ── 8C. Monte Carlo top-up ────────────────────────────────────────────
    mc = monte_carlo_portfolios(candidate_stocks, mu_hat, Sigma, n_sims=N_PORTFOLIOS)
    print(f"  Monte Carlo: {len(mc)} random portfolios evaluated")

    # ── 9. Rank and select ────────────────────────────────────────────────
    portfolios = _rank_portfolios(portfolios, user_risk)
    top = portfolios[:N_TOP_PORTFOLIOS]

    print(f"\n  ✓ Top {len(top)} portfolios selected")
    _print_summary(top)
    return top


def _rank_portfolios(portfolios: list[dict], user_risk: str) -> list[dict]:
    """Score and rank portfolios based on user risk appetite."""
    for p in portfolios:
        if user_risk == "Low":
            # Minimize drawdown + vol, penalize high beta
            p["score"] = (
                -p["volatility"]
                - abs(p["max_drawdown_est"])
                + 0.5 * p["sortino"]
            )
        elif user_risk == "Medium":
            # Sharpe + diversification (low avg weight concentration)
            hhi = (p["weights"] ** 2).sum()  # Herfindahl index (lower = more diverse)
            p["score"] = p["sharpe"] - 2 * hhi
        else:
            # Max return with drawdown guardrail
            drawdown_penalty = max(0, abs(p["max_drawdown_est"]) - 0.25)
            p["score"] = p["expected_return"] + 0.5 * p["sharpe"] - 3 * drawdown_penalty

    return sorted(portfolios, key=lambda p: p["score"], reverse=True)


def _print_summary(portfolios: list[dict]):
    print(f"\n  {'─'*56}")
    print(f"  {'#':<3} {'Stocks':<5} {'Ret%':>6} {'Vol%':>6} {'Sharpe':>7} {'Score':>7}")
    print(f"  {'─'*56}")
    for i, p in enumerate(portfolios):
        print(
            f"  {i+1:<3} {len(p['stocks']):<5} "
            f"{p['expected_return']*100:>5.1f}% "
            f"{p['volatility']*100:>5.1f}% "
            f"{p['sharpe']:>7.2f} "
            f"{p.get('score', 0):>7.3f}"
        )
    print(f"  {'─'*56}")
