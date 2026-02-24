# models/risk_clustering.py — Step 4: ML Risk Bucket Assignment

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import sys
from config import N_RISK_CLUSTERS, RANDOM_STATE

# Risk features to cluster on (ignore return/momentum — just pure risk)
RISK_FEATURE_COLS = [
    "vol_21d", "vol_63d", "beta", "max_drawdown",
    "downside_dev", "var_95", "cvar_95", "corr_market",
]


def assign_risk_clusters(
    feature_table: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: int = N_RISK_CLUSTERS,
    save_path: str = "models/risk_model.pkl",
) -> pd.Series:
    """
    Cluster stocks into Low / Medium / High risk buckets.

    Returns a Series: stock → risk_label ("Low" | "Medium" | "High")
    """
    print(f"\n{'─'*60}")
    print(f"  Risk Clustering  (method={method}, k={n_clusters})")
    print(f"{'─'*60}")

    # ── 1. Select and scale risk features ─────────────────────────────────
    cols_available = [c for c in RISK_FEATURE_COLS if c in feature_table.columns]
    X = feature_table[cols_available].copy().fillna(feature_table[cols_available].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. Optional PCA for visualization ────────────────────────────────
    pca = PCA(n_components=min(2, X_scaled.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # ── 3. Fit clustering model ───────────────────────────────────────────
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)

    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)

    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_scaled)

    else:
        raise ValueError(f"Unknown method: {method}. Choose: kmeans | gmm | hierarchical")

    # ── 4. Silhouette score ───────────────────────────────────────────────
    try:
        sil = silhouette_score(X_scaled, labels)
        print(f"  Silhouette score: {sil:.3f}")
    except Exception:
        pass

    # ── 5. Map cluster IDs to Low/Med/High by avg volatility ──────────────
    #   The cluster with highest avg volatility → High, etc.
    cluster_df = pd.DataFrame({
        "stock":   feature_table.index,
        "cluster": labels,
        "avg_vol": feature_table.get("vol_63d", feature_table.iloc[:, 0]).values,
    })
    cluster_avg_vol = cluster_df.groupby("cluster")["avg_vol"].mean().sort_values()
    vol_rank = {cid: rank for rank, cid in enumerate(cluster_avg_vol.index)}

    RISK_MAP = {0: "Low", 1: "Medium", 2: "High"}
    cluster_df["risk_label"] = cluster_df["cluster"].map(
        lambda c: RISK_MAP.get(vol_rank.get(c, 0), "Medium")
    )

    risk_series = cluster_df.set_index("stock")["risk_label"]

    # ── 6. Print summary ──────────────────────────────────────────────────
    print("\n  Risk Bucket Assignments:")
    for label in ["Low", "Medium", "High"]:
        stocks = risk_series[risk_series == label].index.tolist()
        print(f"  {label:8s} ({len(stocks):2d} stocks): {stocks}")

    # ── 7. Save model ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "pca": pca, "risk_map": vol_rank}, save_path)
    print(f"\n  ✓ Model saved → {save_path}")

    return risk_series


def get_risk_subset(
    risk_series: pd.Series,
    user_risk: str,
) -> list[str]:
    """
    Given user risk appetite ('Low'|'Medium'|'High'),
    return the appropriate stock subset.

    - Low → only Low risk stocks
    - Medium → Low + Medium
    - High → all stocks (Low + Med + High)
    """
    if user_risk == "Low":
        mask = risk_series == "Low"
    elif user_risk == "Medium":
        mask = risk_series.isin(["Low", "Medium"])
    else:  # High
        mask = risk_series.isin(["Low", "Medium", "High"])

    subset = risk_series[mask].index.tolist()
    print(f"\n  User risk='{user_risk}' → {len(subset)} candidate stocks")
    return subset
