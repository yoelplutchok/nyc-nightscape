"""
53_neighborhood_typologies.py — Nighttime neighborhood typology via k-means clustering.

Z-scores key nighttime variables, runs k-means for k=3..8, selects optimal k
via silhouette score, and produces typology map data + radar charts.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df, read_yaml, atomic_write_df
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# One variable per conceptual dimension to avoid collinear pairs
# dominating Euclidean distance in k-means.
# Removed: violent_rate_per_1k (subset of crime_rate), pm25_mean_primary
# (correlated with no2), subway_stations_per_km2 (correlated with entries),
# on_premises_rate_per_1k_pop (overlaps restaurants_per_1k).
CLUSTER_VARS = [
    "rate_per_1k_pop",
    "noise_obj_db_mean",
    "crime_rate_per_1k",
    "crash_rate_per_1k",
    "z_light",
    "streetlight_complaints_per_km2",
    "late_night_entries_per_1k",
    "total_ride_pickups_night_per_1k",
    "no2_mean_primary",
    "restaurants_per_1k",
    "fire_incidents_night_per_1k",
    "poverty_rate",
    "pop_density",
]

K_RANGE = range(3, 9)
TYPOLOGY_LABELS = {
    "high_noise_high_activity": "Bustling & Loud",
    "high_crime_low_transit": "Underserved & Unsafe",
    "quiet_suburban": "Quiet Suburban",
    "moderate_urban": "Moderate Urban",
    "transit_rich_low_crime": "Well-Connected & Safe",
}


def prepare_features(df: pd.DataFrame, logger) -> tuple:
    """Select and z-score clustering features."""
    available = [c for c in CLUSTER_VARS if c in df.columns]
    logger.info(f"Clustering features: {len(available)}/{len(CLUSTER_VARS)}")

    X = df[available].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=available, index=df.index,
    )
    return X_scaled, available, scaler


def find_optimal_k(X: pd.DataFrame, logger) -> dict:
    """Run k-means for k=3..8, return silhouette scores and models."""
    results = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=42, max_iter=500)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        inertia = km.inertia_
        results[k] = {"model": km, "labels": labels, "silhouette": sil, "inertia": inertia}
        logger.info(f"  k={k}: silhouette={sil:.3f}, inertia={inertia:.0f}")
    return results


def plot_silhouette(results: dict, output_path: Path):
    """Plot silhouette scores vs k."""
    ks = sorted(results.keys())
    sils = [results[k]["silhouette"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sils, "o-", color="#1a365d", linewidth=2, markersize=8)
    best_k = ks[np.argmax(sils)]
    ax.axvline(best_k, color="#ed8936", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Cluster Quality by k", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radar_charts(df: pd.DataFrame, cluster_col: str, features: list, output_path: Path):
    """Plot radar/spider charts for each cluster centroid."""
    n_clusters = df[cluster_col].nunique()
    centroids = df.groupby(cluster_col)[features].mean()

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    short_labels = [f.replace("_per_1k", "/1k").replace("_per_km2", "/km²")
                    .replace("_rate", "").replace("_mean", "").replace("primary", "")
                    .replace("_night", "")[:20]
                    for f in features]

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5),
                             subplot_kw={"projection": "polar"})
    if n_clusters == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    for idx, (cluster_id, row) in enumerate(centroids.iterrows()):
        ax = axes[idx]
        values = row.values.tolist()
        values += values[:1]

        ax.plot(angles, values, "o-", color=colors[idx], linewidth=2)
        ax.fill(angles, values, color=colors[idx], alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_labels, fontsize=6)
        n_cds = (df[cluster_col] == cluster_id).sum()
        ax.set_title(f"Type {cluster_id + 1}\n({n_cds} CDs)", fontsize=11, fontweight="bold")

    fig.suptitle("Nighttime Neighborhood Typologies (z-scored centroids)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def label_clusters(df: pd.DataFrame, features: list, cluster_col: str) -> pd.Series:
    """Assign human-readable labels based on centroid characteristics.

    Each cluster gets a unique label — if two clusters would get the same label
    from the heuristic rules, the second is disambiguated with a suffix.
    """
    centroids = df.groupby(cluster_col)[features].mean()
    raw_labels = {}

    for cid, row in centroids.iterrows():
        crime = row.get("crime_rate_per_1k", 0)
        transit = row.get("late_night_entries_per_1k", 0)
        noise = row.get("rate_per_1k_pop", 0)
        restaurants = row.get("restaurants_per_1k", 0)
        poverty = row.get("poverty_rate", 0)
        density = row.get("pop_density", 0)

        if noise > 0.5 and restaurants > 0.5:
            raw_labels[cid] = "Bustling & Loud"
        elif crime > 0.5 and transit < -0.3:
            raw_labels[cid] = "Underserved & Unsafe"
        elif density < -0.5 and crime < -0.3:
            raw_labels[cid] = "Quiet Suburban"
        elif transit > 0.5 and crime < 0:
            raw_labels[cid] = "Well-Connected & Safe"
        elif poverty > 0.3:
            raw_labels[cid] = "High-Need Urban"
        elif density > 0:
            raw_labels[cid] = "Moderate Urban"
        else:
            raw_labels[cid] = "Mixed Residential"

    # Disambiguate duplicates
    seen = {}
    labels = {}
    for cid in sorted(raw_labels):
        base = raw_labels[cid]
        if base in seen:
            seen[base] += 1
            labels[cid] = f"{base} ({seen[base]})"
        else:
            seen[base] = 1
            labels[cid] = base

    return df[cluster_col].map(labels)


def main():
    with get_logger("53_neighborhood_typologies") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        seed = params.get("analysis", {}).get("random_seed", 42)
        df = read_df(MASTER_PATH)
        logger.info(f"Master: {df.shape}")

        X_scaled, features, scaler = prepare_features(df, logger)
        results = find_optimal_k(X_scaled, logger)

        best_k = max(results, key=lambda k: results[k]["silhouette"])
        logger.info(f"Optimal k={best_k} (silhouette={results[best_k]['silhouette']:.3f})")

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_silhouette(results, FIGURES_DIR / "cluster_silhouette.png")

        best_labels = results[best_k]["labels"]
        df["cluster"] = best_labels
        X_scaled["cluster"] = best_labels

        # Warn about small clusters
        cluster_sizes = pd.Series(best_labels).value_counts()
        small_clusters = cluster_sizes[cluster_sizes < 5]
        if len(small_clusters) > 0:
            logger.warning(f"Small clusters (<5 CDs): {dict(small_clusters)} — interpret with caution")

        type_labels = label_clusters(X_scaled, features, "cluster")
        df["typology"] = type_labels

        plot_radar_charts(X_scaled, "cluster", features,
                         FIGURES_DIR / "typology_radar_charts.png")

        # Save all k results for sensitivity analysis
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        sens_rows = []
        for k, res in results.items():
            sens_rows.append({"k": k, "silhouette": res["silhouette"], "inertia": res["inertia"]})
        pd.DataFrame(sens_rows).to_csv(TABLES_DIR / "cluster_sensitivity.csv", index=False)

        # Save typology assignments
        typology_df = df[["boro_cd", "cluster", "typology"]].copy()
        if "borough" in df.columns:
            typology_df["borough"] = df["borough"]
        if "cd_name" in df.columns:
            typology_df["cd_name"] = df["cd_name"]
        typology_df.to_csv(TABLES_DIR / "neighborhood_typologies.csv", index=False)

        # Cluster profile summary (group by cluster to preserve all groups)
        profile = df.groupby("cluster")[features + ["population", "poverty_rate"]].mean()
        profile["n_cds"] = df.groupby("cluster").size()
        profile["typology"] = df.groupby("cluster")["typology"].first()
        profile.to_csv(TABLES_DIR / "typology_profiles.csv")

        logger.log_metrics({
            "optimal_k": best_k,
            "silhouette_score": results[best_k]["silhouette"],
            "features_used": len(features),
        })

        print(f"\nNeighborhood Typologies:")
        print(f"  Features: {len(features)}")
        print(f"  Optimal k: {best_k} (silhouette={results[best_k]['silhouette']:.3f})")
        print(f"\n  Typology distribution:")
        for t, count in df["typology"].value_counts().items():
            print(f"    {t:30s}: {count} CDs")
        print(f"\n  Outputs: {TABLES_DIR}, {FIGURES_DIR}")


if __name__ == "__main__":
    main()
