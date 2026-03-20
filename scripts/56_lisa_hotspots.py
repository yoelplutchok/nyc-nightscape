"""
56_lisa_hotspots.py — LISA (Local Moran's I) spatial autocorrelation analysis.

Computes Local Moran's I for the Nightscape Index and key individual dimensions.
Maps hot spots (high-high), cold spots (low-low), and spatial outliers.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df, read_yaml
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR, GEO_DIR

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

LISA_VARIABLES = [
    "nightscape_index",
    "dim_noise_exposure",
    "dim_safety",
    "dim_transit_access",
    "dim_environmental",
]


def build_spatial_weights(gdf: gpd.GeoDataFrame):
    """Build queen contiguity weights from a GeoDataFrame."""
    try:
        from libpysal.weights import Queen
        w = Queen.from_dataframe(gdf, use_index=False)
        w.transform = "r"
        # Check for island units (no neighbors)
        islands = [i for i in w.neighbors if len(w.neighbors[i]) == 0]
        if islands:
            import logging
            logging.getLogger("nightscape").warning(
                f"Island units with no neighbors (will always be NS): {islands}"
            )
        return w
    except ImportError:
        import logging
        logging.getLogger("nightscape").warning(
            "libpysal not available — falling back to KNN (k=6). "
            "This gives a fundamentally different spatial structure than queen contiguity."
        )
        from sklearn.neighbors import BallTree
        centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        tree = BallTree(np.radians(centroids), metric="haversine")
        k = 6
        dists, indices = tree.query(np.radians(centroids), k=k + 1)

        neighbors = {}
        weights = {}
        for i in range(len(gdf)):
            nbrs = [int(j) for j in indices[i, 1:]]
            neighbors[i] = nbrs
            weights[i] = [1.0 / k] * k

        class SimpleWeights:
            def __init__(self, neighbors, weights, n):
                self.neighbors = neighbors
                self.weights = weights
                self.n = n
                self.transform = "r"
        return SimpleWeights(neighbors, weights, len(gdf))


def compute_local_morans(gdf: gpd.GeoDataFrame, variable: str, w, n_permutations: int = 999, alpha: float = 0.05):
    """Compute Local Moran's I using PySAL or manual implementation."""
    y = gdf[variable].values
    y_std = (y - y.mean()) / y.std()
    n = len(y)

    local_i = np.zeros(n)
    for i in range(n):
        nbrs = w.neighbors[i]
        wts = w.weights[i]
        lag = sum(wt * y_std[j] for j, wt in zip(nbrs, wts))
        local_i[i] = y_std[i] * lag

    # Permutation-based p-values (conditional randomization: hold focal value fixed)
    rng = np.random.default_rng(42)
    sim_Is = np.zeros((n, n_permutations))
    for perm in range(n_permutations):
        for i in range(n):
            # Conditional randomization: shuffle all values except the focal unit
            others = np.delete(y_std, i)
            rng.shuffle(others)
            perm_y = np.insert(others, i, y_std[i])
            nbrs = w.neighbors[i]
            wts = w.weights[i]
            lag = sum(wt * perm_y[j] for j, wt in zip(nbrs, wts))
            sim_Is[i, perm] = y_std[i] * lag

    p_values = np.zeros(n)
    for i in range(n):
        p_values[i] = (np.sum(np.abs(sim_Is[i]) >= np.abs(local_i[i])) + 1) / (n_permutations + 1)

    # Classify: HH, HL, LH, LL, or NS (not significant)
    lag_values = np.zeros(n)
    for i in range(n):
        nbrs = w.neighbors[i]
        wts = w.weights[i]
        lag_values[i] = sum(wt * y_std[j] for j, wt in zip(nbrs, wts))

    cluster = np.full(n, "NS", dtype=object)
    # NOTE: LISA p-values are unadjusted (no FDR correction). This is standard
    # practice in spatial autocorrelation analysis.
    sig_mask = p_values < alpha
    cluster[(sig_mask) & (y_std > 0) & (lag_values > 0)] = "HH"
    cluster[(sig_mask) & (y_std < 0) & (lag_values < 0)] = "LL"
    cluster[(sig_mask) & (y_std > 0) & (lag_values < 0)] = "HL"
    cluster[(sig_mask) & (y_std < 0) & (lag_values > 0)] = "LH"

    return local_i, p_values, cluster


def plot_lisa_map(gdf: gpd.GeoDataFrame, cluster_col: str, title: str, output_path: Path):
    """Plot LISA cluster map."""
    color_map = {"HH": "#d73027", "LL": "#4575b4", "HL": "#fee090", "LH": "#abd9e9", "NS": "#f0f0f0"}
    label_map = {"HH": "Hot Spot (High-High)", "LL": "Cold Spot (Low-Low)",
                 "HL": "Spatial Outlier (High-Low)", "LH": "Spatial Outlier (Low-High)",
                 "NS": "Not Significant"}

    gdf = gdf.copy()
    gdf["color"] = gdf[cluster_col].map(color_map)

    fig, ax = plt.subplots(figsize=(12, 14))
    gdf.plot(ax=ax, color=gdf["color"], edgecolor="white", linewidth=0.5)

    from matplotlib.patches import Patch
    legend_elements = []
    for code in ["HH", "LL", "HL", "LH", "NS"]:
        if code in gdf[cluster_col].values:
            count = (gdf[cluster_col] == code).sum()
            legend_elements.append(
                Patch(facecolor=color_map[code], edgecolor="gray",
                      label=f"{label_map[code]} ({count})")
            )

    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    with get_logger("56_lisa_hotspots") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        df = read_df(MASTER_PATH)
        logger.info(f"Master: {df.shape}")

        # Load nightscape index
        index_path = TABLES_DIR / "nightscape_index.csv"
        if not index_path.exists():
            logger.error("Run 54_composite_index.py first")
            return
        index_df = pd.read_csv(index_path)
        df = df.merge(
            index_df[["boro_cd"] + [c for c in index_df.columns if c in LISA_VARIABLES or c.startswith("dim_")]],
            on="boro_cd", how="left",
        )

        gdf = gpd.read_file(GEO_DIR / "cd59.geojson")
        gdf["boro_cd"] = gdf["boro_cd"].astype(int)
        gdf = gdf.merge(df, on="boro_cd", how="left")
        logger.info(f"GeoDataFrame: {len(gdf)} features")

        w = build_spatial_weights(gdf)
        logger.info("Spatial weights built")

        n_permutations = params.get("analysis", {}).get("lisa_permutations", 999)
        lisa_alpha = params.get("analysis", {}).get("fdr_alpha", 0.05)
        logger.info(f"LISA config: {n_permutations} permutations, alpha={lisa_alpha}")

        all_results = []
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        for var in LISA_VARIABLES:
            if var not in gdf.columns:
                logger.warning(f"Skipping {var} — not in dataframe")
                continue
            if gdf[var].isna().all():
                logger.warning(f"Skipping {var} — all NaN")
                continue

            gdf_clean = gdf.dropna(subset=[var]).copy()
            if len(gdf_clean) < 20:
                logger.warning(f"Skipping {var} — too few observations ({len(gdf_clean)})")
                continue

            w_clean = build_spatial_weights(gdf_clean)
            local_i, p_vals, clusters = compute_local_morans(
                gdf_clean, var, w_clean,
                n_permutations=n_permutations, alpha=lisa_alpha,
            )

            cluster_col = f"lisa_{var}"
            gdf_clean[cluster_col] = clusters
            gdf_clean[f"morans_i_{var}"] = local_i
            gdf_clean[f"p_value_{var}"] = p_vals

            n_hh = (clusters == "HH").sum()
            n_ll = (clusters == "LL").sum()
            n_hl = (clusters == "HL").sum()
            n_lh = (clusters == "LH").sum()
            n_sig = n_hh + n_ll + n_hl + n_lh

            logger.info(f"  {var}: HH={n_hh}, LL={n_ll}, HL={n_hl}, LH={n_lh}, NS={len(clusters) - n_sig}")

            # Global Moran's I = mean of local I values (weighted by row-standardized W)
            global_morans_I = local_i.mean()
            all_results.append({
                "variable": var,
                "n_obs": len(gdf_clean),
                "hot_spots_HH": n_hh,
                "cold_spots_LL": n_ll,
                "outliers_HL": n_hl,
                "outliers_LH": n_lh,
                "not_significant": len(clusters) - n_sig,
                "global_morans_I": global_morans_I,
            })

            var_label = var.replace("dim_", "").replace("_", " ").title()
            plot_lisa_map(
                gdf_clean, cluster_col,
                f"LISA Clusters: {var_label}",
                FIGURES_DIR / f"lisa_{var}.png",
            )

            # Save per-CD LISA results
            per_cd = gdf_clean[["boro_cd", cluster_col, f"morans_i_{var}", f"p_value_{var}"]].copy()
            per_cd.to_csv(TABLES_DIR / f"lisa_{var}_per_cd.csv", index=False)

            # Propagate LISA results back to main gdf for the nightscape index
            if var == "nightscape_index":
                gdf = gdf.merge(
                    per_cd[["boro_cd", cluster_col]],
                    on="boro_cd", how="left",
                )

        # Save summary
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(TABLES_DIR / "lisa_summary.csv", index=False)

        # Save detailed per-CD clusters for the nightscape index
        if "lisa_nightscape_index" in gdf.columns:
            lisa_detail = gdf[["boro_cd", "lisa_nightscape_index"]].copy()
            lisa_detail.to_csv(TABLES_DIR / "lisa_nightscape_index_clusters.csv", index=False)

        logger.log_metrics({
            "variables_analyzed": len(all_results),
            "nightscape_hot_spots": all_results[0]["hot_spots_HH"] if all_results else 0,
            "nightscape_cold_spots": all_results[0]["cold_spots_LL"] if all_results else 0,
        })

        print(f"\nLISA Spatial Autocorrelation:")
        for r in all_results:
            var_label = r["variable"].replace("dim_", "").replace("_", " ").title()
            print(f"  {var_label}:")
            print(f"    Hot spots (HH): {r['hot_spots_HH']}, Cold spots (LL): {r['cold_spots_LL']}")
            print(f"    Outliers: HL={r['outliers_HL']}, LH={r['outliers_LH']}")
        print(f"\n  Outputs: {TABLES_DIR}, {FIGURES_DIR}")


if __name__ == "__main__":
    main()
