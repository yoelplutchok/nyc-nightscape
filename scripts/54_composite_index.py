"""
54_composite_index.py — Compute the Composite Nightscape Index.

Weighted index combining noise, safety, lighting (inverted), transit (inverted),
environmental, and services (inverted). Higher score = worse nighttime conditions.
Normalized 0-100. Sensitivity check with equal weights.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df, read_yaml, atomic_write_df
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Dimension definitions: (variable, direction)
# direction = "higher_worse" means higher values = worse conditions
# direction = "higher_better" means higher values = better (will be inverted)
DIMENSIONS = {
    "noise_exposure": {
        "weight": 0.20,
        "variables": [
            ("rate_per_1k_pop", "higher_worse"),
            ("noise_obj_db_mean", "higher_worse"),
            ("dot_pct_above_65dB", "higher_worse"),
            ("helicopter_complaints_night_per_1k", "higher_worse"),
        ],
    },
    "safety": {
        "weight": 0.25,
        "variables": [
            ("crime_rate_per_1k", "higher_worse"),
            # violent_rate_per_1k dropped: subset of crime_rate (rho=0.97)
            ("crash_rate_per_1k", "higher_worse"),
            # injury_rate_per_1k dropped: subset of crash_rate (rho=0.93)
        ],
    },
    "lighting": {
        "weight": 0.10,
        "variables": [
            ("z_light", "higher_better"),
            ("streetlight_complaints_per_km2", "higher_worse"),
        ],
    },
    "transit_access": {
        "weight": 0.15,
        "variables": [
            ("late_night_entries_per_1k", "higher_better"),
            ("subway_stations_per_km2", "higher_better"),
            ("mean_trains_per_hour_late_night", "higher_better"),
            ("citibike_trips_night_per_1k", "higher_better"),
        ],
    },
    "environmental": {
        "weight": 0.15,
        "variables": [
            ("no2_mean_primary", "higher_worse"),
            # pm25_mean_primary dropped: highly correlated with no2 (rho=0.93)
            ("z_heat", "higher_worse"),
        ],
    },
    "services": {
        "weight": 0.15,
        "variables": [
            ("restaurants_per_1k", "higher_better"),
            ("wifi_hotspots_per_km2", "higher_better"),
            ("ems_response_min_night", "higher_worse"),
        ],
    },
}


def rank_normalize(series: pd.Series) -> pd.Series:
    """Convert to percentile ranks (0-1), handling NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=series.index)
    ranks = rankdata(valid, method="average")
    pct = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.full(len(ranks), 0.5)
    result = pd.Series(np.nan, index=series.index)
    result.loc[valid.index] = pct
    return result


def compute_dimension_score(df: pd.DataFrame, dim_config: dict) -> pd.Series:
    """Compute a single dimension score (0-1, higher = worse)."""
    components = []
    for var, direction in dim_config["variables"]:
        if var not in df.columns:
            continue
        ranked = rank_normalize(df[var])
        if direction == "higher_better":
            ranked = 1 - ranked
        components.append(ranked)

    if not components:
        return pd.Series(np.nan, index=df.index)

    stacked = pd.concat(components, axis=1)
    return stacked.mean(axis=1)


def compute_composite_index(df: pd.DataFrame, weights: dict, logger) -> pd.DataFrame:
    """Compute the full composite Nightscape Index."""
    result = df[["boro_cd"]].copy()

    dim_scores = {}
    for dim_name, dim_config in DIMENSIONS.items():
        score = compute_dimension_score(df, dim_config)
        dim_scores[dim_name] = score
        result[f"dim_{dim_name}"] = score
        n_vars = sum(1 for v, _ in dim_config["variables"] if v in df.columns)
        logger.info(f"  {dim_name}: {n_vars} variables, mean={score.mean():.3f}")

    total_weight = sum(weights.get(d, DIMENSIONS[d]["weight"]) for d in DIMENSIONS)
    dim_frame = pd.DataFrame(dim_scores)
    weight_series = pd.Series(
        {d: weights.get(d, DIMENSIONS[d]["weight"]) / total_weight for d in DIMENSIONS}
    )
    # Weighted mean, skipping NaN dimensions per CD so one missing dimension
    # doesn't null out the entire index
    composite = dim_frame.apply(
        lambda row: (
            np.average(row.dropna(), weights=weight_series[row.dropna().index])
            if row.notna().any() else np.nan
        ),
        axis=1,
    )

    result["nightscape_index"] = (composite * 100).round(2)
    result["nightscape_rank"] = result["nightscape_index"].rank(ascending=False, method="min").astype("Int64")

    return result


def sensitivity_equal_weights(df: pd.DataFrame, logger) -> pd.Series:
    """Compute index with equal weights for sensitivity check."""
    n_dims = len(DIMENSIONS)
    equal = {d: 1.0 / n_dims for d in DIMENSIONS}
    result = compute_composite_index(df, equal, logger)
    return result["nightscape_index"]


def plot_index_distribution(index_df: pd.DataFrame, df: pd.DataFrame, output_path: Path):
    """Plot the Nightscape Index distribution with borough coloring."""
    if "borough" in index_df.columns:
        merged = index_df.copy()
    else:
        merged = index_df.merge(df[["boro_cd", "borough"]], on="boro_cd")
    boro_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
    merged["borough_name"] = merged["borough"].map(boro_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(merged["nightscape_index"], bins=15, color="#1a365d", alpha=0.7, edgecolor="white")
    ax1.axvline(merged["nightscape_index"].median(), color="#ed8936", linestyle="--",
                linewidth=2, label=f"Median: {merged['nightscape_index'].median():.1f}")
    ax1.set_xlabel("Nightscape Index (higher = worse)", fontsize=11)
    ax1.set_ylabel("Number of CDs", fontsize=11)
    ax1.set_title("Distribution of Nightscape Index", fontsize=13, fontweight="bold")
    ax1.legend()

    # Borough boxplot
    boro_order = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]
    boro_order = [b for b in boro_order if b in merged["borough_name"].values]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]

    bp = ax2.boxplot(
        [merged.loc[merged["borough_name"] == b, "nightscape_index"] for b in boro_order],
        labels=boro_order, patch_artist=True, widths=0.6,
    )
    for patch, color in zip(bp["boxes"], colors[:len(boro_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Nightscape Index", fontsize=11)
    ax2.set_title("Nightscape Index by Borough", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_breakdown(index_df: pd.DataFrame, output_path: Path):
    """Plot stacked bar chart of dimension contributions for top/bottom 10 CDs."""
    dim_cols = [c for c in index_df.columns if c.startswith("dim_")]

    top10 = index_df.nlargest(10, "nightscape_index")
    bot10 = index_df.nsmallest(10, "nightscape_index")
    subset = pd.concat([top10, bot10])
    subset = subset.sort_values("nightscape_index", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    x = range(len(subset))
    bottom = np.zeros(len(subset))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(dim_cols)))

    for i, col in enumerate(dim_cols):
        vals = subset[col].values
        label = col.replace("dim_", "").replace("_", " ").title()
        ax.barh(x, vals, left=bottom, color=colors[i], label=label, height=0.7)
        bottom += vals

    ax.set_yticks(x)
    labels = [str(int(r)) for r in subset["boro_cd"]]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Dimension Score (sum → Index)", fontsize=11)
    ax.set_title("Nightscape Index: Top 10 Worst & Best CDs\n(dimension breakdown)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(y=9.5, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    with get_logger("54_composite_index") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        df = read_df(MASTER_PATH)
        logger.info(f"Master: {df.shape}")

        weights = params.get("nightscape_index", {}).get("dimensions", {})
        logger.info(f"Dimension weights: {weights}")

        index_df = compute_composite_index(df, weights, logger)
        logger.info(f"Index range: {index_df['nightscape_index'].min():.1f} - {index_df['nightscape_index'].max():.1f}")

        # Sensitivity: equal weights
        index_df["nightscape_index_equal"] = sensitivity_equal_weights(df, logger)
        corr = index_df[["nightscape_index", "nightscape_index_equal"]].corr("spearman").iloc[0, 1]
        logger.info(f"Weighted vs equal-weight correlation: ρ={corr:.3f}")

        # Save
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        out_cols = ["boro_cd", "nightscape_index", "nightscape_rank",
                    "nightscape_index_equal"] + [c for c in index_df.columns if c.startswith("dim_")]
        index_df[out_cols].to_csv(TABLES_DIR / "nightscape_index.csv", index=False)

        # Plots
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_df = index_df.copy()
        if "borough" in df.columns:
            plot_df = plot_df.merge(df[["boro_cd", "borough"]], on="boro_cd")
        plot_index_distribution(plot_df, df, FIGURES_DIR / "nightscape_index_distribution.png")
        plot_dimension_breakdown(index_df, FIGURES_DIR / "nightscape_dimension_breakdown.png")
        logger.info("Saved index plots")

        logger.log_metrics({
            "index_mean": index_df["nightscape_index"].mean(),
            "index_median": index_df["nightscape_index"].median(),
            "index_std": index_df["nightscape_index"].std(),
            "weighted_equal_corr": corr,
        })

        print(f"\nComposite Nightscape Index:")
        print(f"  Range: {index_df['nightscape_index'].min():.1f} - {index_df['nightscape_index'].max():.1f}")
        print(f"  Mean: {index_df['nightscape_index'].mean():.1f}, Median: {index_df['nightscape_index'].median():.1f}")
        print(f"  Weighted vs equal-weight ρ = {corr:.3f}")
        print(f"\n  Worst 5 CDs:")
        for _, row in index_df.nlargest(5, "nightscape_index").iterrows():
            cd = int(row["boro_cd"])
            idx = row["nightscape_index"]
            print(f"    CD {cd}: {idx:.1f}")
        print(f"\n  Best 5 CDs:")
        for _, row in index_df.nsmallest(5, "nightscape_index").iterrows():
            cd = int(row["boro_cd"])
            idx = row["nightscape_index"]
            print(f"    CD {cd}: {idx:.1f}")
        print(f"\n  Outputs: {TABLES_DIR}, {FIGURES_DIR}")


if __name__ == "__main__":
    main()
