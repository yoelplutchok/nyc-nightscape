"""
51_correlation_matrix.py — Spearman correlation matrix across nighttime dimensions.

Computes pairwise Spearman rank correlations for all analysis-grade variables,
applies FDR correction, and produces a grouped heatmap and results CSV.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df, read_yaml, atomic_write_df
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR
from nightscape.qa import apply_fdr_correction

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

DIMENSION_VARIABLES = {
    "Noise": [
        "rate_per_1k_pop", "rate_per_km2", "noise_obj_db_mean",
        "dot_pct_above_65dB", "late_night_share", "weekend_uplift",
        "warm_season_ratio", "helicopter_complaints_night_per_1k",
    ],
    "Safety": [
        "crime_rate_per_1k", "violent_rate_per_1k", "felony_share",
        "shots_nighttime_share", "crash_rate_per_1k", "injury_rate_per_1k",
        "arrest_rate_per_1k", "drug_arrest_rate_per_1k",
    ],
    "Lighting": [
        "z_light", "radiance_raw", "streetlight_complaints_per_km2",
    ],
    "Transit": [
        "late_night_entries_per_1k", "subway_stations_per_km2",
        "mean_trains_per_hour_late_night", "citibike_trips_night_per_1k",
        "total_ride_pickups_night_per_1k",
    ],
    "Environment": [
        "z_air", "z_heat", "no2_mean_primary", "pm25_mean_primary",
        "tmin_mean_primary",
    ],
    "Services": [
        "non_noise_311_per_1k", "homeless_per_1k", "restaurants_per_1k",
        "late_night_food_per_1k", "film_activity_per_1k",
        "on_premises_rate_per_1k_pop", "wifi_hotspots_per_km2",
        "linknyc_kiosks_per_1k",
    ],
    "Emergency": [
        "fire_incidents_night_per_1k", "fire_night_day_ratio",
        "ems_response_min_night", "ems_night_day_response_ratio",
    ],
    "Demographics": [
        "poverty_rate", "pct_nonhisp_black", "pct_hispanic",
        "rent_burden_rate", "pop_density",
    ],
}


def get_analysis_columns(df: pd.DataFrame) -> list:
    """Return ordered list of analysis columns that exist in the dataframe."""
    ordered = []
    for dim, cols in DIMENSION_VARIABLES.items():
        for c in cols:
            if c in df.columns:
                ordered.append(c)
    return ordered


def compute_spearman_matrix(df: pd.DataFrame, columns: list) -> tuple:
    """Compute pairwise Spearman correlations and p-values."""
    n = len(columns)
    rho_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            x = df[columns[i]].dropna()
            y = df[columns[j]].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 10:
                continue
            r, p = stats.spearmanr(x[common], y[common])
            rho_matrix[i, j] = r
            rho_matrix[j, i] = r
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    rho_df = pd.DataFrame(rho_matrix, index=columns, columns=columns)
    p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)
    return rho_df, p_df


def apply_fdr_to_matrix(p_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply FDR correction to the upper triangle of a p-value matrix."""
    n = len(p_df)
    sig_df = pd.DataFrame(False, index=p_df.index, columns=p_df.columns)

    upper_ps = []
    upper_coords = []
    for i in range(n):
        for j in range(i + 1, n):
            val = p_df.iloc[i, j]
            if not np.isnan(val):
                upper_ps.append(val)
                upper_coords.append((i, j))

    if not upper_ps:
        return sig_df

    p_series = pd.Series(upper_ps)
    fdr_result = apply_fdr_correction(p_series, alpha=alpha)

    for k, (i, j) in enumerate(upper_coords):
        is_sig = fdr_result.iloc[k]["significant_fdr"]
        sig_df.iloc[i, j] = is_sig
        sig_df.iloc[j, i] = is_sig

    return sig_df


def make_dimension_labels(columns: list) -> list:
    """Create dimension prefix labels for each column."""
    col_to_dim = {}
    for dim, cols in DIMENSION_VARIABLES.items():
        for c in cols:
            col_to_dim[c] = dim
    return [col_to_dim.get(c, "Other") for c in columns]


def plot_heatmap(rho_df, sig_df, dim_labels, output_path):
    """Plot grouped correlation heatmap with FDR significance markers."""
    n = len(rho_df)
    fig, ax = plt.subplots(figsize=(20, 18))

    mask = np.triu(np.ones_like(rho_df, dtype=bool), k=1)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        rho_df, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
        xticklabels=True, yticklabels=True, ax=ax,
    )

    for i in range(n):
        for j in range(i):
            if sig_df.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, "·", ha="center", va="center",
                        fontsize=6, color="black", fontweight="bold")

    # Dimension separators
    dims_seen = []
    boundaries = []
    for k, d in enumerate(dim_labels):
        if d not in dims_seen:
            dims_seen.append(d)
            boundaries.append(k)
    boundaries.append(n)

    for b in boundaries[1:-1]:
        ax.axhline(y=b, color="black", linewidth=1.2)
        ax.axvline(x=b, color="black", linewidth=1.2)

    # Dimension labels on right side
    for idx in range(len(dims_seen)):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        mid = (start + end) / 2
        ax.text(n + 0.5, mid, dims_seen[idx], ha="left", va="center",
                fontsize=9, fontweight="bold", color="#333")

    ax.set_title("Nightscape Variable Correlations (Spearman ρ, FDR-corrected)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(labelsize=7)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_correlations(rho_df, p_df, sig_df, output_path, n_top=50):
    """Save the strongest correlations as a flat table."""
    rows = []
    n = len(rho_df)
    cols = rho_df.columns
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "var_a": cols[i],
                "var_b": cols[j],
                "rho": rho_df.iloc[i, j],
                "p_value": p_df.iloc[i, j],
                "fdr_significant": sig_df.iloc[i, j],
                "abs_rho": abs(rho_df.iloc[i, j]),
            })

    result = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    result = result.drop(columns=["abs_rho"])

    # Flag trivially correlated pairs (z-composite vs component, subset relationships)
    trivial_pairs = {
        ("z_light", "radiance_raw"), ("z_heat", "tmin_mean_primary"),
        ("z_air", "no2_mean_primary"), ("z_air", "pm25_mean_primary"),
        ("z_noise_obj", "noise_obj_db_mean"), ("z_noise_obj", "dot_exposure_index"),
        ("no2_mean_primary", "pm25_mean_primary"),
        ("crime_rate_per_1k", "violent_rate_per_1k"),
        ("crash_rate_per_1k", "injury_rate_per_1k"),
        ("arrest_rate_per_1k", "drug_arrest_rate_per_1k"),
        ("arrest_rate_per_1k", "felony_arrest_rate_per_1k"),
        ("arrest_count", "felony_arrests"), ("arrest_count", "drug_arrests"),
        ("rate_per_1k_pop", "rate_per_km2"),
        ("citibike_trips_night_per_1k", "total_ride_pickups_night_per_1k"),
        ("taxi_pickups_night_per_1k", "total_ride_pickups_night_per_1k"),
        ("rideshare_pickups_night_per_1k", "total_ride_pickups_night_per_1k"),
        ("restaurants_per_1k", "late_night_food_per_1k"),
        ("total_licenses", "on_premises_count"),
        ("license_rate_per_1k_pop", "on_premises_rate_per_1k_pop"),
    }
    result["trivial"] = result.apply(
        lambda r: (r.var_a, r.var_b) in trivial_pairs or (r.var_b, r.var_a) in trivial_pairs,
        axis=1,
    )

    result.to_csv(output_path, index=False)
    return result


def main():
    with get_logger("51_correlation_matrix") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        df = read_df(MASTER_PATH)
        logger.info(f"Master: {df.shape[0]} rows x {df.shape[1]} cols")

        columns = get_analysis_columns(df)

        # Drop near-constant columns (< 6 unique values) — unreliable correlations
        min_n = params.get("analysis", {}).get("min_n_correlation", 10)
        dropped_low_card = []
        filtered_columns = []
        for c in columns:
            n_unique = df[c].dropna().nunique()
            if n_unique < 6:
                dropped_low_card.append((c, n_unique))
            else:
                filtered_columns.append(c)
        if dropped_low_card:
            logger.warning(f"Dropped {len(dropped_low_card)} low-cardinality columns: "
                           f"{[(c, n) for c, n in dropped_low_card]}")
        columns = filtered_columns
        logger.info(f"Analysis variables: {len(columns)}")

        rho_df, p_df = compute_spearman_matrix(df, columns)
        logger.info("Spearman correlations computed")

        alpha = params.get("analysis", {}).get("fdr_alpha", 0.05)
        sig_df = apply_fdr_to_matrix(p_df, alpha=alpha)
        n_sig = sig_df.sum().sum() // 2
        n_total = len(columns) * (len(columns) - 1) // 2
        logger.info(f"FDR significant: {n_sig}/{n_total} pairs (α={alpha})")

        # Save correlation matrix
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        rho_df.to_csv(TABLES_DIR / "correlation_matrix_rho.csv")
        p_df.to_csv(TABLES_DIR / "correlation_matrix_pvalues.csv")
        logger.info(f"Saved matrices to {TABLES_DIR}")

        # Save top correlations
        top = save_top_correlations(
            rho_df, p_df, sig_df,
            TABLES_DIR / "top_correlations.csv",
        )
        logger.info(f"Top correlation: {top.iloc[0]['var_a']} vs {top.iloc[0]['var_b']} (ρ={top.iloc[0]['rho']:.3f})")

        # Plot heatmap
        dim_labels = make_dimension_labels(columns)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_heatmap(
            rho_df, sig_df, dim_labels,
            FIGURES_DIR / "correlation_heatmap.png",
        )
        logger.info(f"Saved heatmap to {FIGURES_DIR / 'correlation_heatmap.png'}")

        # Summary stats
        strong_pos = ((rho_df > 0.5) & sig_df).sum().sum() // 2
        strong_neg = ((rho_df < -0.5) & sig_df).sum().sum() // 2
        logger.log_metrics({
            "analysis_variables": len(columns),
            "total_pairs": n_total,
            "fdr_significant_pairs": int(n_sig),
            "strong_positive_pairs": int(strong_pos),
            "strong_negative_pairs": int(strong_neg),
        })

        print(f"\nCorrelation Matrix:")
        print(f"  Variables: {len(columns)} across {len(DIMENSION_VARIABLES)} dimensions")
        print(f"  Total pairs: {n_total}")
        print(f"  FDR significant (α={alpha}): {n_sig}")
        print(f"  Strong positive (ρ>0.5): {strong_pos}")
        print(f"  Strong negative (ρ<-0.5): {strong_neg}")
        print(f"\n  Top 10 correlations:")
        for _, row in top.head(10).iterrows():
            sig = "*" if row["fdr_significant"] else ""
            print(f"    {row['var_a']:40s} vs {row['var_b']:40s}  ρ={row['rho']:+.3f}{sig}")
        print(f"\n  Outputs: {TABLES_DIR}, {FIGURES_DIR}")


if __name__ == "__main__":
    main()
