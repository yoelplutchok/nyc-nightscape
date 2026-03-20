"""
52_demographic_gradients.py — Demographic gradient analysis.

For each nighttime variable: Spearman correlation with poverty, race, and
rent burden; quintile means; Mann-Whitney tests (top vs bottom quintile).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df, read_yaml
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR
from nightscape.qa import apply_fdr_correction

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

DEMOGRAPHIC_VARS = ["poverty_rate", "pct_nonhisp_black", "pct_hispanic", "rent_burden_rate"]

NIGHTTIME_VARS = [
    "rate_per_1k_pop", "rate_per_km2", "noise_obj_db_mean", "dot_pct_above_65dB",
    "late_night_share", "warm_season_ratio", "helicopter_complaints_night_per_1k",
    "crime_rate_per_1k", "violent_rate_per_1k", "shots_nighttime_share",
    "crash_rate_per_1k", "injury_rate_per_1k",
    "arrest_rate_per_1k", "drug_arrest_rate_per_1k",
    "z_light", "radiance_raw", "streetlight_complaints_per_km2",
    "late_night_entries_per_1k", "subway_stations_per_km2",
    "mean_trains_per_hour_late_night", "citibike_trips_night_per_1k",
    "total_ride_pickups_night_per_1k",
    "z_air", "z_heat", "no2_mean_primary", "pm25_mean_primary",
    "non_noise_311_per_1k", "homeless_per_1k",
    "restaurants_per_1k", "late_night_food_per_1k",
    "on_premises_rate_per_1k_pop", "wifi_hotspots_per_km2",
    "fire_incidents_night_per_1k", "fire_night_day_ratio",
    "ems_response_min_night",
]


def compute_gradients(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Compute Spearman correlations between all nighttime vars and demographic vars."""
    rows = []
    for night_var in NIGHTTIME_VARS:
        if night_var not in df.columns:
            continue
        for demo_var in DEMOGRAPHIC_VARS:
            if demo_var not in df.columns:
                continue
            mask = df[night_var].notna() & df[demo_var].notna()
            n = mask.sum()
            if n < 10:
                continue
            r, p = stats.spearmanr(df.loc[mask, night_var], df.loc[mask, demo_var])
            rows.append({
                "nighttime_var": night_var,
                "demographic_var": demo_var,
                "spearman_rho": r,
                "p_value": p,
                "n": n,
            })

    result = pd.DataFrame(rows)

    if len(result) > 0:
        fdr = apply_fdr_correction(result["p_value"])
        result["p_adjusted"] = fdr["p_adjusted"].values
        result["significant_fdr"] = fdr["significant_fdr"].values

    return result


def compute_quintile_means(df: pd.DataFrame, logger) -> tuple:
    """For each demographic variable, compute quintile means of nighttime variables.

    Returns (quintile_df, mw_df) where mw_df has FDR-corrected Mann-Whitney results.
    """
    rows = []
    mw_rows = []
    work = df.copy()
    for demo_var in DEMOGRAPHIC_VARS:
        if demo_var not in work.columns:
            continue
        work["_q"] = pd.qcut(work[demo_var], 5, labels=False, duplicates="drop")
        n_groups = work["_q"].nunique()
        if n_groups < 5:
            logger.warning(f"  {demo_var}: only {n_groups} quintile groups (ties caused fewer bins)")

        for night_var in NIGHTTIME_VARS:
            if night_var not in work.columns:
                continue
            qmeans = work.groupby("_q")[night_var].mean()
            for q, val in qmeans.items():
                rows.append({
                    "demographic_var": demo_var,
                    "nighttime_var": night_var,
                    "quintile": int(q) + 1,
                    "mean_value": val,
                })

            # Mann-Whitney: top vs bottom quintile
            bottom = work.loc[work["_q"] == 0, night_var].dropna()
            top = work.loc[work["_q"] == work["_q"].max(), night_var].dropna()
            if len(bottom) >= 5 and len(top) >= 5:
                u_stat, mw_p = stats.mannwhitneyu(bottom, top, alternative="two-sided")
                mw_rows.append({
                    "demographic_var": demo_var,
                    "nighttime_var": night_var,
                    "mw_u": u_stat,
                    "mw_p": mw_p,
                })

        work.drop(columns=["_q"], inplace=True)

    quintile_df = pd.DataFrame(rows)
    mw_df = pd.DataFrame(mw_rows)

    # Apply FDR correction to Mann-Whitney p-values
    if len(mw_df) > 0:
        fdr = apply_fdr_correction(mw_df["mw_p"])
        mw_df["mw_p_adjusted"] = fdr["p_adjusted"].values
        mw_df["mw_significant_fdr"] = fdr["significant_fdr"].values
        logger.info(f"  Mann-Whitney tests: {mw_df['mw_significant_fdr'].sum()}/{len(mw_df)} FDR-significant")

    return quintile_df, mw_df


def plot_gradient_heatmap(gradients: pd.DataFrame, output_path: Path):
    """Plot demographic gradient heatmap (nighttime vars x demographic vars)."""
    pivot = gradients.pivot_table(
        index="nighttime_var", columns="demographic_var",
        values="spearman_rho", aggfunc="first",
    )
    sig_pivot = gradients.pivot_table(
        index="nighttime_var", columns="demographic_var",
        values="significant_fdr", aggfunc="first",
    ).fillna(False)

    col_order = [c for c in DEMOGRAPHIC_VARS if c in pivot.columns]
    pivot = pivot[col_order]
    sig_pivot = sig_pivot.reindex(columns=col_order)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.35)))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        pivot, cmap=cmap, center=0, vmin=-0.8, vmax=0.8,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.7},
        xticklabels=True, yticklabels=True, ax=ax,
    )

    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            if sig_pivot.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                        fontsize=10, color="black", fontweight="bold")

    ax.set_title("Demographic Gradients of Nighttime Conditions\n(* = FDR-significant)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    with get_logger("52_demographic_gradients") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        df = read_df(MASTER_PATH)
        logger.info(f"Master: {df.shape}")

        gradients = compute_gradients(df, logger)
        logger.info(f"Computed {len(gradients)} gradient pairs")

        n_sig = gradients["significant_fdr"].sum() if "significant_fdr" in gradients.columns else 0
        logger.info(f"FDR significant: {n_sig}/{len(gradients)}")

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        gradients.to_csv(TABLES_DIR / "demographic_gradients.csv", index=False)

        quintiles, mw_results = compute_quintile_means(df, logger)
        quintiles.to_csv(TABLES_DIR / "quintile_means.csv", index=False)
        if len(mw_results) > 0:
            mw_results.to_csv(TABLES_DIR / "mann_whitney_results.csv", index=False)
        logger.info(f"Quintile means: {len(quintiles)} rows, MW tests: {len(mw_results)}")

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_gradient_heatmap(gradients, FIGURES_DIR / "demographic_gradients_heatmap.png")
        logger.info("Saved gradient heatmap")

        poverty_strong = gradients[
            (gradients["demographic_var"] == "poverty_rate") &
            (gradients["significant_fdr"] == True) &
            (gradients["spearman_rho"].abs() > 0.4)
        ]

        logger.log_metrics({
            "total_pairs": len(gradients),
            "fdr_significant": int(n_sig),
            "poverty_strong_correlations": len(poverty_strong),
        })

        print(f"\nDemographic Gradients:")
        print(f"  Pairs tested: {len(gradients)}")
        print(f"  FDR significant: {n_sig}")
        print(f"\n  Strongest poverty correlations:")
        pov = gradients[gradients["demographic_var"] == "poverty_rate"].copy()
        pov["abs_rho"] = pov["spearman_rho"].abs()
        for _, row in pov.nlargest(10, "abs_rho").iterrows():
            sig = "*" if row["significant_fdr"] else ""
            print(f"    {row['nighttime_var']:40s}  ρ={row['spearman_rho']:+.3f}{sig}")
        print(f"\n  Outputs: {TABLES_DIR}, {FIGURES_DIR}")


if __name__ == "__main__":
    main()
