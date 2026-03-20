"""
55_regression_models.py — OLS regression models for Nightscape Index.

Models:
  1. Full model: Nightscape Index ~ poverty + race + density + rent_burden + borough FEs
  2. Reduced: without borough FEs
  3. Per-dimension models: each dimension score as DV

Reports standardized betas, VIF, and diagnostics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df, read_yaml
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"

PREDICTORS = ["poverty_rate", "pct_nonhisp_black", "pct_hispanic", "rent_burden_rate", "pop_density"]


def winsorize_series(s: pd.Series, pct: float = 0.01) -> pd.Series:
    """Winsorize at given percentile (both tails)."""
    return pd.Series(mstats.winsorize(s.dropna(), limits=[pct, pct]), index=s.dropna().index)


def standardize(s: pd.Series) -> pd.Series:
    """Z-score standardize."""
    return (s - s.mean()) / s.std()


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each predictor."""
    from numpy.linalg import lstsq

    vifs = {}
    for col in X.columns:
        others = [c for c in X.columns if c != col]
        if not others:
            vifs[col] = 1.0
            continue
        y = X[col].values
        x = X[others].values
        x = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = lstsq(x, y, rcond=None)
        y_hat = x @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(vifs)


def run_ols(y: pd.Series, X: pd.DataFrame, model_name: str) -> dict:
    """Run OLS regression and return summary dict."""
    import statsmodels.api as sm

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, missing="drop").fit()

    coefficients = []
    for var in X.columns:
        coefficients.append({
            "variable": var,
            "beta": model.params.get(var, np.nan),
            "std_err": model.bse.get(var, np.nan),
            "t_stat": model.tvalues.get(var, np.nan),
            "p_value": model.pvalues.get(var, np.nan),
        })

    return {
        "model_name": model_name,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_p_value": model.f_pvalue,
        "n_obs": int(model.nobs),
        "aic": model.aic,
        "bic": model.bic,
        "coefficients": pd.DataFrame(coefficients),
        "model": model,
    }


def main():
    with get_logger("55_regression_models") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        winsorize_pct = params.get("analysis", {}).get("winsorize_pct", 0.01)

        df = read_df(MASTER_PATH)
        logger.info(f"Master: {df.shape}")

        # Load nightscape index
        index_path = TABLES_DIR / "nightscape_index.csv"
        if not index_path.exists():
            logger.error("Run 54_composite_index.py first to generate nightscape_index.csv")
            return

        index_df = pd.read_csv(index_path)
        df = df.merge(index_df[["boro_cd", "nightscape_index"] +
                               [c for c in index_df.columns if c.startswith("dim_")]],
                      on="boro_cd", how="left")

        # Winsorize predictors
        avail_predictors = [p for p in PREDICTORS if p in df.columns]
        for p in avail_predictors:
            df[p] = winsorize_series(df[p], winsorize_pct)

        # Standardize all variables for comparable betas
        df_std = df[["boro_cd", "borough"]].copy()
        for col in avail_predictors + ["nightscape_index"] + [c for c in df.columns if c.startswith("dim_")]:
            if col in df.columns:
                df_std[col] = standardize(df[col])

        # Borough dummies
        borough_dummies = pd.get_dummies(df["borough"], prefix="boro", drop_first=True, dtype=float)
        for c in borough_dummies.columns:
            df_std[c] = borough_dummies[c].values

        results = []

        # Model 1: Full model with borough FEs
        X_full = df_std[avail_predictors + list(borough_dummies.columns)].dropna()
        y_full = df_std.loc[X_full.index, "nightscape_index"]
        r1 = run_ols(y_full, X_full, "Full (demographics + borough FE)")
        results.append(r1)
        n_params_full = len(X_full.columns) + 1  # +1 for intercept
        n_per_k_full = r1["n_obs"] / n_params_full
        logger.info(f"Model 1 (Full): R²={r1['r_squared']:.3f}, adj R²={r1['adj_r_squared']:.3f}")
        logger.info(f"  n/k ratio: {n_per_k_full:.1f} (n={r1['n_obs']}, k={n_params_full})")
        if n_per_k_full < 10:
            logger.warning(
                f"  WARNING: n/k={n_per_k_full:.1f} is below the recommended minimum of 10. "
                f"Results may be unreliable due to overfitting risk."
            )

        # Model 2: Without borough FEs
        X_reduced = df_std[avail_predictors].dropna()
        y_reduced = df_std.loc[X_reduced.index, "nightscape_index"]
        r2 = run_ols(y_reduced, X_reduced, "Reduced (demographics only)")
        results.append(r2)
        logger.info(f"Model 2 (Reduced): R²={r2['r_squared']:.3f}, adj R²={r2['adj_r_squared']:.3f}")

        # VIF
        vif = compute_vif(X_reduced)
        logger.info(f"VIF: {dict(vif.round(2))}")
        high_vif = vif[vif > 5]
        if len(high_vif) > 0:
            logger.warning(f"  High VIF (>5) predictors: {dict(high_vif.round(1))} — multicollinearity concern")

        # Per-dimension models
        dim_cols = [c for c in df_std.columns if c.startswith("dim_")]
        for dim_col in dim_cols:
            dim_name = dim_col.replace("dim_", "")
            X_dim = df_std[avail_predictors].dropna()
            y_dim = df_std.loc[X_dim.index, dim_col]
            if y_dim.notna().sum() < 20:
                continue
            r_dim = run_ols(y_dim, X_dim, f"Dimension: {dim_name}")
            results.append(r_dim)
            logger.info(f"  {dim_name}: R²={r_dim['r_squared']:.3f}")

        # Save results
        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        # Model summaries
        summary_rows = []
        for r in results:
            summary_rows.append({
                "model": r["model_name"],
                "r_squared": r["r_squared"],
                "adj_r_squared": r["adj_r_squared"],
                "f_stat": r["f_stat"],
                "f_p_value": r["f_p_value"],
                "n_obs": r["n_obs"],
                "aic": r["aic"],
                "bic": r["bic"],
            })
        pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "regression_model_summaries.csv", index=False)

        # Coefficients from all models
        all_coefs = []
        for r in results:
            coefs = r["coefficients"].copy()
            coefs["model"] = r["model_name"]
            all_coefs.append(coefs)
        pd.concat(all_coefs).to_csv(TABLES_DIR / "regression_coefficients.csv", index=False)

        # VIF table
        vif.to_frame("vif").to_csv(TABLES_DIR / "regression_vif.csv")

        logger.log_metrics({
            "full_model_r2": r1["r_squared"],
            "reduced_model_r2": r2["r_squared"],
            "max_vif": vif.max(),
            "n_models": len(results),
        })

        print(f"\nRegression Models:")
        print(f"\n  Model 1 — Full (demographics + borough FE):")
        print(f"    R² = {r1['r_squared']:.3f}, Adj R² = {r1['adj_r_squared']:.3f}")
        print(f"    Significant predictors:")
        for _, row in r1["coefficients"].iterrows():
            if row["p_value"] < 0.05:
                print(f"      {row['variable']:25s} β={row['beta']:+.3f}  p={row['p_value']:.4f}")

        print(f"\n  Model 2 — Reduced (demographics only):")
        print(f"    R² = {r2['r_squared']:.3f}, Adj R² = {r2['adj_r_squared']:.3f}")
        print(f"    VIF: {dict(vif.round(1))}")

        print(f"\n  Dimension models:")
        for r in results[2:]:
            print(f"    {r['model_name']:35s} R²={r['r_squared']:.3f}")

        print(f"\n  Outputs: {TABLES_DIR}")


if __name__ == "__main__":
    main()
