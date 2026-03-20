"""
60_data_quality_report.py — Automated data quality checks on master nightscape dataframe.

Checks:
- No CD has >20% missing values across all variables
- All rates are non-negative
- Population-normalized rates are plausible (flag outliers > 3 SD)
- Cross-validations between related datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import read_df
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR

MASTER_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
REPORT_PATH = REPORTS_DIR / "data_quality_report.csv"


def check_missing(df: pd.DataFrame, logger) -> list:
    """Check missing value rates per CD and per column."""
    issues = []
    numeric = df.select_dtypes(include="number").drop(columns=["boro_cd"], errors="ignore")

    # Per-CD missing rate
    cd_missing = numeric.isna().mean(axis=1)
    bad_cds = df.loc[cd_missing > 0.20, "boro_cd"].tolist()
    if bad_cds:
        issues.append(("FAIL", f"CDs with >20% missing: {bad_cds}"))
        logger.warning(f"CDs with >20% missing: {bad_cds}")
    else:
        logger.info("PASS: No CD has >20% missing values")

    # Per-column missing rate
    col_missing = numeric.isna().sum()
    any_missing = col_missing[col_missing > 0]
    if len(any_missing) > 0:
        logger.info(f"Columns with missing values: {dict(any_missing)}")
        issues.append(("INFO", f"Columns with missing: {dict(any_missing)}"))
    else:
        logger.info("PASS: Zero missing values across all columns")

    return issues


def check_non_negative(df: pd.DataFrame, logger) -> list:
    """Check that all rate columns are non-negative."""
    issues = []
    rate_cols = [c for c in df.columns if any(k in c for k in ["per_1k", "per_km2", "rate", "count", "trips"])]

    for c in rate_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        neg = (vals < 0).sum()
        if neg > 0:
            issues.append(("FAIL", f"{c}: {neg} negative values"))
            logger.warning(f"FAIL: {c} has {neg} negative values")

    if not issues:
        logger.info(f"PASS: All {len(rate_cols)} rate/count columns are non-negative")

    return issues


def check_outliers(df: pd.DataFrame, logger) -> list:
    """Flag extreme outliers (> 3 SD from mean) and zero-variance columns."""
    issues = []
    numeric = df.select_dtypes(include="number").drop(columns=["boro_cd"], errors="ignore")

    for c in numeric.columns:
        vals = numeric[c].dropna()
        if len(vals) < 5:
            continue
        if vals.std() < 1e-10:
            issues.append(("WARN", f"{c}: zero variance (all {len(vals)} values identical = {vals.iloc[0]})"))
            logger.warning(f"ZERO-VARIANCE: {c} — all values identical")
            continue
        z = np.abs(stats.zscore(vals))
        extreme = (z > 3).sum()
        if extreme > 0:
            max_val = vals[z > 3].max()
            issues.append(("WARN", f"{c}: {extreme} values > 3 SD (max={max_val:.2f})"))
            logger.warning(f"OUTLIER: {c} has {extreme} values > 3 SD")

    if not any(i[0] == "WARN" for i in issues):
        logger.info("PASS: No extreme outliers (> 3 SD) or zero-variance columns found")

    return issues


def check_cross_validations(df: pd.DataFrame, logger) -> list:
    """Cross-validate related variables."""
    issues = []

    checks = [
        ("z_light", "streetlight_complaints_per_km2", "VIIRS vs streetlight complaints", 0.2, "any"),
        ("crime_rate_per_1k", "arrest_rate_per_1k", "Crime rate vs arrest rate", 0.5, "positive"),
        ("late_night_ridership", "subway_stations", "Subway ridership vs stations", 0.3, "positive"),
        ("on_premises_rate_per_1k_pop", "restaurants_per_1k", "Bars vs restaurants", 0.3, "positive"),
        ("taxi_pickups_night_per_1k", "rideshare_pickups_night_per_1k", "Taxi vs rideshare", 0.3, "positive"),
        ("pop_density", "total_ride_pickups_night_per_1k", "Pop density vs ride pickups", 0.2, "any"),
        ("fire_incidents_night_per_1k", "ems_response_min_night", "Fire incidents vs EMS response", 0.2, "any"),
    ]

    for col_a, col_b, desc, min_r, direction in checks:
        if col_a not in df.columns or col_b not in df.columns:
            issues.append(("SKIP", f"{desc}: column(s) missing"))
            continue

        a = pd.to_numeric(df[col_a], errors="coerce")
        b = pd.to_numeric(df[col_b], errors="coerce")
        mask = a.notna() & b.notna()

        if mask.sum() < 10:
            issues.append(("SKIP", f"{desc}: too few valid pairs ({mask.sum()})"))
            continue

        r, p = stats.spearmanr(a[mask], b[mask])
        status = "PASS" if abs(r) >= min_r else "WARN"
        if direction == "positive" and r < 0:
            status = "WARN"

        issues.append((status, f"{desc}: r={r:.3f}, p={p:.4f}"))
        logger.info(f"  {status}: {desc} — Spearman r={r:.3f}, p={p:.4f}")

    return issues


def main():
    with get_logger("60_data_quality_report") as logger:
        df = read_df(MASTER_PATH)
        logger.info(f"Master shape: {df.shape}")

        all_issues = []

        logger.info("\n── Missing Value Checks ──")
        all_issues.extend(check_missing(df, logger))

        logger.info("\n── Non-Negative Checks ──")
        all_issues.extend(check_non_negative(df, logger))

        logger.info("\n── Outlier Checks ──")
        all_issues.extend(check_outliers(df, logger))

        logger.info("\n── Cross-Validation Checks ──")
        all_issues.extend(check_cross_validations(df, logger))

        # Check for borough-level variables (≤5 unique values)
        logger.info("\n── Borough-Level Variable Detection ──")
        numeric = df.select_dtypes(include="number").drop(columns=["boro_cd"], errors="ignore")
        for c in numeric.columns:
            vals = numeric[c].dropna()
            n_unique = vals.nunique()
            if 1 < n_unique <= 5 and len(vals) >= 50:
                all_issues.append(("WARN", f"{c}: only {n_unique} unique values — likely borough-level, CD-level correlations unreliable"))
                logger.warning(f"  {c}: {n_unique} unique values (borough-level — correlations unreliable)")

        # Rate vs count consistency check
        logger.info("\n── Rate/Count Consistency ──")
        if "crime_rate_per_1k" in df.columns and "crimes_per_year" in df.columns and "population" in df.columns:
            expected = df["crimes_per_year"] / (df["population"] / 1000)
            diff = (df["crime_rate_per_1k"] - expected).abs()
            max_diff = diff.max()
            if max_diff > 1.0:
                all_issues.append(("WARN", f"crime_rate_per_1k vs crimes_per_year/(pop/1000): max diff={max_diff:.2f}"))
            else:
                all_issues.append(("PASS", f"crime_rate_per_1k consistent with raw counts (max diff={max_diff:.4f})"))

        # Plausibility bounds for health data and percentages
        logger.info("\n── Plausibility Bounds ──")
        PLAUSIBILITY_BOUNDS = {
            "depression": (0, 100),
            "heavy_drinking": (0, 100),
            "obesity": (0, 100),
            "poor_health": (0, 100),
            "sleep_trouble": (0, 100),
            "felony_share": (0, 1),
            "ahv_nighttime_share": (0, 1),
            "shots_nighttime_share": (0, 1),
            "pct_ridership_late_night": (0, 100),
            "commercial_waste_zone_pct": (0, 100),
        }
        for col, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                below = (vals < lo).sum()
                above = (vals > hi).sum()
                if below > 0 or above > 0:
                    all_issues.append(("FAIL", f"{col}: {below} below {lo}, {above} above {hi} — possible scale/unit error"))
                    logger.warning(f"FAIL: {col} out of plausible range [{lo}, {hi}]: {below} below, {above} above")
                else:
                    all_issues.append(("PASS", f"{col}: all values within [{lo}, {hi}]"))

        # Deterministic inequality checks (total >= part)
        logger.info("\n── Inequality Checks ──")
        inequality_pairs = [
            ("nighttime_crimes_total", "violent_crimes", "Total crimes >= violent crimes"),
            ("arrest_count", "felony_arrests", "Arrests >= felony arrests"),
            ("arrest_count", "drug_arrests", "Arrests >= drug arrests"),
            ("arrest_count", "dui_arrests", "Arrests >= DUI arrests"),
            ("ahv_total_permits", "ahv_nighttime_permits", "AHV total >= nighttime permits"),
            ("ems_total", "ems_nighttime", "EMS total >= EMS nighttime"),
            ("fire_total", "fire_nighttime", "Fire total >= fire nighttime"),
            ("shots_total", "shots_nighttime", "Shots total >= shots nighttime"),
        ]
        for total_col, part_col, desc in inequality_pairs:
            if total_col in df.columns and part_col in df.columns:
                violations = (df[part_col] > df[total_col]).sum()
                if violations > 0:
                    all_issues.append(("FAIL", f"{desc}: {violations} CDs where {part_col} > {total_col}"))
                    logger.warning(f"FAIL: {desc} — {violations} violations")
                else:
                    all_issues.append(("PASS", f"{desc}: consistent"))

        # Additive consistency: total_ride_trips == taxi + rideshare
        if all(c in df.columns for c in ["total_ride_trips_night", "taxi_trips_night", "rideshare_trips_night"]):
            expected_total = df["taxi_trips_night"] + df["rideshare_trips_night"]
            diff = (df["total_ride_trips_night"] - expected_total).abs()
            max_diff = diff.max()
            if max_diff > 1.0:
                all_issues.append(("WARN", f"total_ride_trips_night != taxi + rideshare: max diff={max_diff:.2f}"))
            else:
                all_issues.append(("PASS", f"TLC trip totals consistent (max diff={max_diff:.4f})"))

        # Temporal scope warnings
        logger.info("\n── Temporal Scope Warnings ──")
        if "arrest_count" in df.columns:
            all_issues.append(("WARN",
                "arrest_count is all-hours data (not nighttime-filtered) — "
                "direct comparison with nighttime-only metrics may be misleading"))
            logger.warning("arrest_count is all-hours (no time-of-day available in source)")

        # Summary
        report_df = pd.DataFrame(all_issues, columns=["status", "detail"])
        report_df.to_csv(REPORT_PATH, index=False)

        n_fail = (report_df["status"] == "FAIL").sum()
        n_warn = (report_df["status"] == "WARN").sum()
        n_pass = (report_df["status"] == "PASS").sum()
        n_info = (report_df["status"] == "INFO").sum()

        logger.log_metrics({
            "checks_pass": n_pass,
            "checks_warn": n_warn,
            "checks_fail": n_fail,
            "checks_info": n_info,
            "total_columns": len(df.columns),
        })

        print(f"\nData Quality Report:")
        print(f"  Master: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  PASS: {n_pass}  |  WARN: {n_warn}  |  FAIL: {n_fail}  |  INFO: {n_info}")
        for _, row in report_df.iterrows():
            marker = {"PASS": "+", "WARN": "!", "FAIL": "X", "INFO": "i", "SKIP": "-"}[row["status"]]
            print(f"  [{marker}] {row['detail']}")
        print(f"\n  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
