#!/usr/bin/env python3
"""
74_build_findings.py — Build key findings summary page.

Generates a self-contained HTML page presenting the major findings from
the NYC Nightscape analysis in a narrative, data-driven format.

Output: outputs/interactive/findings.html
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nightscape.paths import INTERACTIVE_DIR, REPORTS_DIR, TABLES_DIR
from nightscape.logging_utils import get_logger

logger = get_logger("74_build_findings")


def load_data():
    """Load all analysis outputs needed for the findings page."""
    data = {}
    data["index"] = pd.read_csv(TABLES_DIR / "nightscape_index.csv")
    data["typologies"] = pd.read_csv(TABLES_DIR / "neighborhood_typologies.csv")
    data["top_corr"] = pd.read_csv(TABLES_DIR / "top_correlations.csv")
    data["gradients"] = pd.read_csv(TABLES_DIR / "demographic_gradients.csv")
    data["regression"] = pd.read_csv(TABLES_DIR / "regression_coefficients.csv")
    data["lisa"] = pd.read_csv(TABLES_DIR / "lisa_summary.csv")
    data["master"] = pd.read_parquet(REPORTS_DIR / "master_nightscape_df.parquet")
    data["lookup"] = pd.read_csv(
        Path(REPORTS_DIR).parent / "geo" / "cd_lookup.csv"
    )
    return data


def cd_name(lookup, boro_cd):
    row = lookup[lookup.boro_cd == boro_cd]
    if len(row) > 0:
        return row.iloc[0]["cd_short"]
    return f"CD {boro_cd}"


def build_html(data):
    lookup = data["lookup"]
    idx = data["index"]
    master = data["master"]
    master["boro_cd"] = master["boro_cd"].astype(int)
    idx["boro_cd"] = idx["boro_cd"].astype(int)

    # Worst / best CDs
    worst5 = idx.nlargest(5, "nightscape_index")
    best5 = idx.nsmallest(5, "nightscape_index")

    worst_rows = ""
    for _, r in worst5.iterrows():
        worst_rows += f'<tr><td>{cd_name(lookup, r.boro_cd)}</td><td>{r.nightscape_index:.1f}</td><td>#{int(r.nightscape_rank)}</td></tr>'
    best_rows = ""
    for _, r in best5.iterrows():
        best_rows += f'<tr><td>{cd_name(lookup, r.boro_cd)}</td><td>{r.nightscape_index:.1f}</td><td>#{int(r.nightscape_rank)}</td></tr>'

    # Key correlations (non-trivial, cross-domain)
    corr = data["top_corr"]
    corr_sig = corr[corr.fdr_significant == True].copy()
    # Filter out same-domain trivial pairs
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
    interesting = corr_sig[~corr_sig.apply(
        lambda r: (r.var_a, r.var_b) in trivial_pairs or (r.var_b, r.var_a) in trivial_pairs,
        axis=1
    )].head(12)

    corr_rows = ""
    for _, r in interesting.iterrows():
        corr_rows += f'<tr><td>{r.var_a}</td><td>{r.var_b}</td><td>{r.rho:.2f}</td></tr>'

    # Regression findings
    reg = data["regression"]
    reduced = reg[reg.model == "Reduced (demographics only)"].copy()
    reg_rows = ""
    for _, r in reduced.iterrows():
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
        reg_rows += f'<tr><td>{r.variable}</td><td>{r.beta:+.3f}</td><td>{r.p_value:.4f}{sig}</td></tr>'

    # Get actual R² from model summaries
    reg_summary = pd.read_csv(TABLES_DIR / "regression_model_summaries.csv")
    reduced_summary = reg_summary[reg_summary.model == "Reduced (demographics only)"].iloc[0]
    r_squared = reduced_summary.r_squared
    adj_r_squared = reduced_summary.adj_r_squared

    # Demographic gradient highlights
    grad = data["gradients"]
    grad_sig = grad[grad.significant_fdr == True].copy()
    grad_strong = grad_sig.reindex(grad_sig.spearman_rho.abs().sort_values(ascending=False).index).head(10)
    grad_rows = ""
    for _, r in grad_strong.iterrows():
        grad_rows += f'<tr><td>{r.nighttime_var}</td><td>{r.demographic_var}</td><td>{r.spearman_rho:+.2f}</td></tr>'

    # LISA
    lisa = data["lisa"]
    lisa_rows = ""
    for _, r in lisa.iterrows():
        lisa_rows += f'<tr><td>{r.variable}</td><td>{r.hot_spots_HH}</td><td>{r.cold_spots_LL}</td><td>{r.global_morans_I:.2f}</td></tr>'

    # Typology counts
    typo = data["typologies"]
    typo_counts = typo.groupby("typology")["boro_cd"].count().to_dict()

    # Key summary numbers
    n_vars = master.shape[1]
    idx_range = f"{idx.nightscape_index.min():.1f} - {idx.nightscape_index.max():.1f}"
    n_sig_corr = len(corr_sig)
    n_sig_grad = len(grad_sig)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Nightscape — Key Findings</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0; line-height: 1.7;
  }}

  header {{
    background: #1a365d; padding: 40px 0; text-align: center;
    border-bottom: 3px solid #ed8936;
  }}
  header h1 {{ font-size: 28px; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }}
  header p {{ font-size: 14px; color: #94a3b8; max-width: 700px; margin: 0 auto; }}

  main {{ max-width: 900px; margin: 0 auto; padding: 40px 24px; }}

  /* Hero stats */
  .hero-stats {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin-bottom: 48px;
  }}
  .hero-stat {{
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 20px; text-align: center;
  }}
  .hero-stat .hs-value {{ font-size: 28px; font-weight: 700; color: #ed8936; }}
  .hero-stat .hs-label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}

  /* Section */
  .finding {{
    margin-bottom: 48px;
  }}
  .finding-num {{
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #ed8936; margin-bottom: 6px;
  }}
  .finding h2 {{
    font-size: 22px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;
  }}
  .finding p {{
    font-size: 14px; color: #cbd5e1; margin-bottom: 16px;
  }}

  /* Tables */
  table {{
    width: 100%; border-collapse: collapse; margin-bottom: 16px;
    font-size: 13px;
  }}
  thead th {{
    text-align: left; padding: 8px 12px; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;
    border-bottom: 2px solid #334155; background: #1e293b;
  }}
  tbody td {{
    padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #cbd5e1;
  }}
  tbody tr:hover {{ background: #1e293b; }}

  /* Two-col table layout */
  .table-pair {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px;
  }}
  .table-pair-item h3 {{
    font-size: 14px; font-weight: 600; color: #f8fafc; margin-bottom: 8px;
  }}

  /* Callout */
  .callout {{
    background: #162032; border-left: 4px solid #ed8936; border-radius: 0 8px 8px 0;
    padding: 16px 20px; margin: 16px 0; font-size: 14px; color: #cbd5e1;
  }}
  .callout strong {{ color: #f8fafc; }}

  /* Navigation links */
  .nav-links {{
    display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 48px;
    justify-content: center;
  }}
  .nav-links a {{
    padding: 8px 16px; border-radius: 6px; border: 1px solid #334155;
    background: #1e293b; color: #cbd5e1; text-decoration: none;
    font-size: 13px; transition: all 0.15s;
  }}
  .nav-links a:hover {{ border-color: #ed8936; color: #ed8936; }}

  footer {{
    text-align: center; padding: 32px; border-top: 1px solid #334155;
    font-size: 12px; color: #64748b;
  }}
  footer a {{ color: #94a3b8; }}

  @media (max-width: 768px) {{
    .hero-stats {{ grid-template-columns: 1fr 1fr; }}
    .table-pair {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>NYC Nightscape: Key Findings</h1>
  <p>A comprehensive analysis of nighttime conditions across New York City's 59 community districts, drawing on 20+ public datasets covering noise, crime, transit, lighting, emergency services, and more (2021-2023).</p>
</header>
<main>

  <div class="nav-links">
    <a href="atlas.html">Interactive Atlas</a>
    <a href="profiles.html">Neighborhood Profiles</a>
    <a href="bivariate.html">Bivariate Maps</a>
    <a href="typologies.html">Typologies</a>
  </div>

  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hs-value">59</div>
      <div class="hs-label">Community Districts</div>
    </div>
    <div class="hero-stat">
      <div class="hs-value">{n_vars}</div>
      <div class="hs-label">Variables Analyzed</div>
    </div>
    <div class="hero-stat">
      <div class="hs-value">{n_sig_corr}</div>
      <div class="hs-label">Significant Correlations</div>
    </div>
    <div class="hero-stat">
      <div class="hs-value">{idx_range}</div>
      <div class="hs-label">Nightscape Index Range</div>
    </div>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 1</div>
    <h2>Nighttime conditions vary enormously across NYC</h2>
    <p>The Nightscape Index — a weighted composite of noise, safety, lighting, transit access, environmental quality, and services — ranges from {idx.nightscape_index.min():.1f} to {idx.nightscape_index.max():.1f} across NYC's 59 community districts. The worst nighttime conditions are overwhelmingly concentrated in the Bronx, while the best are found in Queens and Staten Island.</p>

    <div class="table-pair">
      <div class="table-pair-item">
        <h3>Worst Nighttime Conditions</h3>
        <table>
          <thead><tr><th>District</th><th>Index</th><th>Rank</th></tr></thead>
          <tbody>{worst_rows}</tbody>
        </table>
      </div>
      <div class="table-pair-item">
        <h3>Best Nighttime Conditions</h3>
        <table>
          <thead><tr><th>District</th><th>Index</th><th>Rank</th></tr></thead>
          <tbody>{best_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="callout">
      <strong>The gap is stark:</strong> {cd_name(lookup, int(worst5.iloc[0].boro_cd))} (Index: {worst5.iloc[0].nightscape_index:.1f}) scores {worst5.iloc[0].nightscape_index - best5.iloc[0].nightscape_index:.1f} points higher than {cd_name(lookup, int(best5.iloc[0].boro_cd))} (Index: {best5.iloc[0].nightscape_index:.1f}) on the 0–100 scale — a gap wider than any single dimension can explain.
    </div>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 2</div>
    <h2>Race and income predict nighttime quality of life</h2>
    <p>Demographics explain roughly {r_squared*100:.0f}% of the variation in the Nightscape Index (R&sup2; = {r_squared:.2f}, adjusted R&sup2; = {adj_r_squared:.2f}). The strongest predictors are percent Hispanic (beta = +0.68) and percent non-Hispanic Black (beta = +0.48) — meaning neighborhoods with larger Hispanic and Black populations tend to have significantly worse nighttime conditions, even after controlling for population density.</p>

    <table>
      <thead><tr><th>Predictor</th><th>Beta (std.)</th><th>p-value</th></tr></thead>
      <tbody>{reg_rows}</tbody>
    </table>

    <div class="callout">
      <strong>This is an equity finding, not a causal claim.</strong> These demographic variables are markers for decades of disinvestment — underlit streets, slower emergency response, louder environments, and fewer late-night services — not causes of poor nighttime conditions.
    </div>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 3</div>
    <h2>EMS response time is strongly linked to poverty</h2>
    <p>Among all nighttime indicators, EMS response time shows the strongest gradient with poverty (Spearman rho = +0.67). The poorest quintile of neighborhoods waits an average of ~10.7 minutes for an ambulance at night, while the wealthiest quintile waits ~9.4 minutes. In emergencies, those 80 seconds matter.</p>

    <p>The strongest demographic gradient findings (FDR-corrected):</p>
    <table>
      <thead><tr><th>Nighttime Variable</th><th>Demographic</th><th>Spearman rho</th></tr></thead>
      <tbody>{grad_rows}</tbody>
    </table>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 4</div>
    <h2>Nighttime problems cluster spatially</h2>
    <p>Spatial autocorrelation analysis (Local Moran's I) confirms that nighttime conditions are not randomly distributed. The Nightscape Index shows strong spatial clustering (Global Moran's I = 0.48): hot spots (high-high clusters) concentrate in the South Bronx and Upper Manhattan, while cold spots (low-low clusters) anchor southern Brooklyn and Queens.</p>

    <table>
      <thead><tr><th>Variable</th><th>Hot Spots</th><th>Cold Spots</th><th>Moran's I</th></tr></thead>
      <tbody>{lisa_rows}</tbody>
    </table>

    <div class="callout">
      <strong>Environmental quality shows the strongest spatial clustering</strong> (Moran's I = 0.79) because air pollution and temperature patterns are driven by building density and proximity to highways. Transit access is close behind (0.63), reflecting the fixed geography of subway infrastructure. Safety (0.34) is the least spatially concentrated — high-crime areas are not as neatly clustered as environmental or transit patterns.
    </div>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 5</div>
    <h2>Five distinct nighttime neighborhood types emerge</h2>
    <p>K-means clustering on the six Nightscape dimensions identified five neighborhood typologies:</p>

    <table>
      <thead><tr><th>Typology</th><th>CDs</th><th>Character</th></tr></thead>
      <tbody>
        <tr><td>Quiet Suburban</td><td>{typo_counts.get('Quiet Suburban', 0)}</td><td>Low noise, low crime, limited transit — the safest but least connected</td></tr>
        <tr><td>High-Need Urban</td><td>{typo_counts.get('High-Need Urban', 0)}</td><td>Highest crime, slowest EMS, most fires — faces compounding nighttime burdens</td></tr>
        <tr><td>Moderate Urban</td><td>{typo_counts.get('Moderate Urban', 0)}</td><td>Balanced conditions with good transit — the urban sweet spot</td></tr>
        <tr><td>Bustling & Loud</td><td>{typo_counts.get('Bustling & Loud', 0)}</td><td>Midtown Manhattan — extreme on every dimension</td></tr>
        <tr><td>Mixed Residential</td><td>{typo_counts.get('Mixed Residential', 0)}</td><td>Moderate across most dimensions with a unique residential-transit mix</td></tr>
      </tbody>
    </table>

    <div class="callout">
      <strong>The 21 High-Need Urban CDs</strong> — spanning the Bronx, Central Brooklyn, and Upper Manhattan — represent roughly 35% of the city's community districts but bear a disproportionate share of every negative nighttime outcome.
    </div>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 6</div>
    <h2>Cross-domain correlations reveal interconnected systems</h2>
    <p>Of 820 variable pairs tested, {n_sig_corr} showed significant correlations after FDR correction (54%). The strongest cross-domain relationships:</p>

    <table>
      <thead><tr><th>Variable A</th><th>Variable B</th><th>Spearman rho</th></tr></thead>
      <tbody>{corr_rows}</tbody>
    </table>

    <p>These correlations suggest that nighttime conditions form an interconnected system: neighborhoods don't just have one problem — they tend to have many, or few, simultaneously.</p>
  </div>

  <div class="finding">
    <div class="finding-num">Finding 7</div>
    <h2>What this means for policy</h2>
    <p>This analysis points to several actionable insights:</p>
    <div class="callout">
      <strong>Targeted lighting investment:</strong> Neighborhoods with both high crime and low satellite radiance (identifiable via the bivariate maps) are candidates for streetlight upgrades — an intervention with evidence backing its crime-reduction potential.
    </div>
    <div class="callout">
      <strong>EMS equity:</strong> The strong poverty-response time gradient suggests that EMS resource allocation should account for neighborhood disadvantage, not just call volume.
    </div>
    <div class="callout">
      <strong>Late-night transit:</strong> Transit deserts in low-income areas force reliance on less safe and more expensive transportation options. Expanding late-night bus service in High-Need Urban CDs could address a compounding equity gap.
    </div>
    <div class="callout">
      <strong>Holistic approach:</strong> Because nighttime problems cluster together, single-issue interventions may be insufficient. The 21 High-Need Urban neighborhoods would benefit most from coordinated investment across multiple dimensions.
    </div>
  </div>

</main>

<footer>
  NYC Nightscape Project | Data: NYC Open Data, ACS, VIIRS, CDC PLACES, MTA, TLC, and others (2021-2023)<br>
  Analysis: Spearman correlations with FDR correction, OLS regression, k-means clustering, Local Moran's I
</footer>
</body>
</html>"""
    return html


def main():
    logger.info("Building key findings page")
    data = load_data()
    html = build_html(data)

    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERACTIVE_DIR / "findings.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Findings page written to {out_path} ({len(html)/1e3:.0f} KB)")

    print(f"\nFindings built: {out_path}")
    print(f"  File size: {len(html)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
