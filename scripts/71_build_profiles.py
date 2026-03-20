#!/usr/bin/env python3
"""
71_build_profiles.py — Build interactive neighborhood profile cards.

Generates a self-contained HTML page with:
  - Dropdown to select any of 59 community districts
  - Nightscape Index score + rank + typology badge
  - 6-dimension radar chart (vs borough avg and city avg)
  - Key nighttime stats with bar sparklines showing position in city range
  - Borough and city comparison context

Output: outputs/interactive/profiles.html
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nightscape.paths import (
    GEO_DIR,
    INTERACTIVE_DIR,
    REPORTS_DIR,
    TABLES_DIR,
)
from nightscape.logging_utils import get_logger

logger = get_logger("71_build_profiles")

# Borough code → name
BORO_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

# Key stats to show on each profile card
# (col, label, unit, higher_is_worse, description)
KEY_STATS = [
    ("nightscape_index", "Nightscape Index", "", True, "Composite score (higher = worse)"),
    ("noise_obj_db_mean", "Ambient Noise", "dB", True, "Mean modeled noise level"),
    ("rate_per_1k_pop", "Noise Complaints / 1k", "", True, "311 noise complaints per 1k residents"),
    ("crime_rate_per_1k", "Nighttime Crime / 1k", "", True, "Crimes 10PM-7AM per 1k residents"),
    ("violent_rate_per_1k", "Violent Crime / 1k", "", True, "Violent crimes at night per 1k"),
    ("crash_rate_per_1k", "Nighttime Crashes / 1k", "", True, "Vehicle crashes 10PM-7AM per 1k"),
    ("fire_incidents_night_per_1k", "Fire Incidents / 1k", "", True, "Nighttime fire incidents per 1k"),
    ("ems_response_min_night", "EMS Response", "min", True, "Mean nighttime ambulance response"),
    ("radiance_raw", "Light Radiance", "nW", False, "Satellite-measured nighttime brightness"),
    ("streetlight_complaints_per_km2", "Streetlight Issues / km²", "", True, "Broken streetlight reports"),
    ("late_night_entries_per_1k", "Late-Night Subway / 1k", "", False, "Subway entries 1-5AM per 1k"),
    ("subway_stations_per_km2", "Subway Density", "/ km²", False, "Stations per square km"),
    ("total_ride_pickups_night_per_1k", "Ride Pickups / 1k", "", False, "Taxi + rideshare at night per 1k"),
    ("on_premises_rate_per_1k_pop", "Bars & Clubs / 1k", "", None, "On-premises liquor licenses per 1k"),
    ("restaurants_per_1k", "Restaurants / 1k", "", None, "Active restaurants per 1k"),
    ("non_noise_311_per_1k", "Non-Noise 311 / 1k", "", True, "Other nighttime complaints per 1k"),
]

DIMENSIONS = [
    "dim_noise_exposure", "dim_safety", "dim_lighting",
    "dim_transit_access", "dim_environmental", "dim_services",
]
DIM_LABELS = [
    "Noise Exposure", "Safety", "Lighting",
    "Transit Access", "Environmental", "Services",
]


def build_profile_data(
    master: pd.DataFrame,
    index_df: pd.DataFrame,
    typology_df: pd.DataFrame,
    lookup: pd.DataFrame,
) -> list[dict]:
    """Build a list of profile dicts, one per CD."""

    # Merge everything
    df = master.copy()
    df["boro_cd"] = df["boro_cd"].astype(int)

    for extra in [index_df, typology_df]:
        extra = extra.copy()
        extra["boro_cd"] = extra["boro_cd"].astype(int)
        new_cols = [c for c in extra.columns if c not in df.columns or c == "boro_cd"]
        df = df.merge(extra[new_cols], on="boro_cd", how="left")

    lookup = lookup.copy()
    lookup["boro_cd"] = lookup["boro_cd"].astype(int)
    df = df.merge(lookup[["boro_cd", "cd_short", "cd_label", "borough_name"]], on="boro_cd", how="left")

    # Compute city-wide stats
    city_stats = {}
    for col, *_ in KEY_STATS:
        vals = df[col].dropna()
        city_stats[col] = {
            "mean": float(vals.mean()) if len(vals) > 0 else None,
            "min": float(vals.min()) if len(vals) > 0 else None,
            "max": float(vals.max()) if len(vals) > 0 else None,
        }

    # City-wide dimension averages
    city_dims = {}
    for dim in DIMENSIONS:
        city_dims[dim] = float(df[dim].mean())

    # Borough-level stats
    boro_stats = {}
    boro_dims = {}
    for boro_code in sorted(df["borough"].unique()):
        bdf = df[df["borough"] == boro_code]
        boro_stats[int(boro_code)] = {}
        for col, *_ in KEY_STATS:
            vals = bdf[col].dropna()
            boro_stats[int(boro_code)][col] = float(vals.mean()) if len(vals) > 0 else None
        boro_dims[int(boro_code)] = {}
        for dim in DIMENSIONS:
            boro_dims[int(boro_code)][dim] = float(bdf[dim].mean())

    # Build per-CD profiles
    profiles = []
    for _, row in df.iterrows():
        cd = int(row["boro_cd"])
        boro = int(row["borough"])
        profile = {
            "boro_cd": cd,
            "cd_short": row.get("cd_short", f"CD {cd}"),
            "cd_label": row.get("cd_label", ""),
            "borough_name": row.get("borough_name", BORO_NAMES.get(boro, "")),
            "borough_code": boro,
            "typology": row.get("typology", ""),
            "population": int(row["population"]) if pd.notna(row["population"]) else None,
            "nightscape_index": _safe_float(row.get("nightscape_index")),
            "nightscape_rank": int(row["nightscape_rank"]) if pd.notna(row.get("nightscape_rank")) else None,
            "dimensions": [_safe_float(row.get(d)) for d in DIMENSIONS],
            "stats": [],
        }
        for col, label, unit, higher_is_worse, desc in KEY_STATS:
            val = _safe_float(row.get(col))
            cs = city_stats.get(col, {})
            profile["stats"].append({
                "col": col,
                "label": label,
                "unit": unit,
                "higher_is_worse": higher_is_worse,
                "desc": desc,
                "value": val,
                "city_mean": cs.get("mean"),
                "city_min": cs.get("min"),
                "city_max": cs.get("max"),
                "boro_mean": boro_stats.get(boro, {}).get(col),
            })
        profiles.append(profile)

    return profiles, city_dims, boro_dims


def _safe_float(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 4)


def build_html(profiles: list[dict], city_dims: dict, boro_dims: dict) -> str:
    """Generate the profiles HTML page."""

    profiles_json = json.dumps(profiles)
    city_dims_json = json.dumps(city_dims)
    boro_dims_json = json.dumps(boro_dims)
    dim_labels_json = json.dumps(DIM_LABELS)

    # Build dropdown options
    options = []
    for p in sorted(profiles, key=lambda x: x["boro_cd"]):
        options.append(f'<option value="{p["boro_cd"]}">{p["cd_short"]} — {p["cd_label"]}</option>')
    options_html = "\n".join(options)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Nightscape — Neighborhood Profiles</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0; min-height: 100vh;
  }}

  /* Header */
  header {{
    background: #1a365d; padding: 20px 32px; border-bottom: 1px solid #334155;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  }}
  header h1 {{ font-size: 20px; font-weight: 700; color: #f8fafc; }}
  header p {{ font-size: 13px; color: #94a3b8; }}
  #cd-select {{
    padding: 8px 14px; border-radius: 6px; border: 1px solid #475569;
    background: #0f172a; color: #e2e8f0; font-size: 14px; min-width: 340px;
    outline: none; cursor: pointer;
  }}
  #cd-select:focus {{ border-color: #ed8936; }}

  /* Main layout */
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}

  /* Top card row */
  .top-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px;
  }}

  /* Card base */
  .card {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 20px 24px;
  }}
  .card-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #ed8936; margin-bottom: 12px;
  }}

  /* Identity card */
  #identity-card .cd-name {{ font-size: 22px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  #identity-card .cd-full {{ font-size: 13px; color: #94a3b8; margin-bottom: 12px; }}
  .badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }}
  .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
  }}
  .badge-index {{ background: #7c2d12; color: #fed7aa; }}
  .badge-rank {{ background: #1e3a5f; color: #93c5fd; }}
  .badge-typology {{ background: #14532d; color: #86efac; }}
  .badge-pop {{ background: #334155; color: #cbd5e1; }}

  .summary-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .summary-stat {{ }}
  .summary-stat .ss-label {{ font-size: 11px; color: #64748b; }}
  .summary-stat .ss-value {{ font-size: 18px; font-weight: 700; color: #f8fafc; }}
  .summary-stat .ss-context {{ font-size: 11px; color: #64748b; }}

  /* Radar card */
  #radar-card {{ display: flex; flex-direction: column; }}
  #radar-card canvas {{ flex: 1; max-height: 320px; }}
  .radar-legend {{
    display: flex; gap: 16px; margin-top: 10px; justify-content: center;
  }}
  .radar-legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #94a3b8; }}
  .radar-legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

  /* Stats grid */
  .stats-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  }}
  .stat-card {{
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 14px 18px;
  }}
  .stat-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }}
  .stat-label {{ font-size: 13px; font-weight: 600; color: #cbd5e1; }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: #f8fafc; }}
  .stat-unit {{ font-size: 12px; color: #94a3b8; margin-left: 2px; }}
  .stat-desc {{ font-size: 11px; color: #94a3b8; margin-bottom: 8px; }}

  /* Position bar */
  .pos-bar-container {{ position: relative; height: 8px; background: #0f172a; border-radius: 4px; margin-top: 6px; }}
  .pos-bar-fill {{ position: absolute; left: 0; top: 0; height: 100%; border-radius: 4px; }}
  .pos-marker {{
    position: absolute; top: -3px; width: 3px; height: 14px;
    border-radius: 2px; transform: translateX(-50%);
  }}
  .pos-marker-cd {{ background: #ed8936; z-index: 3; }}
  .pos-marker-boro {{ background: #3b82f6; z-index: 2; }}
  .pos-marker-city {{ background: #475569; z-index: 1; }}
  .pos-labels {{ display: flex; justify-content: space-between; margin-top: 4px; }}
  .pos-labels span {{ font-size: 10px; color: #94a3b8; }}
  .pos-legend {{
    display: flex; gap: 12px; margin-top: 4px;
  }}
  .pos-legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 10px; color: #cbd5e1; }}
  .pos-legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}

  /* Section title */
  .section-title {{
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: #ed8936; margin: 24px 0 12px;
  }}

  /* Navigation links */
  .nav-links {{
    display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;
    justify-content: center;
  }}
  .nav-links a {{
    padding: 8px 16px; border-radius: 6px; border: 1px solid #334155;
    background: #1e293b; color: #cbd5e1; text-decoration: none;
    font-size: 13px; transition: all 0.15s;
  }}
  .nav-links a:hover {{ border-color: #ed8936; color: #ed8936; }}

  @media (max-width: 768px) {{
    .top-row {{ grid-template-columns: 1fr; }}
    .stats-grid {{ grid-template-columns: 1fr; }}
    #cd-select {{ min-width: 100%; }}
  }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Neighborhood Profiles</h1>
    <p>Nighttime conditions across 59 NYC community districts</p>
  </div>
  <select id="cd-select">
    {options_html}
  </select>
</header>
<main>
  <div class="nav-links">
    <a href="atlas.html">Atlas</a>
    <a href="profiles.html">Profiles</a>
    <a href="bivariate.html">Bivariate Maps</a>
    <a href="typologies.html">Typologies</a>
    <a href="findings.html">Findings</a>
  </div>
  <div class="top-row">
    <div class="card" id="identity-card">
      <div class="card-title">Community District</div>
      <div class="cd-name" id="p-cd-name"></div>
      <div class="cd-full" id="p-cd-full"></div>
      <div class="badge-row">
        <span class="badge badge-index" id="p-badge-index"></span>
        <span class="badge badge-rank" id="p-badge-rank"></span>
        <span class="badge badge-typology" id="p-badge-typology"></span>
        <span class="badge badge-pop" id="p-badge-pop"></span>
      </div>
      <div class="summary-stats" id="p-summary-stats"></div>
    </div>
    <div class="card" id="radar-card">
      <div class="card-title">Dimension Profile</div>
      <canvas id="radar-canvas"></canvas>
      <div class="radar-legend">
        <div class="radar-legend-item"><div class="radar-legend-dot" style="background:#ed8936;"></div>This CD</div>
        <div class="radar-legend-item"><div class="radar-legend-dot" style="background:#3b82f6;"></div>Borough avg</div>
        <div class="radar-legend-item"><div class="radar-legend-dot" style="background:#475569;"></div>City avg</div>
      </div>
    </div>
  </div>

  <div class="section-title">Key Nighttime Statistics</div>
  <div class="stats-grid" id="stats-grid"></div>
</main>

<script>
const profiles = {profiles_json};
const cityDims = {city_dims_json};
const boroDims = {boro_dims_json};
const dimLabels = {dim_labels_json};
const dimKeys = ['dim_noise_exposure','dim_safety','dim_lighting','dim_transit_access','dim_environmental','dim_services'];

// Index profiles by boro_cd
const profileMap = {{}};
profiles.forEach(p => {{ profileMap[p.boro_cd] = p; }});

// Radar chart
let radarChart = null;

function initRadar() {{
  const ctx = document.getElementById('radar-canvas').getContext('2d');
  radarChart = new Chart(ctx, {{
    type: 'radar',
    data: {{
      labels: dimLabels,
      datasets: [
        {{
          label: 'This CD',
          data: [0,0,0,0,0,0],
          backgroundColor: 'rgba(237,137,54,0.15)',
          borderColor: '#ed8936',
          borderWidth: 2,
          pointBackgroundColor: '#ed8936',
          pointRadius: 4,
        }},
        {{
          label: 'Borough avg',
          data: [0,0,0,0,0,0],
          backgroundColor: 'rgba(59,130,246,0.08)',
          borderColor: '#3b82f6',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointBackgroundColor: '#3b82f6',
          pointRadius: 3,
        }},
        {{
          label: 'City avg',
          data: [0,0,0,0,0,0],
          backgroundColor: 'rgba(71,85,105,0.08)',
          borderColor: '#475569',
          borderWidth: 1,
          borderDash: [2, 2],
          pointBackgroundColor: '#475569',
          pointRadius: 2,
        }},
      ],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        r: {{
          min: 0, max: 1,
          ticks: {{ display: false }},
          grid: {{ color: '#334155' }},
          angleLines: {{ color: '#334155' }},
          pointLabels: {{
            color: '#94a3b8', font: {{ size: 12 }},
          }},
        }},
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              return ctx.dataset.label + ': ' + (ctx.raw !== null ? ctx.raw.toFixed(2) : 'N/A');
            }}
          }}
        }},
      }},
    }},
  }});
}}

function fmt(val, decimals) {{
  if (val === null || val === undefined) return 'N/A';
  if (decimals === 0) return Math.round(val).toLocaleString();
  return val.toFixed(decimals || 1);
}}

function pctPosition(val, mn, mx) {{
  if (val === null || mn === null || mx === null || mx === mn) return 50;
  return Math.max(0, Math.min(100, ((val - mn) / (mx - mn)) * 100));
}}

function renderProfile(cd) {{
  const p = profileMap[cd];
  if (!p) return;

  // Identity
  document.getElementById('p-cd-name').textContent = p.cd_short;
  document.getElementById('p-cd-full').textContent = p.cd_label;
  document.getElementById('p-badge-index').textContent = 'Index: ' + fmt(p.nightscape_index, 1);
  document.getElementById('p-badge-rank').textContent = 'Rank ' + (p.nightscape_rank || '?') + ' / 59';
  document.getElementById('p-badge-typology').textContent = p.typology || 'Unknown';
  document.getElementById('p-badge-pop').textContent = 'Pop: ' + (p.population ? p.population.toLocaleString() : '?');

  // Summary stats (top 4)
  const summaryStats = [
    {{ label: 'Nightscape Index', value: fmt(p.nightscape_index, 1), context: 'City avg: ' + fmt(getCityMean('nightscape_index'), 1) }},
    {{ label: 'City Rank', value: p.nightscape_rank ? '#' + p.nightscape_rank : '?', context: 'of 59 districts' }},
    {{ label: 'Borough', value: p.borough_name, context: '' }},
    {{ label: 'Typology', value: p.typology || '?', context: '' }},
  ];
  const ssEl = document.getElementById('p-summary-stats');
  ssEl.innerHTML = summaryStats.map(s =>
    '<div class="summary-stat">' +
      '<div class="ss-label">' + s.label + '</div>' +
      '<div class="ss-value">' + s.value + '</div>' +
      '<div class="ss-context">' + s.context + '</div>' +
    '</div>'
  ).join('');

  // Radar
  const cdDims = p.dimensions.map(d => d !== null ? d : 0);
  const boroCode = p.borough_code;
  const boroDimVals = dimKeys.map(k => (boroDims[boroCode] && boroDims[boroCode][k]) || 0);
  const cityDimVals = dimKeys.map(k => cityDims[k] || 0);

  radarChart.data.datasets[0].data = cdDims;
  radarChart.data.datasets[1].data = boroDimVals;
  radarChart.data.datasets[2].data = cityDimVals;
  radarChart.update();

  // Stats grid
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = p.stats.filter(s => s.col !== 'nightscape_index').map(s => {{
    const val = s.value;
    const mn = s.city_min;
    const mx = s.city_max;
    const cdPct = pctPosition(val, mn, mx);
    const boroPct = pctPosition(s.boro_mean, mn, mx);
    const cityPct = pctPosition(s.city_mean, mn, mx);

    const valStr = val !== null ? fmt(val, val > 100 ? 0 : 1) : 'N/A';
    const diffFromCity = (val !== null && s.city_mean !== null)
      ? ((val - s.city_mean) / s.city_mean * 100)
      : null;
    let diffStr = '';
    if (diffFromCity !== null && Math.abs(diffFromCity) >= 1) {{
      const sign = diffFromCity > 0 ? '+' : '';
      const color = (s.higher_is_worse === true && diffFromCity > 15) ? '#ef4444' :
                    (s.higher_is_worse === true && diffFromCity < -15) ? '#22c55e' :
                    (s.higher_is_worse === false && diffFromCity > 15) ? '#22c55e' :
                    (s.higher_is_worse === false && diffFromCity < -15) ? '#ef4444' : '#94a3b8';
      diffStr = '<span style="color:' + color + ';font-size:12px;font-weight:600;"> ' + sign + Math.round(diffFromCity) + '% vs city</span>';
    }}

    return '<div class="stat-card">' +
      '<div class="stat-header">' +
        '<div class="stat-label">' + s.label + '</div>' +
        '<div><span class="stat-value">' + valStr + '</span><span class="stat-unit">' + s.unit + '</span>' + diffStr + '</div>' +
      '</div>' +
      '<div class="stat-desc">' + s.desc + '</div>' +
      '<div class="pos-bar-container">' +
        '<div class="pos-marker pos-marker-city" style="left:' + cityPct + '%;"></div>' +
        '<div class="pos-marker pos-marker-boro" style="left:' + boroPct + '%;"></div>' +
        (val !== null ? '<div class="pos-marker pos-marker-cd" style="left:' + cdPct + '%;"></div>' : '') +
      '</div>' +
      '<div class="pos-labels"><span>' + fmt(mn, mn > 100 ? 0 : 1) + '</span><span>' + fmt(mx, mx > 100 ? 0 : 1) + '</span></div>' +
      '<div class="pos-legend">' +
        '<div class="pos-legend-item"><div class="pos-legend-dot" style="background:#ed8936;"></div>This CD</div>' +
        '<div class="pos-legend-item"><div class="pos-legend-dot" style="background:#3b82f6;"></div>' + p.borough_name + ' avg</div>' +
        '<div class="pos-legend-item"><div class="pos-legend-dot" style="background:#475569;"></div>City avg</div>' +
      '</div>' +
    '</div>';
  }}).join('');
}}

function getCityMean(col) {{
  const all = profiles.map(p => {{
    const s = p.stats.find(s => s.col === col);
    return s ? s.city_mean : null;
  }});
  return all.find(v => v !== null);
}}

// --- Init ---
initRadar();

const select = document.getElementById('cd-select');
select.addEventListener('change', () => renderProfile(parseInt(select.value)));

// Default to first CD
renderProfile(parseInt(select.value));
</script>
</body>
</html>"""

    return html


def main():
    logger.info("Building neighborhood profiles")

    master = pd.read_parquet(REPORTS_DIR / "master_nightscape_df.parquet")
    index_df = pd.read_csv(TABLES_DIR / "nightscape_index.csv")
    typology_df = pd.read_csv(TABLES_DIR / "neighborhood_typologies.csv")
    lookup = pd.read_csv(GEO_DIR / "cd_lookup.csv")

    profiles, city_dims, boro_dims = build_profile_data(master, index_df, typology_df, lookup)
    logger.info(f"Built {len(profiles)} profiles")

    html = build_html(profiles, city_dims, boro_dims)

    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERACTIVE_DIR / "profiles.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Profiles written to {out_path} ({len(html)/1e3:.0f} KB)")

    print(f"\nProfiles built: {out_path}")
    print(f"  Districts: {len(profiles)}")
    print(f"  Stats per profile: {len(KEY_STATS)}")
    print(f"  File size: {len(html)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
