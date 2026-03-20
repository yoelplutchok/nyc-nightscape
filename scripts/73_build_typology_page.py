#!/usr/bin/env python3
"""
73_build_typology_page.py — Build interactive neighborhood typology page.

Generates a self-contained HTML page with:
  - Map colored by cluster typology
  - Radar/spider charts comparing each typology's dimension profile
  - Typology descriptions and member CD lists
  - Click-to-highlight CDs within each cluster

Output: outputs/interactive/typologies.html
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nightscape.paths import GEO_DIR, INTERACTIVE_DIR, REPORTS_DIR, TABLES_DIR
from nightscape.logging_utils import get_logger

logger = get_logger("73_build_typology")

# Typology colors
TYPO_COLORS = {
    "Quiet Suburban": "#22c55e",
    "Moderate Urban": "#3b82f6",
    "High-Need Urban": "#ef4444",
    "Bustling & Loud": "#f59e0b",
    "Mixed Residential": "#a855f7",
}

# Typology descriptions
TYPO_DESCRIPTIONS = {
    "Quiet Suburban": "The largest cluster (27 CDs), spanning most of Queens, Staten Island, and southern Brooklyn. These neighborhoods have low noise, low crime, minimal nightlife, but also limited late-night transit. They are the quietest, safest places to sleep in NYC — but getting home late at night can be a challenge.",
    "Moderate Urban": "Nine CDs concentrated in Manhattan (below 110th St) and northwest Brooklyn. These neighborhoods balance moderate noise and activity levels with good transit access. They represent the 'urban sweet spot' — lively but not overwhelming, with infrastructure that supports nighttime life.",
    "High-Need Urban": "The most concerning cluster (21 CDs), dominating the Bronx and extending into Central Brooklyn and Upper Manhattan. These neighborhoods face the highest nighttime crime, slowest EMS response, most fire incidents, and worst Nightscape Index scores — while also having higher poverty rates and fewer late-night services.",
    "Bustling & Loud": "A single CD (Midtown Manhattan, CD 105) so extreme it forms its own category. The highest noise, highest crime rate (driven by non-resident population), most nightlife, brightest lights, and most transit of any neighborhood. The city that never sleeps — literally.",
    "Mixed Residential": "A single CD (Pelham Parkway/Williamsbridge/Eastchester, CD 212) that defies the other categories — moderate across most dimensions but with a distinctive mix of residential character and elevated noise from transportation corridors.",
}

DIMENSIONS = [
    "dim_noise_exposure", "dim_safety", "dim_lighting",
    "dim_transit_access", "dim_environmental", "dim_services",
]
DIM_LABELS = [
    "Noise Exposure", "Safety", "Lighting",
    "Transit Access", "Environmental", "Services",
]

# Key comparison stats
COMPARE_STATS = [
    ("nightscape_index", "Nightscape Index", "float1"),
    ("crime_rate_per_1k", "Crime / 1k", "float1"),
    ("noise_obj_db_mean", "Noise (dB)", "float1"),
    ("ems_response_min_night", "EMS Response (min)", "float1"),
    ("rate_per_1k_pop", "Noise Complaints / 1k", "float1"),
    ("on_premises_rate_per_1k_pop", "Bars / 1k", "float1"),
    ("subway_stations_per_km2", "Subway / km\u00b2", "float1"),
    ("poverty_rate", "Poverty Rate", "pct"),
]


def build_data(geo, master, index_df, typology_df, lookup):
    """Build all data needed for the page."""
    gdf = geo.copy()
    gdf["boro_cd"] = gdf["boro_cd"].astype(int)
    master = master.copy()
    master["boro_cd"] = master["boro_cd"].astype(int)
    gdf = gdf.merge(master, on="boro_cd", how="left")

    index_df = index_df.copy()
    index_df["boro_cd"] = index_df["boro_cd"].astype(int)
    for c in index_df.columns:
        if c != "boro_cd" and c not in gdf.columns:
            gdf = gdf.merge(index_df[["boro_cd", c]], on="boro_cd", how="left")

    typology_df = typology_df.copy()
    typology_df["boro_cd"] = typology_df["boro_cd"].astype(int)
    if "typology" not in gdf.columns:
        gdf = gdf.merge(typology_df[["boro_cd", "typology", "cluster"]], on="boro_cd", how="left")

    lookup = lookup.copy()
    lookup["boro_cd"] = lookup["boro_cd"].astype(int)
    gdf = gdf.merge(lookup[["boro_cd", "cd_short", "cd_label", "borough_name"]], on="boro_cd", how="left")

    for col in gdf.select_dtypes(include=[np.number]).columns:
        gdf[col] = gdf[col].replace([np.inf, -np.inf], np.nan)

    gdf = gdf.to_crs(epsg=4326)
    geojson = json.loads(gdf.to_json())

    # Compute per-typology dimension averages
    merged = gdf.drop(columns="geometry")
    typology_profiles = {}
    for typo_name in sorted(TYPO_COLORS.keys()):
        tdf = merged[merged["typology"] == typo_name]
        profile = {
            "name": typo_name,
            "n_cds": len(tdf),
            "color": TYPO_COLORS.get(typo_name, "#888"),
            "description": TYPO_DESCRIPTIONS.get(typo_name, ""),
            "dimensions": [round(float(tdf[d].mean()), 4) if len(tdf) > 0 else 0 for d in DIMENSIONS],
            "cds": sorted(tdf["boro_cd"].tolist()),
            "cd_names": sorted(tdf["cd_short"].tolist()),
            "stats": {},
        }
        for col, label, fmt in COMPARE_STATS:
            vals = tdf[col].dropna()
            profile["stats"][col] = {
                "label": label,
                "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                "fmt": fmt,
            }
        typology_profiles[typo_name] = profile

    # City averages for dimensions
    city_dims = [round(float(merged[d].mean()), 4) for d in DIMENSIONS]

    return geojson, typology_profiles, city_dims


def build_html(geojson, typology_profiles, city_dims):
    geojson_str = json.dumps(geojson)
    profiles_str = json.dumps(typology_profiles)
    city_dims_str = json.dumps(city_dims)
    dim_labels_str = json.dumps(DIM_LABELS)
    typo_colors_str = json.dumps(TYPO_COLORS)

    # Typology order for display
    display_order = ["Quiet Suburban", "Moderate Urban", "High-Need Urban", "Bustling & Loud", "Mixed Residential"]

    # Build typology cards
    cards = []
    for typo_name in display_order:
        tp = typology_profiles[typo_name]
        color = tp["color"]
        cd_list = ", ".join(tp["cd_names"])

        stats_html = ""
        for col, label, fmt in COMPARE_STATS:
            s = tp["stats"].get(col, {})
            val = s.get("mean")
            if val is None:
                val_str = "N/A"
            elif fmt == "pct":
                val_str = f"{val*100:.1f}%"
            else:
                val_str = f"{val:.1f}"
            stats_html += f'<div class="tc-stat"><span class="tc-stat-label">{label}</span><span class="tc-stat-value">{val_str}</span></div>'

        cards.append(f"""
        <div class="typo-card" data-typology="{typo_name}">
          <div class="tc-header">
            <div class="tc-dot" style="background:{color};"></div>
            <div class="tc-name">{typo_name}</div>
            <div class="tc-count">{tp['n_cds']} CDs</div>
          </div>
          <div class="tc-desc">{tp['description']}</div>
          <div class="tc-stats">{stats_html}</div>
          <div class="tc-cds"><span class="tc-cds-label">Districts:</span> {cd_list}</div>
        </div>""")
    cards_html = "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Nightscape — Neighborhood Typologies</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0;
  }}

  #app {{ display: flex; height: 100vh; }}

  /* Left panel */
  #panel {{
    width: 480px; min-width: 480px; background: #1e293b; overflow-y: auto;
    border-right: 1px solid #334155; display: flex; flex-direction: column;
  }}
  #panel-header {{
    padding: 16px 20px; border-bottom: 1px solid #334155;
    background: #1a365d; position: sticky; top: 0; z-index: 10;
  }}
  #panel-header h1 {{ font-size: 18px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  #panel-header p {{ font-size: 12px; color: #94a3b8; line-height: 1.5; }}

  #panel-content {{ padding: 16px 20px; }}

  /* Radar section */
  #radar-section {{
    background: #162032; border: 1px solid #334155; border-radius: 10px;
    padding: 16px; margin-bottom: 20px;
  }}
  #radar-section .section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #ed8936; margin-bottom: 10px;
  }}
  #radar-canvas-container {{ height: 280px; }}
  .radar-legend {{
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; justify-content: center;
  }}
  .radar-legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: #cbd5e1; cursor: pointer; }}
  .radar-legend-item:hover {{ color: #f8fafc; }}
  .radar-legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

  /* Typology cards */
  .typo-card {{
    background: #0f172a; border: 1px solid #334155; border-radius: 10px;
    padding: 16px; margin-bottom: 12px; cursor: pointer;
    transition: border-color 0.2s;
  }}
  .typo-card:hover {{ border-color: #64748b; }}
  .typo-card.active {{ border-color: #ed8936; }}
  .tc-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .tc-dot {{ width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }}
  .tc-name {{ font-size: 15px; font-weight: 700; color: #f8fafc; flex: 1; }}
  .tc-count {{ font-size: 12px; color: #94a3b8; background: #1e293b; padding: 2px 10px; border-radius: 12px; }}
  .tc-desc {{ font-size: 12px; color: #94a3b8; line-height: 1.6; margin-bottom: 10px; }}
  .tc-stats {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; margin-bottom: 10px;
  }}
  .tc-stat {{ display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }}
  .tc-stat-label {{ color: #94a3b8; }}
  .tc-stat-value {{ color: #cbd5e1; font-weight: 600; }}
  .tc-cds {{ font-size: 11px; color: #64748b; line-height: 1.5; }}
  .tc-cds-label {{ color: #94a3b8; font-weight: 600; }}

  /* Map */
  #map-area {{ flex: 1; display: flex; flex-direction: column; }}
  #map {{ flex: 1; background: #0f172a; position: relative; }}

  /* Map legend */
  #map-legend {{
    position: absolute; bottom: 30px; right: 20px; z-index: 1000;
    background: rgba(30,41,59,0.95); border: 1px solid #475569;
    border-radius: 8px; padding: 12px 16px;
    backdrop-filter: blur(8px);
  }}
  #map-legend h4 {{
    font-size: 11px; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 8px;
  }}
  .ml-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; color: #cbd5e1; cursor: pointer; }}
  .ml-row:hover {{ color: #f8fafc; }}
  .ml-dot {{ width: 14px; height: 14px; border-radius: 50%; }}

  /* Tooltip */
  .leaflet-tooltip {{
    background: rgba(15,23,42,0.95) !important; color: #f8fafc !important;
    border: 1px solid #475569 !important; border-radius: 6px !important;
    padding: 8px 12px !important; font-family: inherit !important;
    font-size: 13px !important;
  }}
  .tt-name {{ font-weight: 700; margin-bottom: 2px; }}
  .tt-typo {{ font-size: 12px; }}

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

  @media (max-width: 900px) {{
    #panel {{ width: 380px; min-width: 380px; }}
  }}
</style>
</head>
<body>
<div id="app">
  <div id="panel">
    <div id="panel-header">
      <h1>Neighborhood Typologies</h1>
      <p>K-means clustering grouped NYC's 59 community districts into 5 distinct nighttime profiles based on noise, safety, lighting, transit, environmental quality, and services. Click a typology to highlight it on the map.</p>
      <div class="nav-links" style="margin-top:10px;">
        <a href="atlas.html">Atlas</a>
        <a href="profiles.html">Profiles</a>
        <a href="bivariate.html">Bivariate Maps</a>
        <a href="typologies.html">Typologies</a>
        <a href="findings.html">Findings</a>
      </div>
    </div>
    <div id="panel-content">
      <div id="radar-section">
        <div class="section-title">Dimension Profiles (all typologies)</div>
        <div id="radar-canvas-container">
          <canvas id="radar-canvas"></canvas>
        </div>
        <div class="radar-legend" id="radar-legend"></div>
      </div>
      {cards_html}
    </div>
  </div>
  <div id="map-area">
    <div id="map">
      <div id="map-legend"></div>
    </div>
  </div>
</div>

<script>
const geojsonData = {geojson_str};
const typoProfiles = {profiles_str};
const cityDims = {city_dims_str};
const dimLabels = {dim_labels_str};
const typoColors = {typo_colors_str};
const displayOrder = ['Quiet Suburban','Moderate Urban','High-Need Urban','Bustling & Loud','Mixed Residential'];

let geojsonLayer = null;
let activeTypology = null;

// --- Map ---
const map = L.map('map', {{
  zoomControl: true, attributionControl: false,
}}).setView([40.7128, -73.95], 11);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 18, subdomains: 'abcd',
}}).addTo(map);

function getTypoColor(typoName) {{
  return typoColors[typoName] || '#64748b';
}}

function styleFeature(feature) {{
  const typo = feature.properties.typology;
  const isActive = !activeTypology || typo === activeTypology;
  return {{
    fillColor: getTypoColor(typo),
    weight: isActive ? 2 : 1,
    opacity: isActive ? 0.9 : 0.3,
    color: isActive ? '#f8fafc' : '#334155',
    fillOpacity: isActive ? 0.7 : 0.15,
  }};
}}

function highlightFeature(e) {{
  e.target.setStyle({{ weight: 3, color: '#ed8936', fillOpacity: 0.85 }});
  e.target.bringToFront();
}}

function resetHighlight(e) {{
  geojsonLayer.resetStyle(e.target);
}}

function initMap() {{
  geojsonLayer = L.geoJSON(geojsonData, {{
    style: styleFeature,
    onEachFeature: function(feature, layer) {{
      const p = feature.properties;
      const cdName = p.cd_short || ('CD ' + p.boro_cd);
      const typo = p.typology || 'Unknown';
      layer.bindTooltip(
        '<div class="tt-name">' + cdName + '</div>' +
        '<div class="tt-typo" style="color:' + getTypoColor(typo) + ';">' + typo + '</div>',
        {{ sticky: true, direction: 'top', offset: [0, -10] }}
      );
      layer.on({{ mouseover: highlightFeature, mouseout: resetHighlight }});
    }}
  }}).addTo(map);
  map.fitBounds(geojsonLayer.getBounds(), {{ padding: [20, 20] }});
}}

function restyleMap() {{
  geojsonLayer.eachLayer(layer => geojsonLayer.resetStyle(layer));
}}

// --- Map legend ---
function buildMapLegend() {{
  const leg = document.getElementById('map-legend');
  let html = '<h4>Typology</h4>';
  displayOrder.forEach(name => {{
    const tp = typoProfiles[name];
    if (!tp) return;
    html += '<div class="ml-row" data-typology="' + name + '">' +
      '<div class="ml-dot" style="background:' + tp.color + ';"></div>' +
      name + ' (' + tp.n_cds + ')' +
    '</div>';
  }});
  leg.innerHTML = html;

  leg.querySelectorAll('.ml-row').forEach(row => {{
    row.addEventListener('click', () => toggleTypology(row.dataset.typology));
  }});
}}

// --- Radar chart ---
let radarChart = null;

function initRadar() {{
  const ctx = document.getElementById('radar-canvas').getContext('2d');
  const datasets = [];

  displayOrder.forEach(name => {{
    const tp = typoProfiles[name];
    if (!tp) return;
    datasets.push({{
      label: name,
      data: tp.dimensions,
      backgroundColor: tp.color + '18',
      borderColor: tp.color,
      borderWidth: 2,
      pointBackgroundColor: tp.color,
      pointRadius: 3,
    }});
  }});

  // City average
  datasets.push({{
    label: 'City Average',
    data: cityDims,
    backgroundColor: 'rgba(148,163,184,0.05)',
    borderColor: '#64748b',
    borderWidth: 1,
    borderDash: [4, 3],
    pointBackgroundColor: '#64748b',
    pointRadius: 2,
  }});

  radarChart = new Chart(ctx, {{
    type: 'radar',
    data: {{ labels: dimLabels, datasets: datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        r: {{
          min: 0, max: 1,
          ticks: {{ display: false }},
          grid: {{ color: '#334155' }},
          angleLines: {{ color: '#334155' }},
          pointLabels: {{ color: '#94a3b8', font: {{ size: 11 }} }},
        }},
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.dataset.label + ': ' + (ctx.raw !== null ? ctx.raw.toFixed(2) : 'N/A'),
          }},
        }},
      }},
    }},
  }});

  // Build custom legend
  const legEl = document.getElementById('radar-legend');
  let legHtml = '';
  displayOrder.forEach(name => {{
    const tp = typoProfiles[name];
    if (!tp) return;
    legHtml += '<div class="radar-legend-item" data-typology="' + name + '">' +
      '<div class="radar-legend-dot" style="background:' + tp.color + ';"></div>' +
      name +
    '</div>';
  }});
  legHtml += '<div class="radar-legend-item"><div class="radar-legend-dot" style="background:#64748b;"></div>City avg</div>';
  legEl.innerHTML = legHtml;

  legEl.querySelectorAll('.radar-legend-item[data-typology]').forEach(item => {{
    item.addEventListener('click', () => toggleTypology(item.dataset.typology));
  }});
}}

// --- Toggle typology highlight ---
function toggleTypology(name) {{
  if (activeTypology === name) {{
    activeTypology = null;
  }} else {{
    activeTypology = name;
  }}
  restyleMap();

  // Update card active state
  document.querySelectorAll('.typo-card').forEach(card => {{
    card.classList.toggle('active', card.dataset.typology === activeTypology);
  }});
}}

// --- Card clicks ---
document.querySelectorAll('.typo-card').forEach(card => {{
  card.addEventListener('click', () => toggleTypology(card.dataset.typology));
}});

// --- Init ---
initMap();
buildMapLegend();
initRadar();
</script>
</body>
</html>"""
    return html


def main():
    logger.info("Building typology page")

    geo = gpd.read_file(GEO_DIR / "cd59.geojson")
    master = pd.read_parquet(REPORTS_DIR / "master_nightscape_df.parquet")
    index_df = pd.read_csv(TABLES_DIR / "nightscape_index.csv")
    typology_df = pd.read_csv(TABLES_DIR / "neighborhood_typologies.csv")
    lookup = pd.read_csv(GEO_DIR / "cd_lookup.csv")

    geojson, typology_profiles, city_dims = build_data(
        geo, master, index_df, typology_df, lookup
    )
    logger.info(f"Built {len(typology_profiles)} typology profiles")

    html = build_html(geojson, typology_profiles, city_dims)

    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERACTIVE_DIR / "typologies.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Typologies page written to {out_path} ({len(html)/1e6:.1f} MB)")

    print(f"\nTypologies built: {out_path}")
    print(f"  Clusters: {len(typology_profiles)}")
    for name, tp in sorted(typology_profiles.items()):
        print(f"    {name}: {tp['n_cds']} CDs")
    print(f"  File size: {len(html)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
