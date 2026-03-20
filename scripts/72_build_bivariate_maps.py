#!/usr/bin/env python3
"""
72_build_bivariate_maps.py — Build interactive bivariate choropleth maps.

Generates a self-contained HTML page with bivariate maps showing
the intersection of two nighttime variables simultaneously using
a 3x3 color grid. Users can select from preset pairings or build
custom combinations.

Output: outputs/interactive/bivariate.html
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

logger = get_logger("72_build_bivariate")

# Preset bivariate pairings: (varX, varY, title, narrative)
PRESETS = [
    {
        "x": "crime_rate_per_1k",
        "y": "radiance_raw",
        "title": "Crime vs Nighttime Brightness",
        "narrative": "Do darker neighborhoods see more crime? This map overlays nighttime crime rates with satellite-measured light radiance. Areas that are both dark (low radiance) and high-crime appear in the upper-left of the legend — potential targets for lighting interventions. Brightly lit, low-crime areas cluster in lower Manhattan and parts of Queens.",
    },
    {
        "x": "violent_rate_per_1k",
        "y": "ems_response_min_night",
        "title": "Violent Crime vs EMS Response Time",
        "narrative": "The cruelest spatial mismatch: neighborhoods with the most nighttime violence also tend to wait longest for an ambulance. This map reveals where high violent crime co-occurs with slow EMS response — a double burden concentrated in the South Bronx and parts of Central Brooklyn.",
    },
    {
        "x": "noise_obj_db_mean",
        "y": "poverty_rate",
        "title": "Noise Levels vs Poverty",
        "narrative": "Who bears the burden of nighttime noise? This map shows that high objective noise levels and high poverty rates are not randomly distributed. Low-income neighborhoods in the Bronx and Upper Manhattan experience some of the loudest nighttime environments, while affluent areas enjoy relative quiet.",
    },
    {
        "x": "subway_stations_per_km2",
        "y": "poverty_rate",
        "title": "Transit Access vs Poverty",
        "narrative": "Late-night transit is an equity issue. This map reveals where poverty and transit scarcity overlap — communities that most need affordable nighttime transportation options often have the fewest subway stations. The outer Bronx, eastern Queens, and Staten Island show the starkest gaps.",
    },
    {
        "x": "on_premises_rate_per_1k_pop",
        "y": "rate_per_1k_pop",
        "title": "Nightlife Density vs Noise Complaints",
        "narrative": "Where nightlife meets residential life, friction follows. This map overlays bar/club density with nighttime noise complaint rates. The hottest spots — high in both — mark the neighborhoods where entertainment economy and residential sleep quality are most in conflict.",
    },
    {
        "x": "fire_incidents_night_per_1k",
        "y": "nycha_pop_share",
        "title": "Fire Incidents vs Public Housing",
        "narrative": "Nighttime fire incidents are sharply concentrated in neighborhoods with large NYCHA (public housing) populations. Aging infrastructure, deferred maintenance, and high residential density contribute to elevated fire risk in public housing communities.",
    },
    {
        "x": "crash_rate_per_1k",
        "y": "on_premises_rate_per_1k_pop",
        "title": "Nighttime Crashes vs Bar Density",
        "narrative": "Does nightlife proximity increase crash risk? This map compares nighttime motor vehicle crash rates with bar and club density. Areas with both high bar density and high crash rates suggest a connection between alcohol availability and road danger after dark.",
    },
    {
        "x": "streetlight_complaints_per_km2",
        "y": "crime_rate_per_1k",
        "title": "Streetlight Issues vs Crime",
        "narrative": "Broken streetlights and nighttime crime: coincidence or connection? This map overlays 311 streetlight outage complaints with crime rates. Neighborhoods reporting more lighting problems tend to have higher crime — suggesting that infrastructure neglect and safety outcomes are linked.",
    },
]

# Variables available for custom selection
VARIABLES = [
    ("nightscape_index", "Nightscape Index"),
    ("crime_rate_per_1k", "Crime Rate / 1k"),
    ("violent_rate_per_1k", "Violent Crime / 1k"),
    ("noise_obj_db_mean", "Mean Noise (dB)"),
    ("rate_per_1k_pop", "Noise Complaints / 1k"),
    ("radiance_raw", "VIIRS Radiance"),
    ("streetlight_complaints_per_km2", "Streetlight Issues / km\u00b2"),
    ("crash_rate_per_1k", "Crash Rate / 1k"),
    ("fire_incidents_night_per_1k", "Fire Incidents / 1k"),
    ("ems_response_min_night", "EMS Response (min)"),
    ("poverty_rate", "Poverty Rate"),
    ("pct_nonhisp_black", "% Non-Hispanic Black"),
    ("pct_hispanic", "% Hispanic"),
    ("rent_burden_rate", "Rent Burden Rate"),
    ("subway_stations_per_km2", "Subway Stations / km\u00b2"),
    ("late_night_entries_per_1k", "Late-Night Subway / 1k"),
    ("total_ride_pickups_night_per_1k", "Ride Pickups / 1k"),
    ("on_premises_rate_per_1k_pop", "Bars & Clubs / 1k"),
    ("restaurants_per_1k", "Restaurants / 1k"),
    ("nycha_pop_share", "NYCHA Pop. Share"),
    ("shots_per_year", "Shots Fired / yr"),
    ("arrest_rate_per_1k", "Arrest Rate / 1k"),
    ("no2_mean_primary", "NO\u2082 (ppb)"),
    ("pm25_mean_primary", "PM2.5 (\u00b5g/m\u00b3)"),
    ("non_noise_311_per_1k", "Non-Noise 311 / 1k"),
    ("homeless_per_1k", "Homeless Reports / 1k"),
    ("wifi_hotspots_per_km2", "Wi-Fi Hotspots / km\u00b2"),
    ("energy_use_per_sqft", "Energy Use (kBtu/sqft)"),
]


def build_geojson(geo, master, lookup):
    """Merge data into GeoJSON."""
    gdf = geo.copy()
    gdf["boro_cd"] = gdf["boro_cd"].astype(int)
    master = master.copy()
    master["boro_cd"] = master["boro_cd"].astype(int)
    gdf = gdf.merge(master, on="boro_cd", how="left")
    lookup = lookup.copy()
    lookup["boro_cd"] = lookup["boro_cd"].astype(int)
    gdf = gdf.merge(lookup[["boro_cd", "cd_short", "cd_label"]], on="boro_cd", how="left")
    for col in gdf.select_dtypes(include=[np.number]).columns:
        gdf[col] = gdf[col].replace([np.inf, -np.inf], np.nan)
    gdf = gdf.to_crs(epsg=4326)
    return json.loads(gdf.to_json())


def build_html(geojson, presets, variables):
    """Generate the bivariate maps HTML."""

    geojson_str = json.dumps(geojson)
    presets_str = json.dumps(presets)
    variables_str = json.dumps(variables)

    # Preset buttons
    preset_btns = []
    for i, p in enumerate(presets):
        preset_btns.append(
            f'<button class="preset-btn" data-idx="{i}">{p["title"]}</button>'
        )
    preset_html = "\n".join(preset_btns)

    # Variable options
    var_options = []
    for col, label in variables:
        var_options.append(f'<option value="{col}">{label}</option>')
    var_options_html = "\n".join(var_options)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Nightscape — Bivariate Maps</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0;
  }}

  #app {{ display: flex; height: 100vh; }}

  /* Sidebar */
  #sidebar {{
    width: 340px; min-width: 340px; background: #1e293b; overflow-y: auto;
    border-right: 1px solid #334155; display: flex; flex-direction: column;
  }}
  #sidebar-header {{
    padding: 16px 20px; border-bottom: 1px solid #334155;
    background: #1a365d; position: sticky; top: 0; z-index: 10;
  }}
  #sidebar-header h1 {{ font-size: 18px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  #sidebar-header p {{ font-size: 12px; color: #94a3b8; }}

  #sidebar-content {{ padding: 16px 20px; flex: 1; }}

  .section-label {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #ed8936; margin-bottom: 8px; margin-top: 16px;
  }}
  .section-label:first-child {{ margin-top: 0; }}

  .preset-btn {{
    display: block; width: 100%; text-align: left;
    padding: 10px 14px; margin-bottom: 6px; border-radius: 8px;
    border: 1px solid #334155; background: #0f172a; color: #cbd5e1;
    font-size: 13px; cursor: pointer; transition: all 0.15s;
  }}
  .preset-btn:hover {{ border-color: #ed8936; color: #f8fafc; }}
  .preset-btn.active {{ border-color: #ed8936; background: #1a365d; color: #ed8936; font-weight: 600; }}

  .custom-row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
  .custom-row label {{ font-size: 12px; color: #94a3b8; min-width: 16px; font-weight: 700; }}
  .custom-row select {{
    flex: 1; padding: 7px 10px; border-radius: 6px;
    border: 1px solid #475569; background: #0f172a; color: #e2e8f0;
    font-size: 12px; outline: none;
  }}
  .custom-row select:focus {{ border-color: #ed8936; }}
  #btn-apply {{
    width: 100%; padding: 8px; border-radius: 6px; border: none;
    background: #ed8936; color: #0f172a; font-weight: 700; font-size: 13px;
    cursor: pointer; margin-top: 4px;
  }}
  #btn-apply:hover {{ background: #f6ad55; }}

  /* Narrative */
  #narrative {{
    margin-top: 16px; padding: 14px; background: #162032; border-radius: 8px;
    border: 1px solid #334155;
  }}
  #narrative-title {{ font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }}
  #narrative-text {{ font-size: 12px; color: #cbd5e1; line-height: 1.6; }}

  /* Map area */
  #map-area {{ flex: 1; display: flex; flex-direction: column; }}
  #map {{ flex: 1; background: #0f172a; position: relative; }}

  /* Bivariate legend */
  #biv-legend {{
    position: absolute; bottom: 30px; right: 20px; z-index: 1000;
    background: rgba(30,41,59,0.95); border: 1px solid #475569;
    border-radius: 10px; padding: 14px 18px;
    backdrop-filter: blur(8px);
  }}
  #biv-legend-title {{ font-size: 11px; color: #94a3b8; margin-bottom: 8px; text-align: center; }}
  .biv-grid {{
    display: grid; grid-template-columns: auto repeat(3, 28px); gap: 0;
    align-items: center;
  }}
  .biv-cell {{
    width: 28px; height: 28px; border: 1px solid rgba(255,255,255,0.1);
  }}
  .biv-y-label {{
    writing-mode: vertical-rl; transform: rotate(180deg);
    font-size: 10px; color: #cbd5e1; text-align: center;
    padding-right: 6px; grid-row: 1 / 4; white-space: nowrap;
  }}
  .biv-x-labels {{
    display: grid; grid-template-columns: auto repeat(3, 28px); gap: 0;
    margin-top: 2px;
  }}
  .biv-x-label {{ font-size: 10px; color: #cbd5e1; text-align: center; }}
  .biv-corner {{ }}

  /* Info panel */
  #info-panel {{
    position: absolute; top: 20px; right: 20px; z-index: 1000;
    background: rgba(30,41,59,0.95); border: 1px solid #475569;
    border-radius: 10px; padding: 14px 18px; width: 260px;
    backdrop-filter: blur(8px);
  }}
  #info-panel.hidden {{ display: none; }}
  #info-panel h3 {{ font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 2px; }}
  #info-panel .info-sub {{ font-size: 12px; color: #94a3b8; margin-bottom: 10px; }}
  .info-stat {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1e293b; }}
  .info-stat-label {{ font-size: 12px; color: #94a3b8; }}
  .info-stat-value {{ font-size: 13px; font-weight: 600; color: #f8fafc; }}
  .info-close {{
    position: absolute; top: 8px; right: 12px; background: none; border: none;
    color: #64748b; font-size: 18px; cursor: pointer;
  }}
  .info-close:hover {{ color: #f8fafc; }}

  /* Tooltip */
  .leaflet-tooltip {{
    background: rgba(15,23,42,0.95) !important; color: #f8fafc !important;
    border: 1px solid #475569 !important; border-radius: 6px !important;
    padding: 8px 12px !important; font-family: inherit !important;
    font-size: 13px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
  }}
  .tt-name {{ font-weight: 700; margin-bottom: 4px; }}
  .tt-row {{ display: flex; justify-content: space-between; gap: 12px; font-size: 12px; }}
  .tt-val {{ color: #ed8936; font-weight: 600; }}

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
    #sidebar {{ width: 280px; min-width: 280px; }}
  }}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">
      <h1>Bivariate Maps</h1>
      <p>Explore how two nighttime variables intersect across NYC's 59 community districts</p>
      <div class="nav-links" style="margin-top:10px;">
        <a href="atlas.html">Atlas</a>
        <a href="profiles.html">Profiles</a>
        <a href="bivariate.html">Bivariate Maps</a>
        <a href="typologies.html">Typologies</a>
        <a href="findings.html">Findings</a>
      </div>
    </div>
    <div id="sidebar-content">
      <div class="section-label">Preset Pairings</div>
      {preset_html}

      <div class="section-label">Custom Pairing</div>
      <div class="custom-row">
        <label>X</label>
        <select id="sel-x">{var_options_html}</select>
      </div>
      <div class="custom-row">
        <label>Y</label>
        <select id="sel-y">{var_options_html}</select>
      </div>
      <button id="btn-apply">Apply Custom Pairing</button>

      <div id="narrative">
        <div id="narrative-title"></div>
        <div id="narrative-text"></div>
      </div>
    </div>
  </div>
  <div id="map-area">
    <div id="map">
      <div id="info-panel" class="hidden">
        <button class="info-close" id="info-close">&times;</button>
        <h3 id="info-cd-name"></h3>
        <div class="info-sub" id="info-cd-label"></div>
        <div id="info-stats"></div>
      </div>
      <div id="biv-legend"></div>
    </div>
  </div>
</div>

<script>
const geojsonData = {geojson_str};
const presets = {presets_str};
const variables = {variables_str};

// --- 3x3 bivariate color scheme ---
// Rows = Y (bottom to top), Cols = X (left to right)
// low-low  low-mid  low-high   (Y-low row)
// mid-low  mid-mid  mid-high   (Y-mid row)
// high-low high-mid high-high  (Y-high row)
const COLORS = [
  ['#e8e8e8', '#b5c0da', '#6c83b5'],  // Y-low
  ['#b8d6be', '#90b2b3', '#567994'],  // Y-mid
  ['#73ae80', '#5a9178', '#2a5a5b'],  // Y-high
];

let currentX = 'crime_rate_per_1k';
let currentY = 'radiance_raw';
let currentXLabel = 'Crime Rate / 1k';
let currentYLabel = 'VIIRS Radiance';
let geojsonLayer = null;
let xBreaks = [];
let yBreaks = [];

const map = L.map('map', {{
  zoomControl: true, attributionControl: false,
}}).setView([40.7128, -73.95], 11);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 18, subdomains: 'abcd',
}}).addTo(map);

function getVarLabel(col) {{
  const v = variables.find(v => v[0] === col);
  return v ? v[1] : col;
}}

function tercileBreaks(col) {{
  const vals = geojsonData.features
    .map(f => f.properties[col])
    .filter(v => v !== null && v !== undefined && !isNaN(v))
    .sort((a, b) => a - b);
  if (vals.length < 3) return [0, 0];
  const t1 = vals[Math.floor(vals.length / 3)];
  const t2 = vals[Math.floor(vals.length * 2 / 3)];
  return [t1, t2];
}}

function getBivClass(val, breaks) {{
  if (val === null || val === undefined || isNaN(val)) return -1;
  if (val <= breaks[0]) return 0;
  if (val <= breaks[1]) return 1;
  return 2;
}}

function getBivColor(xVal, yVal) {{
  const xc = getBivClass(xVal, xBreaks);
  const yc = getBivClass(yVal, yBreaks);
  if (xc < 0 || yc < 0) return '#334155';
  return COLORS[yc][xc];
}}

function fmtVal(v) {{
  if (v === null || v === undefined || isNaN(v)) return 'N/A';
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(2);
}}

function updateLegend() {{
  const leg = document.getElementById('biv-legend');
  // Build 3x3 grid (Y rows top-to-bottom = high, mid, low)
  let html = '<div id="biv-legend-title">' + currentXLabel + ' &rarr; / ' + currentYLabel + ' &uarr;</div>';
  html += '<div class="biv-grid">';
  // Y axis label
  html += '<div class="biv-y-label">' + currentYLabel + '</div>';
  // Rows: high Y at top
  for (let yr = 2; yr >= 0; yr--) {{
    for (let xc = 0; xc < 3; xc++) {{
      html += '<div class="biv-cell" style="background:' + COLORS[yr][xc] + ';"></div>';
    }}
  }}
  html += '</div>';
  // X axis labels
  html += '<div class="biv-x-labels">';
  html += '<div class="biv-corner"></div>';
  html += '<div class="biv-x-label">Low</div>';
  html += '<div class="biv-x-label">Mid</div>';
  html += '<div class="biv-x-label">High</div>';
  html += '</div>';
  leg.innerHTML = html;
}}

function styleFeature(feature) {{
  const xVal = feature.properties[currentX];
  const yVal = feature.properties[currentY];
  return {{
    fillColor: getBivColor(xVal, yVal),
    weight: 1.5, opacity: 0.8, color: '#475569', fillOpacity: 0.8,
  }};
}}

function highlightFeature(e) {{
  e.target.setStyle({{ weight: 3, color: '#ed8936', fillOpacity: 0.9 }});
  e.target.bringToFront();
}}

function resetHighlight(e) {{
  geojsonLayer.resetStyle(e.target);
}}

function showInfo(e) {{
  const props = e.target.feature.properties;
  const panel = document.getElementById('info-panel');
  panel.classList.remove('hidden');
  document.getElementById('info-cd-name').textContent = props.cd_short || ('CD ' + props.boro_cd);
  document.getElementById('info-cd-label').textContent = props.cd_label || '';
  document.getElementById('info-stats').innerHTML =
    '<div class="info-stat"><span class="info-stat-label">' + currentXLabel + '</span><span class="info-stat-value">' + fmtVal(props[currentX]) + '</span></div>' +
    '<div class="info-stat"><span class="info-stat-label">' + currentYLabel + '</span><span class="info-stat-value">' + fmtVal(props[currentY]) + '</span></div>';
}}

function renderMap() {{
  xBreaks = tercileBreaks(currentX);
  yBreaks = tercileBreaks(currentY);
  currentXLabel = getVarLabel(currentX);
  currentYLabel = getVarLabel(currentY);

  if (geojsonLayer) {{
    map.removeLayer(geojsonLayer);
  }}

  geojsonLayer = L.geoJSON(geojsonData, {{
    style: styleFeature,
    onEachFeature: function(feature, layer) {{
      const p = feature.properties;
      const cdName = p.cd_short || ('CD ' + p.boro_cd);
      layer.bindTooltip(
        '<div class="tt-name">' + cdName + '</div>' +
        '<div class="tt-row"><span>' + currentXLabel + '</span><span class="tt-val">' + fmtVal(p[currentX]) + '</span></div>' +
        '<div class="tt-row"><span>' + currentYLabel + '</span><span class="tt-val">' + fmtVal(p[currentY]) + '</span></div>',
        {{ sticky: true, direction: 'top', offset: [0, -10] }}
      );
      layer.on({{ mouseover: highlightFeature, mouseout: resetHighlight, click: showInfo }});
    }}
  }}).addTo(map);

  map.fitBounds(geojsonLayer.getBounds(), {{ padding: [20, 20] }});
  updateLegend();
  document.getElementById('info-panel').classList.add('hidden');
}}

function setNarrative(title, text) {{
  document.getElementById('narrative-title').textContent = title;
  document.getElementById('narrative-text').textContent = text;
  document.getElementById('narrative').style.display = '';
}}

function clearNarrative() {{
  document.getElementById('narrative').style.display = 'none';
}}

// --- Preset buttons ---
document.querySelectorAll('.preset-btn').forEach(btn => {{
  btn.addEventListener('click', function() {{
    const idx = parseInt(this.dataset.idx);
    const p = presets[idx];
    currentX = p.x;
    currentY = p.y;
    document.getElementById('sel-x').value = p.x;
    document.getElementById('sel-y').value = p.y;
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    setNarrative(p.title, p.narrative);
    renderMap();
  }});
}});

// --- Custom apply ---
document.getElementById('btn-apply').addEventListener('click', function() {{
  currentX = document.getElementById('sel-x').value;
  currentY = document.getElementById('sel-y').value;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  clearNarrative();
  renderMap();
}});

// --- Close info panel ---
document.getElementById('info-close').addEventListener('click', function() {{
  document.getElementById('info-panel').classList.add('hidden');
}});

// --- Init with first preset ---
document.querySelector('.preset-btn').classList.add('active');
const p0 = presets[0];
currentX = p0.x;
currentY = p0.y;
document.getElementById('sel-x').value = p0.x;
document.getElementById('sel-y').value = p0.y;
setNarrative(p0.title, p0.narrative);
renderMap();
</script>
</body>
</html>"""
    return html


def main():
    logger.info("Building bivariate maps")

    geo = gpd.read_file(GEO_DIR / "cd59.geojson")
    master = pd.read_parquet(REPORTS_DIR / "master_nightscape_df.parquet")
    lookup = pd.read_csv(GEO_DIR / "cd_lookup.csv")

    geojson = build_geojson(geo, master, lookup)
    logger.info(f"GeoJSON built: {len(geojson['features'])} features")

    html = build_html(geojson, PRESETS, VARIABLES)

    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERACTIVE_DIR / "bivariate.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Bivariate maps written to {out_path} ({len(html)/1e6:.1f} MB)")

    print(f"\nBivariate maps built: {out_path}")
    print(f"  Presets: {len(PRESETS)}")
    print(f"  Custom variables: {len(VARIABLES)}")
    print(f"  File size: {len(html)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
