#!/usr/bin/env python3
"""
70_build_atlas.py — Build interactive Leaflet.js choropleth atlas.

Generates a self-contained HTML page with:
  - 59 community districts as choropleth polygons
  - 60+ mappable nighttime variables grouped by dimension in a sidebar
  - Quantile color breaks (5 classes)
  - Hover tooltips with CD name and value
  - Click info panel with variable description and source
  - Nightscape Index as default layer

Output: outputs/interactive/atlas.html
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
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

logger = get_logger("70_build_atlas")

# ---------------------------------------------------------------------------
# Variable registry: group → list of {col, label, description, source, fmt}
# fmt: "pct" (×100, 1 dp, %), "rate" (1 dp), "int" (comma-separated int),
#       "float1" (1 dp), "float2" (2 dp), "ratio" (2 dp), "min" (1 dp + " min")
# ---------------------------------------------------------------------------
VARIABLE_GROUPS = {
    "Composite Index": [
        {
            "col": "nightscape_index",
            "label": "Nightscape Index",
            "desc": "A weighted composite score summarizing nighttime conditions across six dimensions: noise exposure (20%), safety (25%), lighting (10%), transit access (15%), environmental quality (15%), and services (15%). Higher values indicate worse overall nighttime conditions. The index ranges from roughly 29 to 69 across NYC's 59 community districts.",
            "method": "Each dimension is built from min-max normalized indicators, then combined using expert-informed weights. Sensitivity analysis confirms findings are robust to equal weighting.",
            "source": "NYC Nightscape Project (computed from 20+ NYC Open Data sources, 2021-2023)",
            "fmt": "float1",
        },
        {
            "col": "nightscape_rank",
            "label": "Nightscape Rank",
            "desc": "Rank-ordering of all 59 community districts by their Nightscape Index score. Rank 1 indicates the worst nighttime conditions; rank 59 indicates the best. Bronx CDs dominate the worst ranks, while Queens and Staten Island CDs cluster among the best.",
            "method": "Simple rank of the Nightscape Index (descending). Ties are broken by the raw score.",
            "source": "NYC Nightscape Project (computed)",
            "fmt": "int",
        },
        {
            "col": "dim_noise_exposure",
            "label": "Noise Exposure Score",
            "desc": "Normalized score (0-1) capturing the overall noise burden a neighborhood faces at night. Combines objective noise modeling from the NYC BTS noise map, 311 noise complaint rates, helicopter noise, and traffic noise exposure. Higher values mean louder, more complaint-heavy nighttime environments.",
            "method": "Component indicators are min-max normalized to [0,1] and averaged. Weight in composite index: 20%.",
            "source": "NYC BTS Noise Model, NYC 311, DOT Traffic Volumes (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dim_safety",
            "label": "Safety Score",
            "desc": "Normalized score (0-1) representing nighttime safety risk. Combines nighttime crime rates, violent crime rates, shots-fired incidents, arrest rates, and motor vehicle crash rates. Higher values indicate less safe nighttime conditions.",
            "method": "Component indicators are min-max normalized to [0,1] and averaged. Weight in composite index: 25% (highest weight, reflecting safety's central importance to nighttime quality of life).",
            "source": "NYPD CompStat, NYPD ShotSpotter, NYPD Arrests, NYC Open Data MV Crashes (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dim_lighting",
            "label": "Lighting Score",
            "desc": "Normalized score (0-1) reflecting the state of nighttime lighting infrastructure. Incorporates satellite-measured radiance (VIIRS) and 311 streetlight outage complaints. Higher values suggest poorer lighting conditions (more complaints, less radiance).",
            "method": "VIIRS radiance is inverted (low light = high score). Streetlight complaints are normalized per km². Weight in composite index: 10%.",
            "source": "VIIRS Day-Night Band satellite imagery, NYC 311 streetlight complaints (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dim_transit_access",
            "label": "Transit Access Score",
            "desc": "Normalized score (0-1) measuring how well-served a neighborhood is by late-night transit. Combines subway station density, late-night ridership volume, service frequency after midnight, and transit desert flags. Higher values indicate poorer late-night transit access.",
            "method": "Indicators are inverted so that low access = high score. Includes subway, taxi/rideshare, and Citi Bike availability. Weight in composite index: 15%.",
            "source": "MTA Turnstile Data, MTA GTFS Schedules, TLC Trip Records, Citi Bike System Data (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dim_environmental",
            "label": "Environmental Score",
            "desc": "Normalized score (0-1) capturing nighttime environmental quality. Includes air pollution (NO2, PM2.5), nighttime temperature (urban heat island proxy), and building energy intensity. Higher values indicate worse environmental conditions.",
            "method": "Component indicators min-max normalized and averaged. Weight in composite index: 15%.",
            "source": "NYCCAS Air Quality monitoring, PRISM Climate data, LL84 Energy Benchmarking (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dim_services",
            "label": "Services Score",
            "desc": "Normalized score (0-1) reflecting the burden on nighttime services and quality-of-life indicators. Includes EMS response times, fire incident rates, non-noise 311 complaints, and homeless encampment reports. Higher values indicate more strained nighttime services.",
            "method": "Component indicators min-max normalized and averaged. Weight in composite index: 15%.",
            "source": "FDNY, NYC 311, FDNY EMS Incidents (2021-2023)",
            "fmt": "float2",
        },
    ],
    "Demographics": [
        {
            "col": "population",
            "label": "Population",
            "desc": "Total resident population of the community district. Used as the denominator for all per-capita rates in this atlas. Populations range from ~25,000 (Battery Park / FiDi) to ~250,000 (Flatbush / Midwood).",
            "method": "Census tract populations from ACS 5-year estimates, aggregated to community districts using NYC DCP's tract-to-CD crosswalk.",
            "source": "U.S. Census Bureau, American Community Survey 5-Year Estimates (2017-2021), Table B01003",
            "fmt": "int",
        },
        {
            "col": "pop_density",
            "label": "Pop. Density (per km\u00b2)",
            "desc": "Residents per square kilometer of land area. Manhattan CDs typically exceed 25,000/km\u00b2, while outer-borough CDs may be under 5,000/km\u00b2. Density shapes how nighttime noise, crime, and services are experienced.",
            "method": "Population divided by land area in km\u00b2 (water areas excluded).",
            "source": "ACS 5-Year (2017-2021) / Census TIGER shapefiles",
            "fmt": "int",
        },
        {
            "col": "poverty_rate",
            "label": "Poverty Rate",
            "desc": "Share of the population living below the federal poverty line. A key equity indicator \u2014 the Nightscape Index shows strong positive correlation with poverty (Spearman rho \u2248 0.6), meaning poorer neighborhoods tend to have worse nighttime conditions across multiple dimensions.",
            "method": "ACS poverty-status table, aggregated from tract to CD level.",
            "source": "ACS 5-Year Estimates (2017-2021), Table B17001",
            "fmt": "pct",
        },
        {
            "col": "pct_nonhisp_black",
            "label": "% Non-Hispanic Black",
            "desc": "Share of the population identifying as non-Hispanic Black or African American. Regression analysis shows this is a significant predictor of the Nightscape Index, reflecting structural inequities in how nighttime infrastructure and safety are distributed across NYC.",
            "method": "ACS race/ethnicity table, aggregated from tract to CD level.",
            "source": "ACS 5-Year Estimates (2017-2021), Table B03002",
            "fmt": "pct",
        },
        {
            "col": "pct_hispanic",
            "label": "% Hispanic",
            "desc": "Share of the population identifying as Hispanic or Latino (of any race). The strongest single demographic predictor of the Nightscape Index in regression models (beta = +0.57), indicating a strong association between Hispanic population share and worse nighttime conditions.",
            "method": "ACS race/ethnicity table, aggregated from tract to CD level.",
            "source": "ACS 5-Year Estimates (2017-2021), Table B03002",
            "fmt": "pct",
        },
        {
            "col": "rent_burden_rate",
            "label": "Rent Burden Rate",
            "desc": "Share of renter households paying more than 30% of income on housing costs. A marker of economic stress that co-occurs with poor nighttime conditions \u2014 rent-burdened neighborhoods have higher noise complaints, slower EMS response, and fewer late-night transit options.",
            "method": "ACS gross-rent-as-percentage-of-income table, aggregated from tract to CD level.",
            "source": "ACS 5-Year Estimates (2017-2021), Table B25070",
            "fmt": "pct",
        },
    ],
    "Noise": [
        {
            "col": "noise_obj_db_mean",
            "label": "Mean Noise (dB)",
            "desc": "Average modeled noise level in decibels across the community district, derived from the NYC Department of Environmental Protection's noise model. This is an objective measure (not based on complaints) that accounts for traffic, aircraft, rail, and industrial sources. Values above 65 dB are generally considered disruptive to sleep.",
            "method": "Raster-based noise model averaged over all grid cells within each CD boundary using area-weighted spatial aggregation.",
            "source": "NYC BTS (Building Technical Standards) Noise Model, via NYC DEP",
            "fmt": "float1",
        },
        {
            "col": "rate_per_1k_pop",
            "label": "311 Noise per 1k",
            "desc": "Nighttime noise complaints (10 PM - 7 AM) filed through NYC's 311 system, per 1,000 residents. Captures the subjective experience of noise \u2014 what residents find disruptive enough to report. Includes loud music/parties, construction, car alarms, bar noise, and more. Manhattan and dense Brooklyn CDs see rates above 50 per 1,000.",
            "method": "All 311 noise-category complaints with creation time between 22:00-07:00 (2021-2023), geocoded to CDs via spatial join, divided by population.",
            "source": "NYC Open Data \u2014 311 Service Requests (dataset: erm2-nwe9), 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "late_night_share",
            "label": "Late-Night Noise Share",
            "desc": "The proportion of nighttime noise complaints that occur during the deep-night window of 1-5 AM (versus the broader 10 PM - 7 AM window). Higher values indicate that a neighborhood's noise problem is concentrated in the hours most critical for sleep, suggesting entertainment-district or late-night commercial activity.",
            "method": "Count of 311 noise complaints 01:00-05:00 divided by count 22:00-07:00, per CD (2021-2023).",
            "source": "NYC Open Data \u2014 311 Service Requests (2021-2023)",
            "fmt": "pct",
        },
        {
            "col": "weekend_uplift",
            "label": "Weekend Noise Uplift",
            "desc": "The ratio of weekend (Fri-Sat night) to weekday nighttime noise complaints. A ratio of 1.5 means 50% more complaints on weekend nights. Captures the nightlife/entertainment-driven noise signature of a neighborhood \u2014 high ratios correlate with bar density and late-night food establishments.",
            "method": "Mean weekend-night complaint count divided by mean weekday-night complaint count, per CD (2021-2023).",
            "source": "NYC Open Data \u2014 311 Service Requests (2021-2023)",
            "fmt": "ratio",
        },
        {
            "col": "helicopter_complaints_night_per_1k",
            "label": "Helicopter Complaints per 1k",
            "desc": "Nighttime 311 complaints specifically about helicopter noise, per 1,000 residents. Helicopter noise is distinct from ground-level noise: it is intermittent, difficult to mitigate, and concentrated along flight corridors and near heliports (especially along the East River and in Lower Manhattan).",
            "method": "Subset of 311 noise complaints with descriptor containing 'helicopter' during nighttime hours (22:00-07:00), per 1,000 pop.",
            "source": "NYC Open Data \u2014 311 Service Requests (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dot_exposure_index",
            "label": "Traffic Noise Index",
            "desc": "Composite index of road-traffic noise exposure based on DOT automated traffic volume counts. Captures the contribution of motor vehicles to the nighttime soundscape. Higher values indicate more traffic-generated noise from major roads within or bordering the district.",
            "method": "DOT traffic counter volumes during nighttime hours, weighted by road classification, spatially joined to CDs.",
            "source": "NYC DOT Automated Traffic Volume Counts (2021-2023)",
            "fmt": "float1",
        },
    ],
    "Crime & Safety": [
        {
            "col": "crime_rate_per_1k",
            "label": "Crime Rate per 1k",
            "desc": "Total nighttime crimes (10 PM - 7 AM) per 1,000 residents, including felonies, misdemeanors, and violations. This is the broadest measure of nighttime crime intensity. Rates vary enormously: Midtown Manhattan (CD 105) shows the highest rate due to its massive non-resident nighttime population, while suburban Staten Island CDs see rates near zero.",
            "method": "NYPD complaint data filtered to nighttime hours (22:00-07:00), mapped from precincts to CDs via spatial crosswalk, summed over 2021-2023, divided by population \u00d7 3 years.",
            "source": "NYPD CompStat / NYC Open Data (dataset: qgea-i56i), 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "violent_rate_per_1k",
            "label": "Violent Crime per 1k",
            "desc": "Nighttime violent crimes (murder, rape, robbery, felony assault) per 1,000 residents. A more targeted safety indicator than total crime. The highest-rate CDs are concentrated in the South Bronx, Central Brooklyn, and Northern Manhattan \u2014 areas that also rank worst on the composite Nightscape Index.",
            "method": "Subset of NYPD complaints classified as violent felonies during 22:00-07:00, same crosswalk methodology.",
            "source": "NYPD CompStat / NYC Open Data (2021-2023)",
            "fmt": "float1",
        },
        {
            "col": "felony_share",
            "label": "Felony Share",
            "desc": "The proportion of nighttime crimes classified as felonies (the most serious category). Higher values indicate that a neighborhood's nighttime crime is skewed toward more serious offenses rather than minor violations.",
            "method": "Felony count divided by total nighttime crime count per CD (2021-2023).",
            "source": "NYPD CompStat / NYC Open Data (2021-2023)",
            "fmt": "pct",
        },
        {
            "col": "shots_per_year",
            "label": "Shots Fired per Year",
            "desc": "Average number of confirmed shots-fired incidents per year, from NYPD data (not ShotSpotter). A direct measure of gun violence exposure. Extremely concentrated geographically \u2014 a handful of CDs in the Bronx and Brooklyn account for the majority of citywide incidents.",
            "method": "Total shots-fired incidents (2021-2023) divided by 3 years, mapped to CDs via coordinates.",
            "source": "NYPD Shooting Incident Data / NYC Open Data (dataset: 833y-fsy8), 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "shots_nighttime_share",
            "label": "Shots Nighttime Share",
            "desc": "The proportion of all shots-fired incidents that occur during nighttime hours (10 PM - 7 AM). Citywide, roughly 45% of shootings happen at night. CDs with higher nighttime shares may have activity patterns more tied to late-night street activity or weekend nights.",
            "method": "Nighttime shots count divided by all-hours shots count per CD (2021-2023).",
            "source": "NYPD Shooting Incident Data (2021-2023)",
            "fmt": "pct",
        },
        {
            "col": "arrest_rate_per_1k",
            "label": "Arrest Rate per 1k",
            "desc": "Total arrests per 1,000 residents. Reflects both crime levels and policing intensity. High arrest rates may indicate either more criminal activity or more aggressive enforcement \u2014 this measure captures the combination of both.",
            "method": "NYPD arrest data geocoded to CDs, summed over 2021-2023, per 1,000 pop. Note: arrest data lacks reliable time-of-day field, so this is all-hours.",
            "source": "NYPD Arrest Data / NYC Open Data (dataset: 8h9b-rp9u), 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "felony_arrest_rate_per_1k",
            "label": "Felony Arrest Rate per 1k",
            "desc": "Felony-level arrests per 1,000 residents. A subset of total arrests capturing only the most serious charges. Patterns closely track violent crime rates.",
            "method": "Felony arrests geocoded to CDs, summed over 2021-2023, per 1,000 pop.",
            "source": "NYPD Arrest Data (2021-2023)",
            "fmt": "float1",
        },
        {
            "col": "drug_arrest_rate_per_1k",
            "label": "Drug Arrest Rate per 1k",
            "desc": "Drug-related arrests per 1,000 residents. Includes possession, sale, and paraphernalia charges. Geographic patterns reflect both drug market activity and enforcement focus areas.",
            "method": "Arrests with drug-related offense codes, geocoded to CDs, per 1,000 pop (2021-2023).",
            "source": "NYPD Arrest Data (2021-2023)",
            "fmt": "float2",
        },
        {
            "col": "dui_arrest_rate_per_1k",
            "label": "DUI Arrest Rate per 1k",
            "desc": "Driving-under-the-influence arrests per 1,000 residents. A nightlife-adjacent safety indicator \u2014 higher rates tend to appear in CDs with more bars and restaurants, but also in car-dependent outer-borough areas.",
            "method": "DUI-coded arrests geocoded to CDs, per 1,000 pop (2021-2023).",
            "source": "NYPD Arrest Data (2021-2023)",
            "fmt": "float2",
        },
    ],
    "Road Safety": [
        {
            "col": "crash_rate_per_1k",
            "label": "Crash Rate per 1k",
            "desc": "Nighttime motor vehicle crashes (10 PM - 7 AM) per 1,000 residents, including all collisions reported to NYPD. Nighttime crashes are disproportionately deadly: though they account for ~25% of all crashes, they represent ~40% of fatalities due to higher speeds and impaired driving.",
            "method": "NYPD MV collision reports filtered to 22:00-07:00, geocoded to CDs, summed 2021-2023, per 1,000 pop.",
            "source": "NYC Open Data \u2014 Motor Vehicle Collisions (dataset: h9gi-nx95), 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "injury_rate_per_1k",
            "label": "Injury Rate per 1k",
            "desc": "Persons injured in nighttime motor vehicle crashes per 1,000 residents. Includes drivers, passengers, pedestrians, and cyclists. A more severity-weighted measure than crash count alone.",
            "method": "Sum of persons_injured from nighttime crash reports, per 1,000 pop (2021-2023).",
            "source": "NYC Open Data \u2014 Motor Vehicle Collisions (2021-2023)",
            "fmt": "float1",
        },
        {
            "col": "persons_killed",
            "label": "Persons Killed",
            "desc": "Total fatalities from nighttime motor vehicle crashes over the 3-year study period (2021-2023). Small counts per CD, but each represents a life lost. Queens and Brooklyn CDs with wide, high-speed arterial roads tend to have higher counts.",
            "method": "Sum of persons_killed from nighttime crash reports per CD (2021-2023). Raw count, not rate (small numbers make per-capita rates unstable).",
            "source": "NYC Open Data \u2014 Motor Vehicle Collisions (2021-2023)",
            "fmt": "int",
        },
        {
            "col": "ped_injured",
            "label": "Pedestrians Injured",
            "desc": "Pedestrians injured in nighttime crashes (2021-2023 total). Pedestrian injuries at night are a particular concern because reduced visibility and higher vehicle speeds increase both the likelihood and severity of collisions.",
            "method": "Sum of pedestrians_injured from nighttime crash reports per CD (2021-2023).",
            "source": "NYC Open Data \u2014 Motor Vehicle Collisions (2021-2023)",
            "fmt": "int",
        },
        {
            "col": "cyclist_injured",
            "label": "Cyclists Injured",
            "desc": "Cyclists injured in nighttime crashes (2021-2023 total). Nighttime cycling injuries are concentrated in Manhattan and gentrifying Brooklyn, where late-night cycling (including delivery workers) is more common.",
            "method": "Sum of cyclists_injured from nighttime crash reports per CD (2021-2023).",
            "source": "NYC Open Data \u2014 Motor Vehicle Collisions (2021-2023)",
            "fmt": "int",
        },
    ],
    "Lighting & Environment": [
        {
            "col": "radiance_raw",
            "label": "VIIRS Radiance",
            "desc": "Mean nighttime light radiance measured by the VIIRS Day-Night Band satellite sensor, in nanowatts per square centimeter per steradian (nW/cm\u00b2/sr). This is a direct measurement of how bright a neighborhood appears from space at night \u2014 reflecting streetlights, building lights, signage, and vehicle headlights. Manhattan CDs show values 5-10x higher than outer-borough residential areas.",
            "method": "VIIRS DNB annual composite raster clipped to CD boundaries, area-weighted mean per CD.",
            "source": "NASA/NOAA VIIRS Day-Night Band Annual Composite",
            "fmt": "float1",
        },
        {
            "col": "streetlight_complaints_per_km2",
            "label": "Streetlight Complaints per km\u00b2",
            "desc": "311 complaints about broken, flickering, or non-functioning streetlights, normalized by area (per km\u00b2). This is a proxy for lighting infrastructure quality \u2014 more complaints suggest more outages, darker streets, and potential safety concerns. Does not capture total streetlight stock (which is not publicly available at CD level).",
            "method": "All 311 complaints with type 'Street Light Condition' (2021-2023), geocoded to CDs, divided by CD land area in km\u00b2.",
            "source": "NYC Open Data \u2014 311 Service Requests (2021-2023)",
            "fmt": "float1",
        },
        {
            "col": "no2_mean_primary",
            "label": "NO\u2082 (ppb)",
            "desc": "Mean nitrogen dioxide concentration in parts per billion, from NYC's community-level air quality monitoring network. NO\u2082 is a marker for traffic-related air pollution and is elevated near highways and dense traffic corridors. Nighttime NO\u2082 tends to persist longer due to reduced photochemical breakdown.",
            "method": "NYCCAS monitoring data interpolated to a fine grid via land-use regression, then area-weighted to CD level.",
            "source": "NYC Community Air Survey (NYCCAS), NYC DOHMH",
            "fmt": "float1",
        },
        {
            "col": "pm25_mean_primary",
            "label": "PM2.5 (\u00b5g/m\u00b3)",
            "desc": "Mean fine particulate matter (particles < 2.5 micrometers) in micrograms per cubic meter. PM2.5 penetrates deep into the lungs and is linked to respiratory and cardiovascular disease. Indoor nighttime exposure is influenced by outdoor ambient levels, especially in buildings without central filtration.",
            "method": "NYCCAS monitoring data interpolated via land-use regression, area-weighted to CD level.",
            "source": "NYC Community Air Survey (NYCCAS), NYC DOHMH",
            "fmt": "float1",
        },
        {
            "col": "tmin_mean_primary",
            "label": "Min Temp (\u00b0C)",
            "desc": "Mean daily minimum temperature in degrees Celsius, a proxy for nighttime temperatures and urban heat island intensity. Higher nighttime minimums mean less overnight cooling, which disrupts sleep (especially in buildings without air conditioning) and increases heat-related health risks.",
            "method": "PRISM 4km gridded climate data, area-weighted to CD level, averaged over 2021-2023.",
            "source": "PRISM Climate Group, Oregon State University",
            "fmt": "float1",
        },
        {
            "col": "energy_use_per_sqft",
            "label": "Energy Use (kBtu/sqft)",
            "desc": "Site energy use intensity (EUI) in kBtu per square foot for large buildings (>25,000 sqft) required to report under Local Law 84. A proxy for the built environment's nighttime energy footprint \u2014 commercial districts with office towers and data centers show higher intensities.",
            "method": "LL84 benchmarking submissions geocoded to CDs, median EUI per CD.",
            "source": "NYC Mayor's Office of Sustainability \u2014 LL84 Energy Benchmarking Data",
            "fmt": "float1",
        },
        {
            "col": "large_buildings_per_km2",
            "label": "Large Buildings per km\u00b2",
            "desc": "Density of large buildings (>25,000 sqft) per square kilometer. Indicates the intensity of the built environment and commercial/institutional presence. CDs with more large buildings tend to have higher light pollution, more nighttime activity, and greater energy consumption.",
            "method": "Count of unique buildings in LL84 dataset, geocoded to CDs, divided by land area.",
            "source": "NYC LL84 Energy Benchmarking Data",
            "fmt": "float1",
        },
    ],
    "Transit & Mobility": [
        {
            "col": "late_night_entries_per_1k",
            "label": "Late-Night Subway per 1k",
            "desc": "Subway turnstile entries during late-night hours (1-5 AM) per 1,000 residents, summed over 2021-2023. Measures how much late-night subway activity a neighborhood generates. Manhattan CDs dominate due to their role as entertainment and employment destinations.",
            "method": "MTA turnstile audit data filtered to 01:00-05:00, stations mapped to CDs via spatial join, per 1,000 pop.",
            "source": "MTA Turnstile Usage Data (2021-2023)",
            "fmt": "int",
        },
        {
            "col": "subway_stations_per_km2",
            "label": "Subway Stations per km\u00b2",
            "desc": "Subway station density \u2014 the number of unique subway stations per square kilometer. A fundamental measure of transit accessibility. Manhattan averages ~3-5 stations/km\u00b2 while many Queens and Bronx CDs have fewer than 1.",
            "method": "MTA station coordinates spatially joined to CDs, count divided by land area.",
            "source": "MTA GTFS Static Feed",
            "fmt": "float1",
        },
        {
            "col": "pct_ridership_late_night",
            "label": "Late-Night Ridership %",
            "desc": "The share of a CD's total subway ridership that occurs during late-night hours (1-5 AM). Higher percentages indicate neighborhoods where the subway serves a disproportionately late-night function \u2014 entertainment districts, hospitality worker commuting corridors, or areas with shift-work economies.",
            "method": "Late-night entries (01:00-05:00) divided by all-hours entries per CD.",
            "source": "MTA Turnstile Usage Data (2021-2023)",
            "fmt": "pct",
        },
        {
            "col": "transit_desert_flag",
            "label": "Transit Desert",
            "desc": "Binary flag (1 = yes, 0 = no) indicating whether a community district has zero subway stations within its boundaries. These 'transit deserts' force residents to rely entirely on buses, cars, or for-hire vehicles for late-night travel, creating significant mobility barriers.",
            "method": "1 if no MTA station centroid falls within the CD polygon; 0 otherwise.",
            "source": "MTA GTFS Static Feed",
            "fmt": "int",
        },
        {
            "col": "mean_trains_per_hour_late_night",
            "label": "Trains/Hour (Late Night)",
            "desc": "Average number of scheduled train arrivals per hour across all stations in the CD during late-night service (12 AM - 5 AM). Captures service frequency \u2014 even CDs with stations may have poor late-night service if trains run infrequently.",
            "method": "GTFS stop_times filtered to 00:00-05:00, averaged across all stops and routes within each CD.",
            "source": "MTA GTFS Static Feed (December 2023 schedule)",
            "fmt": "float1",
        },
        {
            "col": "total_ride_pickups_night_per_1k",
            "label": "Ride Pickups per 1k",
            "desc": "Total nighttime for-hire vehicle pickups (taxis + rideshare/app-based services) per 1,000 residents. Captures the combined taxi/Uber/Lyft/Via demand at night. Manhattan CDs see rates 10-50x higher than outer-borough CDs due to entertainment, dining, and nightlife activity.",
            "method": "TLC trip records filtered to 22:00-07:00 pickup times, geocoded to CDs, summed 2021-2023, per 1,000 pop.",
            "source": "NYC Taxi & Limousine Commission Trip Record Data (2021-2023)",
            "fmt": "int",
        },
        {
            "col": "rideshare_to_taxi_ratio",
            "label": "Rideshare-to-Taxi Ratio",
            "desc": "The ratio of rideshare (Uber/Lyft/Via) pickups to yellow/green taxi pickups at night. A ratio above 5 means rideshare dominates; below 2 means taxis remain competitive. Manhattan below 60th Street has the lowest ratios (taxis still competitive); outer boroughs see ratios of 10-30+.",
            "method": "Rideshare trip count divided by taxi trip count per CD, nighttime hours only (2021-2023).",
            "source": "NYC TLC Trip Record Data (2021-2023)",
            "fmt": "ratio",
        },
        {
            "col": "citibike_trips_night_per_1k",
            "label": "Citi Bike Trips per 1k",
            "desc": "Nighttime Citi Bike trips (10 PM - 7 AM) per 1,000 residents. Captures micromobility usage at night \u2014 concentrated in Manhattan and brownstone Brooklyn where station density is highest. Zero in CDs without Citi Bike coverage.",
            "method": "Citi Bike system data filtered to nighttime start times, start stations mapped to CDs, per 1,000 pop (2021-2023).",
            "source": "Citi Bike System Data (Lyft), 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "citibike_stations_per_km2",
            "label": "Citi Bike Stations per km\u00b2",
            "desc": "Citi Bike docking station density per square kilometer. Zero outside the Citi Bike service area (most of Queens, Bronx, Staten Island). Within the service area, density varies from ~5/km\u00b2 to over 50/km\u00b2 in core Manhattan.",
            "method": "Citi Bike station feed geocoded to CDs, count divided by land area.",
            "source": "Citi Bike Station Feed (GBFS), 2023",
            "fmt": "float1",
        },
    ],
    "Emergency Services": [
        {
            "col": "fire_incidents_night_per_1k",
            "label": "Fire Incidents per 1k",
            "desc": "Nighttime fire-related incidents (10 PM - 7 AM) per 1,000 residents, including structural fires, vehicle fires, rubbish fires, and false alarms. CDs with older housing stock, higher poverty, and more NYCHA buildings tend to have higher rates. The Bronx consistently leads.",
            "method": "FDNY incident data filtered to 22:00-07:00, geocoded to CDs, summed 2021-2023, per 1,000 pop.",
            "source": "FDNY Open Data \u2014 Fire Incident Dispatch Data, 2021-2023",
            "fmt": "float1",
        },
        {
            "col": "fire_night_day_ratio",
            "label": "Fire Night/Day Ratio",
            "desc": "Ratio of nighttime to daytime fire incidents. A ratio near 0.5 is typical (fewer fires at night when cooking/industrial activity is lower). Higher ratios may indicate arson risk or heating-related fires in winter. Lower ratios indicate daytime-dominated incident patterns (commercial areas).",
            "method": "Nighttime fire count divided by daytime fire count per CD (2021-2023).",
            "source": "FDNY Open Data (2021-2023)",
            "fmt": "ratio",
        },
        {
            "col": "structural_fires_night",
            "label": "Structural Fires (Night)",
            "desc": "Total nighttime structural fire incidents (building fires) over the 3-year study period. The most dangerous fire type \u2014 nighttime structural fires are especially deadly because occupants may be asleep.",
            "method": "FDNY incidents classified as structural fires during 22:00-07:00, summed per CD (2021-2023).",
            "source": "FDNY Open Data (2021-2023)",
            "fmt": "int",
        },
        {
            "col": "ems_response_min_night",
            "label": "EMS Response (min)",
            "desc": "Mean EMS (ambulance) response time in minutes during nighttime hours. Response time is measured from initial call to unit arrival on scene. Citywide average is ~10 minutes at night. Longer response times correlate strongly with poverty (Spearman rho = +0.67), reflecting both geographic distance from hospitals and call volume pressure.",
            "method": "FDNY EMS incident data, filtered to 22:00-07:00, mean of (arrival_time - dispatch_time) per CD (2021-2023).",
            "source": "FDNY EMS Incident Dispatch Data / NYC Open Data, 2021-2023",
            "fmt": "min",
        },
        {
            "col": "ems_night_day_response_ratio",
            "label": "EMS Night/Day Ratio",
            "desc": "Ratio of nighttime to daytime mean EMS response time. Values below 1.0 mean EMS responds faster at night (less traffic); values above 1.0 mean nighttime response is actually slower (possible in high-volume areas). Most CDs show ratios near 0.85-0.95, indicating modest nighttime improvement.",
            "method": "Mean nighttime response time divided by mean daytime response time per CD.",
            "source": "FDNY EMS Incident Dispatch Data (2021-2023)",
            "fmt": "ratio",
        },
        {
            "col": "ems_incidents_night_est",
            "label": "EMS Incidents (Night)",
            "desc": "Estimated total nighttime EMS incidents (ambulance dispatches) over 2021-2023. Reflects both the health burden and the demand on emergency medical infrastructure at night.",
            "method": "FDNY EMS incidents filtered to 22:00-07:00, summed per CD (2021-2023).",
            "source": "FDNY EMS Incident Dispatch Data (2021-2023)",
            "fmt": "int",
        },
    ],
    "Nightlife & Services": [
        {
            "col": "on_premises_rate_per_1k_pop",
            "label": "Bars/Clubs per 1k",
            "desc": "On-premises liquor licenses (bars, nightclubs, lounges where alcohol is consumed on-site) per 1,000 residents. The primary indicator of nightlife density. Lower Manhattan and Williamsburg/Greenpoint have the highest concentrations, while residential outer-borough CDs have very few.",
            "method": "Active SLA on-premises licenses geocoded to CDs, divided by population.",
            "source": "NYS Liquor Authority (SLA) License Data, via NYC Open Data",
            "fmt": "float1",
        },
        {
            "col": "license_rate_per_1k_pop",
            "label": "All Liquor Licenses per 1k",
            "desc": "All active liquor licenses (on-premises + off-premises like liquor stores and grocery) per 1,000 residents. Broader than bars/clubs alone, this captures the full alcohol retail landscape of a neighborhood.",
            "method": "All active SLA licenses geocoded to CDs, divided by population.",
            "source": "NYS Liquor Authority License Data",
            "fmt": "float1",
        },
        {
            "col": "restaurants_per_1k",
            "label": "Restaurants per 1k",
            "desc": "Active restaurants (with valid DOHMH inspection records) per 1,000 residents. Reflects the commercial dining density of a neighborhood. High restaurant density correlates with nighttime foot traffic, noise complaints, and for-hire vehicle demand.",
            "method": "Unique active restaurant establishments from DOHMH inspection data, geocoded to CDs, per 1,000 pop.",
            "source": "NYC DOHMH Restaurant Inspection Results / NYC Open Data",
            "fmt": "float1",
        },
        {
            "col": "late_night_food_per_1k",
            "label": "Late-Night Food per 1k",
            "desc": "Late-night food establishments (restaurants, bodegas, diners, fast food open past midnight) per 1,000 residents. A measure of late-night service availability and neighborhood 'aliveness' after midnight.",
            "method": "Restaurants with inspection records indicating late-night hours or cuisines associated with late-night service (pizza, diner, fast food), geocoded to CDs.",
            "source": "NYC DOHMH Restaurant Inspection Results",
            "fmt": "float1",
        },
        {
            "col": "cuisine_diversity",
            "label": "Cuisine Diversity",
            "desc": "Count of distinct cuisine types (e.g., Italian, Chinese, Mexican, Ethiopian) among restaurants in the community district. A proxy for cultural diversity and the breadth of the dining scene. Ranges from ~15 in homogeneous areas to 60+ in cosmopolitan Manhattan CDs.",
            "method": "Count of unique cuisine_description values from DOHMH inspection data per CD.",
            "source": "NYC DOHMH Restaurant Inspection Results",
            "fmt": "int",
        },
        {
            "col": "film_activity_per_1k",
            "label": "Night Film Permits per 1k",
            "desc": "Nighttime film and TV production shoot permits per 1,000 residents. Night shoots bring lighting rigs, generators, crew vehicles, and noise to residential streets. Concentrated in Manhattan and brownstone Brooklyn locations favored by the film industry.",
            "method": "Mayor's Office of Media permits with nighttime shooting indicated, geocoded to CDs, per 1,000 pop (2021-2023).",
            "source": "Mayor's Office of Media and Entertainment / NYC Open Data",
            "fmt": "float2",
        },
        {
            "col": "wifi_hotspots_per_km2",
            "label": "Wi-Fi Hotspots per km\u00b2",
            "desc": "Public Wi-Fi hotspot locations (including parks, libraries, and transit hubs) per square kilometer. A proxy for digital infrastructure and public connectivity \u2014 relevant to nighttime because late-night workers, transit users, and unhoused individuals depend on public Wi-Fi.",
            "method": "NYC Wi-Fi Hotspot Locations dataset geocoded to CDs, count divided by land area.",
            "source": "NYC Open Data \u2014 NYC Wi-Fi Hotspot Locations",
            "fmt": "float1",
        },
        {
            "col": "linknyc_kiosks_per_1k",
            "label": "LinkNYC Kiosks per 1k",
            "desc": "LinkNYC smart kiosks (free gigabit Wi-Fi, USB charging, phone calls) per 1,000 residents. These 24/7 streetside kiosks are a nighttime resource for connectivity and wayfinding. Concentrated in Manhattan and parts of Brooklyn; absent from most of the Bronx, Queens, and Staten Island.",
            "method": "LinkNYC kiosk locations geocoded to CDs, divided by population.",
            "source": "NYC Open Data \u2014 LinkNYC Locations",
            "fmt": "float2",
        },
    ],
    "311 (Non-Noise)": [
        {
            "col": "non_noise_311_per_1k",
            "label": "Non-Noise 311 per 1k",
            "desc": "Nighttime 311 complaints excluding noise categories, per 1,000 residents. Captures the full spectrum of nighttime quality-of-life issues residents report: illegal parking, blocked driveways, graffiti, rodents, unsanitary conditions, and more. Provides a broader view of nighttime livability beyond noise alone.",
            "method": "All 311 complaints created 22:00-07:00 with complaint_type not in noise categories, geocoded to CDs, per 1,000 pop (2021-2023).",
            "source": "NYC Open Data \u2014 311 Service Requests (2021-2023)",
            "fmt": "float1",
        },
        {
            "col": "homeless_per_1k",
            "label": "Homeless Reports per 1k",
            "desc": "Nighttime 311 reports related to homeless encampments or individuals per 1,000 residents. A sensitive indicator of visible homelessness at night. High rates cluster in Manhattan (transit hubs, parks) and in CDs with shelters. Reflects both the scale of homelessness and reporting behavior.",
            "method": "311 complaints with homeless-related types during 22:00-07:00, geocoded to CDs, per 1,000 pop (2021-2023).",
            "source": "NYC Open Data \u2014 311 Service Requests (2021-2023)",
            "fmt": "float1",
        },
    ],
    "Housing & Construction": [
        {
            "col": "nycha_pop_share",
            "label": "NYCHA Pop. Share",
            "desc": "Estimated share of the community district's population living in NYCHA (public housing) developments. NYCHA residents face distinct nighttime challenges including aging infrastructure, elevator outages, and concentrated poverty. CDs in the South Bronx and East/Central Brooklyn have the highest shares (20-30%+).",
            "method": "NYCHA building-level resident counts aggregated to CDs, divided by total CD population.",
            "source": "NYCHA Development Data / ACS Population Estimates",
            "fmt": "pct",
        },
        {
            "col": "ahv_nighttime_share",
            "label": "After-Hours Work Share",
            "desc": "The share of DOB (Department of Buildings) permits that authorize after-hours or nighttime construction work. After-hours construction generates noise, vibration, and truck traffic during hours when residents are trying to sleep. Higher shares indicate more nighttime construction disruption.",
            "method": "DOB NOW after-hours variance permits as a share of total permits per CD (2021-2023).",
            "source": "NYC DOB NOW \u2014 After Hours Variances / NYC Open Data",
            "fmt": "pct",
        },
    ],
}


def build_geojson(geo: gpd.GeoDataFrame, master: pd.DataFrame,
                  index_df: pd.DataFrame, lookup: pd.DataFrame,
                  typology_df: pd.DataFrame) -> dict:
    """Merge all data into a single GeoJSON for the atlas."""
    # Start from geo
    gdf = geo.copy()
    gdf["boro_cd"] = gdf["boro_cd"].astype(int)

    # Merge master data
    master = master.copy()
    master["boro_cd"] = master["boro_cd"].astype(int)
    gdf = gdf.merge(master, on="boro_cd", how="left")

    # Merge index dimensions
    idx_cols = [c for c in index_df.columns if c != "boro_cd"]
    for c in idx_cols:
        if c not in gdf.columns:
            gdf = gdf.merge(index_df[["boro_cd", c]], on="boro_cd", how="left")

    # Merge typology
    if "typology" not in gdf.columns:
        gdf = gdf.merge(
            typology_df[["boro_cd", "typology"]].rename(
                columns={"typology": "typology"}
            ),
            on="boro_cd",
            how="left",
        )

    # Merge lookup for names
    lookup = lookup.copy()
    lookup["boro_cd"] = lookup["boro_cd"].astype(int)
    gdf = gdf.merge(lookup[["boro_cd", "cd_short", "cd_label"]], on="boro_cd", how="left")

    # Replace inf / NaN for JSON safety
    for col in gdf.select_dtypes(include=[np.number]).columns:
        gdf[col] = gdf[col].replace([np.inf, -np.inf], np.nan)

    # Convert to GeoJSON dict
    gdf = gdf.to_crs(epsg=4326)
    geojson = json.loads(gdf.to_json())
    return geojson


def compute_quantile_breaks(geojson: dict, col: str, n_classes: int = 5):
    """Return quantile break values for a variable."""
    vals = []
    for feat in geojson["features"]:
        v = feat["properties"].get(col)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    if len(vals) < n_classes:
        return []
    vals = sorted(vals)
    breaks = []
    for i in range(1, n_classes):
        idx = int(len(vals) * i / n_classes)
        breaks.append(round(vals[idx], 4))
    return breaks


# Variables where higher values are positive (better conditions).
# For these, the color scale is reversed: blue=high (good), red=low (bad).
POSITIVE_VALENCE_COLS = {
    "radiance_raw", "z_light",
    "late_night_entries_per_1k", "subway_stations_per_km2",
    "mean_trains_per_hour_late_night", "citibike_trips_night_per_1k",
    "citibike_stations_per_km2", "total_ride_pickups_night_per_1k",
    "pct_ridership_late_night",
    "restaurants_per_1k", "late_night_food_per_1k", "cuisine_diversity",
    "on_premises_rate_per_1k_pop", "license_rate_per_1k_pop",
    "wifi_hotspots_per_km2", "linknyc_kiosks_per_1k",
    "film_activity_per_1k",
    "population", "total_res_units",
}


def build_html(geojson: dict) -> str:
    """Generate the self-contained atlas HTML."""

    # Pre-compute breaks for all variables
    var_meta = {}
    for group_name, variables in VARIABLE_GROUPS.items():
        for v in variables:
            col = v["col"]
            breaks = compute_quantile_breaks(geojson, col)
            var_meta[col] = {
                "label": v["label"],
                "desc": v["desc"],
                "method": v.get("method", ""),
                "source": v["source"],
                "fmt": v["fmt"],
                "group": group_name,
                "breaks": breaks,
                "reverse": col in POSITIVE_VALENCE_COLS,
            }

    geojson_str = json.dumps(geojson)
    var_meta_str = json.dumps(var_meta)

    # Build sidebar HTML
    sidebar_items = []
    for group_name, variables in VARIABLE_GROUPS.items():
        sidebar_items.append(f'<div class="sidebar-group">')
        sidebar_items.append(f'<div class="sidebar-group-title">{group_name}</div>')
        for v in variables:
            sidebar_items.append(
                f'<div class="sidebar-item" data-col="{v["col"]}">{v["label"]}</div>'
            )
        sidebar_items.append("</div>")
    sidebar_html = "\n".join(sidebar_items)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Nightscape Atlas</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}

  #app {{ display: flex; height: 100vh; }}

  /* Sidebar */
  #sidebar {{
    width: 300px; min-width: 300px; background: #1e293b; overflow-y: auto;
    border-right: 1px solid #334155; display: flex; flex-direction: column;
  }}
  #sidebar-header {{
    padding: 16px 20px; border-bottom: 1px solid #334155;
    background: #1a365d; position: sticky; top: 0; z-index: 10;
  }}
  #sidebar-header h1 {{ font-size: 18px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  #sidebar-header p {{ font-size: 12px; color: #94a3b8; }}
  #search-box {{
    width: 100%; padding: 8px 12px; margin-top: 10px; border-radius: 6px;
    border: 1px solid #475569; background: #0f172a; color: #e2e8f0;
    font-size: 13px; outline: none;
  }}
  #search-box:focus {{ border-color: #ed8936; }}
  #search-box::placeholder {{ color: #64748b; }}

  #sidebar-content {{ padding: 8px 0; flex: 1; }}
  .sidebar-group {{ margin-bottom: 4px; }}
  .sidebar-group-title {{
    padding: 10px 20px 6px; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; color: #ed8936;
  }}
  .sidebar-item {{
    padding: 7px 20px 7px 28px; font-size: 13px; cursor: pointer;
    color: #cbd5e1; transition: background 0.15s;
  }}
  .sidebar-item:hover {{ background: #334155; color: #f8fafc; }}
  .sidebar-item.active {{ background: #1a365d; color: #ed8936; font-weight: 600; }}

  /* Map area */
  #map-area {{ flex: 1; display: flex; flex-direction: column; }}
  #map-header {{
    padding: 12px 24px; background: #1e293b; border-bottom: 1px solid #334155;
    display: flex; align-items: center; justify-content: space-between;
  }}
  #map-title {{ font-size: 16px; font-weight: 600; color: #f8fafc; }}
  #map-subtitle {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}

  /* Variable info bar */
  #var-info-bar {{
    background: #162032; border-bottom: 1px solid #334155;
    max-height: 0; overflow: hidden; transition: max-height 0.3s ease;
  }}
  #var-info-bar.open {{ max-height: 300px; }}
  #var-info-inner {{
    padding: 14px 24px; display: grid;
    grid-template-columns: 1fr 1fr; gap: 12px 24px;
  }}
  .var-info-section {{ }}
  .var-info-label {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #ed8936; margin-bottom: 4px;
  }}
  .var-info-text {{ font-size: 12px; color: #cbd5e1; line-height: 1.6; }}
  #var-info-desc {{ grid-column: 1 / -1; }}
  #toggle-info {{
    background: none; border: 1px solid #475569; border-radius: 4px;
    color: #94a3b8; font-size: 11px; padding: 3px 10px; cursor: pointer;
    transition: color 0.15s, border-color 0.15s; white-space: nowrap;
  }}
  #toggle-info:hover {{ color: #ed8936; border-color: #ed8936; }}

  #map {{ flex: 1; background: #0f172a; }}

  /* Info panel */
  #info-panel {{
    position: absolute; top: 20px; right: 20px; z-index: 1000;
    background: rgba(30,41,59,0.95); border: 1px solid #475569;
    border-radius: 10px; padding: 16px 20px; width: 300px;
    backdrop-filter: blur(8px); pointer-events: auto;
    max-height: calc(100% - 100px); overflow-y: auto;
  }}
  #info-panel h3 {{ font-size: 15px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  #info-panel .info-cd {{ font-size: 12px; color: #94a3b8; margin-bottom: 10px; }}
  #info-panel .info-value {{ font-size: 28px; font-weight: 700; color: #ed8936; margin-bottom: 2px; }}
  #info-panel .info-rank {{ font-size: 12px; color: #cbd5e1; margin-bottom: 10px; }}
  #info-panel .info-divider {{ border: none; border-top: 1px solid #334155; margin: 10px 0; }}
  #info-panel .info-section-label {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #ed8936; margin-bottom: 3px;
  }}
  #info-panel .info-desc {{ font-size: 12px; color: #94a3b8; line-height: 1.5; margin-bottom: 8px; }}
  #info-panel .info-source {{ font-size: 11px; color: #64748b; }}
  #info-panel .info-close {{
    position: absolute; top: 10px; right: 14px; background: none; border: none;
    color: #64748b; font-size: 18px; cursor: pointer; line-height: 1;
  }}
  #info-panel .info-close:hover {{ color: #f8fafc; }}
  #info-panel.hidden {{ display: none; }}

  /* Legend */
  #legend {{
    position: absolute; bottom: 30px; right: 20px; z-index: 1000;
    background: rgba(30,41,59,0.95); border: 1px solid #475569;
    border-radius: 8px; padding: 12px 16px; min-width: 160px;
    backdrop-filter: blur(8px);
  }}
  #legend h4 {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
  .legend-row {{ display: flex; align-items: center; margin-bottom: 4px; font-size: 12px; color: #cbd5e1; }}
  .legend-color {{ width: 20px; height: 14px; margin-right: 8px; border-radius: 2px; }}

  /* Tooltip override */
  .leaflet-tooltip {{
    background: rgba(15,23,42,0.95) !important; color: #f8fafc !important;
    border: 1px solid #475569 !important; border-radius: 6px !important;
    padding: 8px 12px !important; font-family: inherit !important;
    font-size: 13px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
  }}
  .leaflet-tooltip::before {{ border-top-color: rgba(15,23,42,0.95) !important; }}
  .tt-name {{ font-weight: 700; margin-bottom: 2px; }}
  .tt-value {{ color: #ed8936; font-size: 15px; font-weight: 600; }}

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

  /* Mobile */
  @media (max-width: 768px) {{
    #sidebar {{ width: 240px; min-width: 240px; }}
    #info-panel {{ width: 220px; top: 70px; right: 10px; }}
  }}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">
      <h1>NYC Nightscape Atlas</h1>
      <p>60+ nighttime variables across 59 community districts</p>
      <input type="text" id="search-box" placeholder="Search variables...">
      <div class="nav-links" style="margin-top:10px;">
        <a href="atlas.html">Atlas</a>
        <a href="profiles.html">Profiles</a>
        <a href="bivariate.html">Bivariate Maps</a>
        <a href="typologies.html">Typologies</a>
        <a href="findings.html">Findings</a>
      </div>
    </div>
    <div id="sidebar-content">
      {sidebar_html}
    </div>
  </div>
  <div id="map-area">
    <div id="map-header">
      <div>
        <div id="map-title">Nightscape Index</div>
        <div id="map-subtitle">Click a variable in the sidebar to map it. Click a district for details.</div>
      </div>
      <button id="toggle-info">About this variable</button>
    </div>
    <div id="var-info-bar" class="open">
      <div id="var-info-inner">
        <div class="var-info-section" id="var-info-desc">
          <div class="var-info-label">What this shows</div>
          <div class="var-info-text" id="var-info-desc-text"></div>
        </div>
        <div class="var-info-section">
          <div class="var-info-label">Methodology</div>
          <div class="var-info-text" id="var-info-method-text"></div>
        </div>
        <div class="var-info-section">
          <div class="var-info-label">Data source</div>
          <div class="var-info-text" id="var-info-source-text"></div>
        </div>
      </div>
    </div>
    <div id="map" style="position:relative;">
      <div id="info-panel" class="hidden">
        <button class="info-close" id="info-close">&times;</button>
        <h3 id="info-cd-name"></h3>
        <div class="info-cd" id="info-cd-label"></div>
        <div class="info-value" id="info-val"></div>
        <div class="info-rank" id="info-rank"></div>
        <hr class="info-divider">
        <div class="info-section-label">About this metric</div>
        <div class="info-desc" id="info-desc"></div>
        <div class="info-section-label">Source</div>
        <div class="info-source" id="info-source"></div>
      </div>
      <div id="legend"></div>
    </div>
  </div>
</div>

<script>
// --- Data ---
const geojsonData = {geojson_str};
const varMeta = {var_meta_str};

// --- Color palettes (5-class sequential, dark-mode friendly) ---
// Default: blue(low) → red(high) — for "higher is worse" variables
const COLORS = ['#1e3a5f', '#2b6cb0', '#ed8936', '#e05d2c', '#c53030'];
// Reversed: red(low) → blue(high) — for "higher is better" variables
const COLORS_REV = ['#c53030', '#e05d2c', '#ed8936', '#2b6cb0', '#1e3a5f'];
const NULL_COLOR = '#334155';

// --- State ---
let currentCol = 'nightscape_index';
let geojsonLayer = null;

// --- Map init ---
const map = L.map('map', {{
  zoomControl: true,
  attributionControl: false,
}}).setView([40.7128, -73.95], 11);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 18,
  subdomains: 'abcd',
}}).addTo(map);

L.control.attribution({{ position: 'bottomleft' }}).addTo(map);

// --- Format value ---
function fmtVal(val, fmt) {{
  if (val === null || val === undefined || (typeof val === 'number' && isNaN(val))) return 'N/A';
  switch(fmt) {{
    case 'pct': return (val * 100).toFixed(1) + '%';
    case 'rate': return val.toFixed(1);
    case 'int': return Math.round(val).toLocaleString();
    case 'float1': return val.toFixed(1);
    case 'float2': return val.toFixed(2);
    case 'ratio': return val.toFixed(2);
    case 'min': return val.toFixed(1) + ' min';
    default: return val.toFixed(1);
  }}
}}

// --- Compute rank for a column ---
function computeRanks(col) {{
  const vals = geojsonData.features
    .map(f => ({{ cd: f.properties.boro_cd, val: f.properties[col] }}))
    .filter(d => d.val !== null && d.val !== undefined && !isNaN(d.val));
  vals.sort((a, b) => b.val - a.val);  // descending (highest = rank 1)
  const ranks = {{}};
  vals.forEach((d, i) => {{ ranks[d.cd] = i + 1; }});
  return ranks;
}}

// --- Get color for value (palette reverses for positive-valence vars) ---
function getColor(val, breaks, reverse) {{
  if (val === null || val === undefined || (typeof val === 'number' && isNaN(val))) return NULL_COLOR;
  const pal = reverse ? COLORS_REV : COLORS;
  if (breaks.length === 0) return pal[2];
  if (val <= breaks[0]) return pal[0];
  if (val <= breaks[1]) return pal[1];
  if (val <= breaks[2]) return pal[2];
  if (val <= breaks[3]) return pal[3];
  return pal[4];
}}

// --- Update legend ---
function updateLegend(col) {{
  const meta = varMeta[col];
  if (!meta) return;
  const breaks = meta.breaks;
  const fmt = meta.fmt;
  const leg = document.getElementById('legend');

  if (breaks.length === 0) {{
    leg.innerHTML = '<h4>' + meta.label + '</h4><div style="font-size:12px;color:#94a3b8;">Insufficient data</div>';
    return;
  }}

  // Compute actual min/max
  let vals = geojsonData.features.map(f => f.properties[col]).filter(v => v !== null && v !== undefined && !isNaN(v));
  let mn = Math.min(...vals);
  let mx = Math.max(...vals);

  const reverse = meta.reverse || false;
  const pal = reverse ? COLORS_REV : COLORS;
  let html = '<h4>' + meta.label + (reverse ? ' <span style="font-size:11px;color:#94a3b8;">(higher = better)</span>' : '') + '</h4>';
  const ranges = [
    [mn, breaks[0]],
    [breaks[0], breaks[1]],
    [breaks[1], breaks[2]],
    [breaks[2], breaks[3]],
    [breaks[3], mx],
  ];
  for (let i = 0; i < 5; i++) {{
    html += '<div class="legend-row">' +
      '<div class="legend-color" style="background:' + pal[i] + ';"></div>' +
      fmtVal(ranges[i][0], fmt) + ' &ndash; ' + fmtVal(ranges[i][1], fmt) +
      '</div>';
  }}
  leg.innerHTML = html;
}}

// --- Style function ---
function styleFeature(feature) {{
  const meta = varMeta[currentCol];
  const val = feature.properties[currentCol];
  const breaks = meta ? meta.breaks : [];
  const reverse = meta ? (meta.reverse || false) : false;
  return {{
    fillColor: getColor(val, breaks, reverse),
    weight: 1.5,
    opacity: 0.8,
    color: '#475569',
    fillOpacity: 0.75,
  }};
}}

// --- Highlight ---
function highlightFeature(e) {{
  const layer = e.target;
  layer.setStyle({{ weight: 3, color: '#ed8936', fillOpacity: 0.9 }});
  layer.bringToFront();
}}

function resetHighlight(e) {{
  geojsonLayer.resetStyle(e.target);
}}

// --- Click → info panel ---
let currentRanks = {{}};

function showInfo(e) {{
  const props = e.target.feature.properties;
  const meta = varMeta[currentCol];
  const val = props[currentCol];
  const panel = document.getElementById('info-panel');
  panel.classList.remove('hidden');

  document.getElementById('info-cd-name').textContent = props.cd_short || ('CD ' + props.boro_cd);
  document.getElementById('info-cd-label').textContent =
    (props.cd_label || '') + (props.typology ? ' | Cluster: ' + props.typology : '');
  document.getElementById('info-val').textContent = meta ? fmtVal(val, meta.fmt) : String(val);

  const rank = currentRanks[props.boro_cd];
  document.getElementById('info-rank').textContent = rank ? 'Rank ' + rank + ' of 59 (1 = highest)' : '';
  document.getElementById('info-desc').textContent = meta ? meta.desc : '';
  document.getElementById('info-source').textContent = meta ? meta.source : '';
}}

// --- Update variable info bar ---
function updateVarInfoBar(meta) {{
  document.getElementById('var-info-desc-text').textContent = meta.desc;
  document.getElementById('var-info-method-text').textContent = meta.method || 'See methodology page for details.';
  document.getElementById('var-info-source-text').textContent = meta.source;
}}

// --- Switch variable ---
function switchVariable(col) {{
  currentCol = col;
  const meta = varMeta[col];
  if (!meta) return;

  // Update header
  document.getElementById('map-title').textContent = meta.label;
  document.getElementById('map-subtitle').textContent = meta.group + ' | ' + meta.source;

  // Update info bar
  updateVarInfoBar(meta);

  // Update sidebar active
  document.querySelectorAll('.sidebar-item').forEach(el => {{
    el.classList.toggle('active', el.dataset.col === col);
  }});

  // Recompute ranks
  currentRanks = computeRanks(col);

  // Restyle map
  geojsonLayer.eachLayer(layer => {{
    geojsonLayer.resetStyle(layer);
    // Update tooltip
    const props = layer.feature.properties;
    const val = props[col];
    const cdName = props.cd_short || ('CD ' + props.boro_cd);
    layer.setTooltipContent(
      '<div class="tt-name">' + cdName + '</div>' +
      '<div class="tt-value">' + fmtVal(val, meta.fmt) + '</div>'
    );
  }});

  // Update legend
  updateLegend(col);

  // Hide info panel
  document.getElementById('info-panel').classList.add('hidden');
}}

// --- Init layer ---
function initLayer() {{
  geojsonLayer = L.geoJSON(geojsonData, {{
    style: styleFeature,
    onEachFeature: function(feature, layer) {{
      const props = feature.properties;
      const meta = varMeta[currentCol];
      const val = props[currentCol];
      const cdName = props.cd_short || ('CD ' + props.boro_cd);

      layer.bindTooltip(
        '<div class="tt-name">' + cdName + '</div>' +
        '<div class="tt-value">' + fmtVal(val, meta ? meta.fmt : 'float1') + '</div>',
        {{ sticky: true, direction: 'top', offset: [0, -10] }}
      );

      layer.on({{
        mouseover: highlightFeature,
        mouseout: resetHighlight,
        click: showInfo,
      }});
    }}
  }}).addTo(map);

  map.fitBounds(geojsonLayer.getBounds(), {{ padding: [20, 20] }});
  currentRanks = computeRanks(currentCol);
  updateLegend(currentCol);
}}

// --- Sidebar clicks ---
document.querySelectorAll('.sidebar-item').forEach(el => {{
  el.addEventListener('click', () => switchVariable(el.dataset.col));
}});

// --- Search filter ---
document.getElementById('search-box').addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  document.querySelectorAll('.sidebar-group').forEach(group => {{
    const items = group.querySelectorAll('.sidebar-item');
    let anyVisible = false;
    items.forEach(item => {{
      const match = item.textContent.toLowerCase().includes(q);
      item.style.display = match ? '' : 'none';
      if (match) anyVisible = true;
    }});
    const title = group.querySelector('.sidebar-group-title');
    if (title) title.style.display = anyVisible ? '' : 'none';
    group.style.display = anyVisible ? '' : 'none';
  }});
}});

// --- Toggle info bar ---
document.getElementById('toggle-info').addEventListener('click', function() {{
  const bar = document.getElementById('var-info-bar');
  bar.classList.toggle('open');
  this.textContent = bar.classList.contains('open') ? 'Hide details' : 'About this variable';
  // Leaflet needs to recalculate size after the bar opens/closes
  setTimeout(() => map.invalidateSize(), 350);
}});

// --- Close info panel ---
document.getElementById('info-close').addEventListener('click', function() {{
  document.getElementById('info-panel').classList.add('hidden');
}});

// --- Mark default active ---
document.querySelector('.sidebar-item[data-col="nightscape_index"]').classList.add('active');

// --- Initialize info bar with default variable ---
updateVarInfoBar(varMeta['nightscape_index']);

// --- GO ---
initLayer();
</script>
</body>
</html>"""

    return html


def main():
    logger.info("Building interactive atlas")

    # Load data
    geo = gpd.read_file(GEO_DIR / "cd59.geojson")
    master = pd.read_parquet(REPORTS_DIR / "master_nightscape_df.parquet")
    lookup = pd.read_csv(GEO_DIR / "cd_lookup.csv")

    index_df = pd.read_csv(TABLES_DIR / "nightscape_index.csv")
    typology_df = pd.read_csv(TABLES_DIR / "neighborhood_typologies.csv")

    # Build merged GeoJSON
    geojson = build_geojson(geo, master, index_df, lookup, typology_df)
    logger.info(f"GeoJSON built: {len(geojson['features'])} features")

    # Validate: check how many variables are present
    sample_props = geojson["features"][0]["properties"]
    present = 0
    missing = []
    for group_name, variables in VARIABLE_GROUPS.items():
        for v in variables:
            if v["col"] in sample_props:
                present += 1
            else:
                missing.append(v["col"])

    logger.info(f"Variables present: {present}, missing: {len(missing)}")
    if missing:
        logger.warning(f"Missing variables: {missing}")

    # Generate HTML
    html = build_html(geojson)

    # Write output
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERACTIVE_DIR / "atlas.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Atlas written to {out_path} ({len(html)/1e6:.1f} MB)")

    # Count total variables
    total_vars = sum(len(vs) for vs in VARIABLE_GROUPS.values())
    logger.info(f"Total mappable variables: {total_vars} across {len(VARIABLE_GROUPS)} groups")

    print(f"\nAtlas built: {out_path}")
    print(f"  Variables: {present}/{total_vars} present ({len(missing)} missing)")
    print(f"  File size: {len(html)/1e6:.1f} MB")
    print(f"  Groups: {len(VARIABLE_GROUPS)}")


if __name__ == "__main__":
    main()
