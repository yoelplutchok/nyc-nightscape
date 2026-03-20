.PHONY: setup migrate collect-t1 collect-t2 collect-t3 process analyze visualize validate all clean

SOURCE_REPO := $(HOME)/Desktop/Sleep_ESI_NYC

setup:
	conda env create -f environment.yml || conda env update -f environment.yml

migrate:
	python scripts/00_build_geo.py
	python scripts/01_migrate_data.py

collect-t1:
	python scripts/10_collect_crashes.py
	python scripts/11_collect_311_all.py
	python scripts/12_collect_fdny.py
	python scripts/13_collect_streetlights.py
	python scripts/14_collect_mta_turnstiles.py

collect-t2:
	python scripts/20_collect_tlc_trips.py
	python scripts/21_collect_arrests.py
	python scripts/22_collect_citibike.py
	python scripts/23_collect_pedestrians.py
	python scripts/24_collect_film_permits.py
	python scripts/25_collect_speed_cameras.py
	python scripts/26_collect_restaurants.py
	python scripts/27_collect_commercial_waste.py

collect-t3:
	python scripts/30_collect_sparcs_er.py
	python scripts/31_collect_energy_benchmarking.py
	python scripts/32_extract_helicopter_noise.py
	python scripts/33_collect_mta_gtfs.py
	python scripts/34_collect_linknyc.py
	python scripts/35_collect_ems_response_times.py

process:
	python scripts/40_build_master_dataset.py

analyze:
	python scripts/50_correlation_matrix.py
	python scripts/51_demographic_gradients.py
	python scripts/52_neighborhood_typologies.py
	python scripts/53_composite_index.py
	python scripts/54_regression_models.py
	python scripts/55_lisa_hotspots.py

visualize:
	python scripts/build_atlas_page.py

validate:
	python scripts/60_data_quality_report.py

all: migrate collect-t1 collect-t2 collect-t3 process analyze visualize validate
