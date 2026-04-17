from __future__ import annotations

import logging
from pathlib import Path

from analysis.model_1 import train_obesity_model
from etl.extract import (
    read_census_zctas,
    read_osm_areas,
    read_osm_points,
    read_places_data,
)
from etl.load import ensure_directories, save_csv
from etl.transform import (
    build_analytic_table,
    clean_census_zctas,
    clean_osm_gyms,
    clean_places_data,
    count_gyms_by_zcta,
)
from vis.visualizations import make_visualizations


def main() -> None:
    """
    Run the full project pipeline from raw data loading to final outputs.

    This function creates the required folders, reads the raw datasets, cleans
    and transforms the data, performs the spatial gym counting step, builds the
    final analytical table, trains the model, and saves visualizations.
    """
    ensure_directories()

    log_path = Path("logs/pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting project pipeline.")


    # Extract CDC
    try:
        places_raw = read_places_data()
        logger.info("CDC PLACES data loaded successfully.")
    except Exception:
        logger.exception("Failed to load CDC PLACES data.")
        return


    # Transform CDC
    try:
        places_long, places_wide = clean_places_data(places_raw)
        save_csv(places_long, "data/transformed/places_filtered_long.csv")
        save_csv(places_wide, "data/transformed/places_wide.csv")
        logger.info("CDC PLACES data cleaned and transformed successfully.")
    except Exception:
        logger.exception("Failed during CDC transformation stage.")
        return


    # Spatial ETL
    gym_counts = None

    try:
        zctas_raw = read_census_zctas()
        logger.info("Census ZCTA shapefile loaded successfully.")
    except Exception:
        logger.exception("Failed to load Census ZCTA shapefile.")
        zctas_raw = None

    try:
        osm_points_raw = read_osm_points()
        osm_areas_raw = read_osm_areas()
        logger.info("OSM POI shapefiles loaded successfully.")
    except Exception:
        logger.exception("Failed to load OSM shapefiles.")
        osm_points_raw = None
        osm_areas_raw = None

    if zctas_raw is not None and osm_points_raw is not None and osm_areas_raw is not None:
        try:
            zctas = clean_census_zctas(zctas_raw)
            gyms = clean_osm_gyms(osm_points_raw, osm_areas_raw)
            gym_counts = count_gyms_by_zcta(gyms, zctas)
            save_csv(gym_counts, "data/transformed/gym_counts_by_zcta.csv")
            logger.info("Spatial gym counting step completed successfully.")
        except Exception:
            logger.exception("Failed during spatial transformation/counting step.")
            gym_counts = None
    else:
        logger.warning("Spatial step skipped because one or more raw geospatial datasets failed to load.")


    # Build analytical dataset
    try:
        analytic_df = build_analytic_table(places_wide, gym_counts)
        save_csv(analytic_df, "data/transformed/analytic_dataset.csv")
        logger.info("Analytical dataset created successfully.")
    except Exception:
        logger.exception("Failed to build analytical dataset.")
        return

    # Modeling
    try:
        metrics = train_obesity_model(analytic_df)
        if metrics is not None:
            logger.info("Model training completed successfully.")
            logger.info("Model metrics: %s", metrics)
        else:
            logger.warning("Model training skipped because required features or enough rows were not available.")
    except Exception:
        logger.exception("Model training failed.")


    # Visualization
    try:
        make_visualizations(analytic_df)
        logger.info("Visualizations saved successfully.")
    except Exception:
        logger.exception("Visualization stage failed.")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()