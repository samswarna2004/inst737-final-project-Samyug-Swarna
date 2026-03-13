from __future__ import annotations

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
    """Run the full project workflow."""
    print("Starting project pipeline...")
    ensure_directories()

    places_raw = read_places_data()
    print("CDC PLACES data loaded.")

    places_long, places_wide = clean_places_data(places_raw)
    save_csv(places_long, "data/transformed/places_filtered_long.csv")
    save_csv(places_wide, "data/transformed/places_wide.csv")
    print("CDC PLACES data cleaned and transformed.")

    gym_counts = None

    try:
        zctas_raw = read_census_zctas()
        osm_points_raw = read_osm_points()
        osm_areas_raw = read_osm_areas()

        zctas = clean_census_zctas(zctas_raw)
        gyms = clean_osm_gyms(osm_points_raw, osm_areas_raw)
        gym_counts = count_gyms_by_zcta(gyms, zctas)

        save_csv(gym_counts, "data/transformed/gym_counts_by_zcta.csv")
        print("Spatial gym count step completed.")
    except Exception as exc:
        print(f"Spatial step skipped for now: {exc}")

    analytic_df = build_analytic_table(places_wide, gym_counts)
    save_csv(analytic_df, "data/transformed/analytic_dataset.csv")
    print("Analytical dataset created.")

    metrics = train_obesity_model(analytic_df)
    if metrics is not None:
        print("Model training complete.")
        print(metrics)
    else:
        print("Model training skipped because not enough data/features were available yet.")

    make_visualizations(analytic_df)
    print("Visualizations saved.")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()