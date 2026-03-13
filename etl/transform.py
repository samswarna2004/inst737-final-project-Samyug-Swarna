from __future__ import annotations

import re

import geopandas as gpd
import pandas as pd

DEFAULT_MEASURES = [
    "Obesity",
    "Physical Inactivity",
]


def _to_zcta5(value) -> str:
    """Convert a value into a 5-digit ZCTA string."""
    value_str = str(value).split(".")[0].strip()
    return value_str.zfill(5)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert values like '60,287' into numeric."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def _metric_to_column(metric_name: str) -> str:
    """Convert human-readable metric names into clean column names."""
    mapping = {
        "Obesity": "obesity_pct",
        "Physical Inactivity": "physical_inactivity_pct",
    }
    if metric_name in mapping:
        return mapping[metric_name]

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", metric_name.lower()).strip("_")
    return f"{cleaned}_pct"


def clean_places_data(
    df: pd.DataFrame,
    measures: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean, filter, and reshape the raw CDC PLACES dataset for downstream analysis.

    Stadnardizes column names, converts key fields to conistent types, filters the data
    to the mot recent year avaiable,limits the data to the selected health measures.

    Alos dtandardizes the ZCTA identifier, cleans population fields, and removes duplicate
    rows, and renames columns to better fit names

    Parameters

        df : DataFrame
            The raw CDC PLACES DataGrame loaded from the soruce csv file

        measures : list[str] None, deafult=None
            A list of the health measures to keep from the dataset, if no list is provided,
            the function uses the deafult roject measures stores in DEAULT_MEASURES
    
    Returns
        tuple[pd.DataFrame, pd.DataFrame]
    
    """
    if measures is None:
        measures = DEFAULT_MEASURES

    working = df.copy()
    working.columns = [col.strip() for col in working.columns]

    working["LocationID"] = working["LocationID"].apply(_to_zcta5)
    working["Data_Value"] = pd.to_numeric(working["Data_Value"], errors="coerce")
    working["TotalPopulation"] = _to_numeric(working["TotalPopulation"])
    working["TotalPop18plus"] = _to_numeric(working["TotalPop18plus"])
    working["Year"] = pd.to_numeric(working["Year"], errors="coerce")

    latest_year = int(working["Year"].max())

    filtered = working[
        (working["Year"] == latest_year)
        & (working["Short_Question_Text"].isin(measures))
        & (working["Data_Value_Type"].astype(str).str.contains("crude", case=False, na=False))
    ].copy()

    filtered = filtered[
        [
            "Year",
            "LocationID",
            "Short_Question_Text",
            "Data_Value",
            "Data_Value_Type",
            "TotalPopulation",
            "TotalPop18plus",
        ]
    ].drop_duplicates()

    populations = (
        filtered.groupby("LocationID", as_index=False)[["TotalPopulation", "TotalPop18plus"]]
        .max()
    )

    pivoted = (
        filtered.pivot_table(
            index="LocationID",
            columns="Short_Question_Text",
            values="Data_Value",
            aggfunc="mean",
        )
        .reset_index()
    )

    pivoted = pivoted.rename(columns={"LocationID": "zcta5"})
    pivoted = pivoted.rename(
        columns={col: _metric_to_column(col) for col in pivoted.columns if col != "zcta5"}
    )

    populations = populations.rename(
        columns={
            "LocationID": "zcta5",
            "TotalPopulation": "totalpopulation",
            "TotalPop18plus": "totalpop18plus",
        }
    )

    wide = pivoted.merge(populations, on="zcta5", how="left")

    filtered = filtered.rename(
        columns={
            "Year": "year",
            "LocationID": "zcta5",
            "Short_Question_Text": "measure",
            "Data_Value": "data_value",
            "Data_Value_Type": "data_value_type",
            "TotalPopulation": "totalpopulation",
            "TotalPop18plus": "totalpop18plus",
        }
    )

    return filtered, wide


def clean_census_zctas(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardize Census ZCTA shapefile columns.
    
    Prepares the raw Census TIGER/Line ZCTA shapefile for use in
    the project by identifying the correct ZCTA identifier column, selecting
    only the needed fields, and renaming the identifier to a consistent column

    Parameter
        gdf : gpd.GeoDataFrame
            The raw Census ZCTA GeoDataFrame read from the shapefile.

    Returns
        gpd.GeoDataFrame
            A simplified GeoDataFrame containing:
            zcta5 as the standardized ZCTA identifier
            geometry as the polygon boundary for each ZCTA


    """
    working = gdf.copy()

    zcta_col = None
    for candidate in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE24", "GEOID20", "GEOID"]:
        if candidate in working.columns:
            zcta_col = candidate
            break

    if zcta_col is None:
        raise KeyError("Could not find a ZCTA column in Census shapefile.")

    working = working[[zcta_col, "geometry"]].copy()
    working = working.rename(columns={zcta_col: "zcta5"})
    working["zcta5"] = working["zcta5"].astype(str).str.zfill(5)

    return working


def clean_osm_gyms(points_gdf: gpd.GeoDataFrame, areas_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter OSM POI layers down to gym/fitness-related records.
    
    It combines the point-based and area-based OSM POI layers and
    searches across common descriptive columns to identify records related to
    gyms, fitness centers, or sports centers.

    Parameter
         points_gdf : gpd.GeoDataFrame
            The OSM GeoDataFrame containing point-based POI features.
        areas_gdf : gpd.GeoDataFrame
            The OSM GeoDataFrame containing area-based or polygon POI features.

    Returns
        gpd.GeoDataFrame
            A GeoDataFrame of filtered gym and fitness-related records.

    """
    frames = []

    for gdf in [points_gdf, areas_gdf]:
        working = gdf.copy()
        text_columns = [col for col in ["fclass", "name", "type", "amenity", "other_tags"] if col in working.columns]

        if not text_columns:
            continue

        mask = pd.Series(False, index=working.index)

        for col in text_columns:
            mask = mask | working[col].astype(str).str.contains(
                "gym|fitness|fitness_centre|sports_centre",
                case=False,
                na=False,
            )

        gyms = working.loc[mask].copy()
        frames.append(gyms)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=points_gdf.crs)

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=points_gdf.crs)

    keep_cols = [col for col in ["name", "fclass", "geometry"] if col in combined.columns]
    if "geometry" not in keep_cols:
        keep_cols.append("geometry")

    return combined[keep_cols].copy()


def count_gyms_by_zcta(
    gyms_gdf: gpd.GeoDataFrame,
    zctas_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatially join gym features to ZCTA polygons and count gyms within each ZCTA.

    This function takes the filtered gym GeoDataFrame and the cleaned ZCTA
    boundary GeoDataFrame, aligns their coordinate reference systems, performs a
    spatial join, and then counts the number of gym-related features associated
    with each ZCTA.

    Parameters
         gyms_gdf : gpd.GeoDataFrame
            A GeoDataFrame containing filtered gym and fitness-related OSM features.
        zctas_gdf : gpd.GeoDataFrame
            A GeoDataFrame containing cleaned Census ZCTA polygons.

    Returns
        pd.DataFrame
            A DataFrame with one row per ZCTA and a `gym_count` column showing
            the number of matched gym features in that ZCTA.
    """
    if gyms_gdf.empty:
        return pd.DataFrame(columns=["zcta5", "gym_count"])

    gyms = gyms_gdf.to_crs(zctas_gdf.crs)

    joined = gpd.sjoin(
        gyms,
        zctas_gdf[["zcta5", "geometry"]],
        how="left",
        predicate="intersects",
    )

    counts = (
        joined.groupby("zcta5", as_index=False)
        .size()
        .rename(columns={"size": "gym_count"})
    )

    return counts


def build_analytic_table(
    places_wide: pd.DataFrame,
    gym_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge PLACES metrics with gym counts and create final analysis fields.

    This creates the projects main modeling table. It starts with the
    wide-form CDC PLACES dataset, merges in the ZCTA-level gym counts when they
    are available, fills missing gym counts with zero

    Parameters
        places_wide : pd.DataFrame
            A wide-form PLACES dataset with one row per ZCTA and selected health
            indicators stored as columns.
        gym_counts : pd.DataFrame | None, default=None
            A DataFrame containing ZCTA-level gym counts. If no gym counts are
            provided, the function creates a default `gym_count` column filled with
            zero.
    
    Returns
        pd.DataFrame
            A merged analytical dataset containing health indicators, population
            fields, gym counts, and engineered competition-related features.

    """
    analytic = places_wide.copy()

    if gym_counts is not None and not gym_counts.empty:
        analytic = analytic.merge(gym_counts, on="zcta5", how="left")
    else:
        analytic["gym_count"] = 0

    analytic["gym_count"] = analytic["gym_count"].fillna(0)

    if "totalpop18plus" in analytic.columns:
        analytic["gyms_per_10000_adults"] = (
            analytic["gym_count"] / analytic["totalpop18plus"].replace({0: pd.NA})
        ) * 10000

    return analytic