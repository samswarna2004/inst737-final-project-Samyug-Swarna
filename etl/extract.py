from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW_DIR = Path("data/raw")


def _find_first(search_dir: Path, patterns: list[str]) -> Path:
    """
    Find the first file matching any pattern in a directory tree.
    
    Parameters:

    search_dir : Path
        The root directory to search within. The function will look through
        this directory and all nested subdirectories.

    patterns : list [str]
         A list of glob patterns to test, in priority order.

    
    Returns

        Path
             The path to the first matching file found in the directory tree.
    
        Raises
            Raised if no files matching any of the provided patterns are found in
            the target directory.

    """
    for pattern in patterns:
        matches = sorted(search_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file found in {search_dir} for patterns: {patterns}")



def get_places_csv_path(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the CDC PLACES CSV file."""
    return _find_first(raw_dir / "cdc", ["*.csv"])


def get_census_shapefile_path(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the Census ZCTA shapefile."""
    return _find_first(raw_dir / "census", ["*zcta520*.shp", "*.shp"])


def get_osm_point_shapefile_path(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the OSM POI point shapefile."""
    return _find_first(raw_dir / "osm", ["gis_osm_pois_free_1.shp"])


def get_osm_area_shapefile_path(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the OSM POI area shapefile."""
    return _find_first(raw_dir / "osm", ["gis_osm_pois_a_free_1.shp"])


def read_places_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the CDC PLACES CSV."""
    return pd.read_csv(get_places_csv_path(raw_dir))


def read_census_zctas(raw_dir: Path = RAW_DIR) -> gpd.GeoDataFrame:
    """Read the Census ZCTA shapefile."""
    return gpd.read_file(get_census_shapefile_path(raw_dir))


def read_osm_points(raw_dir: Path = RAW_DIR) -> gpd.GeoDataFrame:
    """Read the OSM POI point shapefile."""
    return gpd.read_file(get_osm_point_shapefile_path(raw_dir))


def read_osm_areas(raw_dir: Path = RAW_DIR) -> gpd.GeoDataFrame:
    """Read the OSM POI area shapefile."""
    return gpd.read_file(get_osm_area_shapefile_path(raw_dir))