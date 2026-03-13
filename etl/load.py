from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directories() -> None:
    """
    Create the main project directory structure if the folders do not already exist.

    Used for the load stage, to make sure it has the storage for raw data, transformed data,
    model outputs, visualizations, amd reference tables.

    Directories Made

        "data/raw/cdc",
        "data/raw/census",
        "data/raw/osm",
        "data/transformed",
        "data/model_outputs",
        "data/visualizations",
        "data/reference-tables"

    Returns nothing becayse its just creating the directories
    """
    directories = [
        "data/raw/cdc",
        "data/raw/census",
        "data/raw/osm",
        "data/transformed",
        "data/model_outputs",
        "data/visualizations",
        "data/reference-tables",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save a DataFrame to CSV.
    
    This function gives a reusable way to write DataFrame outputs

    Parameters
        df : pd.DataFrame
        
        path : str Path
            The destination file path for hte csv output
    
    Returns 
        nothing

    CSV is written with "index = False" so that the pandas row index is not included
    as an extra column in the saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)