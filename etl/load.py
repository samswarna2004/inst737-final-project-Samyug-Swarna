from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directories() -> None:
    """
    Create the main project directory structure if the folders do not already exist.
    """
    directories = [
        "data/raw/cdc",
        "data/raw/census",
        "data/raw/osm",
        "data/transformed",
        "data/model_outputs",
        "data/visualizations",
        "data/reference-tables",
        "logs",
    ]

    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    except Exception as error:
        raise RuntimeError(f"Could not create project directories: {error}") from error


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save a DataFrame to CSV.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as error:
        raise RuntimeError(f"Could not save CSV to {path}: {error}") from error