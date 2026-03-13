from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_obesity_model(
    df: pd.DataFrame,
    output_dir: str | Path = "data/model_outputs",
) -> dict | None:
    """
    Train a Random Forest model to predict obesity percentage by ZCTA.

    This function selects the available predictor columns from the analytical
    dataset, splits the data into training and testing sets, fits the model,
    evaluates it, and saves the results to CSV files.

    Parameters
        df : pd.DataFrame
            The dataset used for modeling.
        output_dir : str | Path, default="data/model_outputs"
            The folder where model outputs will be saved.
    
    Returns
        dict
            A dictionary of model evaluation metrics if the model runs successfully,
            or None if the required columns are missing or there are too few rows.


    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = "obesity_pct"
    candidate_features = [
        "physical_inactivity_pct",
        "gym_count",
        "gyms_per_10000_adults",
        "totalpopulation",
        "totalpop18plus",
    ]

    available_features = [col for col in candidate_features if col in df.columns]

    if target not in df.columns or not available_features:
        return None

    model_df = df[["zcta5"] + available_features + [target]].copy()
    model_df = model_df.dropna(subset=[target])

    if len(model_df) < 25:
        return None

    X = model_df[available_features].fillna(model_df[available_features].median(numeric_only=True))
    y = model_df[target]

    X_train, X_test, y_train, y_test, zcta_train, zcta_test = train_test_split(
        X,
        y,
        model_df["zcta5"],
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rows_used": len(model_df),
        "feature_count": len(available_features),
        "r2": r2_score(y_test, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": mean_absolute_error(y_test, predictions),
    }

    pd.DataFrame([metrics]).to_csv(output_dir / "model_1_metrics.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "zcta5": zcta_test.values,
            "actual_obesity_pct": y_test.values,
            "predicted_obesity_pct": predictions,
        }
    )
    predictions_df.to_csv(output_dir / "model_1_predictions.csv", index=False)

    feature_importance_df = pd.DataFrame(
        {
            "feature": available_features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(output_dir / "model_1_feature_importance.csv", index=False)

    return metrics