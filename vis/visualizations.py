from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def make_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path = "data/visualizations",
) -> None:
    """
    Create project visualizations.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if "obesity_pct" in df.columns:
            top_obesity = df.sort_values("obesity_pct", ascending=False).head(15)

            plt.figure(figsize=(12, 6))
            plt.bar(top_obesity["zcta5"].astype(str), top_obesity["obesity_pct"])
            plt.xticks(rotation=45)
            plt.title("Top 15 ZCTAs by Obesity Percentage")
            plt.xlabel("ZCTA")
            plt.ylabel("Obesity %")
            plt.tight_layout()
            plt.savefig(output_dir / "top_15_obesity_zctas.png")
            plt.close()

        if {"physical_inactivity_pct", "obesity_pct"}.issubset(df.columns):
            plot_df = df.dropna(subset=["physical_inactivity_pct", "obesity_pct"])

            plt.figure(figsize=(8, 6))
            plt.scatter(plot_df["physical_inactivity_pct"], plot_df["obesity_pct"])
            plt.title("Obesity vs Physical Inactivity")
            plt.xlabel("Physical Inactivity %")
            plt.ylabel("Obesity %")
            plt.tight_layout()
            plt.savefig(output_dir / "obesity_vs_inactivity.png")
            plt.close()

        if {"gym_count", "obesity_pct"}.issubset(df.columns):
            plot_df = df.dropna(subset=["gym_count", "obesity_pct"])

            plt.figure(figsize=(8, 6))
            plt.scatter(plot_df["gym_count"], plot_df["obesity_pct"])
            plt.title("Obesity vs Gym Count")
            plt.xlabel("Gym Count")
            plt.ylabel("Obesity %")
            plt.tight_layout()
            plt.savefig(output_dir / "obesity_vs_gym_count.png")
            plt.close()

    except Exception as error:
        raise RuntimeError(f"Could not create project visualizations: {error}") from error