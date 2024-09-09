"""Functions for generating plots."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_exercise_plots(
    exercise_df: pd.DataFrame,
    exercise_name: str,
    unit: str,
    dst_dir: Path,
) -> None:
    """Write plot of exercise maxes to dst_dir.

    Save a plot to dst dir that plots maximum weight lifted for the
    given exercise over time.

    Args:
        exercise_df: DataFrame with Date and total_volume to plot.
        exercise_name: Name of the exerices to plot.
        unit: Unit to display for the volume (e.g. 'tons').
        dst_dir: Directory where to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(exercise_df["Date"], exercise_df["total_volume"])
    plt.title(f"{exercise_name} progress")
    plt.xlabel("Date")
    plt.ylabel(f"Volume ({unit})")
    plt.savefig(dst_dir / f"{exercise_name}.png")
