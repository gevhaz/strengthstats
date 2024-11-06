"""Functions for generating plots."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def generate_exercise_plots(
    exercise_df: pd.DataFrame,
    exercise_name: str,
    unit: str,
    dst_dir: str,
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
    this_exc = exercise_df["Exercise"] == exercise_name
    plt.plot(exercise_df[this_exc]["Date"], exercise_df[this_exc]["total_volume"])
    plt.title(f"{exercise_name} progress")
    plt.xlabel("Date")
    plt.ylabel(f"Volume ({unit})")
    plt.savefig(os.path.join(dst_dir, f"{exercise_name}.png"))
    plt.close()
