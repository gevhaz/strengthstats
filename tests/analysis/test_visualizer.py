"""Tests for visualizer, for generating plots."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strengthstats.analysis.visualizer import generate_exercise_plots


def test_generate_exercise_plots():
    """Test generating plot for exercise progress."""
    day1 = datetime(year=2024, month=1, day=1)
    day2 = datetime(year=2024, month=1, day=2)
    day3 = datetime(year=2024, month=1, day=3)
    day4 = datetime(year=2024, month=1, day=4)
    deadlift_df = pd.DataFrame(
        [
            (day1, 1, "Deadlift", 3, 30, 100, 3000),
            (day2, 2, "Deadlift", 3, 30, 110, 3300),
            (day3, 3, "Deadlift", 3, 30, 110, 3300),
            (day4, 4, "Deadlift", 3, 30, 120, 3600),
        ],
        columns=pd.Index(
            [
                "Date",
                "workout_index",
                "Exercise",
                "sets",
                "total_reps",
                "max_weight",
                "total_volume",
            ]
        ),
    )
    exercise_name = "Deadlift"
    dst_dir = Path("/path/to/nowhere")
    unit = "ton"

    expected_x_values = pd.Series([day1, day2, day3, day4])
    expected_y_values = pd.Series([3000, 3300, 3300, 3600])

    with patch("strengthstats.analysis.visualizer.plt") as plotmock:
        generate_exercise_plots(deadlift_df, exercise_name, unit, dst_dir)

        plotmock.figure.assert_called_once_with(figsize=(8, 6))

        actual_x_values = plotmock.plot.call_args[0][0]
        actual_y_values = plotmock.plot.call_args[0][1]
        assert actual_x_values.equals(expected_x_values)
        assert actual_y_values.equals(expected_y_values)

        plotmock.title.assert_called_once_with("Deadlift progress")
        plotmock.xlabel.assert_called_once_with("Date")
        plotmock.ylabel.assert_called_once_with("Volume (ton)")
        plotmock.savefig.assert_called_once_with("/path/to/nowhere/Deadlift.png")
