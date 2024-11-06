"""Tests for the main app logic."""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from strengthstats.analysis.constants import ET
from strengthstats.webapp.app import generate_plots


def test_generate_plots():
    """Test that plots get generated and saved correctly."""
    day1 = datetime(year=2024, month=1, day=1)
    day2 = datetime(year=2024, month=1, day=2)
    day3 = datetime(year=2024, month=1, day=3)
    day4 = datetime(year=2024, month=1, day=4)
    sets_df = pd.DataFrame(
        [
            (1, "Deadlift", 1, np.nan, np.nan, np.nan, 10, 100, np.nan, np.nan, day1),
        ],
        columns=pd.Index(
            [
                "workout_index",
                "Exercise",
                "Set",
                "bodyweight",
                "extraWeight",
                "time",
                "reps",
                "weight",
                "distanceMeter",
                "height",
                "Date",
            ]
        ),
    )
    exercise_dfs = {
        ET.WREPS: pd.DataFrame(
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
    }
    with tempfile.TemporaryDirectory() as tempdir:
        session = {
            "id": "test-id",
            "user_folder": tempdir,
        }
        generate_plots(
            sets_df=sets_df,
            exercise_dfs=exercise_dfs,
            plots_dir=tempdir,
            session=session,
        )
        assert os.path.exists(os.path.join(tempdir, "Deadlift.png"))
