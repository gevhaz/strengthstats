"""Tests for the main app logic."""

import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strengthstats.analysis.constants import ET
from strengthstats.webapp.app import app, generate_plots


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


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


class TestErrorHandling:
    """Tests for error handling in web app."""

    def test_report_without_csv_returns_404(self, client):
        """Test that accessing report without CSV file returns 404."""
        response = client.get("/report")
        assert response.status_code == 404

    def test_invalid_csv_format_returns_400(self, client):
        """Test that uploading invalid CSV format returns 400 error.

        When a CSV file doesn't match the StrengthLog format,
        the app should return 400 (bad request) instead of crashing.
        """
        invalid_csv_content = b"col1,col2,col3\nval1,val2,val3\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["DATA_FOLDER"] = tmpdir

            # Upload the invalid CSV
            data = {
                "strengthlog_csv": (
                    io.BytesIO(invalid_csv_content),
                    "invalid.csv",
                )
            }
            response = client.post("/upload_csv", data=data, follow_redirects=False)

            # Upload should succeed (redirect)
            assert response.status_code == 302

            # But report generation should fail with 400
            response = client.get("/report")
            assert response.status_code == 400
            assert b"Invalid CSV file format" in response.data

    def test_missing_csv_file_returns_404(self, client):
        """Test that report with missing CSV returns 404."""
        with client.session_transaction() as sess:
            sess["csv_path"] = "/nonexistent/file.csv"
        response = client.get("/report")
        assert response.status_code == 404
