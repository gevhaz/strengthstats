"""Tests for the main app logic."""

import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strengthstats.analysis.constants import ET
from strengthstats.webapp.app import (
    app,
    generate_plots,
    get_safe_user_path,
    is_valid_csv_content,
)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False  # Disable CSRF for testing
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_csv_content():
    """Return valid CSV content for testing."""
    return b"col1,col2,col3\nval1,val2,val3\nval4,val5,val6\n"


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


class TestCSVValidation:
    """Tests for CSV content validation."""

    def test_valid_csv_content(self):
        """Test that valid CSV content is accepted."""
        valid_csv = b"header1,header2,header3\nval1,val2,val3\nval4,val5,val6\n"
        assert is_valid_csv_content(valid_csv) is True

    def test_invalid_csv_binary_content(self):
        """Test that binary (non-CSV) content is rejected."""
        binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # PNG header
        assert is_valid_csv_content(binary_content) is False

    def test_invalid_csv_encoding(self):
        """Test that content with invalid encoding is rejected."""
        invalid_encoding = b"\xff\xfe\x00\x00"  # Invalid UTF-8
        assert is_valid_csv_content(invalid_encoding) is False

    def test_empty_csv_content(self):
        """Test that empty content is handled correctly.

        Empty content returns True as it's technically valid CSV.
        """
        empty_csv = b""
        # Empty CSV is technically valid, just has no rows
        assert is_valid_csv_content(empty_csv) is True


class TestPathSecurity:
    """Tests for path traversal protection."""

    def test_safe_path_normal_filename(self):
        """Test that normal filenames generate safe paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["DATA_FOLDER"] = tmpdir
            safe_path = get_safe_user_path("user123", "data.csv")
            assert tmpdir in safe_path
            assert "user123" in safe_path
            assert "data.csv" in safe_path

    def test_safe_path_prevents_traversal(self):
        """Test that path traversal attempts are sanitized.

        secure_filename strips path separators, so "../../../etc/passwd"
        becomes "etc_passwd". No path traversal occurs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["DATA_FOLDER"] = tmpdir
            # secure_filename sanitizes "../../../etc/passwd"
            safe_path = get_safe_user_path("user123", "../../../etc/passwd")
            # Ensure the result is within the user folder
            assert tmpdir in safe_path
            assert "user123" in safe_path
            # And doesn't actually reference the system passwd file
            assert "/etc/passwd" not in safe_path

    def test_safe_path_sanitizes_filename(self):
        """Test that filenames are properly sanitized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["DATA_FOLDER"] = tmpdir
            safe_path = get_safe_user_path("user123", "my file.csv")
            # secure_filename converts spaces to underscores
            assert "my_file.csv" in safe_path

    def test_safe_path_rejects_empty_inputs(self):
        """Test that empty user ID or filename is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["DATA_FOLDER"] = tmpdir
            with pytest.raises(ValueError, match="Invalid user ID or filename"):
                get_safe_user_path("", "data.csv")
            with pytest.raises(ValueError, match="Invalid user ID or filename"):
                get_safe_user_path("user123", "")


class TestFileUpload:
    """Tests for file upload security and validation."""

    def test_upload_no_file(self, client):
        """Test that upload without file returns 400."""
        response = client.post("/upload_csv", data={})
        assert response.status_code == 400

    def test_upload_empty_filename(self, client):
        """Test that upload with empty filename returns 400."""
        data = {"strengthlog_csv": (io.BytesIO(b""), "")}
        response = client.post("/upload_csv", data=data)
        assert response.status_code == 400

    def test_upload_non_csv_extension(self, client):
        """Test that upload with non-CSV extension returns 400."""
        data = {"strengthlog_csv": (io.BytesIO(b"content"), "file.txt")}
        response = client.post("/upload_csv", data=data)
        assert response.status_code == 400

    def test_upload_empty_file(self, client):
        """Test that upload of empty file returns 400."""
        data = {"strengthlog_csv": (io.BytesIO(b""), "file.csv")}
        response = client.post("/upload_csv", data=data)
        assert response.status_code == 400

    def test_upload_invalid_csv_content(self, client):
        """Test that upload of non-CSV content returns 400."""
        binary_data = b"\x89PNG\r\n\x1a\n"  # PNG header
        data = {"strengthlog_csv": (io.BytesIO(binary_data), "file.csv")}
        response = client.post("/upload_csv", data=data)
        assert response.status_code == 400

    def test_upload_valid_csv(self, client, valid_csv_content):
        """Test that valid CSV upload succeeds and redirects."""
        data = {"strengthlog_csv": (io.BytesIO(valid_csv_content), "export.csv")}
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["DATA_FOLDER"] = tmpdir
            response = client.post("/upload_csv", data=data, follow_redirects=False)
            # Should redirect to report page
            assert response.status_code == 302
            assert "/report" in response.location


class TestReportGeneration:
    """Tests for report generation endpoint."""

    def test_report_without_csv(self, client):
        """Test that accessing report without CSV returns 404."""
        response = client.get("/report")
        assert response.status_code == 404

    def test_report_with_missing_csv_file(self, client):
        """Test that report with missing CSV file returns 404."""
        with client.session_transaction() as sess:
            sess["csv_path"] = "/nonexistent/path/file.csv"
        response = client.get("/report")
        assert response.status_code == 404


class TestSecurityConfiguration:
    """Tests for security configuration."""

    def test_secret_key_configured(self):
        """Test that secret key is configured."""
        assert app.secret_key is not None
        assert len(app.secret_key) > 0

    def test_max_content_length_configured(self):
        """Test that max content length prevents huge uploads."""
        assert "MAX_CONTENT_LENGTH" in app.config
        # Should be 16MB (16 * 1024 * 1024)
        assert app.config["MAX_CONTENT_LENGTH"] == 16 * 1024 * 1024
