"""Main logic of the web app."""

import csv
import os
from typing import NoReturn
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, redirect, render_template, request, session, url_for
from flask.sessions import SessionMixin
from flask_wtf.csrf import CSRFProtect  # type: ignore[import-untyped]
from werkzeug.utils import secure_filename
from werkzeug.wrappers.response import Response

from strengthstats.analysis.constants import ET, Units
from strengthstats.analysis.preprocessor import get_all_exercises_dfs, preprocess_data
from strengthstats.analysis.visualizer import generate_exercise_plots

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Enable CSRF protection
csrf = CSRFProtect(app)


DATA_FOLDER = "data"
EXPORT_CSV_NAME = "strengthlog_export.csv"
app.config["DATA_FOLDER"] = DATA_FOLDER
os.makedirs(DATA_FOLDER, exist_ok=True)


def is_valid_csv_content(file_content: bytes) -> bool:
    """Validate that file content is actually CSV format.

    Args:
        file_content: Raw bytes from uploaded file.

    Returns:
        True if content appears to be valid CSV, False otherwise.
    """
    try:
        # Decode and try to parse as CSV
        text_content = file_content.decode("utf-8")
        # Try to read at least the first few lines as CSV
        csv_reader = csv.reader(text_content.splitlines())
        # Read first 5 lines to validate structure
        for i, _ in enumerate(csv_reader):
            if i >= 5:
                break
        return True
    except (UnicodeDecodeError, csv.Error):
        return False


def get_safe_user_path(user_id: str, filename: str) -> str:
    """Get a safe file path within user's directory.

    Args:
        user_id: UUID string for the user session.
        filename: Filename to store (will be sanitized).

    Returns:
        Absolute path that is guaranteed to be within user's folder.

    Raises:
        ValueError: If the resulting path would escape user's directory.
    """
    # Sanitize user_id and filename
    safe_user_id = secure_filename(user_id)
    safe_filename = secure_filename(filename)

    if not safe_user_id or not safe_filename:
        raise ValueError("Invalid user ID or filename")

    # Build the path
    user_folder = os.path.join(app.config["DATA_FOLDER"], safe_user_id)
    target_path = os.path.abspath(os.path.join(user_folder, safe_filename))
    expected_prefix = os.path.abspath(user_folder)

    # Ensure the path doesn't escape the user folder
    if not target_path.startswith(expected_prefix + os.sep):
        raise ValueError("Path traversal attempt detected")

    return target_path


@app.route("/")
def index() -> str:
    """Render the homepage of the app."""
    return render_template("index.html")


@app.route("/upload_csv", methods=["POST"])
def upload_strenghlog_export() -> Response:
    """Route handler for accepting the Strengthlog export CSV file."""
    # Check that the file is available
    if "strengthlog_csv" not in request.files:
        app.logger.warning("No file provided in request")
        abort(400, "No file provided")

    f = request.files["strengthlog_csv"]

    # Validate filename exists and has .csv extension
    if not f.filename or f.filename == "":
        app.logger.warning("Empty filename provided")
        abort(400, "No file selected")

    if not f.filename.endswith(".csv"):
        app.logger.warning(f"Invalid file extension: {f.filename}")
        abort(400, "File must be a CSV file")

    # Read and validate file content
    file_content = f.read()
    if not file_content:
        app.logger.warning("Empty file uploaded")
        abort(400, "File is empty")

    if not is_valid_csv_content(file_content):
        app.logger.warning("File content is not valid CSV")
        abort(400, "File does not contain valid CSV data")

    # Ensure user folder exists and get safe path
    ensure_user_folder(session)
    try:
        csv_path = get_safe_user_path(session["id"], EXPORT_CSV_NAME)
    except ValueError as e:
        app.logger.error(f"Path validation failed: {e}")
        abort(400, "Invalid file path")

    # Save the validated file
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "wb") as output_file:
            output_file.write(file_content)
        session["csv_path"] = csv_path
        app.logger.info(f"Saved CSV file to {csv_path}")
    except OSError as e:
        app.logger.error(f"Failed to save file: {e}")
        abort(500, "Failed to save file")

    return redirect(url_for("generate_report"))


@app.route("/report")
def generate_report() -> str | NoReturn:
    """Generate report."""
    if "csv_path" not in session or not os.path.exists(session["csv_path"]):
        abort(404, "No CSV file found for this session")

    sets_df, _ = preprocess_data(session["csv_path"])
    exercise_dfs = get_all_exercises_dfs(sets_df)
    plots_dir = os.path.join(session["user_folder"], "plots")
    generate_plots(sets_df, exercise_dfs, plots_dir, session)
    app.logger.info(f"Generated and saved plots to {plots_dir}")

    return render_template("report.html")


def generate_plots(
    sets_df: pd.DataFrame,
    exercise_dfs: dict[ET, pd.DataFrame],
    plots_dir: str,
    session: SessionMixin,
) -> None:
    """Generate plots for user session and save."""
    exercise_type_map = {}
    for exercise_type, exercice_df in exercise_dfs.items():
        for exercise_name in exercice_df["Exercise"]:
            exercise_type_map[exercise_name] = exercise_type

    top_ten_exercises = list(sets_df["Exercise"].value_counts(sort=True)[:10].index)
    for exercise_name in top_ten_exercises:
        exc_type = exercise_type_map.get(exercise_name)
        if exc_type is None:
            app.logger.warning(
                f"Couldn't find exercise type for exercise {exercise_name}"
            )
            continue
        generate_exercise_plots(
            exercise_df=exercise_dfs[exc_type],
            exercise_name=exercise_name,
            unit=Units.short[exc_type],
            dst_dir=plots_dir,
        )


def ensure_user_folder(session: SessionMixin) -> None:
    """Ensure folder structure for user data exists when session starts.

    Ensure folder structure for user data exists when session starts,
    including subfolder for plots.
    """
    if "id" not in session:
        session["id"] = str(uuid4())

    user_folder = os.path.join(app.config["DATA_FOLDER"], session["id"])
    plots_folder = os.path.join(user_folder, "plots")

    # Create both directories if they don't exist (race-condition safe)
    os.makedirs(user_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    if "user_folder" not in session:
        session["user_folder"] = user_folder
        app.logger.info(f"User directory initialized: {user_folder}")
