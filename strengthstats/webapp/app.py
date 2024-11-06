"""Main logic of the web app."""

import os
from typing import NoReturn
from uuid import uuid4

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, session, url_for
from flask.sessions import SessionMixin
from werkzeug.wrappers.response import Response

from strengthstats.analysis.constants import ET, Units
from strengthstats.analysis.preprocessor import get_all_exercises_dfs, preprocess_data
from strengthstats.analysis.visualizer import generate_exercise_plots

app = Flask(__name__)
app.secret_key = "replace_with_something_secure"


DATA_FOLDER = "data"
EXPORT_CSV_NAME = "strengthlog_export.csv"
app.config["DATA_FOLDER"] = DATA_FOLDER
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)


@app.route("/")
def index() -> str:
    """Render the homepage of the app."""
    return render_template("index.html")


@app.route("/upload_csv", methods=["POST"])
def upload_strenghlog_export() -> Response:
    """Route handler for accepting the Strengthlog export CSV file."""
    # Check that the file is available
    if "strengthlog_csv" not in request.files:
        app.logger.warning("Something went wrong with submitting the file")
        return redirect(url_for("index"))
    f = request.files["strengthlog_csv"]
    if f.filename == "" or not f.filename.endswith(".csv"):
        app.logger.warning("The file does not have the .csv extension")
        return redirect(url_for("index"))

    # Save the file to disk for use in report generation
    ensure_user_folder(session)
    session["csv_path"] = os.path.join(session["user_folder"], EXPORT_CSV_NAME)
    f.save(session["csv_path"])
    app.logger.info(f"Saved/overwrote CSV file {session['csv_path']}")

    return redirect(url_for("generate_report"))


@app.route("/report")
def generate_report() -> str | NoReturn:
    """Generate report."""
    if "csv_path" not in session or not os.path.exists(session["csv_path"]):
        abort(500, "No CSV file found for this session")

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

    if not os.path.exists(user_folder):
        os.mkdir(user_folder)
        app.logger.info("User directory created")

    if not os.path.exists(os.path.join(user_folder, "plots")):
        os.mkdir(os.path.join(user_folder, "plots"))
        app.logger.info("User plots directory created")

    if "user_folder" not in session:
        session["user_folder"] = user_folder
