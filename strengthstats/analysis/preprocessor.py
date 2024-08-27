import csv
import logging
import sys
from io import StringIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def divide_up_csv_lines(data_path: str) -> tuple[str, str]:
    """Extract lines in CSV relevant to either workouts or exercises.

    Args:
        data_path: Path to a StrengthLog app exported CSV.

    Return:
        A tuple with one string containing lines relevant to exercises,
        and one string containing lines relevant to full workouts.
    """
    with open(data_path, "r") as f:
        raw_content = f.read()

    dividing_line = "Name,Date,Body weight,Shape,Sleep,Calories,Stress"
    try:
        raw_workout_data = raw_content.strip().split(dividing_line)[1]
    except IndexError:
        logger.error("The CSV file does not appear to be a StrengthLog app export file")
        sys.exit(1)

    raw_workouts = raw_workout_data.split("\n\n")
    if len(raw_workouts) < 1:
        logger.error("The CSV file does not contain any workouts")
        sys.exit(1)

    workouts_lines: list[str] = []
    exercises_lines: list[str] = []
    for index, raw_workout in enumerate(raw_workouts):
        raw_workout_lines = raw_workout.strip().split("\n")
        workouts_lines.append(f"{index},{raw_workout_lines[0]}")
        for line in raw_workout_lines[1:]:
            exercises_lines.append(f"{index},{line}")

    exercises_csv = "\n".join(exercises_lines)
    workouts_csv = "\n".join(workouts_lines)

    return exercises_csv, workouts_csv


def preprocess_workouts(workouts_csv: str) -> pd.DataFrame:
    """Create a DataFrame from workout-related CSV lines and clean it.

    Create a DataFrame from workout-related CSV lines, clean the data,
    and set data types.

    Args:
        exercises_csv: Lines from the StrengthLog app export with data
        about whole workouts.

    Return:
        DataFrame with one exercise per row, and its associated data.
    """
    column_names = [
        "Index",
        "Name",
        "Date",
        "Body weight",
        "Shape",
        "Sleep",
        "Calories",
        "Stress",
    ]
    workouts_s = StringIO(workouts_csv)
    workouts_df = pd.read_csv(
        workouts_s,
        header=None,
        names=column_names,
        index_col="Index",
        dtype={
            "Name": "string",
            "Date": "string",
            "Body weight": "int16",
            "Shape": "int8",
            "Sleep": "int8",
            "Calories": "int8",
            "Stress": "int8",
        },
    )

    workouts_df["Date"] = pd.to_datetime(workouts_df["Date"])

    return workouts_df


def preprocess_exercises(exercises_csv) -> pd.DataFrame:
    """Create a DataFrame from exercise-related CSV lines and clean it.

    Create a DataFrame from exercise-related CSV lines, clean the data,
    and set data types.

    Args:
        exercises_csv: Lines from the StrengthLog app export starting
        with '"Exercise, '.

    Return:
        DataFrame with one exercise per row, and its associated data.
    """
    exercises_s = StringIO(exercises_csv)
    reader = csv.reader(exercises_s)
    records: list[dict[str, Any]] = []
    for row in reader:
        record: dict[str, Any] = {}
        record["workout_index"] = row[0]
        record["Exercise"] = row[1]
        for i in range(2, len(row), 2):
            record[row[i]] = row[i + 1]
        records.append(record)

    exercises_df = pd.DataFrame(records)

    # We only deal with exercises that have reps
    exercises_df = exercises_df[exercises_df["reps"].notna()]
    exercises_df = pd.DataFrame(exercises_df)  # Help mypy

    exercises_df["workout_index"] = pd.to_numeric(exercises_df["workout_index"])
    exercises_df["Set"] = pd.to_numeric(exercises_df["Set"], downcast="integer")

    exercises_df["Exercise"] = exercises_df["Exercise"].astype("string")
    exercises_df["Exercise"] = exercises_df["Exercise"].map(
        lambda x: x.replace("Exercise, ", "")
    )

    if "reps" in exercises_df.columns:
        exercises_df["reps"] = pd.to_numeric(exercises_df["reps"], downcast="integer")

    if "bodyweight" in exercises_df.columns:
        exercises_df["bodyweight"] = pd.to_numeric(exercises_df["bodyweight"])
    if "weight" in exercises_df.columns:
        exercises_df["weight"] = pd.to_numeric(exercises_df["weight"])
    if "extraWeight" in exercises_df.columns:
        exercises_df["extraWeight"] = pd.to_numeric(exercises_df["extraWeight"])
    if "time" in exercises_df.columns:
        exercises_df["time"] = exercises_df["time"].astype("string")
    if "distanceMeter" in exercises_df.columns:
        exercises_df["distanceMeter"] = pd.to_numeric(exercises_df["distanceMeter"])
    if "height" in exercises_df.columns:
        exercises_df["height"] = pd.to_numeric(exercises_df["height"])

    return exercises_df


def preprocess_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-process StrengthLog data at path.

    Process a CSV file to produce two DataFrames with structured data.
    The CSV file is not proper CSV so lines need to be divided up
    according to what data set it actually belongs to
    (is the line for an exercise or a workout?). Appropriate types will
    be added to columns. Indices will be added to the workouts, and
    a corresponding 'workout_index' to the exercises.

    Param:
        data_path: Path to CSV file exported from the StrengthLog app.

    Return:
        Two DataFrames – one with all exercises and associated data,
        and one with all workouts and associated data.
    """
    exercises_csv, workouts_csv = divide_up_csv_lines(data_path)

    workouts_df = preprocess_workouts(workouts_csv)
    exercises_df = preprocess_exercises(exercises_csv)

    return exercises_df, workouts_df
