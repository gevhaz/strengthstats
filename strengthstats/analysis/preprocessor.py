import csv
import logging
import sys
from io import StringIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def divide_up_csv_lines(data_path: str) -> tuple[str, str]:
    """Extract lines in CSV relevant to either workouts or sets.

    Args:
        data_path: Path to a StrengthLog app exported CSV.

    Return:
        A tuple with one string containing lines relevant to sets,
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
    sets_lines: list[str] = []
    for index, raw_workout in enumerate(raw_workouts):
        raw_workout_lines = raw_workout.strip().split("\n")
        workouts_lines.append(f"{index},{raw_workout_lines[0]}")
        for line in raw_workout_lines[1:]:
            sets_lines.append(f"{index},{line}")

    sets_csv = "\n".join(sets_lines)
    workouts_csv = "\n".join(workouts_lines)

    return sets_csv, workouts_csv


def preprocess_workouts(workouts_csv: str) -> pd.DataFrame:
    """Create a DataFrame from workout-related CSV lines and clean it.

    Create a DataFrame from workout-related CSV lines, clean the data,
    and set data types.

    Args:
        sets_csv: Lines from the StrengthLog app export with data
        about whole workouts.

    Return:
        DataFrame with one set per row, and its associated data.
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


def preprocess_sets(sets_csv) -> pd.DataFrame:
    """Create a DataFrame from set-related CSV lines and clean it.

    Create a DataFrame from set-related CSV lines, clean the data,
    and set data types.

    Args:
        sets_csv: Lines from the StrengthLog app export starting
        with '"Exercise, '.

    Return:
        DataFrame with one set per row, and its associated data.
    """
    sets_s = StringIO(sets_csv)
    reader = csv.reader(sets_s)
    records: list[dict[str, Any]] = []
    for row in reader:
        record: dict[str, Any] = {}
        record["workout_index"] = row[0]
        record["Exercise"] = row[1]
        for i in range(2, len(row), 2):
            record[row[i]] = row[i + 1]
        records.append(record)

    sets_df = pd.DataFrame(records)

    # We only deal with sets that have reps
    sets_df = sets_df[sets_df["reps"].notna()]
    sets_df = pd.DataFrame(sets_df)  # Help mypy

    sets_df["workout_index"] = pd.to_numeric(sets_df["workout_index"])
    sets_df["Set"] = pd.to_numeric(sets_df["Set"], downcast="integer")

    sets_df["Exercise"] = sets_df["Exercise"].astype("string")
    sets_df["Exercise"] = sets_df["Exercise"].map(
        lambda x: x.replace("Exercise, ", "")
    )

    if "reps" in sets_df.columns:
        sets_df["reps"] = pd.to_numeric(sets_df["reps"], downcast="integer")

    if "bodyweight" in sets_df.columns:
        sets_df["bodyweight"] = pd.to_numeric(sets_df["bodyweight"])
    if "weight" in sets_df.columns:
        sets_df["weight"] = pd.to_numeric(sets_df["weight"])
    if "extraWeight" in sets_df.columns:
        sets_df["extraWeight"] = pd.to_numeric(sets_df["extraWeight"])
    if "time" in sets_df.columns:
        sets_df["time"] = sets_df["time"].astype("string")
    if "distanceMeter" in sets_df.columns:
        sets_df["distanceMeter"] = pd.to_numeric(sets_df["distanceMeter"])
    if "height" in sets_df.columns:
        sets_df["height"] = pd.to_numeric(sets_df["height"])

    return sets_df


def preprocess_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-process StrengthLog data at path.

    Process a CSV file to produce two DataFrames with structured data.
    The CSV file is not proper CSV so lines need to be divided up
    according to what data set it actually belongs to
    (is the line for an set or a workout?). Appropriate types will
    be added to columns. Indices will be added to the workouts, and
    a corresponding 'workout_index' to the sets.

    Param:
        data_path: Path to CSV file exported from the StrengthLog app.

    Return:
        Two DataFrames – one with all sets and associated data,
        and one with all workouts and associated data.
    """
    sets_csv, workouts_csv = divide_up_csv_lines(data_path)

    workouts_df = preprocess_workouts(workouts_csv)
    sets_df = preprocess_sets(sets_csv)

    return sets_df, workouts_df
