"""Functions for preprocessing StrengthLog export CSV and create DFs.

The functions in this file are used for preprocessing the StrenghtLog
export CSV file and create DataFrames that can be used for analysis
later.

There are three general types of DataFrames defined in this file,
'workouts', 'sets', and 'exercises'. The DataFrames of type 'exercise'
have different subtypes, but their schema is the same. They differ in
how the exercise 'volume' is calculated.

The three types:

- workouts: Each row has a full workout, with general workoutout data
  such as sleep quality and stress level.
- sets: Has one workout-exercise-set per row, that is, per-set data for
  a set in a given exercise and workout.
- exercises: Has one workout-exercise per line, that is, data for each
  set of a given exercise in a given workout.
"""

import csv
import logging
import sys
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from strengthstats.analysis.constants import ET

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


def preprocess_workouts(workouts_csv: str) -> DataFrame:
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


def preprocess_sets(sets_csv: str, workouts_df: DataFrame) -> DataFrame:
    """Create a DataFrame from set-related CSV lines and clean it.

    Create a DataFrame from set-related CSV lines, clean the data,
    and set data types.

    Args:
        sets_csv: Lines from the StrengthLog app export starting
        with '"Exercise, '.

    Return:
        DataFrame with one set per row, and its associated data.
        Matching the original CSV file.
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

    sets_df = DataFrame(records)

    # We only deal with sets that have reps
    sets_df = sets_df[sets_df["reps"].notna()]
    sets_df = DataFrame(sets_df)  # Help mypy

    sets_df["workout_index"] = pd.to_numeric(sets_df["workout_index"])
    sets_df["Set"] = pd.to_numeric(sets_df["Set"], downcast="integer")

    sets_df["Exercise"] = sets_df["Exercise"].astype("string")
    sets_df["Exercise"] = sets_df["Exercise"].map(lambda x: x.replace("Exercise, ", ""))

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

    sets_df = sets_df.merge(
        workouts_df["Date"],
        left_on="workout_index",
        right_index=True,
    )

    return sets_df


def preprocess_data(data_path: str) -> tuple[DataFrame, DataFrame]:
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
    sets_df = preprocess_sets(sets_csv, workouts_df)

    return sets_df, workouts_df


def separate_sets_by_exercise_type(sets_df: DataFrame) -> dict[ET, DataFrame]:
    """Divide sets into separate DataFrames based on exercise type.

    There are five exercise types:

    - Exercises with weight specified but no time, nor reps.
    - Exercises with weight and reps specified.
    - Exercises with weight and time specified.
    - Exercises with time specified but no weight, nor reps.
    - The rest.

    This division is necessary because the way of calculating 'volume'
    is different in each case.
    """
    sets_dfs: dict[ET, DataFrame] = {}

    has_time = pd.notna(sets_df["time"])
    has_reps = pd.notna(sets_df["reps"])
    has_weight = (pd.notna(sets_df["weight"]) & sets_df["weight"] > 0) | (
        pd.notna(sets_df["extraWeight"]) & sets_df["extraWeight"] > 0
    )

    time_filter = has_time & ~has_weight & ~has_reps
    reps_filter = ~has_time & ~has_weight & has_reps
    weight_time_filter = has_time & has_weight & ~has_reps
    weight_reps_filter = ~has_time & has_weight & has_reps

    sets_dfs[ET.TIME] = DataFrame(sets_df[time_filter])
    sets_dfs[ET.REPS] = DataFrame(sets_df[reps_filter])
    sets_dfs[ET.WTIME] = DataFrame(sets_df[weight_time_filter])
    sets_dfs[ET.WREPS] = DataFrame(sets_df[weight_reps_filter])

    sets_dfs[ET.OTHER] = DataFrame(
        sets_df[~(time_filter | reps_filter | weight_time_filter | weight_reps_filter)]
    )

    return sets_dfs


def add_anyweight_column(sets_df: DataFrame) -> None:
    """Calculate volume when weight is involved."""
    sets_df["weight"] = sets_df["weight"].fillna(0)
    sets_df["extraWeight"] = sets_df["extraWeight"].fillna(0)
    sets_df["anyWeight"] = sets_df["weight"] + sets_df["extraWeight"]


def get_all_exercises_dfs(sets_df: DataFrame) -> dict[ET, DataFrame]:
    """Generate dict of DataFrames in 'exercise' format.

    Generate DataFrames with one weight-only workout-exercise per row,
    with summarized values for that exercise – in particular:

        - Total number of sets
        - Maximum weight lifted per exercise
        - Total number of reps
        - Total volume lifted

    Args:
        sets_df: A DataFrame with one workout-exercise-set per row.

    Return:
        Dictionary of all the generated workout-exercise DataFrames.
    """
    # Divide the sets into categories
    split_sets_dfs = separate_sets_by_exercise_type(sets_df)

    # Fill in reps and anyWeight columns for all sub-DataFrames.
    split_sets_dfs[ET.OTHER]["anyWeight"] = np.nan
    split_sets_dfs[ET.OTHER]["volume"] = np.nan
    split_sets_dfs[ET.TIME]["anyWeight"] = np.nan
    split_sets_dfs[ET.TIME]["volume"] = split_sets_dfs[ET.TIME]["time"]
    split_sets_dfs[ET.REPS]["anyWeight"] = np.nan
    split_sets_dfs[ET.REPS]["volume"] = split_sets_dfs[ET.REPS]["reps"]
    add_anyweight_column(split_sets_dfs[ET.WREPS])
    split_sets_dfs[ET.WREPS]["volume"] = (
        split_sets_dfs[ET.WREPS]["anyWeight"] * split_sets_dfs[ET.WREPS]["reps"]
    )
    add_anyweight_column(split_sets_dfs[ET.WTIME])
    split_sets_dfs[ET.WTIME]["volume"] = (
        split_sets_dfs[ET.WTIME]["anyWeight"] * split_sets_dfs[ET.WTIME]["time"]
    )

    # Generate the exercise type DataFrames
    exercise_dfs: dict[ET, DataFrame] = {}
    for exercise_type, sets_df in split_sets_dfs.items():
        grouped_exercise_df = sets_df.groupby(
            ["Date", "workout_index", "Exercise"],
        )[["Set", "reps", "anyWeight", "volume"]]
        exercise_df = grouped_exercise_df.agg(
            sets=pd.NamedAgg(column="Set", aggfunc="max"),
            total_reps=pd.NamedAgg(column="reps", aggfunc="sum"),
            max_weight=pd.NamedAgg(column="anyWeight", aggfunc="max"),
            total_volume=pd.NamedAgg(column="volume", aggfunc="sum"),
        )
        exercise_df = exercise_df.reset_index()
        exercise_dfs[exercise_type] = exercise_df

    return exercise_dfs
