from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strengthstats.analysis.constants import ET
from strengthstats.analysis.preprocessor import (
    add_anyweight_column, divide_up_csv_lines, get_all_exercises_dfs,
    preprocess_data, preprocess_sets, preprocess_workouts,
    separate_sets_by_exercise_type)

TEST_DATA = "tests/analysis/resources/sample_export.csv"
SETS_LINES = (
    '0,"Exercise, Plank",Set,1,bodyweight,70,extraWeight,0,time,00:01:00'
    '\n0,"Exercise, Push-Up",Set,1,reps,10,bodyweight,70,extraWeight,0'
    '\n0,"Exercise, Push-Up",Set,2,reps,10,bodyweight,70,extraWeight,0'
    '\n0,"Exercise, Push-Up",Set,3,reps,12,bodyweight,70,extraWeight,0'
    '\n0,"Exercise, Push-Up",Set,4,reps,10,bodyweight,70,extraWeight,0'
    '\n0,"Exercise, Back Extension",Set,1,reps,15,bodyweight,70,extraWeight,10'
    '\n0,"Exercise, Back Extension",Set,2,reps,10,bodyweight,70,extraWeight,15'
    '\n0,"Exercise, Back Extension",Set,3,reps,10,bodyweight,70,extraWeight,15'
    '\n0,"Exercise, Back Extension",Set,4,reps,10,bodyweight,70,extraWeight,15'
    '\n0,"Exercise, Seated Leg Curl",Set,1,reps,10,weight,50'
    '\n0,"Exercise, Seated Leg Curl",Set,2,reps,9,weight,50'
    '\n0,"Exercise, Dumbbell Lunge",Set,1,reps,9,weight,20'
    '\n0,"Exercise, Dumbbell Lunge",Set,2,reps,9,weight,20'
    '\n0,"Exercise, Dumbbell Lunge",Set,3,reps,6,weight,24'
    '\n0,"Exercise, Squat",Set,1,reps,10,weight,110'
    '\n0,"Exercise, Squat",Set,2,reps,7,weight,100'
    '\n0,"Exercise, Squat",Set,3,reps,7,weight,110'
    '\n1,"Exercise, Deadlift",Set,1,reps,10,weight,100'
    '\n1,"Exercise, Deadlift",Set,2,reps,10,weight,100'
    '\n1,"Exercise, Deadlift",Set,3,reps,10,weight,105'
    '\n1,"Exercise, Bench Press",Set,1,reps,7,weight,80'
    '\n1,"Exercise, Bench Press",Set,2,reps,8,weight,80'
    '\n1,"Exercise, Bench Press",Set,3,reps,7,weight,80'
    '\n1,"Exercise, Bench Press",Set,4,reps,6,weight,80'
    '\n1,"Exercise, Squat",Set,1,reps,10,weight,100'
    '\n1,"Exercise, Squat",Set,2,reps,7,weight,100'
    '\n1,"Exercise, Squat",Set,3,reps,7,weight,110'
    '\n2,"Exercise, Plank",Set,1,bodyweight,70,extraWeight,0,time,00:01:00'
    '\n2,"Exercise, Push-Up",Set,1,reps,10,bodyweight,70,extraWeight,0'
    '\n2,"Exercise, Push-Up",Set,2,reps,10,bodyweight,70,extraWeight,0'
    '\n2,"Exercise, Push-Up",Set,3,reps,10,bodyweight,70,extraWeight,0'
    '\n2,"Exercise, Push-Up",Set,4,reps,10,bodyweight,70,extraWeight,10'
    '\n2,"Exercise, Back Extension",Set,1,reps,15,bodyweight,70,extraWeight,15'
    '\n2,"Exercise, Back Extension",Set,2,reps,10,bodyweight,70,extraWeight,15'
    '\n2,"Exercise, Back Extension",Set,3,reps,10,bodyweight,70,extraWeight,15'
    '\n2,"Exercise, Back Extension",Set,4,reps,10,bodyweight,70,extraWeight,15'
    '\n2,"Exercise, Seated Leg Curl",Set,1,reps,10,weight,50'
    '\n2,"Exercise, Seated Leg Curl",Set,2,reps,9,weight,50'
    '\n2,"Exercise, Dumbbell Lunge",Set,1,reps,9,weight,20'
    '\n2,"Exercise, Dumbbell Lunge",Set,2,reps,9,weight,20'
    '\n2,"Exercise, Dumbbell Lunge",Set,3,reps,6,weight,20'
    '\n2,"Exercise, Squat",Set,1,reps,10,weight,100'
    '\n2,"Exercise, Squat",Set,2,reps,7,weight,100'
    '\n2,"Exercise, Squat",Set,3,reps,7,weight,100'
    '\n3,"Exercise, Chin-Ups",Set,1,reps,9,bodyweight,70'
    '\n3,"Exercise, Chin-Ups",Set,2,reps,7,bodyweight,70'
    '\n3,"Exercise, Chin-Ups",Set,3,reps,7,bodyweight,70'
    '\n3,"Exercise, Bar Dip",Set,1,reps,10,bodyweight,70,extraWeight,0'
    '\n3,"Exercise, Bar Dip",Set,2,reps,10,bodyweight,70,extraWeight,0'
    '\n3,"Exercise, Bar Dip",Set,3,reps,8,bodyweight,70,extraWeight,10'
    '\n3,"Exercise, Dumbbell Curl",Set,1,reps,12,weight,10'
    '\n3,"Exercise, Dumbbell Curl",Set,2,reps,9,weight,10'
    '\n3,"Exercise, Dumbbell Curl",Set,3,reps,6,weight,10'
    '\n3,"Exercise, Sit-Up",Set,1,reps,20,bodyweight,70,extraWeight,0'
    '\n3,"Exercise, Sit-Up",Set,2,reps,20,bodyweight,70,extraWeight,0'
    '\n3,"Exercise, Sit-Up",Set,3,reps,20,bodyweight,70,extraWeight,10'
    '\n3,"Exercise, Leaning Plank",Set,1,reps,10,bodyweight,70,distanceMeter,0,time'
    ",00:00:05"
    '\n3,"Exercise, One-arm Push-Up",Set,1,reps,10,bodyweight,70,height,50'
    '\n3,"Exercise, One-arm Push-Up",Set,2,reps,8,bodyweight,70,height,50'
    '\n3,"Exercise, One-arm Push-Up",Set,3,reps,9,bodyweight,70,height,50'
    '\n3,"Exercise, One-arm Push-Up",Set,4,reps,7,bodyweight,70,height,50'
)

WORKOUTS_LINES = (
    "0,Program 1: Workout 1,2024-01-15,70,1,2,3,-1"
    "\n1,Program 1: Workout 2,2024-01-08,70,1,-1,1,-1"
    "\n2,Program 1: Workout 1,2024-01-01,70,-1,3,2,1"
    "\n3,Other program: Workout 1,2023-12-15,70,-1,-1,-1,-1"
)
EXPECTED_SET_COLUMNS = [
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
]
EXPECTED_WORKOUT_COLUMNS = [
    "Name",
    "Date",
    "Body weight",
    "Shape",
    "Sleep",
    "Calories",
    "Stress",
]


def test_preprocess_data():
    """Test that DataFrames look as expected when valid data is used."""
    sets, workouts = preprocess_data(TEST_DATA)

    assert list(sets.columns) == EXPECTED_SET_COLUMNS
    assert len(sets) == 59

    # Spot check
    assert sets.at[5, "Exercise"] == "Back Extension"
    assert sets.at[5, "Set"] == 1
    assert sets.at[5, "reps"] == 15
    assert sets.at[5, "bodyweight"] == 70
    assert sets.at[5, "extraWeight"] == 10
    assert not pd.notna(sets.at[5, "time"])
    assert not pd.notna(sets.at[5, "distanceMeter"])
    assert not pd.notna(sets.at[5, "height"])

    assert list(workouts.columns) == EXPECTED_WORKOUT_COLUMNS
    assert len(workouts) == 4

    # Spot check
    assert workouts.at[1, "Name"] == "Program 1: Workout 2"
    assert workouts.at[1, "Date"] == datetime(2024, 1, 8)
    assert workouts.at[1, "Body weight"] == 70
    assert workouts.at[1, "Shape"] == 1
    assert workouts.at[1, "Sleep"] == -1
    assert workouts.at[1, "Calories"] == 1
    assert workouts.at[1, "Stress"] == -1


def test_divide_up_csv_lines():
    """Test that the lines are divided as expected."""
    actual_sets_lines, actual_workouts_lines = divide_up_csv_lines(TEST_DATA)

    assert actual_sets_lines == SETS_LINES
    assert actual_workouts_lines == WORKOUTS_LINES


def test_preprocess_workouts():
    """Test that DataFrame looks as expected given correct lines."""
    workouts_df = preprocess_workouts(WORKOUTS_LINES)

    assert list(workouts_df.columns) == EXPECTED_WORKOUT_COLUMNS
    assert len(workouts_df) == 4

    # Spot check
    assert workouts_df.at[1, "Name"] == "Program 1: Workout 2"
    assert workouts_df.at[1, "Date"] == datetime(2024, 1, 8)
    assert workouts_df.at[1, "Body weight"] == 70
    assert workouts_df.at[1, "Shape"] == 1
    assert workouts_df.at[1, "Sleep"] == -1
    assert workouts_df.at[1, "Calories"] == 1
    assert workouts_df.at[1, "Stress"] == -1


def test_preprocess_sets():
    """Test that DataFrame looks as expected given correct lines."""
    sets_df = preprocess_sets(SETS_LINES)
    assert list(sets_df.columns) == EXPECTED_SET_COLUMNS
    assert len(sets_df) == 59

    # Spot check
    assert sets_df.at[5, "Exercise"] == "Back Extension"
    assert sets_df.at[5, "Set"] == 1
    assert sets_df.at[5, "reps"] == 15
    assert sets_df.at[5, "bodyweight"] == 70
    assert sets_df.at[5, "extraWeight"] == 10
    assert pd.isna(sets_df.at[5, "time"])
    assert pd.isna(sets_df.at[5, "distanceMeter"])
    assert pd.isna(sets_df.at[5, "height"])


def test_separate_sets_by_exercise_type():
    thirty_sec = pd.Timedelta(seconds=30)
    sets_df = pd.DataFrame(
        [
            (1, "Deadlift", 1, np.nan, np.nan, np.nan, 10, 100, np.nan, np.nan),
            (1, "Deadlift", 2, np.nan, np.nan, np.nan, 11, 110, np.nan, np.nan),
            (1, "Deadlift", 3, np.nan, np.nan, np.nan, 12, 130, np.nan, np.nan),
            (1, "Squat", 1, np.nan, np.nan, np.nan, 18, 100, np.nan, np.nan),
            (1, "Squat", 2, np.nan, np.nan, np.nan, 13, 100, np.nan, np.nan),
            (1, "Push-Up", 1, 70, 0, np.nan, 8, np.nan, np.nan, np.nan),
            (1, "Push-Up", 2, 70, 0, np.nan, 3, np.nan, np.nan, np.nan),
            (1, "Back Extension", 1, 70, 10, np.nan, 10, np.nan, np.nan, np.nan),
            (1, "Back Extension", 2, 70, 10, np.nan, 11, np.nan, np.nan, np.nan),
            (1, "Plank", 1, 70, 10, thirty_sec, np.nan, np.nan, np.nan, np.nan),
            (1, "Plank", 2, 70, 10, thirty_sec, np.nan, np.nan, np.nan, np.nan),
            (2, "Deadlift", 1, np.nan, np.nan, np.nan, 15, 110, np.nan, np.nan),
            (2, "Deadlift", 2, np.nan, np.nan, np.nan, 11, 120, np.nan, np.nan),
            (2, "Deadlift", 3, np.nan, np.nan, np.nan, 14, 115, np.nan, np.nan),
            (2, "Handstand", 1, 70, np.nan, thirty_sec, np.nan, np.nan, np.nan, np.nan),
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
            ]
        ),
    )
    split_sets_dfs = separate_sets_by_exercise_type(sets_df)

    assert list(split_sets_dfs.keys()) == [
        ET.TIME,
        ET.REPS,
        ET.WTIME,
        ET.WREPS,
        ET.OTHER,
    ]
    assert len(split_sets_dfs[ET.TIME]) == 1
    assert len(split_sets_dfs[ET.REPS]) == 2
    assert len(split_sets_dfs[ET.WREPS]) == 10
    assert len(split_sets_dfs[ET.WTIME]) == 2
    assert len(split_sets_dfs[ET.OTHER]) == 0


def test_generate_exercises_dataframe():
    """Test that expected columns exist and have expected values."""
    thirty_sec = pd.Timedelta(seconds=30)
    sets_df = pd.DataFrame(
        [
            (1, "Deadlift", 1, np.nan, np.nan, np.nan, 10, 100, np.nan, np.nan),
            (1, "Deadlift", 2, np.nan, np.nan, np.nan, 11, 110, np.nan, np.nan),
            (1, "Deadlift", 3, np.nan, np.nan, np.nan, 12, 130, np.nan, np.nan),
            (1, "Squat", 1, np.nan, np.nan, np.nan, 18, 100, np.nan, np.nan),
            (1, "Squat", 2, np.nan, np.nan, np.nan, 13, 100, np.nan, np.nan),
            (1, "Push-Up", 1, 70, 0, np.nan, 8, np.nan, np.nan, np.nan),
            (1, "Push-Up", 2, 70, 0, np.nan, 3, np.nan, np.nan, np.nan),
            (1, "Back Extension", 1, 70, 10, np.nan, 10, np.nan, np.nan, np.nan),
            (1, "Back Extension", 2, 70, 10, np.nan, 11, np.nan, np.nan, np.nan),
            (1, "Plank", 1, 70, 10, thirty_sec, np.nan, np.nan, np.nan, np.nan),
            (1, "Plank", 2, 70, 10, thirty_sec, np.nan, np.nan, np.nan, np.nan),
            (2, "Deadlift", 1, np.nan, np.nan, np.nan, 15, 110, np.nan, np.nan),
            (2, "Deadlift", 2, np.nan, np.nan, np.nan, 11, 120, np.nan, np.nan),
            (2, "Deadlift", 3, np.nan, np.nan, np.nan, 14, 115, np.nan, np.nan),
            (2, "Handstand", 1, 70, np.nan, thirty_sec, np.nan, np.nan, np.nan, np.nan),
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
            ]
        ),
    )

    exercise_dfs = get_all_exercises_dfs(sets_df)

    assert list(exercise_dfs[ET.WREPS].columns) == [
        "workout_index",
        "Exercise",
        "sets",
        "total_reps",
        "max_weight",
        "total_volume",
    ]

    # Spot check one row in weight-reps DataFrame
    assert exercise_dfs[ET.WREPS].at[3, "Exercise"] == "Deadlift"
    assert exercise_dfs[ET.WREPS].at[3, "sets"] == 3
    assert exercise_dfs[ET.WREPS].at[3, "total_reps"] == 40
    assert exercise_dfs[ET.WREPS].at[3, "max_weight"] == 120
    assert exercise_dfs[ET.WREPS].at[3, "total_volume"] == 4580

    # Check reps DataFrame
    assert exercise_dfs[ET.REPS].at[0, "Exercise"] == "Push-Up"
    assert exercise_dfs[ET.REPS].at[0, "sets"] == 2
    assert exercise_dfs[ET.REPS].at[0, "total_reps"] == 11
    assert pd.isnull(exercise_dfs[ET.REPS].at[0, "max_weight"])
    assert exercise_dfs[ET.REPS].at[0, "total_volume"] == 11

    # Check time Dataframe
    assert exercise_dfs[ET.TIME].at[0, "Exercise"] == "Handstand"
    assert exercise_dfs[ET.TIME].at[0, "sets"] == 1
    assert exercise_dfs[ET.TIME].at[0, "total_reps"] == 0
    assert pd.isnull(exercise_dfs[ET.TIME].at[0, "max_weight"])
    assert exercise_dfs[ET.TIME].at[0, "total_volume"] == pd.Timedelta(seconds=30)

    # Check weight-time DataFrame"
    assert exercise_dfs[ET.WTIME].at[0, "Exercise"] == "Plank"
    assert exercise_dfs[ET.WTIME].at[0, "sets"] == 2
    assert exercise_dfs[ET.WTIME].at[0, "total_reps"] == 0
    assert exercise_dfs[ET.WTIME].at[0, "max_weight"] == 10
    assert exercise_dfs[ET.WTIME].at[0, "total_volume"] == pd.Timedelta(minutes=10)


def test_add_anyweight_column():
    """Test adding the column works."""
    sets_df = pd.DataFrame(
        [
            (1, "exercise 1", 1, np.nan, np.nan, np.nan, 10, 100, np.nan, np.nan),
            (1, "exercise 2", 1, 100, np.nan, np.nan, 10, 100, np.nan, np.nan),
            (1, "exercise 3", 1, np.nan, 30, np.nan, 10, 100, np.nan, np.nan),
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
            ]
        ),
    )

    add_anyweight_column(sets_df)

    assert "anyWeight" in sets_df.columns
    assert sets_df.at[0, "anyWeight"] == 100
    assert sets_df.at[1, "anyWeight"] == 100
    assert sets_df.at[2, "anyWeight"] == 130
