from datetime import datetime

import pandas as pd
import pytest

from strengthstats.analysis.preprocessor import (divide_up_csv_lines,
                                                 preprocess_data,
                                                 preprocess_sets,
                                                 preprocess_workouts)

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
    assert not pd.notna(sets_df.at[5, "time"])
    assert not pd.notna(sets_df.at[5, "distanceMeter"])
    assert not pd.notna(sets_df.at[5, "height"])
