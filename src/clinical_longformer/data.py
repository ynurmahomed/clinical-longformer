import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def sort(input_df, columns):
    """Sort dataframe by specified columns

    Args:
        input_df (pandas.Dataframe): Dataframe to sort
        columns (list): Columns to sort by

    Returns:
        pandas.Dataframe: Sorted dataframe
    """
    df = input_df.sort_values(columns)
    df = df.reset_index(drop=True)
    return df


def set_next_admission(input_df):
    """Add next admission time and type to dataframe, ignores 'ELECTIVE' admissions

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with each admission pointing to the next
    """
    by_subject = input_df.groupby("SUBJECT_ID")
    df = input_df.assign(
        NEXT_ADMITTIME=by_subject["ADMITTIME"].shift(-1),
        NEXT_ADMISSION_TYPE=by_subject["ADMISSION_TYPE"].shift(-1),
    )
    # Disregard elective
    rows = df.NEXT_ADMISSION_TYPE == "ELECTIVE"
    df.loc[rows, "NEXT_ADMITTIME"] = pd.NaT
    df.loc[rows, "NEXT_ADMISSION_TYPE"] = np.NaN
    # Correct rows that pointed to elective, point to next admission
    df = sort(df, ["SUBJECT_ID", "ADMITTIME"])
    next_columns = ["NEXT_ADMITTIME", "NEXT_ADMISSION_TYPE"]
    df[next_columns] = df.groupby(["SUBJECT_ID"])[next_columns].fillna(method="bfill")
    return df


def filter_newborn(input_df):
    """Remove NEWBORN type admissions

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with newborns removed
    """
    return input_df[input_df.ADMISSION_TYPE != "NEWBORN"]


def filter_death(input_df):
    """Remove deaths

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with deaths removed
    """
    return input_df[input_df.DEATHTIME.isnull()]


def set_days_to_next_admission(input_df):
    """Add column with number of days until next admission

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with number of days until next admission
    """
    diff = input_df.NEXT_ADMITTIME - input_df.DISCHTIME
    df = input_df.assign(DAYS_NEXT_ADMIT=diff.dt.days)
    return df


def set_output_label(input_df):
    """Add column with the admission label (1 for readmited or 0 for not readmited)

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with labeled admissions
    """
    return input_df.assign(OUTPUT_LABEL=(input_df.DAYS_NEXT_ADMIT < 30).astype("int"))


def set_duration(input_df):
    """Add column with duration of the admission

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with durations
    """
    diff = input_df.DISCHTIME - input_df.ADMITTIME
    df = input_df.assign(DURATION=diff.dt.days)
    return df


def set_token_length(input_df):
    """Add column with the length of each notes in number of tokens.

    Args:
        input_df (pandas.Dataframe): Dataframe with clinical notes

    Returns:
        pandas.Dataframe: Dataframe with number of tokens column
    """
    return input_df.assign(LEN=input_df.TEXT.str.count(r"\w+"))


def preprocess1(x):
    y = re.sub("\\[(.*?)\\]", "", x)  # remove de-identified brackets
    y = re.sub(
        "[0-9]+\.", "", y
    )  # remove 1.2. since the segmenter segments based on this
    y = re.sub("dr\.", "doctor", y)
    y = re.sub("m\.d\.", "md", y)
    y = re.sub("admission date:", "", y)
    y = re.sub("discharge date:", "", y)
    y = re.sub("--|__|==", "", y)
    return y


def preprocessing(input_df):
    input_df["TEXT"] = input_df["TEXT"].fillna(" ")
    input_df["TEXT"] = input_df["TEXT"].str.replace("\n", " ")
    input_df["TEXT"] = input_df["TEXT"].str.replace("\r", " ")
    input_df["TEXT"] = input_df["TEXT"].apply(str.strip)
    input_df["TEXT"] = input_df["TEXT"].str.lower()
    input_df["TEXT"] = input_df["TEXT"].apply(lambda x: preprocess1(x))
    return input_df


def read_admissions(mimic_path):
    admissisons = os.path.join(mimic_path, "ADMISSIONS.csv")
    columns = [
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "ADMISSION_TYPE",
        "DEATHTIME",
    ]
    dtypes = {"ADMISSION_TYPE": "category"}
    dates = ["ADMITTIME", "DISCHTIME", "DEATHTIME"]
    return (
        pd.read_csv(admissisons, usecols=columns, dtype=dtypes, parse_dates=dates)
        .pipe(sort, ["SUBJECT_ID", "ADMITTIME"])
        .pipe(filter_newborn)
        .pipe(filter_death)
        .pipe(set_next_admission)
        .pipe(set_days_to_next_admission)
        .pipe(set_duration)
        .pipe(set_output_label)
    )


def read_notes(mimic_path):
    notes = os.path.join(mimic_path, "NOTEEVENTS.csv")
    columns = ["SUBJECT_ID", "HADM_ID", "CHARTDATE", "TEXT", "CATEGORY"]
    dtypes = {"CATEGORY": "category"}
    return pd.read_csv(
        notes, usecols=columns, dtype=dtypes, parse_dates=["CHARTDATE"]
    ).pipe(sort, ["SUBJECT_ID", "HADM_ID", "CHARTDATE"])


def build_discharge_summary_dataset(mimic_path):
    pass