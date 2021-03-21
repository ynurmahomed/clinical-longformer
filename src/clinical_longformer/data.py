import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path


_logger = logging.getLogger(__name__)


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
        NEXT_ADMISSION_ID=by_subject["HADM_ID"].shift(-1),
        NEXT_ADMITTIME=by_subject["ADMITTIME"].shift(-1),
        NEXT_ADMISSION_TYPE=by_subject["ADMISSION_TYPE"].shift(-1),
    )
    # Disregard elective.
    rows = df.NEXT_ADMISSION_TYPE == "ELECTIVE"
    df.loc[rows, "NEXT_ADMISSION_ID"] = pd.NaT
    df.loc[rows, "NEXT_ADMITTIME"] = pd.NaT
    df.loc[rows, "NEXT_ADMISSION_TYPE"] = np.NaN
    # Correct rows that pointed to elective, point to next admission.
    df = sort(df, ["SUBJECT_ID", "ADMITTIME"])
    next_columns = ["NEXT_ADMISSION_ID", "NEXT_ADMITTIME", "NEXT_ADMISSION_TYPE"]
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
    # Similar to len(str.split()) but str.count is vectorized so its faster
    return input_df.assign(LEN=input_df.TEXT.str.count(r"\S+"))


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
    """Read MIMIC-III admissions

    Args:
        mimic_path (str): MIMIC-III dataset location

    Returns:
        pandas.Dataframe: Admissions dataframe
    """
    admissions = Path(mimic_path) / "ADMISSIONS.csv"
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
        pd.read_csv(admissions, usecols=columns, dtype=dtypes, parse_dates=dates)
        .pipe(sort, ["SUBJECT_ID", "ADMITTIME"])
        .pipe(filter_newborn)
        .pipe(filter_death)
        .pipe(set_next_admission)
        .pipe(set_days_to_next_admission)
        .pipe(set_duration)
        .pipe(set_output_label)
    )


def read_notes(mimic_path):
    """Read MIMIC-III noteevents

    Args:
        mimic_path (str): MIMIC-III dataset location

    Returns:
        pandas.Dataframe: Note events dataframe
    """
    notes = Path(mimic_path) / "NOTEEVENTS.csv"
    columns = ["SUBJECT_ID", "HADM_ID", "CHARTDATE", "TEXT", "CATEGORY"]
    dtypes = {"CATEGORY": "category"}
    return pd.read_csv(
        notes, usecols=columns, dtype=dtypes, parse_dates=["CHARTDATE"]
    ).pipe(sort, ["SUBJECT_ID", "HADM_ID", "CHARTDATE"])


def chunk_text(input_df, note_length):
    """Chunk dataframe `TEXT` column into equal parts with `note_length` size

    Args:
        input_df (pandas.Dataframe): Dataframe with `HADM_ID`, `TEXT` and `OUTPUT_LABEL`
        note_length (int): Size of the chunks

    Returns:
        pandas.Dataframe: Dataframe with chunked text
    """
    chunked_df = pd.DataFrame({"ID": [], "TEXT": [], "LABEL": []})
    for i in range(len(input_df)):
        text = input_df.TEXT.iloc[i].split()
        n = int(len(text) / note_length)
        for j in range(n):
            row = {
                "TEXT": " ".join(text[j * note_length : (j + 1) * note_length]),
                "LABEL": input_df.OUTPUT_LABEL.iloc[i],
                "ID": input_df.HADM_ID.iloc[i],
            }
            chunked_df = chunked_df.append(row, ignore_index=True)

        # Add remaining < note_length chunk
        remain = len(text) % note_length
        if remain > 10:
            row = {
                "TEXT": " ".join(text[-(remain):]),
                "LABEL": input_df.OUTPUT_LABEL.iloc[i],
                "ID": input_df.HADM_ID.iloc[i],
            }
            chunked_df = chunked_df.append(row, ignore_index=True)

    return chunked_df


def split_admissions(df_adm, df_discharge, note_length, random_state=1):
    """Split admissions into training, validation and test.

    The training dataset will have balanced pos/neg examples.

    Args:
        df_adm (pandas.Dataframe): Admissions dataframe
        df_discharge (pandas.Dataframe): Discharge dataframe
        note_length (int): Size of the chunks
        random_state (int, optional): Random state for sampling. Defaults to 1.

    Returns:
        tuple: [description]
    """
    readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 1].HADM_ID
    not_readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 0].HADM_ID

    # Subsampling to get the balanced pos/neg numbers of patients for each dataset.
    positives = len(readmit_ID)
    not_readmit_ID_use = not_readmit_ID.sample(n=positives, random_state=random_state)

    # Sample 20% for validation and testing.
    id_val_test_t = readmit_ID.sample(frac=0.2, random_state=random_state)
    id_val_test_f = not_readmit_ID_use.sample(frac=0.2, random_state=random_state)

    # Remove validation and testing set from training.
    id_train_t = readmit_ID.drop(id_val_test_t.index)
    id_train_f = not_readmit_ID_use.drop(id_val_test_f.index)

    # Sample 50% of validation + testing, use rest for test.
    id_val_t = id_val_test_t.sample(frac=0.5, random_state=random_state)
    id_test_t = id_val_test_t.drop(id_val_t.index)

    # Sample 50% of validation + testing, use rest for test.
    id_val_f = id_val_test_f.sample(frac=0.5, random_state=random_state)
    id_test_f = id_val_test_f.drop(id_val_f.index)

    # Ensure no overlap between train and test.
    intersection = (pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values
    assert len(intersection) == 0

    # Join positive and negative for each split.
    id_train = pd.concat([id_train_t, id_train_f])
    id_val = pd.concat([id_val_t, id_val_f])
    id_test = pd.concat([id_test_t, id_test_f])
    discharge_train = df_discharge[df_discharge.ID.isin(id_train)]
    discharge_val = df_discharge[df_discharge.ID.isin(id_val)]
    discharge_test = df_discharge[df_discharge.ID.isin(id_test)]

    # Positive and negative examples might get unbalanced after chunking.
    # Positive usually have longer notes. Need to sample more negative examples.
    positive = len(discharge_train[discharge_train.LABEL == 1])
    negative = len(discharge_train[discharge_train.LABEL == 0])
    diff = positive - negative
    assert diff > 0

    # Discard already used examples.
    concat = pd.concat([not_readmit_ID_use, not_readmit_ID])
    unused = concat.drop_duplicates(keep=False)
    intersection = pd.Index(unused).intersection(pd.Index(not_readmit_ID_use))
    assert len(intersection) == 0

    # Sample remaining negative examples. The sum of the lengths must not
    # exceed diff * note_length. Wont get perfect balance between positive and
    # negative because of varying length of notes.
    unused_chunks = df_discharge[df_discharge.ID.isin(unused)]
    unused_chunks = set_token_length(unused_chunks)
    total_lengths = unused_chunks.groupby("ID").agg(totallen=("LEN", "sum"))
    total_lengths = total_lengths.sample(frac=1, random_state=random_state)
    cumsum = total_lengths.cumsum()
    not_readmit_ID_more = cumsum[cumsum.totallen <= diff * note_length].reset_index().ID
    remaining = df_discharge[df_discharge.ID.isin(not_readmit_ID_more)]
    discharge_train = pd.concat([remaining, discharge_train])

    # Shuffle.
    discharge_train = discharge_train.sample(frac=1, random_state=random_state)
    discharge_train = discharge_train.reset_index(drop=True)

    positive = len(discharge_train[discharge_train.LABEL == 1])
    negative = len(discharge_train[discharge_train.LABEL == 0])
    readmit = positive / len(discharge_train)
    not_readmit = negative / len(discharge_train)
    _logger.info(
        f"readmitted={readmit*100:,.0f}% not_readmitted={not_readmit*100:,.0f}%"
    )

    return discharge_train, discharge_val, discharge_test


def process_notes(mimic_path, category, note_length, out_path):
    if category == "ds":
        build_discharge_summary_dataset(mimic_path, note_length, out_path)
    else:
        pass


def build_discharge_summary_dataset(mimic_path, note_length, out_path):

    _logger.info(f"mimic_path={mimic_path} length={note_length}")

    df_adm = read_admissions(mimic_path)
    df_notes = read_notes(mimic_path)
    df_adm_notes = pd.merge(df_adm, df_notes, on=["SUBJECT_ID", "HADM_ID"], how="left")

    df_discharge = df_adm_notes[df_adm_notes["CATEGORY"] == "Discharge summary"]

    # If multiple discharge summaries for same admission, replace with last
    df_discharge = (
        df_discharge.groupby(["SUBJECT_ID", "HADM_ID"]).nth(-1)
    ).reset_index()
    df_discharge = df_discharge[df_discharge["TEXT"].notnull()]
    df_discharge = df_discharge.pipe(preprocessing).pipe(chunk_text, note_length)

    train, valid, test = split_admissions(df_adm, df_discharge, note_length)

    path = Path(".") / out_path / "discharge" / str(note_length)
    path.mkdir(parents=True, exist_ok=True)

    train.to_csv(path / "train.csv", index=False)
    valid.to_csv(path / "valid.csv", index=False)
    test.to_csv(path / "test.csv", index=False)
