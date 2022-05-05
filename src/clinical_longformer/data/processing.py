import logging
import re

import numpy as np
import pandas as pd

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
    return input_df.assign(LABEL=(input_df.DAYS_NEXT_ADMIT < 30).astype("int"))


def set_duration(input_df):
    """Add column with duration of the admission

    Args:
        input_df (pandas.Dataframe): Dataframe with admissions

    Returns:
        pandas.Dataframe: Dataframe with durations
    """
    diff = input_df.DISCHTIME - input_df.ADMITTIME
    df = input_df.assign(DURATION=diff)
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
        input_df (pandas.Dataframe): Dataframe with `HADM_ID`, `TEXT` and `LABEL`
        note_length (int): Size of the chunks

    Returns:
        pandas.Dataframe: Dataframe with chunked text
    """

    def chunked(arr, size):
        for i in range(0, len(arr), size):
            yield arr[i : i + size]

    chunked_df = input_df
    chunked_df.loc[:, "TEXT"] = chunked_df.TEXT.str.split()
    chunked_df.loc[:, "TEXT"] = chunked_df.TEXT.apply(
        lambda txt: [" ".join(c) for c in chunked(txt, note_length)]
    )
    chunked_df = chunked_df.explode("TEXT")
    return chunked_df


def get_admissions_split(readmitted, not_readmitted, random_state):
    """Splits admissions into train, validation and test sets.

    Args:
        readmitted (pandas.Series): Admissions resulting in readmission.
        not_readmitted (pandas.Series): Admissions not resulting in readmission.
        random_state (int): Random state for sampling.

    Returns:
        tuple: Series for each train/validation/test set.
    """
    # Sample 20% for validation and testing.
    id_val_test_t = readmitted.sample(frac=0.2, random_state=random_state)
    id_val_test_f = not_readmitted.sample(frac=0.2, random_state=random_state)

    # Remove validation and testing set from training.
    id_train_t = readmitted.drop(id_val_test_t.index)
    id_train_f = not_readmitted.drop(id_val_test_f.index)

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

    return id_train, id_val, id_test


def get_unused_admissions(all_adm, used):
    """Unused admissions.

    Args:
        all_adm (pandas.Series): All admission ids.
        used (pandas.Series): Admissions already used.

    Returns:
        Series: Unused admissions.
    """
    concat = pd.concat([used, all_adm])
    unused = concat.drop_duplicates(keep=False)
    intersection = pd.Index(unused).intersection(pd.Index(used))
    assert len(intersection) == 0
    return unused


def get_remaining_chunks_to_balance(chunked, unused, diff, random_state):
    """Returns additional chunks of notes needed to balance positive and negative
    examples, considering to the `diff` parameter.

    Args:
        chunked (pandas.Dataframe): Chunked notes.
        unused (pandas.Series): Admissions to draw additional chunks from.
        diff (int): Difference in note length between positive and negative examples.
        random_state (int): Random state for sampling.

    Returns:
        pandas.Dataframe: Additional chunks needed for balancing examples.
    """
    unused_chunks = chunked[chunked.HADM_ID.isin(unused)]
    unused_chunks = set_token_length(unused_chunks)
    total_lengths = unused_chunks.groupby("HADM_ID").agg(totallen=("LEN", "sum"))
    total_lengths = total_lengths.sample(frac=1, random_state=random_state)
    cumsum = total_lengths.cumsum()
    not_readmit_ID_more = cumsum[cumsum.totallen <= diff].reset_index().HADM_ID
    remaining = chunked[chunked.HADM_ID.isin(not_readmit_ID_more)]
    return remaining


def split_discharge_summaries(admissions, chunked, random_state=1):
    """Split chunked discharge summaries into training, validation and test sets.

    The training dataset will have balanced pos/neg examples.

    Args:
        admissions (pandas.Dataframe): Admissions dataframe
        chunked (pandas.Dataframe): Dataframe with chunked text
        random_state (int, optional): Random state for sampling. Defaults to 1.

    Returns:
        tuple: Dataframes for each train/validation/test split
    """
    readmit_ID = admissions[admissions.LABEL == 1].HADM_ID
    not_readmit_ID = admissions[admissions.LABEL == 0].HADM_ID

    # Subsampling to get the balanced pos/neg numbers of patients for each dataset.
    positives = len(readmit_ID)
    not_readmit_ID_use = not_readmit_ID.sample(n=positives, random_state=random_state)

    id_train, id_val, id_test = get_admissions_split(
        readmit_ID, not_readmit_ID_use, random_state
    )
    discharge_train = chunked[chunked.HADM_ID.isin(id_train)]
    discharge_val = chunked[chunked.HADM_ID.isin(id_val)]
    discharge_test = chunked[chunked.HADM_ID.isin(id_test)]

    # Positive and negative examples might get unbalanced after chunking.
    # Positive usually have longer notes. Need to sample more negative examples.
    discharge_train = set_token_length(discharge_train)
    positive = discharge_train[discharge_train.LABEL == 1].LEN.sum()
    negative = discharge_train[discharge_train.LABEL == 0].LEN.sum()
    diff = positive - negative
    assert diff > 0

    # Discard already used examples.
    unused = get_unused_admissions(not_readmit_ID, not_readmit_ID_use)

    # Sample remaining negative examples. The sum of the lengths must not
    # exceed diff. Wont get perfect balance between positive and
    # negative because of varying length of notes.
    remaining = get_remaining_chunks_to_balance(
        chunked, unused, diff, random_state
    )
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


def split_all_notes(admissions, chunked, n_days, random_state=1):
    """Split chunked notes into training, validation and test sets.

    The training dataset will have balanced pos/neg examples.

    Args:
        admissions (pandas.Dataframe): Admissions dataframe
        chunked (pandas.Dataframe): Dataframe with chunked text
        n_days(int): Number of days of notes to collect.
        random_state (int, optional): Random state for sampling. Defaults to 1.

    Returns:
        tuple: Dataframes for each train/validation/test split.
    """
    readmit_ID = admissions[admissions.LABEL == 1].HADM_ID
    not_readmit_ID = admissions[admissions.LABEL == 0].HADM_ID

    # Subsampling to get the balanced pos/neg numbers of patients for each dataset.
    positives = len(readmit_ID)
    not_readmit_ID_use = not_readmit_ID.sample(n=positives, random_state=random_state)

    id_train, id_val, id_test = get_admissions_split(
        readmit_ID, not_readmit_ID_use, random_state
    )
    all_notes_train = chunked[chunked.HADM_ID.isin(id_train)]
    all_notes_val = chunked[chunked.HADM_ID.isin(id_val)]

    # We want to test on admissions that are not discharged already, so we
    # filter out admissions discharged within n_days. Since each set of < n_days
    # is a subset of the n_days set, we only need to train on n_days to be able
    # make predictions on < n_days. No need for train and validation sets for < n_days.
    all_notes_test = []
    for i in reversed(range(1, n_days + 1)):
        time_delta = pd.Timedelta(i, "days")
        actionable = admissions[admissions["DURATION"] >= time_delta].HADM_ID
        id_test_actionable = id_test[id_test.isin(actionable)]
        test_set = chunked[chunked.HADM_ID.isin(id_test_actionable)]
        all_notes_test.append(test_set)

    # Positive and negative examples might get unbalanced after chunking.
    # Positive usually have longer notes. Need to sample more negative examples.
    positive = len(all_notes_train[all_notes_train.LABEL == 1])
    negative = len(all_notes_train[all_notes_train.LABEL == 0])
    diff = positive - negative
    assert diff > 0

    # Discard already used examples.
    unused = get_unused_admissions(not_readmit_ID, not_readmit_ID_use)

    # Sample remaining negative examples. The sum of the lengths must not
    # exceed diff * note_length. Wont get perfect balance between positive and
    # negative because of varying length of notes.
    remaining = get_remaining_chunks_to_balance(
        chunked, unused, diff, random_state
    )
    all_notes_train = pd.concat([remaining, all_notes_train])

    # Shuffle.
    all_notes_train = all_notes_train.sample(frac=1, random_state=random_state)
    all_notes_train = all_notes_train.reset_index(drop=True)

    positive = len(all_notes_train[all_notes_train.LABEL == 1])
    negative = len(all_notes_train[all_notes_train.LABEL == 0])
    readmit = positive / len(all_notes_train)
    not_readmit = negative / len(all_notes_train)
    _logger.info(
        f"readmitted={readmit*100:,.0f}% not_readmitted={not_readmit*100:,.0f}%"
    )

    return all_notes_train, all_notes_val, all_notes_test


def process_notes(mimic_path, category, note_length, n_days, out_path):
    if category == "ds":
        build_discharge_summary_dataset(mimic_path, note_length, out_path)
    else:
        build_all_notes_dataset(mimic_path, note_length, n_days, out_path)


def build_discharge_summary_dataset(mimic_path, note_length, out_path):

    _logger.debug(f"mimic_path={mimic_path} length={note_length}, out={out_path}")

    df_adm = read_admissions(mimic_path)
    df_notes = read_notes(mimic_path)
    df_adm_notes = pd.merge(df_adm, df_notes, on=["SUBJECT_ID", "HADM_ID"], how="left")

    df_discharge = df_adm_notes[df_adm_notes["CATEGORY"] == "Discharge summary"]

    # If multiple discharge summaries for same admission, replace with last
    df_discharge = df_discharge.groupby(["SUBJECT_ID", "HADM_ID"]).nth(-1)
    df_discharge = df_discharge.reset_index()

    df_discharge = df_discharge[df_discharge["TEXT"].notnull()]

    df_discharge = df_discharge.pipe(preprocessing)

    if note_length != -1:
        df_discharge = df_discharge.pipe(chunk_text, note_length)

    train, valid, test = split_discharge_summaries(df_adm, df_discharge)

    path = Path(".") / out_path / "discharge" / str(note_length)
    path.mkdir(parents=True, exist_ok=True)

    columns = ["HADM_ID", "TEXT", "LABEL"]
    train.to_csv(path / "train.csv", index=False, columns=columns)
    valid.to_csv(path / "valid.csv", index=False, columns=columns)
    test.to_csv(path / "test.csv", index=False, columns=columns)


def build_all_notes_dataset(mimic_path, note_length, n_days, out_path):

    _logger.debug(
        f"mimic_path={mimic_path} length={note_length} days={n_days} out={out_path}"
    )

    df_adm = read_admissions(mimic_path)
    df_notes = read_notes(mimic_path)
    df_adm_notes = pd.merge(df_adm, df_notes, on=["SUBJECT_ID", "HADM_ID"], how="left")

    chartdate = df_adm_notes.CHARTDATE.dt.date
    admittime = df_adm_notes.ADMITTIME.dt.date
    less_n_days = chartdate - admittime < pd.Timedelta(n_days, "days")
    df_n_days = df_adm_notes[less_n_days]
    df_n_days = df_n_days[df_n_days.TEXT.notnull()]
    df_n_days = df_n_days.pipe(preprocessing)

    if note_length != -1:
        df_n_days = df_n_days.pipe(chunk_text, note_length)

    train, valid, test = split_all_notes(df_adm, df_n_days, n_days, note_length)

    base = Path(".") / out_path / "all" / str(note_length)
    path = base / f"{n_days}days"
    path.mkdir(parents=True, exist_ok=True)

    columns = ["HADM_ID", "TEXT", "LABEL"]
    train.to_csv(path / "train.csv", index=False, columns=columns)
    valid.to_csv(path / "valid.csv", index=False, columns=columns)

    for i in reversed(range(len(test))):
        path = base / f"{n_days - i}days"
        path.mkdir(parents=True, exist_ok=True)
        test[i].to_csv(path / "test.csv", index=False, columns=columns)
