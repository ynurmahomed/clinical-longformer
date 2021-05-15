import pandas as pd
import pytest

from clinical_longformer.data.processing import split_discharge_summaries


def test_split_discharge_summaries(admissions, chunked):

    note_length = 1

    train, valid, test = split_discharge_summaries(admissions, chunked, note_length)

    pos = len(train[train.LABEL == 1])
    neg = len(train[train.LABEL == 0])

    assert pos == neg
