import pytest
import pandas as pd


@pytest.fixture
def admissions():
    return pd.DataFrame(
        [
            [1, 0],
            [2, 0],
            [3, 1],
        ],
        columns=["HADM_ID", "LABEL"],
    )


@pytest.fixture
def chunked():
    return pd.DataFrame(
        [
            [1, "1", 0],
            [2, "1 2", 0],
            [2, "3", 0],
            # Positives
            [3, "1 2", 1],
            [3, "3 4", 1],
        ],
        columns=["HADM_ID", "TEXT", "LABEL"],
    )
