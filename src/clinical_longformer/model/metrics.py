import torch
import pandas as pd

from typing import Tuple
from torch import Tensor


def per_admission_predictions(
    hadm_ids: Tensor, preds: Tensor, target: Tensor, c=None
) -> Tuple[Tensor, Tensor]:
    """ClinicalBERT per admission prediction scaling."""

    device = preds.device

    df = pd.DataFrame(
        {
            "hadm_id": hadm_ids.cpu().numpy(),
            "preds": preds.cpu().numpy(),
            "target": target.cpu().numpy(),
        }
    )

    groupby = df.groupby("hadm_id")

    p_max = groupby.preds.max()
    p_mean = groupby.preds.mean()
    n = groupby.preds.count()

    p_readmit = get_p_readmit(p_max, p_mean, n, c)

    target = groupby.target.first()

    return torch.tensor(p_readmit.values, device=device), torch.tensor(
        target.values, device=device
    )


def get_p_readmit(p_max, p_mean, n, c=2):
    return (p_max + p_mean * n / c) / (1 + n / c)
