import torch
import pandas as pd

from torchmetrics.functional import precision_recall_curve
from torchmetrics import Metric


class ClinicalBERTBinnedPRCurve(Metric):
    def __init__(self, c=2):
        super().__init__()

        # Controls scaling factor for number of subsequences
        self.c = c

        self.add_state("hadm_id", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, hadm_id, preds, target):

        if not (hadm_id.size() == preds.size() == target.size()):
            raise ValueError("hadm_id preds and target should have same shape")

        self.hadm_id = hadm_id
        self.preds = preds.detach()
        self.target = target

    def compute(self):

        df = pd.DataFrame(
            {
                "hadm_id": self.hadm_id,
                "preds": self.preds,
                "target": self.target,
            }
        )

        groupby = df.groupby("hadm_id")

        p_max = groupby.preds.max()
        p_mean = groupby.preds.mean()
        n = groupby.preds.count()

        p_readmit = (p_max + p_mean * n / self.c) / (1 + n / self.c)

        target = groupby.target.first()

        return precision_recall_curve(
            torch.tensor(p_readmit.values), torch.tensor(target.values)
        )
