import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from torchmetrics.functional import auc
from matplotlib.axes import Axes
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def macro_auc_pr(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    return torch.tensor([auc(r, p) for r, p in zip(recall, precision)]).mean()


def auc_pr(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    return auc(recall, precision)


def plot_pr_curve(precision, recall):

    plt.figure(figsize=(10, 7))

    disp = PrecisionRecallDisplay(precision=precision, recall=recall)

    disp.plot()

    disp.ax_.legend([f"AUC {auc(recall, precision):.2f}"])

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    fig = disp.figure_
    plt.close(fig)

    return fig

def plot_roc_curve(fpr, tpr, roc_auc):

    plt.figure(figsize=(10, 7))

    disp = RocCurveDisplay(fpr=fpr, tpr=tpr)

    disp.plot()

    disp.ax_.legend([f"ROC AUC {roc_auc:.2f}"])

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    fig = disp.figure_
    plt.close(fig)

    return fig

def plot_confusion_matrix(confmat: torch.Tensor, index: list, columns: list) -> Axes:
    """Draw confusion matrix.

    Args:
        confmat (torch.Tensor): Confusion matrix data.
        index (list): The labels for the x axis.
        columns (list): The labels for the y axis.

    Returns:
        matplotlib Axes: Axes object with the confusion matrix.
    """

    sns.set_theme()

    # Flip so that when binary classification TP comes first.
    cm = confmat.flip(0, 1)
    df = pd.DataFrame(
        cm.cpu().numpy(), index=list(reversed(index)), columns=list(reversed(columns))
    )

    plt.figure(figsize=(10, 7))
    fig = sns.heatmap(df, annot=True, cmap="Spectral").get_figure()
    plt.close(fig)

    return fig
