import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from torchmetrics.functional import auc
from matplotlib.axes import Axes


def macro_auc_pr(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    return torch.tensor([auc(r, p) for r, p in zip(recall, precision)]).mean()


def auc_pr(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    return auc(recall, precision)


def plot_pr_curve(precision, recall):

    sns.set_theme()

    d = {"Precision": precision.cpu(), "Recall": recall.cpu()}
    df = pd.DataFrame(d)

    plt.figure(figsize=(10, 7))

    ax = sns.lineplot(data=df, x="Recall", y="Precision")

    ax.legend([f"AUC {auc(recall, precision):.2f}"])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig = ax.get_figure()
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
