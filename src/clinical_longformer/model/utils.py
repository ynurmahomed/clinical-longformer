import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from torchmetrics.functional import auc


def macro_auc_pr(precision, recall):
    return torch.tensor([auc(r, p) for r, p in zip(recall, precision)]).mean()


def auc_pr(precision, recall):
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


def plot_confusion_matrix(confmat, index, columns):

    sns.set_theme()

    df = pd.DataFrame(confmat.cpu().numpy(), index=index, columns=columns)

    plt.figure(figsize=(10, 7))
    fig = sns.heatmap(df, annot=True, cmap="Spectral").get_figure()
    plt.close(fig)

    return fig
