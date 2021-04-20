import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from torchmetrics.functional import auc


def macro_auc_pr(precision, recall):
    return torch.tensor([auc(r, p) for r, p in zip(recall, precision)]).mean()


def plot_pr_curve(precision, recall, labels):

    sns.set_theme()

    pr_per_class = []

    for i, l in enumerate(labels):
        d = {"Precision": precision[i].cpu(), "Recall": recall[i].cpu(), "Label": l}
        df = pd.DataFrame(d)
        pr_per_class.append(df)

    pr = pd.concat(pr_per_class)

    plt.figure(figsize=(10, 7))

    ax = sns.lineplot(data=pr, x="Recall", y="Precision", hue="Label")

    h, l = ax.get_legend_handles_labels()
    legend_labels = [
        f"{c} (AUC {auc(recall[i], precision[i]):.2f})" for i, c in enumerate(l)
    ]
    ax.legend(h, legend_labels)

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
