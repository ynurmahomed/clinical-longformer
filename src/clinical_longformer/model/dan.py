import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import sys
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from pathlib import Path
from torchmetrics.functional import auc
from torchtext.experimental.vectors import GloVe
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.module import AGNNewsDataModule, MIMICIIIDataModule


BATCH_SIZE = 50
EMBED_DIM = 300
LEARNING_RATE = 5e-3
NUM_HIDDEN = 1
W_DECAY = 1e-5
WORD_DROPOUT = 0.3


def get_macro_auc_pr(precision, recall):
    return torch.tensor([auc(r, p) for r, p in zip(recall, precision)]).mean()


class DAN(pl.LightningModule):
    """Deep averaging network.

    Returns:
        DAN: Deep averaging network.
    """

    def __init__(
        self,
        vectors,
        vocab,
        embed_dim,
        labels,
        lr=LEARNING_RATE,
        num_hidden=NUM_HIDDEN,
        weight_decay=W_DECAY,
        p=WORD_DROPOUT,
        loss=F.cross_entropy,
    ):

        super().__init__()

        self.labels = labels

        num_class = len(labels)

        self.lr = lr
        self.weight_decay = weight_decay
        self.p = p
        self.save_hyperparameters("lr", "num_hidden", "weight_decay", "p")

        self.loss = loss

        if vectors is not None:
            pre_trained = vectors(vocab.itos)
            self.embedding = nn.EmbeddingBag.from_pretrained(pre_trained)
        else:
            self.embedding = nn.EmbeddingBag(len(vocab), embed_dim)

        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(embed_dim, num_class))

        self.feed_forward = nn.Sequential(*layers)

        self.softmax = nn.Softmax(dim=1)

        pr_curve = torchmetrics.PrecisionRecallCurve(num_class)

        self.train_pr_curve = pr_curve.clone()
        self.valid_pr_curve = pr_curve.clone()
        self.test_pr_curve = pr_curve.clone()

        self.confmat = torchmetrics.ConfusionMatrix(num_class, normalize="true")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DAN")
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--num_hidden", type=int, default=NUM_HIDDEN)
        parser.add_argument("--weight_decay", type=float, default=W_DECAY)
        parser.add_argument("--p", type=float, default=WORD_DROPOUT)
        return parent_parser

    @staticmethod
    def get_model_kwargs(namespace):
        kwargs = vars(namespace)
        return {
            k: kwargs[k]
            for k in kwargs.keys()
            & {"lr", "num_hidden", "weight_decay", "p"}
        }

    def forward(self, x, offsets):

        # Word dropout
        p = torch.full_like(x, self.p, dtype=torch.float)
        b = torch.bernoulli(p)
        x[b == 0] = 0

        embedded = self.embedding(x, offsets)

        ff = self.feed_forward(embedded)

        softmax = self.softmax(ff)

        return softmax

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/auc_pr": 0})

    def training_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        loss = self.loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.train_pr_curve(outputs["preds"], outputs["target"])

        auc_pr = get_macro_auc_pr(precision, recall)

        self.log("AUC-PR (macro)/train", auc_pr)

        self.log("Loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        loss = self.loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.valid_pr_curve(outputs["preds"], outputs["target"])

        auc_pr = get_macro_auc_pr(precision, recall)

        self.log("AUC-PR (macro)/valid", auc_pr)

        self.log("hp/auc_pr", auc_pr)

        self.log("Loss/valid", loss)

        return loss

    def test_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        return {"preds": preds, "target": y}

    def test_epoch_end(self, outputs):

        y = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        self.log_pr_curve(preds, y)

        self.log_confusion_matrix(preds.argmax(1), y)

    def log_pr_curve(self, preds, y):

        precision, recall, _ = self.test_pr_curve(preds, y)

        fig = self.get_pr_curve(precision, recall)

        auc_pr = get_macro_auc_pr(precision, recall)

        self.log("AUC-PR (macro)/test", auc_pr)

        self.logger.experiment.add_figure("PR Curve/test", fig, self.current_epoch)

    def get_pr_curve(self, precision, recall):

        pr_per_class = []

        for i, l in enumerate(self.labels):
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

    def log_confusion_matrix(self, preds, y):

        self.confmat(preds, y)

        cm_df = pd.DataFrame(
            self.confmat.compute().cpu().numpy(), index=self.labels, columns=self.labels
        )

        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(cm_df, annot=True, cmap="Spectral").get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure(
            "Confusion Matrix/test", fig, self.current_epoch
        )

    def configure_optimizers(self):
        return torch.optim.Adagrad(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def get_data_module():
    p = Path("/home/yassin/Projects/Masters/code/discharge")
    note_length = 512
    dm = MIMICIIIDataModule(p, note_length, BATCH_SIZE)
    dm.setup()
    return dm


def log_model_graph(datamodule, trainer, model):
    _, x, offsets = next(iter(datamodule.train_dataloader()))
    trainer.logger.log_graph(model, (x, offsets))


def parse_args(args):

    parser = ArgumentParser()

    parser.add_argument("--no_vectors", action="store_true")

    parser = DAN.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )

    return parser.parse_args(args)


def main(args):

    args = parse_args(args)

    dm = get_data_module()

    if args.no_vectors:
        vectors = None
    else:
        vectors = GloVe(name="6B", dim=EMBED_DIM)

    model = DAN(vectors, dm.vocab, EMBED_DIM, dm.labels, **DAN.get_model_kwargs(args))

    logger = TensorBoardLogger("lightning_logs", name="DAN", default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    log_model_graph(dm, trainer, model)

    trainer.fit(model, datamodule=dm)

    trainer.test(model, ckpt_path=None, datamodule=dm)


def run():
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
