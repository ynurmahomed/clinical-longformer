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
from torchtext.experimental.vectors import GloVe

from ..data.module import AGNNewsDataModule, MIMICIIIDataModule

BATCH_SIZE = 50
EMBED_DIM = 300
LEARNING_RATE = 0.01

"""Deep averaging network.

Returns:
    DAN: Deep averaging network.
"""


class DAN(pl.LightningModule):
    def __init__(
        self, vectors, vocab, embed_dim, labels, lr=LEARNING_RATE, loss=F.cross_entropy
    ):

        super().__init__()

        self.labels = labels

        num_class = len(labels)

        self.lr = lr

        self.loss = loss

        if vectors is not None:
            pre_trained = vectors(vocab.itos)
            self.embedding = nn.EmbeddingBag.from_pretrained(pre_trained)
        else:
            self.embedding = nn.EmbeddingBag(len(vocab), embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_class),
        )

        self.softmax = nn.Softmax(dim=1)

        auc = torchmetrics.AUC(reorder=True)

        self.train_auc = auc.clone()
        self.valid_auc = auc.clone()
        self.test_auc = auc.clone()

        pr_curve = torchmetrics.PrecisionRecallCurve(num_class)

        self.test_pr_curve = pr_curve.clone()

        self.confmat = torchmetrics.ConfusionMatrix(num_class, normalize="true")

    def forward(self, x, offsets):

        embedded = self.embedding(x, offsets)

        ff = self.feed_forward(embedded)

        softmax = self.softmax(ff)

        return softmax

    def training_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        loss = self.loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):

        loss = outputs["loss"]

        self.train_auc(outputs["preds"].argmax(1), outputs["target"])

        self.log("AUC/train", self.train_auc)

        self.log("Loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        loss = self.loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):

        loss = outputs["loss"]

        self.valid_auc(outputs["preds"].argmax(1), outputs["target"])

        self.log("AUC/valid", self.valid_auc)

        self.log("Loss/valid", loss)

        return loss

    def test_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        return {"preds": preds, "target": y}

    def test_epoch_end(self, outputs):

        y = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        auc = self.test_auc(preds.argmax(1), y)

        self.log_pr_curve(preds, y)

        self.log_confusion_matrix(preds.argmax(1), y)

        self.log("AUC/test", auc)

    def log_pr_curve(self, preds, y):

        precision, recall, thresholds = self.test_pr_curve(preds, y)

        pr_per_class = []
        for i, l in enumerate(self.labels):
            d = {"Precision": precision[i], "Recall": recall[i], "Label": l}
            df = pd.DataFrame(d)
            pr_per_class.append(df)

        pr = pd.concat(pr_per_class)

        plt.figure(figsize=(10, 7))
        fig = sns.lineplot(data=pr, x="Recall", y="Precision", hue="Label").get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure("PR Curve/Test", fig, self.current_epoch)

    def log_confusion_matrix(self, preds, y):

        self.confmat(preds, y)

        cm_df = pd.DataFrame(
            self.confmat.compute().numpy(), index=self.labels, columns=self.labels
        )

        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(cm_df, annot=True, cmap="Spectral").get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure(
            "Confusion Matrix/Test", fig, self.current_epoch
        )

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self.lr)

    def get_offsets(self, xb):
        offsets = [0] + [len(entry) for entry in xb]
        return torch.tensor(offsets[:-1]).cumsum(dim=0)


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

    model = DAN(vectors, dm.vocab, EMBED_DIM, dm.labels)

    trainer = pl.Trainer.from_argparse_args(args)

    log_model_graph(dm, trainer, model)

    trainer.fit(model, datamodule=dm)

    trainer.test(model, ckpt_path=None, datamodule=dm)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
