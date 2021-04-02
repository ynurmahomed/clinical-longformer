import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from torchtext.experimental.vectors import GloVe

from ..data.module import AGNNewsDataModule

BATCH_SIZE = 50
EMBED_DIM = 300
LEARNING_RATE = 0.01

"""Deep averaging network.

Returns:
    DAN: Deep averaging network.
"""


class DAN(pl.LightningModule):
    def __init__(
        self, vectors, vocab, embed_dim, labels, lr=LEARNING_RATE, loss=F.nll_loss
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

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_class),
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.train_f1 = pl.metrics.F1(num_class, average="weighted")
        self.valid_f1 = pl.metrics.F1(num_class, average="weighted")
        self.test_f1 = pl.metrics.F1(num_class, average="weighted")

        self.confmat = pl.metrics.ConfusionMatrix(num_class, normalize="true")

    def forward(self, x, offsets):

        embedded = self.embedding(x, offsets)

        ff = self.ff(embedded)

        log_softmax = self.log_softmax(ff)

        return log_softmax

    def training_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        loss = self.loss(preds, y)

        self.train_f1(preds, y)

        self.log(
            "F1Score/train", self.train_f1, on_step=False, on_epoch=True, prog_bar=True
        )

        self.log("Loss/train", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        loss = self.loss(preds, y)

        self.valid_f1(preds, y)

        self.log("F1Score/valid", self.valid_f1, on_epoch=True, prog_bar=True)

        self.log("Loss/valid", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets)

        return {"preds": preds, "y": y}

    def test_epoch_end(self, outputs):

        y = torch.cat([o["y"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])
        preds = torch.argmax(preds, dim=1)

        f1 = self.test_f1(preds, y)

        self.log_confusion_matrix(preds, y)

        self.log("F1Score/test", f1)

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
    dm = AGNNewsDataModule(BATCH_SIZE)
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

    trainer.test(ckpt_path=None, datamodule=dm)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
