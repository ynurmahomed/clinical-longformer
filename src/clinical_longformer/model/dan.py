import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from argparse import ArgumentParser
from pathlib import Path
from torchmetrics.functional import auc
from torchtext.experimental.vectors import GloVe
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.module import MIMICIIIDataModule, YelpReviewPolarityDataModule
from .utils import auc_pr, plot_pr_curve, plot_confusion_matrix


# Default hyperparameters
LEARNING_RATE = 1e-2
NUM_HIDDEN = 2
W_DECAY = 1e-5
WORD_DROPOUT = 0.7
BATCH_SIZE = 50
EMBED_DIM = 300


class DAN(pl.LightningModule):
    """Deep averaging network.

    Returns:
        DAN: Deep averaging network.
    """

    def __init__(
        self,
        vectors,
        vocab,
        labels,
        hparams,
    ):

        super().__init__()

        self.labels = labels

        embed_dim = hparams["embed_dim"]
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.p = hparams["p"]
        self.hparams = hparams

        if vectors is not None:
            pre_trained = vectors(vocab.itos)
            self.embedding = nn.EmbeddingBag.from_pretrained(pre_trained)
        else:
            self.embedding = nn.EmbeddingBag(len(vocab), embed_dim)

        layers = []
        for _ in range(hparams["num_hidden"]):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(embed_dim, 1))

        self.feed_forward = nn.Sequential(*layers)

        self.sigmoid = nn.Sigmoid()

        self.bce_loss = F.binary_cross_entropy

        # Metrics
        pr_curve = torchmetrics.PrecisionRecallCurve()

        self.train_pr_curve = pr_curve.clone()
        self.valid_pr_curve = pr_curve.clone()
        self.test_pr_curve = pr_curve.clone()

        self.confmat = torchmetrics.ConfusionMatrix(2, normalize="true")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DAN")
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--num_hidden", type=int, default=NUM_HIDDEN)
        parser.add_argument("--weight_decay", type=float, default=W_DECAY)
        parser.add_argument("--p", type=float, default=WORD_DROPOUT)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument(
            "--embed_dim",
            type=int,
            default=EMBED_DIM,
            choices=[50, 100, 200, 300],  # GloVe
        )
        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        return {
            k: hparams[k]
            for k in hparams.keys()
            & {"lr", "num_hidden", "weight_decay", "p", "batch_size", "embed_dim"}
        }

    def forward(self, x, offsets):

        # Word dropout
        p = torch.full_like(x, self.p, dtype=torch.float)
        b = torch.bernoulli(p)
        x[b == 0] = 0

        embedded = self.embedding(x, offsets)

        ff = self.feed_forward(embedded)

        sigmoid = self.sigmoid(ff)

        return sigmoid

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"AUC-PR/valid": 0})

    def training_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets).view(-1)

        loss = self.bce_loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.train_pr_curve(outputs["preds"], outputs["target"])

        self.log("AUC-PR/train", auc_pr(precision, recall))

        self.log("Loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets).view(-1)

        loss = self.bce_loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.valid_pr_curve(outputs["preds"], outputs["target"])

        self.log("AUC-PR/valid", auc_pr(precision, recall))

        self.log("Loss/valid", loss)

        return loss

    def test_step(self, batch, batch_idx):

        y, x, offsets = batch

        preds = self(x, offsets).view(-1)

        return {"preds": preds, "target": y}

    def test_epoch_end(self, outputs):

        y = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        self.log_pr_curve(preds, y)

        self.log_confusion_matrix(preds, y)

    def log_pr_curve(self, preds, y):

        precision, recall, _ = self.test_pr_curve(preds, y)

        fig = plot_pr_curve(precision, recall)

        self.log("AUC-PR/test", auc_pr(precision, recall))

        self.logger.experiment.add_figure("PR Curve/test", fig, self.current_epoch)

    def log_confusion_matrix(self, preds, y):

        cm = self.confmat(preds, y.int())

        fig = plot_confusion_matrix(cm, self.labels, self.labels)

        self.logger.experiment.add_figure(
            "Confusion Matrix/test", fig, self.current_epoch
        )

    def configure_optimizers(self):
        return torch.optim.Adagrad(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def get_data_module(args):
    p = Path(args.mimic_path)
    dm = MIMICIIIDataModule(p, args.batch_size, args.num_workers)
    dm.setup()
    return dm


def set_example_input_array(datamodule, model):
    _, x, offsets = next(iter(datamodule.train_dataloader()))
    model.example_input_array = [x, offsets]


def parse_args(args):

    parser = ArgumentParser()

    parser.add_argument(
        dest="mimic_path",
        help="Path containing train/valid/test datasets",
        type=str,
        default=os.getcwd(),
    )

    parser.add_argument("--no_vectors", action="store_true")

    parser.add_argument(
        "--num_workers",
        help="How many subprocesses to use for data loading. `0` means that the data will be loaded in the main process.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--logdir",
        help="Where to store pytorch lightning logs",
        type=str,
        default="lightning_logs",
    )

    parser = DAN.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )

    return parser.parse_args(args)


def main(args):

    args = parse_args(args)

    dm = get_data_module(args)

    if args.no_vectors:
        vectors = None
    else:
        vectors = GloVe(name="6B", dim=EMBED_DIM)

    model = DAN(vectors, dm.vocab, dm.labels, DAN.get_model_hparams(args))

    logger = TensorBoardLogger(
        args.logdir, name="DAN", default_hp_metric=False, log_graph=True
    )

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    set_example_input_array(dm, model)

    trainer.fit(model, datamodule=dm)

    trainer.test(model, ckpt_path=None, datamodule=dm)


def run():
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
