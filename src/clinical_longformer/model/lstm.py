import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb

from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchtext.vocab import GloVe
from torchmetrics import (
    AveragePrecision,
    MetricCollection,
    PrecisionRecallCurve,
)

from ..data.module import MIMICIIIDataModule
from .utils import auc_pr, plot_confusion_matrix, plot_pr_curve


SEED = 42
# Default hyperparameters
LEARNING_RATE = 5e-4
HIDDEN_DIM = 200
BATCH_SIZE = 64
EMBED_DIM = 300
DROPOUT = 5e-1


class LSTMClassifier(pl.LightningModule):
    """Bi-LSTM with global max-pooling."""

    def __init__(self, vectors, vocab, labels, hparams):
        super().__init__()

        self.labels = labels

        self.save_hyperparameters(hparams, ignore=["vectors", "vocab", "labels"])

        if vectors is not None:
            pre_trained = vectors.get_vecs_by_tokens(list(vocab.get_stoi()))
            self.embedding = nn.Embedding.from_pretrained(pre_trained)
        else:
            self.embedding = nn.Embedding(len(vocab), self.hparams.embed_dim)

        self.lstm = nn.LSTM(
            hparams["embed_dim"],
            self.hparams.hidden_dim,
            bidirectional=True,
            dropout=self.hparams.dropout,
        )

        # Global max pool
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.linear = nn.Sequential(
            # Double input size for bi-LSTM
            nn.Linear(2 * self.hparams.hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()

        # Metrics
        metrics = MetricCollection([AveragePrecision(pos_label=1), PrecisionRecallCurve(pos_label=1)])

        self.train_metrics = metrics.clone()
        self.valid_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

        self.confmat = torchmetrics.ConfusionMatrix(2, normalize="true")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LSTMClassifier")
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument(
            "--embed_dim", type=int, default=EMBED_DIM, choices=[50, 100, 200, 300]
        )
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        want_keys = {"lr", "hidden_dim", "batch_size", "embed_dim", "dropout"}
        return {k: hparams[k] for k in hparams.keys() & want_keys}

    def forward(self, x):

        embedding = self.embedding(x)

        out, _ = self.lstm(embedding)

        # Swap sequence length and feature size dimensions
        transpose = out.transpose(0, 2)

        # Global max pool over the sequence length dimension
        max_pool = self.max_pool(transpose)

        # Swap back seq length and feature size
        transpose = max_pool.transpose(0, 2)

        linear = self.linear(transpose)

        return linear

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x).view(-1)

        loss = self.bce_loss(preds, y)

        self.log("Loss/train", loss)

        return {"loss": loss, "preds": preds.detach(), "target": y}

    def training_epoch_end(self, outputs):

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        metrics = self.train_metrics(preds, target)

        precision, recall, _ = metrics["PrecisionRecallCurve"]

        self.log("AUC-PR/train", auc_pr(precision, recall))

    def validation_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x).view(-1)

        loss = self.bce_loss(preds, y)

        self.log("Loss/valid", loss)

        return {"loss": loss, "preds": preds.detach(), "target": y}

    def validation_epoch_end(self, outputs):

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        metrics = self.valid_metrics(preds, target)

        precision, recall, _ = metrics["PrecisionRecallCurve"]

        self.log("AUC-PR/valid", auc_pr(precision, recall))

        self.log("AVG-Precision/valid", metrics["AveragePrecision"])

    def test_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x).view(-1)

        loss = self.bce_loss(preds, y)

        return {"preds": preds.detach(), "target": y}

    def test_epoch_end(self, outputs):

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        self.log_test_metrics(preds, target)

    def log_test_metrics(self, preds, target):

        metrics = self.test_metrics(preds, target)

        precision, recall, _ = metrics["PrecisionRecallCurve"]

        fig = plot_pr_curve(precision.cpu(), recall.cpu())

        self.log("AUC-PR/test", auc_pr(precision, recall))

        self.log("AVG-Precision/test", metrics["AveragePrecision"])

        self.logger.experiment.log(
            {"PR Curve/test": wandb.Image(fig), "global_step": self.global_step}
        )

        self.log_confusion_matrix(preds, target)

    def log_confusion_matrix(self, preds, target):

        cm = self.confmat(preds, target.int())

        fig = plot_confusion_matrix(cm.cpu(), self.labels, self.labels)

        self.logger.experiment.log(
            {"Confusion Matrix/test": wandb.Image(fig), "global_step": self.global_step}
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def get_data_module(mimic_path, batch_size, num_workers):
    p = Path(mimic_path)
    dm = MIMICIIIDataModule(p, batch_size, num_workers, pad_batch=True)
    dm.setup()
    return dm


def get_vectors(dim, cache):
    return GloVe(name="6B", dim=dim, cache=cache)


def set_example_input_array(datamodule, model):
    _, x = next(iter(datamodule.train_dataloader()))
    model.example_input_array = x


def add_arguments():

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

    parser.add_argument(
        "--vectors_root",
        help="Where pre-trained vectors are stored",
        type=str,
        default=".data",
    )

    parser.add_argument(
        "--random_seed",
        help="The integer value seed for global random state.",
        type=str,
        default=SEED,
    )

    parser = LSTMClassifier.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    return parser


def main(args):

    parser = add_arguments()

    args = parser.parse_args(args)

    dm = get_data_module(args.mimic_path, args.batch_size, args.num_workers)

    if args.no_vectors:
        vectors = None
    else:
        vectors = get_vectors(args.embed_dim, args.vectors_root)

    hparams = LSTMClassifier.get_model_hparams(args)
    model = LSTMClassifier(vectors, dm.vocab, dm.labels, hparams)

    callbacks = []
    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.default_root_dir,
        filename="AVG-Precision_valid={AVG-Precision/valid:.2f}-epoch={epoch}-step={step}",
        monitor="AVG-Precision/valid",
        mode="max",
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    logger = WandbLogger(project="clinical-longformer", name="LSTM", entity="yass")

    logger.watch(model)

    logger.experiment.config["seed"] = args.random_seed

    seed_everything(args.random_seed, workers=True)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)

    set_example_input_array(dm, model)

    trainer.fit(model, datamodule=dm)

    trainer.test(model, ckpt_path=None, datamodule=dm)


def run():
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
