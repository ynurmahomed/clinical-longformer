import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from argparse import ArgumentParser
from pathlib import Path
from torchtext.experimental.vectors import GloVe
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.module import AGNNewsDataModule, MIMICIIIDataModule
from .utils import macro_auc_pr, plot_confusion_matrix, plot_pr_curve


# Default hyperparameters
LEARNING_RATE = 1e-2
HIDDEN_DIM = 100
BATCH_SIZE = 64
EMBED_DIM = 300


class LSTMClassifier(pl.LightningModule):
    def __init__(self, vectors, vocab, labels, hparams):
        super().__init__()

        self.labels = labels
        num_class = len(labels)

        hidden_dim = hparams["hidden_dim"]
        self.lr = hparams["lr"]
        self.hparams = hparams

        if vectors is not None:
            pre_trained = vectors(vocab.itos)
            self.embedding = nn.Embedding.from_pretrained(pre_trained)
        else:
            self.embedding = nn.Embedding(len(vocab), hparams["embed_dim"])

        self.lstm = nn.LSTM(hparams["embed_dim"], hidden_dim)

        self.linear = nn.Linear(hidden_dim, num_class)

        self.softmax = nn.LogSoftmax(dim=1)

        self.nll_loss = F.nll_loss

        # Metrics
        pr_curve = torchmetrics.PrecisionRecallCurve(num_class)

        self.train_pr_curve = pr_curve.clone()
        self.valid_pr_curve = pr_curve.clone()
        self.test_pr_curve = pr_curve.clone()

        self.confmat = torchmetrics.ConfusionMatrix(num_class, normalize="true")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DAN")
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument(
            "--embed_dim", type=int, default=EMBED_DIM, choices=[50, 100, 200, 300]
        )
        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        want_keys = {"lr", "hidden_dim", "batch_size", "embed_dim"}
        return {k: hparams[k] for k in hparams.keys() & want_keys}

    def forward(self, x):

        embedding = self.embedding(x)

        out, _ = self.lstm(embedding)

        linear = self.linear(out[-1])

        softmax = self.softmax(linear)

        return softmax

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"AUC-PR (macro)/valid": 0})

    def training_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x)

        loss = self.nll_loss(preds, y)

        return {"loss": loss, "preds": preds.detach(), "target": y}

    def training_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.train_pr_curve(outputs["preds"], outputs["target"])

        auc_pr = macro_auc_pr(precision, recall)

        self.log("Loss/train", loss)

        self.log("AUC-PR (macro)/train", auc_pr)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x)

        loss = self.nll_loss(preds, y)

        return {"loss": loss, "preds": preds.detach(), "target": y}

    def validation_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.valid_pr_curve(outputs["preds"], outputs["target"])

        auc_pr = macro_auc_pr(precision, recall)

        self.log("AUC-PR (macro)/valid", auc_pr)

        self.log("Loss/valid", loss)

        return loss

    def test_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x)

        loss = self.nll_loss(preds, y)

        return {"preds": preds.detach(), "target": y}

    def test_epoch_end(self, outputs):

        y = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        self.log_pr_curve(preds, y)

        self.log_confusion_matrix(preds.argmax(1), y)

    def log_pr_curve(self, preds, y):

        precision, recall, _ = self.test_pr_curve(preds, y)

        fig = plot_pr_curve(precision, recall, self.labels)

        auc_pr = macro_auc_pr(precision, recall)

        self.log("AUC-PR (macro)/test", auc_pr)

        self.logger.experiment.add_figure("PR Curve/test", fig, self.current_epoch)

    def log_confusion_matrix(self, preds, y):

        cm = self.confmat(preds, y)

        fig = plot_confusion_matrix(cm, self.labels, self.labels)

        self.logger.experiment.add_figure(
            "Confusion Matrix/test", fig, self.current_epoch
        )

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self.lr)


def get_data_module(args):
    p = Path(args.mimic_path)
    dm = MIMICIIIDataModule(p, args.batch_size, args.num_workers, pad_batch=True)
    dm.setup()
    return dm


def set_example_input_array(datamodule, model):
    _, x = next(iter(datamodule.train_dataloader()))
    model.example_input_array = x


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

    parser = LSTMClassifier.add_model_specific_args(parser)

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
        vectors = GloVe(name="6B", dim=args.embed_dim)

    hparams = LSTMClassifier.get_model_hparams(args)
    model = LSTMClassifier(vectors, dm.vocab, dm.labels, hparams)

    logger = TensorBoardLogger(
        args.logdir, name="LSTM", default_hp_metric=False, log_graph=True
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
