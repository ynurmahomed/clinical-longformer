import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from pathlib import Path
from torchtext.experimental.vectors import GloVe
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler

from ..data.module import AGNNewsDataModule, MIMICIIIDataModule


BATCH_SIZE = 50
EMBED_DIM = 300
HIDDEN_DIM = 150

# Default hyperparameters
LEARNING_RATE = 1e-3


class LSTMClassifier(pl.LightningModule):
    def __init__(self, vectors, vocab, embed_dim, labels, hparams):
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
            self.embedding = nn.Embedding(len(vocab), embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim)

        self.linear = nn.Linear(hidden_dim, num_class)

        self.softmax = nn.LogSoftmax(dim=1)

        self.nll_loss = F.nll_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DAN")
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        want_keys = {"lr", "hidden_dim"}
        return {k: hparams[k] for k in hparams.keys() & want_keys}

    def forward(self, x):

        embedding = self.embedding(x)

        out, hidden = self.lstm(embedding)

        linear = self.linear(out[-1])

        softmax = self.softmax(linear)

        return softmax

    def training_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x)

        loss = self.nll_loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):

        loss = outputs["loss"]

        self.log("Loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x)

        loss = self.nll_loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):

        loss = outputs["loss"]

        self.log("Loss/valid", loss)

        return loss

    def test_step(self, batch, batch_idx):

        y, x = batch

        preds = self(x)

        loss = self.nll_loss(preds, y)

        return {"loss": loss, "preds": preds, "target": y}

    def test_epoch_end(self, outputs):

        loss = outputs[0]["loss"]

        self.log("Loss/test", loss)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self.lr)


def get_data_module(args):
    p = Path(args.mimic_path)
    # dm = MIMICIIIDataModule(p, BATCH_SIZE, args.num_workers, pad_batch=True)
    dm = AGNNewsDataModule(BATCH_SIZE, args.num_workers, pad_batch=True)
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
        vectors = GloVe(name="6B", dim=EMBED_DIM)

    hparams = LSTMClassifier.get_model_hparams(args)
    model = LSTMClassifier(vectors, dm.vocab, EMBED_DIM, dm.labels, hparams)

    logger = TensorBoardLogger(
        "lightning_logs", name="LSTM", default_hp_metric=False, log_graph=True
    )

    profiler = PyTorchProfiler(profile_memory=True)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, profiler=profiler)

    set_example_input_array(dm, model)

    trainer.fit(model, datamodule=dm)

    trainer.test(model, ckpt_path=None, datamodule=dm)


def run():
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
