import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from ..data.module import TransformerMIMICIIIDataModule
from .utils import auc_pr, plot_pr_curve, plot_confusion_matrix


MAX_LENGTH = 512
CLINICAL_BERT_PATH = ".data/model/pretraining"

# Default hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16


class ClinicalBERT(pl.LightningModule):
    """ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission (Huang et al, 2020)

    Returns:
        ClinicalBERT:
    """

    def __init__(
        self,
        clinical_bert_path,
        labels,
        hparams,
    ):

        super().__init__()

        self.labels = labels

        self.lr = hparams["lr"]
        self.hparams = hparams

        # Model
        self.clinical_bert = AutoModelForSequenceClassification.from_pretrained(
            clinical_bert_path, num_labels=len(labels)
        )

        # Metrics
        pr_curve = torchmetrics.PrecisionRecallCurve()

        self.train_pr_curve = pr_curve.clone()
        self.valid_pr_curve = pr_curve.clone()
        self.test_pr_curve = pr_curve.clone()

        self.confmat = torchmetrics.ConfusionMatrix(2, normalize="true")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClinicalBERT")
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        return {k: hparams[k] for k in hparams.keys() & {"lr", "clinical_bert_path"}}

    def forward(self, labels, encodings):
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        token_type_ids = encodings["token_type_ids"]
        return self.clinical_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"AUC-PR/valid": 0})

    def training_step(self, batch, batch_idx):

        y, x = batch

        outputs = self(y, x)

        preds = torch.argmax(outputs.logits, -1)

        loss = outputs.loss

        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.train_pr_curve(outputs["preds"], outputs["target"])

        self.log("AUC-PR/train", auc_pr(precision, recall))

        self.log("Loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        y, x = batch

        outputs = self(y, x)

        preds = torch.argmax(outputs.logits, -1)

        loss = outputs.loss

        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):

        loss = outputs["loss"]

        precision, recall, _ = self.valid_pr_curve(outputs["preds"], outputs["target"])

        self.log("AUC-PR/valid", auc_pr(precision, recall))

        self.log("Loss/valid", loss)

        return loss

    def test_step(self, batch, batch_idx):

        y, x = batch

        outputs = self(y, x)

        preds = torch.argmax(outputs.logits, -1)

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
        return AdamW(self.parameters(), lr=self.lr)


def get_data_module(mimic_path, clinical_bert_path, batch_size, num_workers):
    p = Path(mimic_path)
    tokenizer = AutoTokenizer.from_pretrained(clinical_bert_path)
    dm = TransformerMIMICIIIDataModule(
        p, batch_size, tokenizer, MAX_LENGTH, num_workers
    )
    dm.setup()
    return dm


def set_example_input_array(datamodule, model):
    y, x = next(iter(datamodule.train_dataloader()))
    model.example_input_array = [y, x]


def add_arguments():

    parser = ArgumentParser()

    parser.add_argument(
        dest="mimic_path",
        help="Path containing train/valid/test datasets",
        type=str,
        default=os.getcwd(),
    )

    parser.add_argument(
        dest="clinical_bert_path",
        help="Path containing ClinicalBERT pre-trained model",
        type=str,
        default=CLINICAL_BERT_PATH,
    )

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

    parser = ClinicalBERT.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )

    return parser


def main(args):

    parser = add_arguments()

    args = parser.parse_args(args)

    dm = get_data_module(
        args.mimic_path, args.clinical_bert_path, args.batch_size, args.num_workers
    )

    model = ClinicalBERT(
        args.clinical_bert_path, dm.labels, ClinicalBERT.get_model_hparams(args)
    )

    logger = TensorBoardLogger(
        args.logdir, name="ClinicalBERT", default_hp_metric=False, log_graph=True
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
