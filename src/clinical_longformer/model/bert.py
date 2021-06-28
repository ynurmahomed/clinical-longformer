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
from .metrics import ClinicalBERTBinnedPRCurve
from .utils import auc_pr, plot_pr_curve, plot_confusion_matrix


MAX_LENGTH = 512
BERT_PRETRAINED_PATH = ".data/model/pretraining"

# Default hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16


class BertPretrainedModule(pl.LightningModule):
    """Lightning module for any bert pre-trained model from the transformers library.

    Returns:
        BertPretrainedModule:
    """

    def __init__(
        self,
        bert_pretrained_path,
        labels,
        hparams,
    ):

        super().__init__()

        self.labels = labels

        self.save_hyperparameters(hparams, ignore=["bert_pretrained_path", "labels"])

        # BERT type Model
        self.bert_pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            bert_pretrained_path, num_labels=1
        )

        # 3 layer classifier like in Huang, K., Altosaar, J., & Ranganath, R. (2019).
        self.bert_pretrained_model.classifier = nn.Sequential(
            nn.Linear(self.bert_pretrained_model.config.hidden_size, 2048),
            nn.Linear(2048, 768),
            nn.Linear(768, self.bert_pretrained_model.config.num_labels)
        )

        self.sigmoid = nn.Sigmoid()

        self.bce_loss = F.binary_cross_entropy

        # Metrics
        pr_curve = ClinicalBERTBinnedPRCurve()

        self.train_pr_curve = pr_curve.clone()
        self.valid_pr_curve = pr_curve.clone()
        self.test_pr_curve = pr_curve.clone()

        self.confmat = torchmetrics.ConfusionMatrix(2, normalize="true")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClinicalBERT")
        parser.add_argument(
            "--bert_pretrained_path",
            help="Path containing ClinicalBERT pre-trained model",
            type=str,
            default=BERT_PRETRAINED_PATH,
        )
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        return {
            k: hparams[k]
            for k in hparams.keys() & {"lr", "bert_pretrained_path", "batch_size"}
        }

    def forward(self, labels, encodings):

        input_ids = encodings["input_ids"]

        attention_mask = encodings["attention_mask"]

        token_type_ids = encodings["token_type_ids"]

        output = self.bert_pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # Unsqueeze labels so dimensions align when computing loss in
            # (BertForSequenceClassification#forward).
            labels=labels.unsqueeze(1),
        )

        sigmoid = self.sigmoid(output.logits)

        return sigmoid

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"AUC-PR/valid": 0})

    def training_step(self, batch, batch_idx):

        hadm_id, y, x = batch

        preds = self(y, x).squeeze()

        loss = self.bce_loss(preds, y)

        self.log("Loss/train", loss)

        return {"loss": loss, "hadm_id": hadm_id, "preds": preds, "target": y}

    def training_epoch_end(self, outputs):

        hadm_ids = torch.cat([o["hadm_id"] for o in outputs])

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        precision, recall, _ = self.train_pr_curve(hadm_ids, preds, target)

        self.log("AUC-PR/train", auc_pr(precision, recall))

    def validation_step(self, batch, batch_idx):

        hadm_id, y, x = batch

        preds = self(y, x).squeeze()

        loss = self.bce_loss(preds, y)

        self.log("Loss/valid", loss)

        return {"loss": loss, "hadm_id": hadm_id, "preds": preds, "target": y}

    def validation_epoch_end(self, outputs):

        hadm_ids = torch.cat([o["hadm_id"] for o in outputs])

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        precision, recall, _ = self.valid_pr_curve(hadm_ids, preds, target)

        self.log("AUC-PR/valid", auc_pr(precision, recall))

    def test_step(self, batch, batch_idx):

        hadm_id, y, x = batch

        preds = self(y, x).squeeze()

        return {"hadm_id": hadm_id, "preds": preds, "target": y}

    def test_epoch_end(self, outputs):

        hadm_ids = torch.cat([o["hadm_id"] for o in outputs])

        y = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        self.log_pr_curve(hadm_ids, preds, y)

        self.log_confusion_matrix(preds, y)

    def log_pr_curve(self, hadm_ids, logits, y):

        precision, recall, _ = self.test_pr_curve(hadm_ids, logits, y)

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
        return AdamW(self.parameters(), lr=self.hparams.lr)


def get_data_module(mimic_path, bert_pretrained_path, batch_size, num_workers):
    p = Path(mimic_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_path)
    dm = TransformerMIMICIIIDataModule(
        p, batch_size, tokenizer, MAX_LENGTH, num_workers
    )
    dm.setup()
    return dm


def set_example_input_array(datamodule, model):
    _, y, x = next(iter(datamodule.train_dataloader()))
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

    parser = BertPretrainedModule.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    return parser


def main(args):

    parser = add_arguments()

    args = parser.parse_args(args)

    dm = get_data_module(
        args.mimic_path, args.bert_pretrained_path, args.batch_size, args.num_workers
    )

    model = BertPretrainedModule(
        args.bert_pretrained_path, dm.labels, BertPretrainedModule.get_model_hparams(args)
    )

    logger = TensorBoardLogger(
        args.logdir, name="ClinicalBERT", default_hp_metric=False
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
