import logging
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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    SchedulerType,
    get_scheduler,
)
from torchmetrics import AveragePrecision, MetricCollection, Precision, Recall

from ..data.module import TransformerMIMICIIIDataModule
from .metrics import ClinicalBERTBinnedPRCurve
from .utils import auc_pr, plot_pr_curve, plot_confusion_matrix

_logger = logging.getLogger(__name__)

SEED = 42
MAX_LENGTH = 1024
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
            bert_pretrained_path,
            num_labels=1,
            attention_window=self.hparams.attention_window,
            attention_probs_dropout_prob=self.hparams.attention_probs_dropout_prob,
            hidden_dropout_prob=self.hparams.hidden_dropout_prob,
        )

        # 3 layer classifier like in Huang, K., Altosaar, J., & Ranganath, R. (2019).
        self.bert_pretrained_model.classifier = nn.Sequential(
            nn.Linear(self.bert_pretrained_model.config.hidden_size, 2048),
            nn.Linear(2048, 768),
            nn.Linear(768, self.bert_pretrained_model.config.num_labels),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()

        # Metrics
        pr_curve = ClinicalBERTBinnedPRCurve()
        # Using separate metric collection as inputs for the following
        # are different
        metrics = MetricCollection([AveragePrecision(), Precision(), Recall()])

        self.train_pr_curve = pr_curve.clone()

        self.valid_pr_curve = pr_curve.clone()
        self.valid_metrics = metrics.clone()

        self.test_pr_curve = pr_curve.clone()
        self.test_metrics = metrics.clone()

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
        parser.add_argument(
            "--max_length",
            help="Max input length",
            type=int,
            default=MAX_LENGTH,
            choices=[1024, 2048, 4096],
        )
        parser.add_argument("--lr", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default=None,
            help="The scheduler type to use.",
            # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10%% of training.",
        )
        parser.add_argument(
            "--attention_probs_dropout_prob",
            default=0.1,
            type=float,
            help="The dropout ratio for the attention probabilities.",
        )
        parser.add_argument(
            "--hidden_dropout_prob",
            default=0.1,
            type=float,
            help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        )
        parser.add_argument(
            "--attention_window",
            default=512,
            type=int,
            help="Size of an attention window around each token.",
        )

        return parent_parser

    @staticmethod
    def get_model_hparams(namespace):
        hparams = vars(namespace)
        return {
            k: hparams[k]
            for k in hparams.keys()
            & {
                "lr",
                "bert_pretrained_path",
                "batch_size",
                "lr_scheduler_type",
                "warmup_proportion",
                "attention_probs_dropout_prob",
                "hidden_dropout_prob",
                "accumulate_grad_batches",
                "attention_window",
            }
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

        return output.logits

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):

        hadm_id, y, x = batch

        preds = self(y, x).reshape(y.shape)

        loss = self.bce_loss(preds, y)

        self.log("Loss/train", loss)

        return {
            "loss": loss,
            "hadm_id": hadm_id.detach(),
            "preds": preds.detach(),
            "target": y.detach(),
        }

    def training_epoch_end(self, outputs):

        hadm_ids = torch.cat([o["hadm_id"] for o in outputs])

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        precision, recall, _ = self.train_pr_curve(hadm_ids, preds, target)

        self.log("AUC-PR/train", auc_pr(precision, recall))

    def validation_step(self, batch, batch_idx):

        hadm_id, y, x = batch

        preds = self(y, x).reshape(y.shape)

        loss = self.bce_loss(preds, y)

        self.log("Loss/valid", loss)

        return {"loss": loss, "hadm_id": hadm_id, "preds": preds, "target": y}

    def validation_epoch_end(self, outputs):

        hadm_ids = torch.cat([o["hadm_id"] for o in outputs])

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        precision, recall, _ = self.valid_pr_curve(hadm_ids, preds, target)

        metrics = self.valid_metrics(preds, target.int())

        self.log("AUC-PR/valid", auc_pr(precision, recall))

        self.log("AVG-Precision/valid", metrics["AveragePrecision"])

        self.log("Precision/valid", metrics["Precision"])

        self.log("Recall/valid", metrics["Recall"])

    def test_step(self, batch, batch_idx):

        hadm_id, y, x = batch

        preds = self(y, x).reshape(y.shape)

        return {"hadm_id": hadm_id, "preds": preds, "target": y}

    def test_epoch_end(self, outputs):

        hadm_ids = torch.cat([o["hadm_id"] for o in outputs])

        target = torch.cat([o["target"] for o in outputs])

        preds = torch.cat([o["preds"] for o in outputs])

        self.log_test_metrics(hadm_ids, preds, target)

    def log_test_metrics(self, hadm_ids, preds, target):

        precision, recall, _ = self.test_pr_curve(hadm_ids, preds, target)

        fig = plot_pr_curve(precision, recall)

        metrics = self.test_metrics(preds, target.int())

        self.log("AUC-PR/test", auc_pr(precision, recall))

        self.log("AVG-Precision/test", metrics["AveragePrecision"])

        self.log("Precision/test", metrics["Precision"])

        self.log("Recall/test", metrics["Recall"])

        self.logger.experiment.log(
            {"PR Curve/test": wandb.Image(fig), "global_step": self.global_step}
        )

        self.log_confusion_matrix(preds, target)

    def log_confusion_matrix(self, preds, y):

        cm = self.confmat(preds, y.int())

        fig = plot_confusion_matrix(cm, self.labels, self.labels)

        self.logger.experiment.log(
            {"Confusion Matrix/test": wandb.Image(fig), "global_step": self.global_step}
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)

        if self.hparams.lr_scheduler_type is not None:
            lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

        return optimizer


def get_data_module(
    mimic_path, bert_pretrained_path, batch_size, max_length, num_workers
):
    p = Path(mimic_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_path)
    dm = TransformerMIMICIIIDataModule(
        p, batch_size, tokenizer, max_length, num_workers
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
        "--random_seed",
        help="The integer value seed for global random state.",
        type=str,
        default=SEED,
    )

    parser = BertPretrainedModule.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    return parser


def setup_lr_scheduler(model, datamodule, args):
    n_batches = len(datamodule.train_dataloader())
    n_devices = args.gpus if args.gpus else 1  # cpu or tpu_cores
    num_training_steps = (
        n_batches // args.accumulate_grad_batches * args.max_epochs // n_devices
    )
    model.num_training_steps = num_training_steps
    model.num_warmup_steps = int(args.warmup_proportion * num_training_steps)

    _logger.info(
        f"n_batches={n_batches}\n batch_size * gpus={args.batch_size * max(1, (args.gpus or 0))}\n max_epochs={args.max_epochs}\n num_training_steps={num_training_steps}\n num_warmup_steps={model.num_warmup_steps}"
    )


def main(args):

    parser = add_arguments()

    args = parser.parse_args(args)

    dm = get_data_module(
        args.mimic_path,
        args.bert_pretrained_path,
        args.batch_size,
        args.max_length,
        args.num_workers,
    )

    model = BertPretrainedModule(
        args.bert_pretrained_path,
        dm.labels,
        BertPretrainedModule.get_model_hparams(args),
    )

    callbacks = []

    # Setup learning rate scheduler
    if args.lr_scheduler_type is not None:
        setup_lr_scheduler(model, dm, args)
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Setup early stopping
    # callbacks.append(EarlyStopping("Loss/valid", stopping_threshold=0.65))

    name = "Longformer-" + str(args.max_length)
    logger = WandbLogger(project="clinical-longformer", name=name, entity="yass")

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
