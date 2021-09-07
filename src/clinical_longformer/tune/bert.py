import logging
import math
import pytorch_lightning as pl
import os
import ray
import sys
import torch

from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from ..model.bert import (
    BertPretrainedModule,
    get_data_module,
    add_arguments,
    setup_lr_scheduler,
)
from .utils import trial_name_string


def train_tune(config, args):

    dm = get_data_module(
        args.mimic_path,
        args.bert_pretrained_path,
        config["batch_size"],
        args.max_length,
        args.num_workers,
    )

    model = BertPretrainedModule(args.bert_pretrained_path, dm.labels, config)

    if config["lr_scheduler_type"] is not None:
        args.lr_scheduler_type = config["lr_scheduler_type"]
        args.warmup_proportion = config["warmup_proportion"]
        setup_lr_scheduler(model, dm, args)

    logger = TensorBoardLogger(
        tune.get_trial_dir(), name="", version=".", default_hp_metric=False
    )

    tune_callback = TuneReportCallback(
        {"loss": "Loss/valid", "AUC-PR": "AUC-PR/valid"}, on="validation_end"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=math.ceil(args.gpus),
        logger=logger,
        callbacks=[tune_callback],
    )

    trainer.fit(model, datamodule=dm)


def tune_clinical_bert(args):

    config = {
        "lr": tune.loguniform(2e-5, 5e-5),
        "batch_size": tune.choice([16, 24, 32, 40]),
        "lr_scheduler_type": tune.choice([None, "linear", "polynomial"]),
        "warmup_proportion": tune.loguniform(0.1, 0.3),
        "attention_probs_dropout_prob": tune.loguniform(0.1, 0.5),
        "hidden_dropout_prob": tune.loguniform(0.1, 0.5),
    }

    scheduler = ASHAScheduler(max_t=args.max_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            "lr",
            "batch_size",
            "lr_scheduler_type",
            "warmup_proportion",
        ],
        metric_columns=["loss", "AUC-PR", "training_iteration"],
    )

    trainable = tune.with_parameters(
        train_tune,
        args=args,
    )

    tune.run(
        trainable,
        resources_per_trial={"cpu": 1, "gpu": args.gpus},
        metric="AUC-PR",
        mode="max",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=args.local_dir,
        name="clinical_bert_tune",
        trial_dirname_creator=trial_name_string,
    )


def main(args):

    parser = add_arguments()

    parser.add_argument(
        "--num_samples",
        help="Number of times to sample from the hyperparameter space. Defaults to 1.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--local_dir",
        help="Local dir to save training results to. Defaults to ``./ray_results``.",
        type=str,
        default="./ray_results",
    )

    args = parser.parse_args()

    ray.init(log_to_driver=False)

    tune_clinical_bert(args)


def run():
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
