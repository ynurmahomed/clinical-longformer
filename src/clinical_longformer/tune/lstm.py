import math
import pytorch_lightning as pl
import sys
import torch

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from ..model.lstm import LSTMClassifier, get_data_module, get_vectors, add_arguments
from .utils import trial_name_string


def train_tune(config, mimic_path, vectors_root, num_workers, max_epochs, gpus, logdir):

    dm = get_data_module(mimic_path, config["batch_size"], num_workers)

    vectors = get_vectors(config["embed_dim"], vectors_root)

    model = LSTMClassifier(vectors, dm.vocab, dm.labels, config)

    logger = TensorBoardLogger(
        tune.get_trial_dir(), name="", version=".", default_hp_metric=False
    )

    tune_callback = TuneReportCallback(
        {"loss": "Loss/valid", "AUC-PR": "AUC-PR/valid"}, on="validation_end"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=math.ceil(gpus),
        logger=logger,
        callbacks=[tune_callback],
    )

    trainer.fit(model, datamodule=dm)


def tune_lstm(args):

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_dim": tune.choice([100, 200, 300]),
        "batch_size": tune.choice([32, 64, 128]),
        "embed_dim": tune.choice([50, 100, 200, 300]),
        "dropout": tune.loguniform(0.3, 0.7),
    }

    scheduler = ASHAScheduler(max_t=args.max_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            "lr",
            "hidden_dim",
            "batch_size",
            "embed_dim",
            "dropout",
        ],
        metric_columns=["loss", "AUC-PR", "training_iteration"],
    )

    trainable = tune.with_parameters(
        train_tune,
        mimic_path=args.mimic_path,
        vectors_root=args.vectors_root,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        logdir=args.logdir,
    )

    tune.run(
        trainable,
        resources_per_trial={"cpu": 1, "gpu": args.gpus},
        metric="loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=args.local_dir,
        name="lstm_tune",
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

    tune_lstm(args)


def run():
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
