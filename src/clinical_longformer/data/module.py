import pandas as pd
import pytorch_lightning as pl
import sys
import torch

from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.datasets.text_classification import (
    build_vocab,
    TextClassificationDataset,
)
from torchtext.experimental.functional import sequential_transforms, vocab_func
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.vocab import Vocab

NUM_WORKERS = 4
DATA_PATH = Path("/home/yassin/Projects/AI/Project/tedtalks")


def get_collate_batch(label_transform, text_transform):
    def func(batch):

        label_list, text_list, offsets = [], [], [0]

        for (_label, _text) in batch:

            label_list.append(label_transform(_label))
            processed_text = text_transform(_text)

            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        return (
            torch.tensor(label_list),
            torch.cat(text_list),
            torch.tensor(offsets[:-1]).cumsum(dim=0),
        )

    return func


class AGNNewsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self):

        train, test = AG_NEWS()

        self.vocab = train.get_vocab()

        self.labels = ["World", "Sports", "Business", "Sci/Tech"]

        self.train = train

        self.valid = test

        self.test = test

        self.label_transform = lambda l: int(l) - 1

        self.text_transform = lambda t: t

    def train_dataloader(self):
        collate_fn = get_collate_batch(self.label_transform, self.text_transform)
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        collate_fn = get_collate_batch(self.label_transform, self.text_transform)
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        collate_fn = get_collate_batch(self.label_transform, self.text_transform)
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )
