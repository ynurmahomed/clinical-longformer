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


def collate_batch(batch):

    label_list, text_list, offsets = [], [], [0]

    for (_label, _text) in batch:

        label_list.append(_label)
        processed_text = _text

        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    return (
        torch.tensor(label_list),
        torch.cat(text_list),
        torch.tensor(offsets[:-1]).cumsum(dim=0),
    )


class AGNNewsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.vocab = None
        self.label_vocab = None
        self.batch_size = batch_size

    def setup(self):

        train, test = AG_NEWS()

        self.vocab = train.get_vocab()

        counter = Counter(["World", "Sports", "Business", "Sci/Tech"])
        self.label_vocab = Vocab(counter, specials=["<unk>"])

        self.train = train

        self.valid = test

        self.test = test

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=collate_batch,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=collate_batch,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_batch,
            num_workers=NUM_WORKERS,
        )
