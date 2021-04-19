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
from torchtext.experimental.functional import (
    sequential_transforms,
    vocab_func,
    totensor,
)
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.vocab import Vocab


NUM_WORKERS = 4


class MIMICIIIDataModule(pl.LightningDataModule):
    def __init__(self, path, note_length, batch_size):

        super().__init__()

        self.path = path
        self.note_length = note_length
        self.vocab = None
        self.label_vocab = None
        self.batch_size = batch_size

    def setup(self):

        columns = ["LABEL", "TEXT"]

        self.labels = ["Not Readmitted", "Readmitted"]

        train = pd.read_csv(self.path / str(self.note_length) / "train.csv")
        valid = pd.read_csv(self.path / str(self.note_length) / "valid.csv")
        test = pd.read_csv(self.path / str(self.note_length) / "test.csv")

        train_data = self.get_tuples(train[columns])
        valid_data = self.get_tuples(valid[columns])
        test_data = self.get_tuples(test[columns])

        tokenizer = basic_english_normalize()
        data = self.get_tuples(train[columns])
        self.vocab = build_vocab(data, tokenizer)

        token_transform = sequential_transforms(
            tokenizer, vocab_func(self.vocab), totensor(torch.long)
        )

        label_transform = totensor(torch.long)

        transforms = (label_transform, token_transform)

        self.train = TextClassificationDataset(train_data, self.vocab, transforms)

        self.valid = TextClassificationDataset(valid_data, self.vocab, transforms)

        self.test = TextClassificationDataset(test_data, self.vocab, transforms)

    def get_tuples(
        self,
        dataframe,
    ):
        return list(dataframe.itertuples(index=False))

    def collate_fn(self, batch):
        label_list, text_list, offsets = [], [], [0]

        for (_label, _text) in batch:

            label_list.append(_label)

            text_list.append(_text)
            offsets.append(_text.size(0))

        return (
            torch.tensor(label_list),
            torch.cat(text_list),
            torch.tensor(offsets[:-1]).cumsum(dim=0),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=NUM_WORKERS,
        )


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

    def collate_fn(self, batch):
        label_list, text_list, offsets = [], [], [0]

        for (_label, _text) in batch:

            label_list.append(self.label_transform(_label))

            processed_text = self.text_transform(_text)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        return (
            torch.tensor(label_list),
            torch.cat(text_list),
            torch.tensor(offsets[:-1]).cumsum(dim=0),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=NUM_WORKERS,
        )
