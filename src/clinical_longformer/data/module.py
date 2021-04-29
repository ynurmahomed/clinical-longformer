import pandas as pd
import pytorch_lightning as pl
import random
import sys
import torch

from collections import Counter
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
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

from .utils import BatchRandomPooledSampler


class MIMICIIIDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, num_workers, pad_batch=False):
        """MIMIC-III DataModule.

        Args:
            path (Path): MIMIC-III dataset location.
            batch_size (int): Batch size.
            pad_batch (bool, optional): If sequences inside batch should be padded.
                If set to `True`, sequences in batch will be padded with 0 to match
                the longest sequence. Also sequences with similar lengths will be
                batched together to minimize padding. Defaults to False.
        """

        super().__init__()

        self.path = path
        self.vocab = None
        self.label_vocab = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_batch = pad_batch

    def setup(self):

        columns = ["LABEL", "TEXT"]

        self.labels = ["Not Readmitted", "Readmitted"]

        train = pd.read_csv(self.path / "train.csv")
        valid = pd.read_csv(self.path / "valid.csv")
        test = pd.read_csv(self.path / "test.csv")

        train_data = self.get_tuples(train[columns])
        valid_data = self.get_tuples(valid[columns])
        test_data = self.get_tuples(test[columns])

        tokenizer = basic_english_normalize()
        data = self.get_tuples(train[columns])
        self.vocab = build_vocab(data, tokenizer)

        token_transform = sequential_transforms(
            tokenizer, vocab_func(self.vocab), totensor(torch.long)
        )

        label_transform = totensor(torch.float)

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

    def collate_padded(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(_label)
            text_list.append(_text)
        return torch.tensor(label_list), pad_sequence(text_list)

    def get_dataloader(self, dataset, shuffle=False):

        if self.pad_batch:
            batch_sampler = BatchRandomPooledSampler(dataset, self.batch_size)
            collate_fn = self.collate_padded
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
            )
        else:
            collate_fn = self.collate_fn
            return DataLoader(
                dataset,
                self.batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                shuffle=shuffle,
            )

    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.valid)

    def test_dataloader(self):
        return self.get_dataloader(self.test)


class AGNNewsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pad_batch=False):

        super().__init__()

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.pad_batch = pad_batch

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

    def collate_padded(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_transform(_label))
            text_list.append(self.text_transform(_text))
        return torch.tensor(label_list), pad_sequence(text_list)

    def get_dataloader(self, dataset, shuffle=False):

        if self.pad_batch:
            batch_sampler = BatchRandomPooledSampler(dataset, self.batch_size)
            collate_fn = self.collate_padded
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
            )
        else:
            collate_fn = self.collate_fn
            return DataLoader(
                dataset,
                self.batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                shuffle=shuffle,
            )

    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.valid)

    def test_dataloader(self):
        return self.get_dataloader(self.test)
