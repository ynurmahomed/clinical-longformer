import pandas as pd
import pytorch_lightning as pl
import random
import sys
import torch

from collections import Counter
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import AG_NEWS, YelpReviewPolarity
from torchtext.legacy.datasets.text_classification import TextClassificationDataset
from torchtext.data.utils import get_tokenizer
from .torchtext_experimental import (
    build_vocab,
    sequential_transforms,
    totensor,
    vocab_func,
)
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

        tokenizer = get_tokenizer("basic_english")
        data = self.get_tuples(pd.concat([train, valid, test])[columns])
        self.vocab = build_vocab(data, tokenizer)

        self.text_transform = sequential_transforms(
            tokenizer, vocab_func(self.vocab), totensor(torch.long)
        )

        self.label_transform = totensor(torch.float)

        self.train = TextClassificationDataset(self.vocab, train_data, self.labels)

        self.valid = TextClassificationDataset(self.vocab, valid_data, self.labels)

        self.test = TextClassificationDataset(self.vocab, test_data, self.labels)

    def get_tuples(
        self,
        dataframe,
    ):
        return list(dataframe.itertuples(index=False))

    def collate_fn(self, batch):
        label_list, text_list, offsets = [], [], [0]

        for (_label, _text) in batch:

            label_list.append(self.label_transform(_label))

            txt = self.text_transform(_text)
            text_list.append(txt)
            offsets.append(txt.size(0))

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


class NoteEventsDataset(Dataset):
    def __init__(self, hadm_ids, text, labels):
        self.hadm_ids = hadm_ids
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        return self.hadm_ids[idx], self.labels[idx], self.text[idx]

    def __len__(self):
        return len(self.labels)


class TransformerMIMICIIIDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, tokenizer, num_workers):

        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    def setup(self):

        self.labels = ["Not Readmitted", "Readmitted"]

        train = pd.read_csv(self.path / "train.csv")
        valid = pd.read_csv(self.path / "valid.csv")
        test = pd.read_csv(self.path / "test.csv")

        train_hadm_ids, train_texts, train_labels = (
            train.HADM_ID.to_list(),
            train.TEXT.to_list(),
            train.LABEL.to_list(),
        )
        valid_hadm_ids, valid_texts, valid_labels = (
            valid.HADM_ID.to_list(),
            valid.TEXT.to_list(),
            valid.LABEL.to_list(),
        )
        test_hadm_ids, test_texts, test_labels = (
            test.HADM_ID.to_list(),
            test.TEXT.to_list(),
            test.LABEL.to_list(),
        )

        self.train_dataset = NoteEventsDataset(
            train_hadm_ids, train_texts, train_labels
        )
        self.valid_dataset = NoteEventsDataset(
            valid_hadm_ids, valid_texts, valid_labels
        )
        self.test_dataset = NoteEventsDataset(test_hadm_ids, test_texts, test_labels)

    def collate_fn(self, batch):

        hadm_id_list, label_list, text_list = [], [], []

        for (_hadm_id, _label, _text) in batch:
            hadm_id_list.append(_hadm_id)
            label_list.append(_label)
            text_list.append(_text)

        encoding = self.tokenizer(text_list, padding='max_length', truncation=True)

        encoding_dict = {key: torch.tensor(val) for key, val in encoding.items()}

        return (
            torch.tensor(hadm_id_list, dtype=torch.float),
            torch.tensor(label_list, dtype=torch.float),
            encoding_dict,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )


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


class YelpReviewPolarityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pad_batch=False):

        super().__init__()

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.pad_batch = pad_batch

    def setup(self):

        train, test = YelpReviewPolarity()

        self.vocab = train.get_vocab()

        self.labels = ["Positive", "Negative"]

        self.train = train

        self.valid = test

        self.test = test

        self.label_transform = sequential_transforms(
            lambda l: int(l) - 1, totensor(torch.float)
        )

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
