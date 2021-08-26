"""This module contains function copied from torchtext.experimental"""
import torch

from torchtext.vocab import build_vocab_from_iterator


def build_vocab(data, transforms):
    def apply_transforms(data):
        for _, line in data:
            yield transforms(line)

    return build_vocab_from_iterator(apply_transforms(data))


def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def totensor(dtype):
    def func(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return func


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func
