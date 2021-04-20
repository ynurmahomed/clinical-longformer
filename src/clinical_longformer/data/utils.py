import random

from torch.utils.data import Sampler


class BatchRandomPooledSampler(Sampler):
    """A Sampler that batches examples with text of similar sizes together in order
    to minimize padding. Examples should be in the form of tuples (label, text).
    """
    def __init__(self, data_source, batch_size):

        self.data_source = data_source

        self.batch_size = batch_size

    def __iter__(self):

        indices = [(i, len(s[1])) for i, s in enumerate(self.data_source)]

        random.shuffle(indices)

        pooled_indices = []

        p_size = self.batch_size * 100

        # create pool of indices with similar lengths
        for i in range(0, len(indices), p_size):

            by_len = sorted(indices[i : i + p_size], key=lambda x: x[1])

            pooled_indices.extend(by_len)

        pooled_indices = [x[0] for x in pooled_indices]

        batches = []
        for i in range(0, len(pooled_indices), self.batch_size):
            batches.append(pooled_indices[i : i + self.batch_size])

        return iter(batches)

    def __len__(self):
        return len(self.data_source) // self.batch_size
