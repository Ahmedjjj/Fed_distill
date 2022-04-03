import random
from abc import ABC, abstractmethod
from typing import List
from collections import Counter

import numpy as np
import torch


class TargetSampler(ABC):
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()


class BalancedSampler(TargetSampler):
    def __init__(self, batch_size: int, num_classes: int) -> None:
        if not batch_size % num_classes == 0:
            raise ValueError("Batch size has to be a multiple of the number of classes")
        self.targets = torch.tensor(
            [[i] * (batch_size // 10) for i in range(num_classes)]
        ).reshape(-1)

    def __iter__(self):
        while True:
            yield self.targets


class RandomSampler(TargetSampler):
    def __init__(self, batch_size: int, num_classes: int = 10) -> None:
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __iter__(self):
        while True:
            yield torch.LongTensor(
                [
                    random.randint(0, self.num_classes - 1)
                    for _ in range(self.batch_size)
                ]
            )


class WeightedSampler(TargetSampler):
    def __init__(self, batch_size: int, labels: List[int]) -> None:
        label_counts = Counter(labels)
        sorted_counts = [label_counts[k] for k in sorted(label_counts)]
        self.probabilities = np.array(label_counts) / len(labels)
        self.drawing_labels = len(sorted_counts)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield torch.from_numpy(
                np.random.Generator.choice(
                    self.drawing_labels,
                    size=self.batch_size,
                    replace=True,
                    p=self.probabilities,
                )
            )
