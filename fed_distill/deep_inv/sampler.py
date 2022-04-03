import random
from abc import ABC, abstractmethod

import torch
from fed_distill import cifar10


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
