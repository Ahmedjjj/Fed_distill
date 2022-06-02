from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import torch


class TargetSampler(ABC):
    def __init__(self, batch_size: int, classes: Union[int, Iterable[int]]) -> None:
        self.batch_size = batch_size
        if isinstance(classes, int):
            self.classes = range(classes)
        else:
            self.classes = tuple(set(classes))

    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        raise NotImplementedError()


class BalancedSampler(TargetSampler):
    def __init__(self, batch_size: int, classes: Union[int, Iterable[int]]) -> None:
        super().__init__(batch_size, classes)
        if batch_size % len(self.classes) != 0:
            raise ValueError("Batch size has to be a multiple of the number of classes")

        self.__targets = torch.from_numpy(
            np.repeat(self.classes, batch_size // len(self.classes))
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield self.__targets

class WeightedSampler(TargetSampler):
    def __init__(
        self,
        batch_size: int,
        classes: Union[int, Iterable[int]],
        p: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__(batch_size, classes)
        self.p = p

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.from_numpy(
                np.random.choice(self.classes, self.batch_size, replace=True, p=self.p)
            )


class RandomSampler(WeightedSampler):
    def __init__(self, batch_size: int, classes: Union[int, Iterable[int]]) -> None:
        super().__init__(batch_size, classes, p=None)


def probs_from_labels(labels: Iterable[int]) -> Dict[int, float]:
    counter = Counter(labels)
    probs = dict()
    for k, v in counter.items():
        probs[k] = v / len(labels)
    
    return probs
