"""
This module contains classes that endlessly yield tensors of labels, used for choosing batches for deep inversion
"""
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import torch


class TargetSampler(ABC):
    def __init__(self, batch_size: int, classes: Union[int, Iterable[int]]) -> None:
        self.batch_size = batch_size
        if isinstance(classes, int):
            self.classes = tuple(range(classes))
        else:
            self.classes = tuple(set(classes))

    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        raise NotImplementedError()


class BalancedSampler(TargetSampler):
    def __init__(self, batch_size: int, classes: Union[int, Iterable[int]]) -> None:
        """
        A balanced sampler returns the same amount of targets from each class

        Args:
            batch_size (int): length of returned vector
            classes (Union[int, Iterable[int]]): classes to sample from, if int samples from range(classes)
        Raises:
            ValueError: if batch_size is not divisible by the number of classes
        """
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
        """
        Sample labels according to a probability distribution
        Args:
            batch_size (int): length of returned vector
            classes (Union[int, Iterable[int]]): classes to sample from, if int samples from range(classes)
            p (Optional[Iterable[float]], optional): probability over the range of classes. Defaults to None.
        """
        super().__init__(batch_size, classes)
        self.p = tuple(p) if p else None

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.from_numpy(
                np.random.choice(self.classes, self.batch_size, replace=True, p=self.p)
            )


class RandomSampler(WeightedSampler):
    # Sample uniformly at random
    def __init__(self, batch_size: int, classes: Union[int, Iterable[int]]) -> None:
        super().__init__(batch_size, classes, p=None)


def probs_from_labels(labels: Iterable[int]) -> Dict[int, float]:
    """
    From a list of labels returns the frequency of each label

    Args:
        labels (Iterable[int]): input labels

    Returns:
        Dict[int, float]: Dict of the form {class:frequency}
    """
    labels = tuple(labels)
    counter = Counter(labels)
    probs = dict()
    for k, v in counter.items():
        probs[k] = v / len(labels)

    return probs
