from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import Dataset, Subset
import numpy as np


class DataSplitter(ABC):
    @abstractmethod
    def split(self, num_nodes: int) -> List[Dataset]:
        raise NotImplementedError


class HeterogenousDistribution(DataSplitter):
    def __init__(self, base_dataset: Dataset, unif_percentage=0.1):
        self.base_dataset = base_dataset
        self.unif_percentage = unif_percentage
        self.len_heter = len(base_dataset) - int(len(base_dataset) * unif_percentage)

    def split(self, num_nodes: int) -> List[Dataset]:
        targets, unif_targets = np.split(self.base_dataset.targets, [self.len_heter])
        sorted_targets = np.argsort(targets[: self.len_heter])
        np.random.shuffle(unif_targets)
        return [
            Subset(self.base_dataset, np.concatenate((h, r)))
            for (h, r) in zip(
                np.array_split(sorted_targets, num_nodes),
                np.array_split(unif_targets, num_nodes),
            )
        ]
