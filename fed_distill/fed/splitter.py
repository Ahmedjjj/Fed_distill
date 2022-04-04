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
        targets, unif_targets = np.split(
            np.copy(self.base_dataset.targets), [self.len_heter]
        )
        sorted_targets = np.argsort(targets[: self.len_heter])
        rng = np.random.default_rng(42)
        unif_targets = rng.permutation(unif_targets)
        subsets = []
        sorted_split = np.array_split(sorted_targets, num_nodes)
        unif_split = np.array_split(unif_targets, num_nodes)

        for (h, r) in zip(sorted_split, unif_split):
            all_indices = np.concatenate((h, r))
            subset = Subset(self.base_dataset, all_indices)
            subset.targets = np.array(self.base_dataset.targets)[all_indices]
            subsets.append(subset)

        return subsets

