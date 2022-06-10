from typing import Iterable, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

def index_split_heter(
    dataset: Dataset, num_nodes: int, unif_percentage: float
) -> Tuple[Tuple[int]]:
    assert hasattr(dataset, "targets")

    # Shuffle in order to get a random uniformly distributed
    targets = np.random.permutation(range(len(dataset)))
    unif_targets_indxs, heter_target_idxs = np.split(
        targets, [int(len(dataset) * unif_percentage)]
    )

    unif_split = np.array_split(unif_targets_indxs, num_nodes)

    sorted_targets = np.argsort(dataset.targets)
    sorted_targets = sorted_targets[np.in1d(sorted_targets, heter_target_idxs)]
    sorted_split = np.array_split(sorted_targets, num_nodes)

    return tuple(tuple(s.tolist() + h.tolist()) for (s,h) in zip(sorted_split, unif_split))


def extract_subset(dataset: Dataset, indices: Iterable[int]) -> Dataset:
    subset = Subset(dataset, indices)
    if hasattr(dataset, "targets"):
        subset.targets = np.array(dataset.targets)[indices]
    return subset


def split_heter(
    dataset: Dataset, num_nodes: int, unif_percentage: float
) -> Tuple[Dataset]:
    return tuple(
        extract_subset(dataset, indices)
        for indices in index_split_heter(dataset, num_nodes, unif_percentage)
    )


def split_by_class(dataset: Dataset, num_nodes: int) -> Tuple[Dataset]:
    return split_heter(dataset, num_nodes, 0.0)
