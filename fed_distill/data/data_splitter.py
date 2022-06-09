from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


def split_heter(
    dataset: Dataset, num_nodes: int, unif_percentage: float
) -> Tuple[Dataset]:
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

    subsets = []

    for (h, r) in zip(sorted_split, unif_split):
        all_indices = np.concatenate((h, r))
        subset = Subset(dataset, all_indices)
        subset.targets = np.array(dataset.targets)[all_indices]
        subsets.append(subset)

    return tuple(subsets)


def split_by_class(dataset: Dataset, num_nodes: int) -> Tuple[Dataset]:
    return split_heter(dataset, num_nodes, 0.0)
