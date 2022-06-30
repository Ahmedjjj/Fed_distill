""" 
This module contains helper functions in order to split a dataset among nodes according to a given strategy
"""
from typing import Iterable, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


def index_split_heter(
    dataset: Dataset, num_nodes: int, unif_percentage: float
) -> Tuple[Tuple[int]]:
    """
    Perform a heterogeneous distribution "by index", i.e return indices of the splits in the dataset
    rather than the subsets

    Args:
        dataset (Dataset): Dataset to split, should have a "targets" attribute, similar to the torchvision
        datasets
        num_nodes (int): Number of nodes to split the dataset on
        unif_percentage (float): Should be in the range [0..1], this percentage of the dataset will be split
        uniformly at random between the nodes

    Returns:
        Tuple[Tuple[int]]: For each node return the indices of its split
    """
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

    return tuple(
        tuple(s.tolist() + h.tolist()) for (s, h) in zip(sorted_split, unif_split)
    )


def extract_subset(dataset: Dataset, indices: Iterable[int]) -> Dataset:
    """
    Extract a subset from a given dataset, with the given indices

    Args:
        dataset (Dataset): Parent dataset
        indices (Iterable[int]): Indices to extract

    Returns:
        Dataset: Child dataset, if the parent has a "target" attribute,
        the corresponding labels are also copied
    """
    subset = Subset(dataset, indices)
    if hasattr(dataset, "targets"):
        subset.targets = np.array(dataset.targets)[indices]
    return subset


def split_heter(
    dataset: Dataset, num_nodes: int, unif_percentage: float
) -> Tuple[Dataset]:
    """
    Performs the actions corresponding to the above two functions
    i.e:
    . Get indices for each node
    . Extract the corresponding subset for each node

    Args:
        dataset (Dataset): Dataset to split, should have a "targets" attribute, similar to the torchvision
        datasets
        num_nodes (int): Number of nodes to split the dataset on
        unif_percentage (float): Should be in the range [0..1], this percentage of the dataset will be split
        uniformly at random between the nodes

    Returns:
        Tuple[Dataset]: Local datasets
    """
    return tuple(
        extract_subset(dataset, indices)
        for indices in index_split_heter(dataset, num_nodes, unif_percentage)
    )


def split_by_class(dataset: Dataset, num_nodes: int) -> Tuple[Dataset]:
    """
    Perform a fully heterogeneous split, i.e nodes will have not intersecting classes

    Args:
        dataset (Dataset): Dataset to split, should have a "targets" attribute, similar to the torchvision
        datasets
        num_nodes (int): Number of nodes to split the dataset on
    Returns:
       Tuple[Dataset]: Local datasets
    """
    return split_heter(dataset, num_nodes, 0.0)
