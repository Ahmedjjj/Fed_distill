from os import PathLike
from torch.utils.data import Dataset, TensorDataset
import torch
from typing import Callable, Tuple
from torch import nn
from fed_distill.deep_inv.deep_inv import NonAdaptiveDeepInversion
import logging

logger = logging.getLogger(__name__)


class GrowingDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform: Callable) -> None:
        self.dataset = base_dataset
        self.transform = transform
        self.last_batch = None
        self.new_batch_min_index = len(self.dataset)

    def __getitem__(self, k) -> torch.Tensor:
        images, labels = self.dataset[k]
        return (
            (images, labels)
            if k > self.new_batch_min_index
            else (self.transform(images), labels)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.dataset += TensorDataset(images.detach().cpu(), labels.detach().cpu())
        self.last_batch = images, labels

    def get_last_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.last_batch[0]), self.last_batch[1]

    def save(self, path: PathLike) -> None:
        new_images = []
        new_labels = []
        for k in range(self.new_batch_min_index, len(self.dataset)):
            images, labels = self.dataset[k]
            new_images.append(images)
            new_labels.append(labels)

        torch.save(
            {"images": torch.cat(new_images), "labels": torch.cat(new_labels)}, path
        )


class DeepInversionGrowingDataset(GrowingDataset):
    def __init__(
        self,
        base_dataset: Dataset,
        teacher_net: nn.Module,
        deep_inversion: NonAdaptiveDeepInversion,
        num_initial_batches: int,
        transform: Callable,
    ) -> None:
        """
        Generate initial batches using (non_adaptive) deep inversion

        Args:
            deep_inversion (DeepInversion): _description_
            num_initial_batches (int): _description_
        """
        logger.info("Generating %i initial batches", num_initial_batches)
        all_images, all_labels = [], []
        for batch in range(num_initial_batches):
            logger.info("Generating initial batch %i", batch)

            images, labels = deep_inversion.compute_batch(teacher_net)
            all_images.append(images.detach().cpu())
            all_labels.append(labels.detach().cpu())

        initial_dataset = base_dataset + TensorDataset(
            torch.cat(images), torch.cat(labels)
        )
        super().__init__(base_dataset=initial_dataset, transform=transform)
        self.new_batch_min_index = len(base_dataset)
