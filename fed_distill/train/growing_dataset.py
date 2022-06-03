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
        self.base_dataset = base_dataset
        self.transform = transform
        self.last_batch = None
        self.new_images = None
        self.new_labels = None
        self.new_batch_min_index = len(self.base_dataset)

    def __getitem__(self, k) -> torch.Tensor:
        if k < self.new_batch_min_index:
            image, label = self.base_dataset[k]
        else: 
            image, label = self.new_images[k - self.new_batch_min_index], self.new_labels[k - self.new_batch_min_index]
            image = self.transform(image)

        if isinstance(label, int):
            label = torch.tensor(label)
        return image, label

    def __len__(self) -> int:
        return len(self.base_dataset) + (self.new_images.shape[0] if self.new_images is not None else 0) 

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        
        self.new_images = torch.cat((self.new_images, images)) if self.new_images is not None else images
        self.new_labels = torch.cat((self.new_labels, labels)) if self.new_labels is not None else labels
        self.last_batch = images, labels

    def get_last_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.last_batch[0]), self.last_batch[1]

    def save(self, path: PathLike) -> None:
        torch.save(
            {"images": self.new_images, "labels": self.new_labels}, path
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
        super().__init__(base_dataset=base_dataset, transform=transform)

        logger.info("Generating %i initial batches", num_initial_batches)
        all_images, all_labels = [], []
        for batch in range(num_initial_batches):
            logger.info("Generating initial batch %i", batch)
            images, labels = deep_inversion.compute_batch(teacher_net)
            all_images.append(images.detach().cpu())
            all_labels.append(labels.detach().cpu())

        self.add_batch(torch.cat(all_images), torch.cat(all_labels))