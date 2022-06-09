import logging
import math
from abc import ABC, abstractmethod
from os import PathLike
import os
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class GrowingDataset(ABC, Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        base_dataset: Optional[Dataset[Tuple[torch.Tensor, torch.Tensor]]] = None,
        new_batch_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.base = base_dataset if base_dataset else dict()
        self.transform = new_batch_transform if new_batch_transform else lambda x: x

        self._images = torch.empty(0)
        self._labels = torch.empty(0).type(torch.LongTensor)

    def __len__(self) -> int:
        return len(self.base) + len(self._labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < len(self.base):
            return self.base[index]
        else:
            shifted_index = index - len(self.base)
            image, label = self._images[shifted_index], self._labels[shifted_index]
            return self.transform(image), label

    def _add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self._images = torch.cat((self._images, images))
        self._labels = torch.cat((self._labels, labels))

    @abstractmethod
    def grow(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
    
    def save(self, path: PathLike) -> None:
        torch.save({"images": self._images, "labels": self._labels}, path)

    def load(self, path: PathLike) -> bool:
        if os.path.exists(path):
            self._images, self._labels = torch.load(path).values()
            logger.info("Successfully loaded %i images", len(self._labels))
            return True
        else:
            logger.info("File does not exist, not loading any images...")
            return False

class DeepInversionDataset(GrowingDataset):
    def __init__(
        self,
        di: Union[
            Iterator[Tuple[torch.Tensor, torch.Tensor]],
            Iterable[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
        ],
        base_dataset: Optional[Dataset[Tuple[torch.Tensor, torch.Tensor]]] = None,
        new_batch_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(base_dataset, new_batch_transform)
        if isinstance(di, Iterator):
            di = (di,)
        self.di = tuple(di)
        self._cur_teacher = 0

    def grow(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("Generating new batch from teacher %i", self._cur_teacher)
        images, labels = next(self.di[self._cur_teacher])
        images = images.cpu().detach()
        labels = labels.cpu().detach()
        self._add_batch(images, labels)
        self._cur_teacher = (self._cur_teacher + 1) % len(self.di)
        return images, labels


class GrowingDatasetDataLoader(Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: GrowingDataset,
        epoch_samples: int,
        new_batches_per_epoch: int = 2,
        batch_size: int = 1,
        shuffle: bool = True,
        **kwargs
    ):
        self.dataset = dataset
        self._epoch_num_batches = math.floor(epoch_samples / batch_size)
        logger.info("Number of batches per epoch: %i", self._epoch_num_batches)

        self._generation_steps = {}
        if new_batches_per_epoch > 0:
            self._generation_steps = set(range(
                0,
                self._epoch_num_batches,
                math.ceil(self._epoch_num_batches / new_batches_per_epoch),
            ))

        logger.info(
            "Generating new batches at steps %s", str(self._generation_steps))
        

        self._loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
        )
 
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        loader_iter = iter(self._loader)

        for step in range(self._epoch_num_batches):
            if step in self._generation_steps:
                images, labels = self.dataset.grow()
            else:
                try:
                    images, labels = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self._loader)
                    images, labels = next(loader_iter)

            yield images, labels

